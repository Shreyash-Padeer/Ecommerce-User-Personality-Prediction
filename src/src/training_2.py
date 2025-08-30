import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm
from torch_geometric.utils import add_self_loops
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class userShoppingGCNModel(nn.Module):
    def __init__(self, user_feat_dim, product_feat_dim, hidden_dim, out_dim):
        super().__init__()
        self.user_projection = nn.Linear(user_feat_dim, hidden_dim)
        self.prod_projection = nn.Linear(product_feat_dim, hidden_dim)

        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, user_feats, prod_feats, edge_index):
        user_x = self.user_projection(user_feats)
        prod_x = self.prod_projection(prod_feats)
        x = torch.cat([user_x, prod_x], dim=0)  
        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.gcn2(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.gcn3(x, edge_index))
        x = self.out(x)
        return x
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train the user shopping GCN model")
    parser.add_argument('train_folder',help="Path to Train Folder")
    parser.add_argument('output_path', help="Path to save the output model")
    args = parser.parse_args()

    train_folder_path = args.train_folder
    output_file_path = args.output_path

    user_features = np.load(train_folder_path + "/user_features.npy")
    product_features = np.load(train_folder_path + "/product_features.npy")
    user_product_edges = np.load(train_folder_path + "/user_product.npy")
    user_labels = np.load(train_folder_path + "/label.npy")

    num_users = user_features.shape[0]
    num_products = product_features.shape[0]
    edge_index = torch.tensor(user_product_edges, dtype=torch.long)
    user_feats = torch.tensor(user_features, dtype=torch.float32)
    prod_feats = torch.tensor(product_features, dtype=torch.float32)
    user_labels = torch.from_numpy(user_labels).long()
    user_labels = user_labels.type(torch.float32)

    edge_index = edge_index.t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_users+num_products)

    user_indices = torch.arange(num_users)
    train_idx, val_idx = train_test_split(user_indices, test_size=0.1, random_state=42)
    train_idx = torch.tensor(train_idx, dtype=torch.long).to(device)
    val_idx = torch.tensor(val_idx, dtype=torch.long).to(device)

    src, dst = edge_index
    val_user_mask = torch.zeros(num_users + num_products, dtype=torch.bool)
    val_user_mask[val_idx] = True

    mask = ~(val_user_mask[src] | val_user_mask[dst])
    train_edge_index = edge_index[:, mask]
    validation_edge_mask =  edge_index[:,~(mask)]

    model = userShoppingGCNModel(
        user_feat_dim=user_feats.shape[1],
        product_feat_dim=prod_feats.shape[1],
        hidden_dim=128,
        out_dim=user_labels.shape[1]
    ).to(device)

    user_feats = user_feats.to(device)
    prod_feats = prod_feats.to(device)
    user_labels = user_labels.to(device)
    train_edge_index = train_edge_index.to(device)
    validation_edge_mask = validation_edge_mask.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 400
    best_val_f1 = 0.0
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()

        logits = model(user_feats, prod_feats, train_edge_index)[:num_users]

        train_logits = logits[train_idx]
        train_labels = user_labels[train_idx]

        loss = criterion(train_logits, train_labels)
        loss.backward()
        optimizer.step()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_logits = model(user_feats, prod_feats, validation_edge_mask)[:num_users][val_idx]
            val_labels = user_labels[val_idx]

            val_probs = torch.sigmoid(val_logits)
            val_preds = (val_probs > 0.5).long().cpu().numpy()
            val_true = val_labels.cpu().long().numpy()
            val_f1 = f1_score(val_true, val_preds, average="weighted")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                state = {
                    'model_state_dict': model.state_dict(),
                    'out_dim': user_labels.shape[1],
                }
                torch.save(state, output_file_path)

