import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm
from torch_geometric.utils import add_self_loops
import pandas as pd

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

    parser = argparse.ArgumentParser(description="Test the user shopping GCN model")
    parser.add_argument('test_folder',help="Path to Train Folder")
    parser.add_argument('model_path', help="Path to save the output model")
    parser.add_argument('output_path', help="Path to save the csv file")
    args = parser.parse_args()

    test_folder_path = args.test_folder
    model_path = args.model_path
    submission_file = args.output_path

    user_features = np.load(test_folder_path + "/user_features.npy")
    product_features = np.load(test_folder_path + "/product_features.npy")
    user_product_edges = np.load(test_folder_path + "/user_product.npy")


    num_users = user_features.shape[0]
    num_products = product_features.shape[0]
    edge_index = torch.tensor(user_product_edges, dtype=torch.long)
    user_feats = torch.tensor(user_features, dtype=torch.float32)
    prod_feats = torch.tensor(product_features, dtype=torch.float32)

    edge_index = edge_index.t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_users+num_products)

    checkpoint = torch.load(model_path, map_location=device)
    out_dim = checkpoint['out_dim']

    model = userShoppingGCNModel(
        user_feat_dim=user_feats.shape[1],
        product_feat_dim=prod_feats.shape[1],
        hidden_dim=128,
        out_dim=out_dim
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    user_feats = user_feats.to(device)
    prod_feats = prod_feats.to(device)
    edge_index = edge_index.to(device)
    
    with torch.no_grad():
        test_logits = model(user_feats, prod_feats, edge_index)[:num_users]
        test_probs = torch.sigmoid(test_logits)
        test_preds = (test_probs >= 0.5).int().cpu().numpy()
        pd.DataFrame(test_preds).to_csv(submission_file, index=False, header=False)
