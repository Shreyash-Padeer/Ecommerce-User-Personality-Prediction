# Personality Prediction in E-Commerce
> The main aim of the project is to understand the personality of the users for a Ecommerce website based on their interation with the prodcucts. The problem has been represented in terms of graph and **Graph Neural Networks(GNNs)** has been applied to understand their relation, and make the prediction.

## Abstract
Modern recommendation systems in e-commerce often lack a deeper understanding of the user. While click and purchase history offer behavior signals, they don't explicitly capture  **personality traits** that drive user decisions. Knowing whether a user is a budget-shopper or fashion-forward can refine recommendations and personalize the shopping experience.

## Datasets
As given in the training dataset, the original human-annotated labels for users along with features related to user and prodcuts has been given. Along with it, seperate files has been given representing the user product interaction. Datasets can be accessed in dataset folder. Shapes for the test dataset has been also given.


## Mathematical Description
Design a Graph Neural Network which:
- Uses a bipartite graph of users and products.
- Leverages features of users (`X_U`) and products (`X_P`).
- Learns to predict **multi-label personality vectors** `y ∈ ℝ^d` for each user.

Formally, given:
- `G = (X_U ∈ ℝ^{m × d1}, X_P ∈ ℝ^{n × d2}, E ⊆ U × P)` – the bipartite graph.
- `Y ∈ ℝ^{m × d}` – personality labels for users.
Train a model to minimize prediction error on unseen users.

## Methodology
### Training and Preprocessing
- Modeled the e-commerce domain as a **heterogeneous graph**:
  - User nodes `U` with features `X_U`
  - Product nodes `P` with features `X_P`
  - Edges `E` representing interactions like purchases or clicks
- Implemented a **Graph Neural Network (GNN)** in **PyTorch**, capable of:
  - Encoding user-product relationships via message passing
  - Combining node features with neighborhood information
  - Outputting multi-dimensional personality predictions for users
- Explored and ran experiments on other variants such as:
  - **GCN**, **GraphSAGE**, and **GAT** architectures
  - Dropout, layer normalization.
- Used **binary cross-entropy loss** for multi-label personality prediction

### Experiments Summary

| Model     | Accuracy@k | Micro-F1 |
|-----------|------------|----------|
| GCN       | 78.08%      | 0.72     |
| GraphSAGE | 76.67%      | 0.71     |
| GAT       | 74.56%      | 0.69     |

## Results
- Achieved **high accuracy** on two differnt test set user personality vectors with **78.08%** and **85.67%**.
- **Graph Convolution Network** seems to work best over all the networks and achiieved the best result on it. 
- The model generalized well to **unseen users**, indicating effective learning from user-product interactions


## Folder Structure

Overview:
- environemnt.yml: Create the conda environemnt given all the libraries.
- src/ : This folder contains all the python and bash scripts used to run the experiments.
- datasets/: Contains the dataset for running the model
- models/ : Contains the best model trained on the dataset

## Language and Libraries Used
- Programming Language: Python
- Libraries: Pytorch, Numpy, Pandas, PyTorch-Geometric


## Setup Environment
``` bash
conda create -n gnn python=3.10
conda activate gnn
pip install -r requirements.txt
```

## Run Code
Run the test script and train script 

- Train Data: In dataset folder
- Model path: Path to folder 
- Output File: Output File must be in CSV Format.

Train Script
``` bash
cd src
bash train.sh <path_to_train_graph> <output_model_file_path>
```

Test Script
```bash
cd src
bash test.sh <path_to_test_graph> <path_to_model> <output_file_path>
```

## References

- https://www.geeksforgeeks.org/deep-learning/graph-neural-networks-with-pytorch/

## Developer and License

This project is a part of the academic course assignment submitted at the end of the course "COL761- Data Mining", taught by the Prof. Sayan Ranu.

Developer Name: Somesh Agrawal

