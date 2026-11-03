import time
start = time.time()

import pickle
import os
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import combinations
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset

from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_edge, add_remaining_self_loops
from torch.utils.data import random_split
from torch_geometric.utils.augmentation import mask_feature
print(f"Libraries imported in {(time.time() - start):.2f} seconds")
'''
    PREPROCESSING   
'''
def read_data(data_path, score_path, score_type, dist_threshold=10, family=False):
    """
    Function to load all the data.
    Assumes both the graph structure of the PDBs and the superposition data of those PDBs are in the same directory
    """
    # Initialize variables
    graph_files = []
    graph_data = {}
    superposition_df = pd.DataFrame()

    # Traverse the directory
    for file in sorted(os.listdir(pdb_path)):
        # Load graphs
        if file.endswith(".pkl"):
            filepath = os.path.join(pdb_path, file)
            graph_files.append(filepath)
            # Keep name, index and family for labeling
            if family:
                temp = file.split('.')[0]
                family_name = temp.split('-')[0]
                protein_name = temp.split('-')[1]
                graph_data[protein_name] = [len(graph_files) - 1, family_name]
            else:
                protein_name = file.split('.')[0]
                graph_data[protein_name] = [len(graph_files) - 1]

    # Load superpositions (single CSV)
    for file in os.listdir(score_path):
        if file.endswith(".csv"):
            filepath = os.path.join(score_path, file)
            superposition_df = pd.read_csv(filepath)

    # Check consistency between graphs and dataframe
    loaded_pdbs = set(graph_data.keys())
    expected_pairs = {frozenset(pair) for pair in combinations(loaded_pdbs, 2)}
    filtered_df = superposition_df[superposition_df["PDB_1"].str.split(".").str[0].isin(loaded_pdbs) & superposition_df["PDB_2"].str.split(".").str[0].isin(loaded_pdbs)]
    df_pairs = {frozenset((row["PDB_1"].split(".")[0], row["PDB_2"].split(".")[0])) for _, row in filtered_df.iterrows()}
    missing_pairs = expected_pairs - df_pairs
    assert not missing_pairs, f"Missing superposition pairs: {missing_pairs}"

    # Initialize matrix (pass dataframe to tensor form for use with model etc.)
    matrix = torch.zeros((len(graph_files), len(graph_files)))
    # Fill the matrix
    for _, row in superposition_df.iterrows():
        pdb1 = row['PDB_1'].split(".")[0]
        pdb2 = row['PDB_2'].split(".")[0]
        if pdb1 not in graph_data or pdb2 not in graph_data:
            continue
        idx1 = graph_data[pdb1][0]
        idx2 = graph_data[pdb2][0]
        matrix[idx1, idx2] = row[score_type]
        matrix[idx2, idx1] = row[score_type]

    # Create the dataset that will dynamically load each graph
    dataset = GraphDataset(graph_files, graph_data, dist_threshold)

    return dataset, matrix

def build_graph(raw_graph, name, metadata, dist_threshold=10, verbose=1):
    """
     Generate a pytorch geometric graph structure, ready to train GNNs and such
     It assumes that the edges of the graph are symmetric (that should otherwise be enforced)
     dist_threshold ensures only the relations between CVs that are at most at n distance are taken into account
     """
    # Collect, for each node, a list [start of CV, end of CV, length/module of CV]
    node_features = []
    for cv in raw_graph[0]:
        # Assume cv IDs start at zero and are of the form 0, 1, 2,..., N. Without gaps.
        node_features.append([cv.start, cv.end, cv.cvl])

    # Collect, for each edge, a list [distance between CVs, angle1, angle2]
    edge_index = []
    edge_attr = []

    for node1, neighbors in raw_graph[1].items():
        for node2, edge in neighbors.items():
            if edge.dist < dist_threshold:
                edge_index.append([node1, node2])
                edge_attr.append([1/edge.dist, edge.angle1, edge.angle2]) # Inverse of distance since the less, the better

    # Node: with shape (num_nodes, 7), where x,y,z coordinates are flattened
    flattened = [np.concatenate([a, b, [c]]) for a, b, c in node_features]
    node_features = np.array(flattened)
    #Edge: with shape (num_edges, 2) for edge_index and (num_edges, 3) for edge_attr
    edge_index = np.array(edge_index)
    edge_attr = np.array(edge_attr)
    if verbose == 1:
        print(f"Graph {name}: Nodes -> {node_features.shape} Edges -> {edge_index.shape}, {edge_attr.shape}")

    # Convert to pytorch tensors
    x = torch.tensor(node_features, dtype=torch.float)
    # Transpose so that shape is [2, n_edges] and not [n_edges, 2], necessary for the edge index list
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() # Ensure it is contiguous in memory
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Create pytorch geometric graph object
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Attach PDB information
    graph.name = name
    if len(metadata) == 2: # Families present
        graph.idx = metadata[0]
        graph.family = metadata[1]
    else:
        graph.idx = metadata[0]
        graph.family = None

    return graph

class GraphDataset(Dataset):
    def __init__(self, graph_files, graph_data, dist_threshold=10):
        super().__init__()
        self.graph_files = graph_files # List with the filepaths of all PDBs
        self.graph_data = graph_data # Data based on name: PDB_name -> [idx, family]
        self.dist_threshold = dist_threshold # Maximum distance to be considered neighbors
        self.index_to_info = {}
        # Reverse lookup: idx -> (PDB_name, metadata)
        for name, metadata in graph_data.items():
            idx = metadata[0]
            self.index_to_info[idx] = (name, metadata)

    def len(self):
        return len(self.graph_files)

    def get(self, idx):
        # Lazy-load the graph
        filepath = self.graph_files[idx]
        with open(filepath, "rb") as f:
            raw_graph = pickle.load(f)

        # Create the pytorch geometric graph object
        name, metadata = self.index_to_info[idx]
        graph = build_graph(raw_graph, name, metadata, dist_threshold=self.dist_threshold, verbose=0)

        return graph

'''
    MODEL
'''
# Encoder
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.2):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.dropout = dropout
        # set add_self_loops to false to handle edge attributes
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=False)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.conv3 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)

    def forward(self, x, edge_index, edge_weight, batch):
        # Handle edge attributes. This layer only accepts a scalar. rn we pass the distance only
        # We could make any reduction to convert the original 3 features into a single one, but otherwise we will need to change layers
        # Set distance of self_loops to 0, since they are edges from and to the same node
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value=0.0, num_nodes=x.size(0))
        # Obtain node embeddings. Add dropout to reduce overfitting
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index, edge_weight)
        # Pool them together
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        return x

'''
    LOSSES
'''
def augment(data, edge_drop_p=0.2, feat_mask_p=0.2): # maybe make these probabilities a bit random
    """
    Returns an augmented version of the input graph, in order to obtain a loss without ground truth
    Requires a single graph
    See https://arxiv.org/abs/2010.13902
    """
    data_aug = data.clone()

    # Edge dropout and feature masking
    edge_index, edge_mask = dropout_edge(data_aug.edge_index, p=edge_drop_p)
    x, _ = mask_feature(data_aug.x, p=feat_mask_p, mode='col')

    # Return new graph
    data_aug = data.clone()
    data_aug.x = x
    data_aug.edge_index = edge_index
    # Modify also the edge attributes to maintain coherent shape
    if data_aug.edge_attr is not None:
        data_aug.edge_attr = data_aug.edge_attr[edge_mask]

    return data_aug

def info_nce_loss(z1, z2, idx1, idx2, family1, family2, temperature=0.5): # do not forget temperature, it could be used somewhere or modified
    """
    Custom loss based on the augmentation idea (see augment() function)
    Both z1 and z2 are augmented versions of the same graph
    Should punish whenever they are different, since they come from the same graph and should be similar
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    similarity = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)

    return F.cross_entropy(similarity, labels)

class ScoreLoss(nn.Module):
    """
    Loss function based in any score that computes the similarity between two PDBs (a priori) and the cosine similarity of the embeddings of these PDBs
    Both the score and the cosine similarity should be as equal as possible
    Warning: It is still unclear to me whether the ranges mean the same in both scores:
        Does 0 (opposed vectors), 0.5 (orthogonal vectors) and 1 (aligned vectors), in cosine similarity,
        equal 0 (absolute dissimilarity) and 1 (absolute similarity) in the other score?
    """
    def __init__(self, matrix, device):
        super(ScoreLoss, self).__init__()
        # Pass the score matrix to device, so that the model can have access to it
        self.matrix = matrix.to(device)

    def forward(self, embed, idx, family):
        # Create the pairs, aligned by index
        emb1 = embed[:-1]
        emb2 = embed[1:]
        # Compute the cosine similarity
        pred_score = F.cosine_similarity(emb1, emb2, dim=-1)
        pred_score = F.relu(pred_score) # Beginning at zero option
        #pred_score = (pred_score + 1.0) / 2.0 # Rescale option
        # Compute loss
        true_score = self.matrix[idx[:-1], idx[1:]]
        loss = F.mse_loss(pred_score, true_score)

        return loss

class FamilyContrastiveLoss(nn.Module):
    """
    Loss function based on the assumption that PDBs from the same families should be similar while different from PDBs from other families
    Contrastive loss, where families are the labels to categorize similarity
    'margin' determines the point at which two PDBs from different families are different enough and the penalization stops
    """
    def __init__(self, margin=0.2):
        super(FamilyContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embed1, embed2, idx1, idx2, family1, family2):
        similarity = F.cosine_similarity(embed1, embed2, dim=-1)

        if family1 == family2:
            return 1.0 - similarity
        else:
            # max(0, similarity-margin) with relu
            return F.relu(similarity - self.margin)



'''
    TRAINING AND TESTING
'''
def train(model, train_loader, val_loader, criterion, device, epochs=10, eval_every=5, lr=0.01, writer=None, save_model=False, model_path="graph_embed.pth"):
    # Hyperparameters
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Investigate AdamW, weight decay and so
    # Do not forget the relation between patience and eval_every, maintain it at 5 epochs, maybe it should be enforced by code
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1) # 1 (patience) * 5 (eval_every) = 5 epochs
    best_loss = float('inf')
    global_step = 0

    # Begin training
    for epoch in range(0, epochs):
        # Initialize variables
        start = time.time()
        model.train()
        epoch_loss = 0

        for data in train_loader:
            data = data.to(device)

            # Forward pass
            optimizer.zero_grad()
            z = model(data.x, data.edge_index, data.edge_attr[:, 0], data.batch)

            # Update loss
            train_loss = criterion(z, data.idx, data.family)
            train_loss.backward()
            optimizer.step()
            epoch_loss += train_loss.item()

            # Step-level logging
            if writer is not None:
                writer.add_scalar("Train Loss/train_step", train_loss.item(), global_step)
            global_step += 1

        # Epoch loss
        avg_train_loss = epoch_loss / len(train_loader)

        # Epoch-level logging
        elapsed = time.time() - start
        if writer is not None:
            writer.add_scalar("Train Loss/train_epoch", avg_train_loss, epoch+1)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch+1)
            writer.add_scalar("Time/epoch_seconds", elapsed, epoch+1)
            if epoch % 10 == 0: # Avoid doing it too much since it can be computationally expensive
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param, epoch+1)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Elapsed Time: {elapsed:.2f} seconds")

        # Evaluate every eval_every epochs
        if (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs:
            start = time.time()
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)

                    # Forward pass
                    z = model(data.x, data.edge_index, data.edge_attr[:, 0], data.batch)

                    # Update loss
                    val_loss = criterion(z, data.idx, data.family)
                    total_val_loss += val_loss.item()

                avg_val_loss = total_val_loss / len(val_loader)
                scheduler.step(avg_val_loss)

                # Logging
                elapsed = time.time() - start
                if writer is not None:
                    writer.add_scalar("Validation Loss/validation_epoch", avg_val_loss, epoch+1)
                    writer.add_scalar("Time/validation_seconds", elapsed, epoch+1)
                print(f"Validation Loss: {avg_val_loss:.4f}, Elapsed Time: {elapsed:.2f} seconds")

                # Save our best model, given validation loss (more trustworthy but misses some epochs)
                if save_model and avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    torch.save(model.state_dict(), model_path)
                    print(f"New best model saved in {model_path} (Validation Loss {best_loss:.4f})")

def test(model, loader, device, criterion, writer=None, load=False, model_path="graph_embed.pth", save=False):
    # Load model weights
    if load:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")

    if save:
        embeddings = []
        families = []
        indices = []
        names = []

    # Hyperparameters
    start = time.time()
    model = model.to(device)
    model.eval()
    total_loss = 0

    # Begin testing
    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # Forward pass
            z = model(data.x, data.edge_index, data.edge_attr[:, 0], data.batch)

            # Update loss
            loss = criterion(z, data.idx, data.family)
            total_loss += loss.item()

            if save:
                for emb, family, idx, name in zip(z.detach().cpu(), data.family, data.idx.cpu(), data.name):
                    embeddings.append(emb)
                    families.append(family)
                    indices.append(idx.item())
                    names.append(name)

    avg_test_loss = total_loss / len(loader)

    # Logging
    elapsed = time.time() - start
    if writer is not None:
        # Epoch 0 since we do not iterate
        writer.add_scalar("Loss/test", avg_test_loss, 0)
        writer.add_scalar("Time/test", elapsed, 0)
    print(f"Test Loss: {avg_test_loss:.4f}, Elapsed Time: {elapsed:.2f} seconds")

    if save:
        return {"embeddings": torch.stack(embeddings), "families": families, "indices": indices, "names": names}

'''
    MAIN    
'''
'''
print('-'*50)
print("Starting main execution...")
main_start = time.time()
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define paths
#pdb_path = "/localdata1/arnau/graph_database/annotation/"
#score_path = "/localdata1/arnau/graph_database/annotation/"
#output_path = "/localdata1/arnau/graph_database/log/"
pdb_path = "/cri4/arnau/repo/test/annotation/"
score_path = "/cri4/arnau/repo/test/score/"
output_path = "/cri4/arnau/repo/test/logs/"

# Load graph dataset and dataframe of scores
start = time.time()
graph_database, matrix_score = read_data(pdb_path, score_path, score_type="Max_TM", dist_threshold=10, family=True)
print(f"Data loaded in {(time.time() - start):.2f} seconds")

# Split data
'''
'''
start = time.time()
train_size = int(0.4 * len(graph_database))
val_size = int(0.3 * len(graph_database))
test_size = len(graph_database) - (train_size + val_size)
assert train_size >= 2 and val_size >= 2 and test_size >= 2, "All data splits must contain at least two elements"
generator = torch.Generator().manual_seed(42)
train_data, validation_data, test_data = random_split(graph_database, lengths=[train_size, val_size, test_size], generator=generator)
print(f"Data split in {(time.time() - start):.2f} seconds")
'''
'''
def split_graphs(graphs, train_size=0.7, val_size=0.15, test_size=0.15):
    assert train_size + val_size + test_size == 1, "Data splits should be equal to 1"
    # Extract family labels and indices
    families = [graph.family for graph in graphs]
    indices = list(range(len(graphs)))
    print(families)

    # First split: train/temp
    train_idx, temp_idx = train_test_split(indices, train_size=train_size, stratify=families, random_state=42)
    temp_families = [families[i] for i in temp_idx]
    # Second split: validation/test
    val_idx, test_idx = train_test_split(temp_idx, test_size=test_size, stratify=temp_families, random_state=42)

    assert len(train_idx) >= 2 and len(val_idx) >= 2 and len(test_idx) >= 2, "All data splits must contain at least two elements"

    # Create PyTorch datasets
    train_data = Subset(graph_database, train_idx)
    validation_data = Subset(graph_database, val_idx)
    test_data = Subset(graph_database, test_idx)

    print(len(train_data), len(validation_data), len(test_data))
    return train_data, validation_data, test_data

train_data, validation_data, test_data = split_graphs(graph_database, train_size=0.7, val_size=0.15, test_size=0.15)
print(train_data, validation_data, test_data)
# Create dataloaders
start = time.time()
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=16, shuffle=False)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
print(f"Dataloaders created in {(time.time() - start):.2f} seconds")


# TensorBoard writer
start = time.time()
run_name = datetime.now().strftime("%d%m%Y_%H%M%S")
writer = SummaryWriter(log_dir=f"{output_path}{run_name}")
print(f"TensorBoard writer initialized in {(time.time() - start):.2f} seconds")

# Initialize model and loss
start = time.time()
sample_graph = graph_database[0] # maybe loading a graph here is slow, should we hardcode the input number?
basic_model = GCN(in_channels=sample_graph.x.shape[1], hidden_channels=128)
criterion = ScoreLoss(matrix=matrix_score, device=device)
print(f"Model and loss function initialized in {(time.time() - start):.2f} seconds")

# Learning
print("Starting training...")
model_path = f"{output_path}{run_name}.pth"
train(basic_model, train_loader, validation_loader, criterion=criterion, epochs=10, lr=0.01, device=device, writer=writer, save_model=True, model_path=model_path)
save = True
output_data = test(basic_model, test_loader, device=device, criterion=criterion, writer=writer, load=True, model_path=model_path, save=True)
if save:
    with open(f"{output_path}/output_all_data.pkl", "wb") as file:
        pickle.dump(output_data, file)

print(f"Total execution time: {(time.time() - main_start):.2f} seconds")
'''
