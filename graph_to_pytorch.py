import time
import random
import argparse
import json
import numpy as np
from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
from aleph_classes_graph import AlephPDB

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GINEConv, global_max_pool, global_mean_pool
from torch.utils.tensorboard import SummaryWriter
from retrieval_metrics import embed_graphs, family_sensitivity



def read_selection(path):
    rows = [tuple(line.split()) for line in Path(path).read_text().splitlines() if line.strip()]
    if any(len(row) != 2 for row in rows):
        raise ValueError(f"Expected 'SID family' rows in {path}")
    if len({sid for sid, family in rows}) != len(rows):
        raise ValueError(f"Duplicate SIDs in {path}")
    return rows


def graph_to_pytorch(data_path, output_path, dist_threshold=10.0, representation="cv", annotation_dir=None):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    selection = []
    excluded = []
    for path in sorted(Path(data_path).glob("*/*")):
        if not path.is_file():
            continue
        name = path.name.removesuffix(".pkl")
        name = Path(name).stem
        family = path.parent.name
        graph = build_graph(path, name, family, dist_threshold, representation=representation, annotation_dir=annotation_dir)
        if graph.num_nodes == 0:
            excluded.append(f"{name} {family}")
            continue
        torch.save(graph, output_path / f"{name}--{family}.pt")
        selection.append(f"{name} {family}")
    (output_path / "selection.txt").write_text("".join(row + "\n" for row in selection))
    (output_path / "excluded.txt").write_text("".join(row + "\n" for row in excluded))
    return selection


def load_annotation(path, family="", annotation_dir=None):
    path = Path(path).resolve()
    if path.suffix.lower() == ".pkl":
        return AlephPDB.load_annotation(path)
    saved = Path(annotation_dir).resolve() / family / f"{path.name}.pkl" if annotation_dir is not None else None
    if saved is not None and saved.is_file():
        return AlephPDB.load_annotation(saved)
    with TemporaryDirectory(prefix="aleph-annotation-") as directory:
        with chdir(directory):
            annotation = AlephPDB(str(path), directory)
    if saved is not None:
        annotation.folder = str(saved.parent)
        annotation.save_annotation(saved)
    annotation.read_graphs()
    return annotation


def build_graph(path, name, family, dist_threshold=10.0, representation="cv", annotation_dir=None):
    builders = {"cv": build_graph_cv, "sse": build_graph_sse}
    if representation not in builders:
        raise ValueError("Use representation='cv' or representation='sse'")
    annotation = load_annotation(path, family, annotation_dir)
    graph = builders[representation](annotation, dist_threshold=dist_threshold)
    graph.name = name
    graph.family = family
    return graph


def build_graph_cv(annotation, dist_threshold=10.0):
    """One non-coil CV per node, using the standalone annotation classes."""
    cvs = annotation.get_cvs(include_coil=False)
    index = {cv.cvid: i for i, cv in enumerate(cvs)}
    node_features = [[annotation.raw_cvs[cv.cvid][1]] for cv in cvs]
    edge_index, edge_attr = [], []
    for cvid in index:
        for other, edge in annotation.distances_between_cv_dict.get(cvid, {}).items():
            if other in index and edge.dist <= dist_threshold:
                edge_index.append([index[cvid], index[other]])
                edge_attr.append([edge.dist, edge.angle1, edge.angle2, edge.weight,
                                  edge.fragment_count])
    return Data(x=torch.tensor(node_features, dtype=torch.float32).reshape(-1, 1),
                edge_index=torch.tensor(edge_index, dtype=torch.long).reshape(-1, 2).T.contiguous(),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float32).reshape(-1, 5))


def build_graph_sse(annotation, dist_threshold=10.0):
    """One non-coil fragment per node; connect by the closest CV-pair distance."""
    fragments = [fragment for fragment in annotation.get_sse_fragments() if fragment.cvids]
    index = {fragment.fragid: i for i, fragment in enumerate(fragments)}
    vectors = {cv[0]: cv[2] - cv[3] for cv in annotation.raw_cvs}
    node_features = []
    for fragment in fragments:
        cvs = np.asarray([vectors[cvid] for cvid in fragment.cvids])
        cv_sum, cv_mean = cvs.sum(axis=0), cvs.mean(axis=0)
        node_features.append([
            np.mean(fragment.cvls), np.linalg.norm(cv_sum), fragment.vecLength[0],
            np.linalg.norm(cv_mean),
        ])
    edge_index, edge_attr = [], []
    for fragid in index:
        for other, edge in annotation.distances_between_fragments.get(fragid, {}).items():
            if other in index and edge.min_distance <= dist_threshold:
                edge_index.append([index[fragid], index[other]])
                edge_attr.append([edge.dist, edge.angle1, edge.angle2,
                                  edge.min_distance, edge.max_distance, edge.weight])
    return Data(x=torch.tensor(node_features, dtype=torch.float32).reshape(-1, 4),
                edge_index=torch.tensor(edge_index, dtype=torch.long).reshape(-1, 2).T.contiguous(),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float32).reshape(-1, 6))


class GraphDataset(Dataset):
    """Build the selected graphs once from complete AlephPDB annotation pickles."""
    def __init__(self, data_path, selection_path, representation="cv", dist_threshold=10.0):
        super().__init__()
        self.data_path = Path(data_path)
        if not self.data_path.is_dir():
            raise FileNotFoundError(f"Annotation folder does not exist: {self.data_path}")
        rows = read_selection(selection_path)
        self.missing = [(sid, family) for sid, family in rows
                        if not (self.data_path / family / f"{sid}.ent.pkl").is_file()]
        missing = set(self.missing)
        self.graphs = []
        self.data = []
        self.graph_by_fam = {}
        self.excluded = []
        print(f"Reading annotations for {selection_path}", flush=True)
        for name, fam in rows:
            if (name, fam) in missing:
                continue
            path = self.data_path / fam / f"{name}.ent.pkl"
            graph = build_graph(path, name, fam, dist_threshold, representation=representation)
            graph.validate(raise_on_error=True)
            if graph.num_nodes == 0:
                self.excluded.append((name, fam))
                continue
            idx = len(self.graphs)
            self.graphs.append((name, fam))
            self.data.append(graph)
            self.graph_by_fam.setdefault(fam, []).append(idx)
            if len(self.graphs) % 500 == 0:
                print(f"Built {len(self.graphs)} graphs", flush=True)
        self.families = list(self.graph_by_fam.keys())
        print(f"Loaded {len(self.graphs)} graphs; missing {len(self.missing)} annotations; "
              f"excluded {len(self.excluded)} empty graphs", flush=True)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.data[idx]

def fit_feature_scaling(training_graphs):
    statistics = {}

    for field in ("x", "edge_attr"):
        values = torch.cat(
            [graph[field] for graph in training_graphs], dim=0
        ).double()

        mean = values.mean(dim=0)
        std = values.std(dim=0, unbiased=False)

        # Constant columns use a scale of 1.
        std[std < 1e-8] = 1.0

        statistics[field] = {"mean": mean, "std": std}

    return statistics

def scale_graph(graph, statistics):
    graph = graph.clone()

    for field in ("x", "edge_attr"):
        values = graph[field]
        mean = statistics[field]["mean"].to(values)
        std = statistics[field]["std"].to(values)

        graph[field] = (values - mean) / std

    return graph

class FamilyBatchSampler(Sampler):
    def __init__(self, dataset, num_per_family = 6, other_families = 5):
        self.dataset = dataset
        self.num_per_family = num_per_family
        self.total_families = other_families + 1 # Me and others

        if len(dataset.families) < self.total_families:
            raise ValueError("Not enough families to create a batch")

    def __iter__(self):
        anchors = self.dataset.families.copy()
        random.shuffle(anchors)

        for anchor in anchors:
            other_families = random.sample([f for f in self.dataset.families if f != anchor], self.total_families - 1)

            batch_families = [anchor] + other_families
            batch_indices = []

            for family in batch_families:
                indices = self.dataset.graph_by_fam[family]

                if len(indices) < self.num_per_family:
                    sampled = random.choices(indices, k = self.num_per_family)
                else:
                    sampled = random.sample(indices, self.num_per_family)

                batch_indices.extend(sampled)

            yield batch_indices

    def __len__(self):
        return len(self.dataset.families)

class AlephModel(nn.Module):
    """Shared CV/fragment encoder with mean/max graph pooling."""
    def __init__(self, node_dim: int = 3, edge_dim: int = 3, hidden_dim: int = 128, embedding_dim: int = 128, num_layers: int = 3, heads: int = 4, dropout: float = 0.0, operator: str = "gine"):
        super().__init__()
        if operator not in ("gatv2", "gine"):
            raise ValueError("Use operator='gatv2' or operator='gine'")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if operator == "gatv2" and (heads < 1 or hidden_dim % heads):
            raise ValueError("GATv2 hidden_dim must be divisible by positive heads")
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.operator = operator

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()

        in_dim = hidden_dim
        for _ in range(num_layers):
            if operator == "gine":
                conv = GINEConv(nn.Sequential(
                    nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)), edge_dim=edge_dim)
            else:
                conv = GATv2Conv(
                    in_channels=in_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.residuals.append(
                nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
            )
            in_dim = hidden_dim

        self.readout = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.input_projection = nn.Linear(node_dim, hidden_dim)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        if isinstance(x, Data):
            data = x
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.input_projection(x)
        for conv, norm, res in zip(self.convs, self.norms, self.residuals):
            h = conv(x, edge_index, edge_attr=edge_attr)
            h = norm(h + res(x))
            h = F.elu(h)
            x = F.dropout(h, p=self.dropout, training=self.training)
        if batch is None:
            batch = x.new_zeros(len(x), dtype=torch.long)
        g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        return F.normalize(self.readout(g), dim=-1)


class FamilyContrastiveLoss(nn.Module):
    """
    Pairwise supervised contrastive loss for PDB graph embeddings.

    Same-family graphs are positive pairs.
    Different-family graphs are negative pairs.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, families):
        embeddings = F.normalize(embeddings, dim=-1)
        similarity = embeddings @ embeddings.T

        same_family = torch.tensor(
            [[a == b for b in families] for a in families],
            device=embeddings.device,
        )
        signs = same_family.float() * 2 - 1
        loss = -F.logsigmoid(signs * similarity / self.temperature)

        # Compare each pair once; exclude comparisons with itself.
        pairs = torch.triu(torch.ones_like(same_family), diagonal=1)
        if not pairs.any():
            return embeddings.sum() * 0

        return loss[pairs].mean()


class SupervisedContrastiveLoss(nn.Module):
    # we learn with this
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = 0.07

    def forward(self, embeddings, families):
        z = F.normalize(embeddings, dim=-1)
        if not isinstance(families, torch.Tensor):
            ids = {family: i for i, family in enumerate(dict.fromkeys(families))}
            families = [ids[family] for family in families]
        families = torch.as_tensor(families, device=z.device).reshape(-1)

        # Cosine similarity / temperature.
        logits = (z @ z.T) / self.temperature

        # Numerical stability.
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        # Exclude self-comparisons.
        self_mask = torch.eye(z.size(0), device=z.device, dtype=torch.bool)
        logits = logits.masked_fill(self_mask, float("-inf"))

        # All other graphs of the same family are positives.
        positive_mask = (families[:, None] == families[None, :]) & ~self_mask

        positive_count = positive_mask.sum(dim=1)
        valid = positive_count > 0
        if not valid.any():
            return embeddings.sum() * 0

        log_prob = logits - torch.logsumexp(
            logits, dim=1, keepdim=True
        )

        mean_positive_log_prob = (
            log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1)
            / positive_count.clamp_min(1)
        )

        return -(self.temperature / self.base_temperature) * mean_positive_log_prob[valid].mean()


def train(model, train_loader, val_loader, criterion, device, gallery_loader, epochs=50, eval_every=1, lr=1e-4, writer=None, save_model=False, model_path="graph_embed.pth"):
    # Hyperparameters
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Investigate AdamW, weight decay and so
    # Patience ony cares about evaluations, so be careful with eval_every
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6) # (patience) * (eval_every) = real reduce_every
    best_sensitivity = None
    val_names = {graph.name for graph in val_loader.dataset}
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
            z = model(data)

            # Update loss
            train_loss = criterion(z, data.family)
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
                    if param.numel():
                        writer.add_histogram(name, param, epoch+1)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Elapsed Time: {elapsed:.2f} seconds")

        # Evaluate every eval_every epochs
        if (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs:
            start = time.time()
            model.eval()
            with torch.no_grad():
                embeddings, rows = embed_graphs(model, gallery_loader, device)
                indices = [i for i, (name, _) in enumerate(rows) if name in val_names]
                avg_val_loss = criterion(embeddings[indices].to(device), [rows[i][1] for i in indices]).item()
                sensitivity = family_sensitivity(embeddings, rows, val_names)
                if writer is not None:
                    writer.add_scalar("Validation/family_sensitivity", sensitivity, epoch+1)
                print(f"Validation family sensitivity: {sensitivity:.4f}")
                scheduler.step(avg_val_loss)

                # Logging
                elapsed = time.time() - start
                if writer is not None:
                    writer.add_scalar("Validation Loss/validation_epoch", avg_val_loss, epoch+1)
                    writer.add_scalar("Time/validation_seconds", elapsed, epoch+1)
                print(f"Validation Loss: {avg_val_loss:.4f}, Elapsed Time: {elapsed:.2f} seconds")

                # Select by mean validation family sensitivity, as in Progres.
                if save_model and (best_sensitivity is None or sensitivity > best_sensitivity):
                    best_sensitivity = sensitivity
                    torch.save(model.state_dict(), model_path)
                    print(f"New best model saved in {model_path} (Validation sensitivity {sensitivity:.4f})")

def evaluate_loss(model, loader, criterion, device):
    model.eval()
    embeddings, families = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            embeddings.append(model(data))
            families.extend(data.family)
        return criterion(torch.cat(embeddings), families).item()


def main():
    start = time.time()
    parser = argparse.ArgumentParser(description="Train CV or SSE graphs from ALEPH annotations")
    parser.add_argument("--annotations", type=Path, required=True, help="Annotation folder containing family/SID.ent.pkl")
    parser.add_argument("--dataset", choices=("scope95_full", "scope95_small"), default="scope95_small",
                        help="SID selections in the datasets folder beside this script")
    parser.add_argument("--representation", choices=("cv", "sse"), default="cv")
    parser.add_argument("--operator", choices=("gatv2", "gine"), default="gine")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--scale", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path("training_output"))
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    project_dir = Path(__file__).resolve().parent
    split_dir = project_dir / "datasets" / args.dataset
    if args.output.resolve() in (project_dir, split_dir, args.annotations.resolve()):
        raise ValueError("Use a separate output folder to preserve the input annotations and SID lists")
    split_names = ("train_sid", "val_unseen_sid", "test_unseen_sid")
    print(f"Dataset: {args.dataset}; selections: {split_dir}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for name in split_names:
        dataset = GraphDataset(args.annotations, split_dir / f"{name}.txt", representation=args.representation)
        datasets[name] = dataset
        if not len(dataset):
            raise ValueError(f"No usable graphs in {name}")

    if args.scale:
        statistics = fit_feature_scaling(datasets['train_sid'].data)
        torch.save(statistics, args.output / "feature_scaling.pt")
        for dataset in datasets.values():
            dataset.data = [scale_graph(graph, statistics) for graph in dataset.data]

    train_loader = DataLoader(datasets["train_sid"], batch_sampler=FamilyBatchSampler(datasets["train_sid"]))
    val_loader = DataLoader(datasets["val_unseen_sid"], batch_size=64, shuffle=False)
    test_loader = DataLoader(datasets["test_unseen_sid"], batch_size=64, shuffle=False)
    gallery = [graph for dataset in datasets.values() for graph in dataset]
    gallery_loader = DataLoader(gallery, batch_size=32, shuffle=False)
    model = AlephModel(node_dim=datasets["train_sid"][0].x.shape[1],
                       edge_dim=datasets["train_sid"][0].edge_attr.shape[1], hidden_dim=128, embedding_dim=128, dropout=0.0, operator=args.operator)
    criterion = SupervisedContrastiveLoss(temperature=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.output / "graph_embed.pth"
    log_directory = args.output / "tensorboard"
    print(f"Everything ready to train in {time.time() - start:.2f} seconds")
    print(f"TensorBoard logs: {log_directory.resolve()}")
    with SummaryWriter(log_dir=str(log_directory)) as writer:
        configuration = {**vars(args), "device": str(device),
                         "split_graph_counts": {name: len(dataset) for name, dataset in datasets.items()}}
        writer.add_text("Run/config", json.dumps(configuration, indent=2, default=str), 0)
        train(model, train_loader, val_loader, criterion, device, gallery_loader, epochs=args.epochs,
              eval_every=1, lr=1e-2, writer=writer, save_model=True, model_path=model_path)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        test_loss = evaluate_loss(model, test_loader, criterion, device)
        writer.add_scalar("Test Loss/best_validation_checkpoint", test_loss, args.epochs)
        embeddings, rows = embed_graphs(model, gallery_loader, device)
        results = {name: {"family_sensitivity": family_sensitivity(embeddings, rows, {graph.name for graph in datasets[name]})}
                   for name in ("val_unseen_sid", "test_unseen_sid")}
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        writer.add_scalar("Test/family_sensitivity", results["test_unseen_sid"]["family_sensitivity"], args.epochs)
        (args.output / "test_loss.txt").write_text(f"{test_loss}\n")
        print(f"Test loss for the best validation checkpoint: {test_loss:.4f}")


if __name__ == "__main__":
    main()
