"""Progres family-search sensitivity for CV and SSE embeddings."""
import numpy as np
import torch


@torch.no_grad()
def embed_graphs(model, loader, device):
    model.eval()
    embeddings, rows = [], []
    for data in loader:
        rows.extend(zip(data.name, data.family))
        embeddings.append(model(data.to(device)).cpu())
    return torch.cat(embeddings), rows


@torch.no_grad()
def family_sensitivity(embeddings, rows, query_names):
    """Same-family hits before the first different-fold hit, including self."""
    z = embeddings.detach().cpu()
    families = [family for _, family in rows]
    query_names = set(query_names)
    sensitivities = []
    for i, (name, family) in enumerate(rows):
        if name not in query_names:
            continue
        total_pos = sum(1 if matched_family == family else 0 for matched_family in families)
        distances = (1 - (z[i:i + 1] * z).sum(-1)) / 2
        count_tp = 0
        for j in distances.argsort():
            matched_family = families[j]
            if matched_family == family:
                count_tp += 1
            elif matched_family.rsplit('.', 2)[0] != family.rsplit('.', 2)[0]:
                break
        sensitivities.append(count_tp / total_pos)
    return float(np.mean(sensitivities))
