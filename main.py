import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from umap import UMAP
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import Att_inject


SEED = 42
HIDDEN_DIM = 256
LR = 5e-4
WEIGHT_DECAY = 5e-5
MAX_EPOCHS = 50
PATIENCE = 8
BATCH_SIZE = 256
GRAD_CLIP = 5.0
SIM_PCA_DIM = 64
TOPK = 5
PCA_DIM = 32
UMAP_DIM = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "model_checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)

DATASET_PATH = r"E:\python project\datasets\dataset3"




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_descriptor(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            vals = [float(x) for x in line.strip().split(',')[1:]]
            data.append(vals)
    return np.array(data, dtype=np.float32)


def extract_proximity_features(feat_mat, k):
    sim = cosine_similarity(feat_mat)
    m_val = sim.mean(axis=1, keepdims=True)
    k_val = np.sort(sim, axis=1)[:, -k:].mean(axis=1, keepdims=True)
    return m_val, k_val


def get_manifold_pca(feat_mat, dim):
    sim = np.nan_to_num(cosine_similarity(feat_mat))
    return PCA(n_components=min(dim, sim.shape[1]), random_state=SEED).fit_transform(sim)


def evaluate_Att_inject(model, c_data, m_data, pairs, labels):
    model.eval()
    y_scores = []
    with torch.no_grad():
        for i in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[i:i + BATCH_SIZE]
            mi, ci = batch[:, 0].astype(int), batch[:, 1].astype(int)
            out = model(torch.tensor(c_data[ci]).to(DEVICE), torch.tensor(m_data[mi]).to(DEVICE))
            y_scores.extend(torch.sigmoid(out).cpu().numpy().flatten())

    y_scores = np.array(y_scores)
    y_pred = (y_scores > 0.5).astype(int)
    return roc_auc_score(labels, y_scores), accuracy_score(labels, y_pred), f1_score(labels, y_pred), recall_score(
        labels, y_pred), y_scores



def run_Att_inject_experiment():
    print("Feature creating...")
    c_raw = np.concatenate([
        load_descriptor(os.path.join(DATASET_PATH, "circRNA_ctd_dict.txt")),
        load_descriptor(os.path.join(DATASET_PATH, "circRNA_doc2vec_dict.txt")),
        load_descriptor(os.path.join(DATASET_PATH, "circRNA_kmer_dict.txt")),
        load_descriptor(os.path.join(DATASET_PATH, "circRNA_role2vec_dict.txt")),
        np.array(pd.read_csv(os.path.join(DATASET_PATH, "circRNA_sequence_similarity.csv"), header=None),
                 dtype=np.float32)
    ], 1)

    m_raw = np.concatenate([
        load_descriptor(os.path.join(DATASET_PATH, "miRNA_ctd_dict.txt")),
        load_descriptor(os.path.join(DATASET_PATH, "miRNA_doc2vec_dict.txt")),
        load_descriptor(os.path.join(DATASET_PATH, "miRNA_kmer_dict.txt")),
        load_descriptor(os.path.join(DATASET_PATH, "miRNA_role2vec_dict.txt")),
        np.array(pd.read_csv(os.path.join(DATASET_PATH, "miRNA_sequence_similarity.csv"), header=None),
                 dtype=np.float32)
    ], 1)

    matrix_file = [os.path.join(DATASET_PATH, f) for f in os.listdir(DATASET_PATH) if "matrix" in f][0]
    adj_mat = np.array(pd.read_csv(matrix_file, header=None), dtype=np.float32)

    pos = np.argwhere(adj_mat == 1)
    neg = np.argwhere(adj_mat == 0)
    np.random.shuffle(neg)
    neg = neg[:len(pos)]
    X, y = np.vstack([pos, neg]), np.hstack([np.ones(len(pos)), np.zeros(len(neg))])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n[Fold {fold}] Performing Dynamic Manifold Calibration...")

        cur_adj = adj_mat.copy()
        for mi, ci in X[val_idx]: cur_adj[int(mi), int(ci)] = 0

        c_deg, m_deg = cur_adj.sum(0).reshape(-1, 1), cur_adj.sum(1).reshape(-1, 1)
        c_mean, c_topk = extract_proximity_features(c_raw, TOPK)
        m_mean, m_topk = extract_proximity_features(m_raw, TOPK)
        c_sim_pca, m_sim_pca = get_manifold_pca(c_raw, SIM_PCA_DIM), get_manifold_pca(m_raw, SIM_PCA_DIM)

        c_integrated = np.hstack([c_raw, c_deg, c_mean, c_topk, c_sim_pca])
        m_integrated = np.hstack([m_raw, m_deg, m_mean, m_topk, m_sim_pca])

        c_aug = np.hstack([c_integrated, PCA(PCA_DIM, random_state=SEED).fit_transform(c_integrated),
                           UMAP(n_components=UMAP_DIM, random_state=SEED).fit_transform(c_integrated)])
        m_aug = np.hstack([m_integrated, PCA(PCA_DIM, random_state=SEED).fit_transform(m_integrated),
                           UMAP(n_components=UMAP_DIM, random_state=SEED).fit_transform(m_integrated)])

        c_in, m_in = StandardScaler().fit_transform(c_aug), StandardScaler().fit_transform(m_aug)



        model = Att_inject(c_in.shape[1], m_in.shape[1], HIDDEN_DIM).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=50)
        criterion = nn.BCEWithLogitsLoss()

        best_auc, patience = -1, 0
        best_save_path = os.path.join(OUT_DIR, f"best_model_fold_{fold}.pth")

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            idx_shuf = np.random.permutation(train_idx)
            for i in range(0, len(idx_shuf), BATCH_SIZE):
                b = idx_shuf[i:i + BATCH_SIZE]
                mi, ci = X[b][:, 0].astype(int), X[b][:, 1].astype(int)
                lbl = torch.tensor(y[b], dtype=torch.float32).to(DEVICE)
                out = model(torch.tensor(c_in[ci]).to(DEVICE), torch.tensor(m_in[mi]).to(DEVICE)).squeeze()
                loss = criterion(out, lbl)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            scheduler.step()

            if epoch % 2 == 0:
                auc, _, _, _, _ = evaluate_Att_inject(model, c_in, m_in, X[val_idx], y[val_idx])
                if auc > best_auc:
                    best_auc = auc
                    patience = 0
                    torch.save(model.state_dict(), best_save_path)
                else:
                    patience += 1
                if patience >= PATIENCE: break

        model.load_state_dict(torch.load(best_save_path))
        f_auc, f_acc, f_f1, f_rec, _ = evaluate_Att_inject(model, c_in, m_in, X[val_idx], y[val_idx])
        print(f"Fold {fold} Final Result: AUC={f_auc:.4f}, ACC={f_acc:.4f}, F1={f_f1:.4f}, Recall={f_rec:.4f}")
        all_results.append([f_auc, f_acc, f_f1, f_rec])

    final_metrics = np.mean(all_results, axis=0)
    print(f"\n" + "=" * 30)
    print(f"5-Fold Average Results:")
    print(f"AUC:    {final_metrics[0]:.4f}")
    print(f"ACC:    {final_metrics[1]:.4f}")
    print(f"F1:     {final_metrics[2]:.4f}")
    print(f"Recall: {final_metrics[3]:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    set_seed(SEED)
    run_Att_inject_experiment()
