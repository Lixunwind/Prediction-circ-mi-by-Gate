import torch
import torch.nn as nn

class LatentManifoldEncoder(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.encoder_layer = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden)
        )

    def forward(self, x):
        return self.encoder_layer(x)

class GatedCrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)

    def forward(self, q_input, kv_input):

        q = self.w_q(q_input)
        k = self.w_k(kv_input)
        v = self.w_v(kv_input)

        attn_score = (q * k).sum(1, keepdim=True) / (q_input.size(1) ** 0.5)

        gating_weight = torch.sigmoid(attn_score)
        return gating_weight * v

class Att_inject(nn.Module):
    def __init__(self, circ_in_dim, mir_in_dim, hidden):
        super().__init__()

        self.circ_manifold_enc = LatentManifoldEncoder(circ_in_dim, hidden)
        self.mir_manifold_enc = LatentManifoldEncoder(mir_in_dim, hidden)

        self.attention_gate = GatedCrossAttention(hidden)

        self.predictor = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden, 1)
        )

    def forward(self, circ_x, mir_x):
        z_circ = self.circ_manifold_enc(circ_x)
        z_mir = self.mir_manifold_enc(mir_x)

        z_aligned = self.attention_gate(z_circ, z_mir)

        f_combined = torch.cat([z_circ, z_aligned], dim=1)
        return self.predictor(f_combined)