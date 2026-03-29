
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from layers import GraphConvolution, SimilarityAdj, DistanceAdj


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.1)

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        n_features = args.feature_size
        n_class = args.num_classes
        
        self.scoring_mode = args.scoring_mode

        # Optimized: 512->256 (fewer params), add BN instead of dropout for regularization
        self.conv1d1 = nn.Conv1d(in_channels=n_features, out_channels=256, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv1d2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Optimized GCN: parameter sharing across 3 paths (gc3-gc6 removed)
        self.gc1 = GraphConvolution(128, 32, residual=True)
        self.gc2 = GraphConvolution(32, 32, residual=True)
        self.disAdj = DistanceAdj()

        self.classifier = nn.Linear(32*3, n_class)
        
        # Simplified approximator: kernel=1 for speed, add BN
        self.approximator = nn.Sequential(
            nn.Conv1d(128, 32, 1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.conv1d_approximator = nn.Conv1d(32, 1, 1, padding=0)
        self.conv1d_approximatorMulti = nn.Conv1d(32, 7, 1, padding=0)
        
        # Lower dropout (0.6->0.3) + BN for better regularization
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs, seq_len):
        # Feature extraction with BN (skip dropout here)
        x = inputs.permute(0, 2, 1)  # (B, C, T)
        x = self.relu(self.bn1(self.conv1d1(x)))
        x = self.relu(self.bn2(self.conv1d2(x)))  # (B, 128, T)

        # Approximator: kernel=1 (no padding trick needed)
        logits_feat = self.approximator(x)  # (B, 32, T)
        logitsMulti = self.conv1d_approximatorMulti(logits_feat)  # (B, 7, T)
        logits = self.conv1d_approximator(logits_feat)  # (B, 1, T)
        
        logitsMulti = logitsMulti.permute(0, 2, 1)  # (B, T, 7)
        logits = logits.permute(0, 2, 1)  # (B, T, 1)
        
        x = x.permute(0, 2, 1)  # (B, T, 128)

        # GCN with optimized adjacency
        adj = self.adj(inputs, seq_len)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        scoadj = self.sadj(logits.detach(), seq_len)

        # Multi-path GCN with parameter sharing (reuse gc1/gc2)
        x1 = self.relu(self.gc1(x, adj))
        x1 = self.dropout(x1)
        x1 = self.relu(self.gc2(x1, adj))
        x1 = self.dropout(x1)

        x2 = self.relu(self.gc1(x, disadj))
        x2 = self.dropout(x2)
        x2 = self.relu(self.gc2(x2, disadj))
        x2 = self.dropout(x2)
        
        x3 = self.relu(self.gc1(x, scoadj))
        x3 = self.dropout(x3)
        x3 = self.relu(self.gc2(x3, scoadj))
        x3 = self.dropout(x3)

        x = torch.cat((x1, x2, x3), 2)
        x = self.classifier(x)

        if self.scoring_mode == 'Binary':
            return x, logits
        elif self.scoring_mode == 'Multi':
            return x, logitsMulti

    def sadj(self, logits, seq_len):
        """Optimized: vectorized operations, no repeat/loop overhead"""
        soft = nn.Softmax(1)
        logits_sig = self.sigmoid(logits).squeeze(-1)  # (B, T)
        
        # Broadcasting: (B, T, 1) - (B, 1, T) -> (B, T, T)
        dist = torch.abs(logits_sig.unsqueeze(2) - logits_sig.unsqueeze(1))
        adj = 1.0 - dist
        
        # Sigmoid smoothing
        adj = 1.0 / (1.0 + torch.exp(-((adj - 0.5) / 0.1)))
        adj = soft(adj)
        
        # Mask by seq_len if provided
        if seq_len is not None:
            for i in range(logits.shape[0]):
                n = int(seq_len[i].item()) if isinstance(seq_len[i], torch.Tensor) else int(seq_len[i])
                if n < adj.shape[1]:
                    adj[i, n:, :] = 0
                    adj[i, :, n:] = 0
        return adj

    def adj(self, x, seq_len):
        """Optimized: vectorized normalization, minimal loops"""
        soft = nn.Softmax(1)
        
        # Vectorized similarity with normalization
        x2 = x.matmul(x.permute(0, 2, 1))  # (B, T, T)
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # (B, T, 1)
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))  # (B, T, T)
        x2 = x2 / (x_norm_x + 1e-20)
        
        # Threshold and softmax (vectorized)
        adj = F.threshold(x2, 0.7, 0)
        adj = soft(adj)
        
        # Mask by seq_len if provided
        if seq_len is not None:
            for i in range(x.shape[0]):
                n = int(seq_len[i].item()) if isinstance(seq_len[i], torch.Tensor) else int(seq_len[i])
                if n < adj.shape[1]:
                    adj[i, n:, :] = 0
                    adj[i, :, n:] = 0
        return adj


