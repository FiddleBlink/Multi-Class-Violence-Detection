
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from layers import GraphConvolution, SimilarityAdj, DistanceAdj
import math


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0.1)

class PositionalEncoding(nn.Module):
    """Add positional encoding to temporal features (handles variable sequence lengths)"""
    def __init__(self, d_model, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Pre-compute for the expected max length
        pe = self._compute_pe(max_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def _compute_pe(self, length, d_model):
        """Compute positional encoding for a given length"""
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        """
        Args:
            x: (batch, time, features)
        Returns:
            x + positional_encoding
        """
        seq_len = x.size(1)
        
        # If sequence is longer than pre-computed, compute on-the-fly
        if seq_len > self.pe.size(1):
            pe = self._compute_pe(seq_len, self.d_model).to(x.device)
            return x + pe.unsqueeze(0).detach()
        else:
            return x + self.pe[:, :seq_len].detach()


class ModalityFusion(nn.Module):
    """Learn to fuse multiple modalities"""
    def __init__(self, input_dim, output_dim):
        super(ModalityFusion, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),  # LayerNorm works on feature dimension
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x shape: (B, T, F)
        attn_weights = self.attention(x)  # (B, T, 1)
        weighted = x * attn_weights
        return self.fc(weighted)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        n_features = args.feature_size
        n_class = args.num_classes
        
        self.online_mode = args.online_mode
        self.max_seqlen = getattr(args, 'max_seqlen', 200)

        # Feature projection and fusion
        self.fusion = ModalityFusion(n_features, 256)
        self.bn_initial = nn.BatchNorm1d(256)
        
        # Positional encoding for temporal awareness
        self.pos_encoding = PositionalEncoding(256, self.max_seqlen)
        
        # Main feature extraction pipeline
        self.conv1d1 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.conv1d2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.conv1d3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.conv1d4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        
        # Graph Convolution with shared weights (more efficient)
        self.gc1 = GraphConvolution(64, 64, residual=True)
        self.gc2 = GraphConvolution(64, 64, residual=True)
        self.gc_bn1 = nn.BatchNorm1d(64)
        self.gc_bn2 = nn.BatchNorm1d(64)
        
        self.gc3 = GraphConvolution(64, 64, residual=True)
        self.gc4 = GraphConvolution(64, 64, residual=True)
        self.gc_bn3 = nn.BatchNorm1d(64)
        self.gc_bn4 = nn.BatchNorm1d(64)
        
        self.gc5 = GraphConvolution(64, 64, residual=True)
        self.gc6 = GraphConvolution(64, 64, residual=True)
        self.gc_bn5 = nn.BatchNorm1d(64)
        self.gc_bn6 = nn.BatchNorm1d(64)
        
        # Adjacency matrix generation (now as layers)
        self.simAdj = SimilarityAdj(256, 32)
        self.disAdj = DistanceAdj()

        # Enhanced classifier
        # Use LayerNorm here because inputs are (B, T, channels) after concat
        self.classifier = nn.Sequential(
            nn.Linear(64*3, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_class)
        )
        
        # Enhanced approximator network
        self.approximator = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 32, 1, padding=0)
        )
        
        self.conv1d_approximator = nn.Conv1d(32, 1, 3, padding=1)
        self.conv1d_approximatorMulti = nn.Conv1d(32, 7, 3, padding=1)
        
        # Regularization
        self.dropout_heavy = nn.Dropout(0.5)
        self.dropout_light = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.apply(weight_init)

    def forward(self, inputs, seq_len):
        # Modality fusion and projection
        x = self.fusion(inputs)  # (B, T, 256)
        x = x.permute(0, 2, 1)  # (B, 256, T)
        x = self.bn_initial(x)
        x = x.permute(0, 2, 1)  # (B, T, 256)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = x.permute(0, 2, 1)  # (B, 256, T) for conv1d

        # Feature extraction
        x = self.relu(self.bn1(self.conv1d1(x)))
        x = self.dropout_light(x)
        
        x = self.relu(self.bn2(self.conv1d2(x)))
        x = self.dropout_light(x)
        
        x = self.relu(self.bn3(self.conv1d3(x)))
        x = self.dropout_light(x)
        
        x = self.relu(self.bn4(self.conv1d4(x)))
        x = self.dropout_light(x)

        # Anomaly score approximation
        logits = self.approximator(x)

        logitsMulti = self.conv1d_approximatorMulti(logits)
        logits = self.conv1d_approximator(logits)

        logitsMulti = logitsMulti.permute(0, 2, 1)
        logits = logits.permute(0, 2, 1)

        x = x.permute(0, 2, 1)  # (B, T, 64)

        # Generate adjacency matrices
        adj = self._compute_adj(inputs, seq_len)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        scoadj = self._compute_score_adj(logits, seq_len)

        # Multi-source graph convolution
        x1_h = self.relu(self.gc_bn1(self.gc1(x, adj).permute(0, 2, 1)).permute(0, 2, 1))
        x1_h = self.dropout_light(x1_h)
        x1 = self.relu(self.gc_bn2(self.gc2(x1_h, adj).permute(0, 2, 1)).permute(0, 2, 1))
        x1 = self.dropout_light(x1)

        x2_h = self.relu(self.gc_bn3(self.gc3(x, disadj).permute(0, 2, 1)).permute(0, 2, 1))
        x2_h = self.dropout_light(x2_h)
        x2 = self.relu(self.gc_bn4(self.gc4(x2_h, disadj).permute(0, 2, 1)).permute(0, 2, 1))
        x2 = self.dropout_light(x2)
        
        x3_h = self.relu(self.gc_bn5(self.gc5(x, scoadj).permute(0, 2, 1)).permute(0, 2, 1))
        x3_h = self.dropout_light(x3_h)
        x3 = self.relu(self.gc_bn6(self.gc6(x3_h, scoadj).permute(0, 2, 1)).permute(0, 2, 1))
        x3 = self.dropout_light(x3)

        # Concatenate multi-source features
        x = torch.cat((x1, x2, x3), 2)
        x = self.classifier(x)

        if self.online_mode == 'Binary':
            return x, logits
        elif self.online_mode == 'Multi':
            return x, logitsMulti

    def _compute_adj(self, x, seq_len):
        """Compute self-similarity adjacency matrix"""
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1))  # (B, T, T)
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # (B, T, 1)
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2 / (x_norm_x + 1e-20)
        
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj = F.threshold(tmp, 0.7, 0)
                adj = soft(adj)
                output[i] = adj
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj = F.threshold(tmp, 0.7, 0)
                adj = soft(adj)
                output[i, :seq_len[i], :seq_len[i]] = adj
        return output

    def _compute_score_adj(self, logits, seq_len):
        """Compute score-based adjacency matrix"""
        soft = nn.Softmax(1)
        lens = logits.shape[1]
        logits2 = self.sigmoid(logits).repeat(1, 1, lens)
        tmp = logits2.permute(0, 2, 1)
        adj = 1.0 - torch.abs(logits2 - tmp)
        
        # Sigmoid function for adjacency weighting
        sig = lambda x: 1.0 / (1.0 + torch.exp(-((x - 0.5) / 0.1)))
        adj = sig(adj)
        
        output = torch.zeros_like(adj)
        if seq_len is None:
            for i in range(logits.shape[0]):
                tmp = adj[i]
                adj_out = soft(tmp)
                output[i] = adj_out
        else:
            for i in range(len(seq_len)):
                tmp = adj[i, :seq_len[i], :seq_len[i]]
                adj_out = soft(tmp)
                output[i, :seq_len[i], :seq_len[i]] = adj_out
        return output


