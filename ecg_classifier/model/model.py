"""
ECG Transformer分类模型

融合传统算法ECGBrain505的设计思路:
1. 多尺度patch embedding (借鉴分块策略)
2. 频域特征分支 (借鉴FFT特征)
3. 位置编码 (保持时序信息)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """将ECG信号分割成patches并嵌入"""

    def __init__(self, seq_len=7500, patch_size=50, hidden_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size

        # 1D卷积实现patch embedding
        self.projection = nn.Conv1d(
            in_channels=1,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [B, seq_len]
        x = x.unsqueeze(1)  # [B, 1, seq_len]
        x = self.projection(x)  # [B, hidden_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, hidden_dim]
        x = self.norm(x)
        return x


class MultiScalePatchEmbedding(nn.Module):
    """
    多尺度Patch Embedding
    借鉴传统算法的多尺度分块策略 (chunk_size=5,10,50)
    """

    def __init__(self, seq_len=7500, hidden_dim=256):
        super().__init__()

        # 三种尺度: 25(0.1s), 50(0.2s), 150(0.6s) for 250Hz
        self.patch_sizes = [25, 50, 150]

        self.embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, hidden_dim // len(self.patch_sizes), kernel_size=ps, stride=ps),
                nn.GELU()
            )
            for ps in self.patch_sizes
        ])

        # 融合不同尺度
        self.fusion = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [B, seq_len]
        x = x.unsqueeze(1)  # [B, 1, seq_len]

        multi_scale_features = []
        for embed in self.embeddings:
            feat = embed(x)  # [B, hidden_dim/3, num_patches]
            # 全局平均池化
            feat = feat.mean(dim=-1)  # [B, hidden_dim/3]
            multi_scale_features.append(feat)

        # 拼接多尺度特征
        combined = torch.cat(multi_scale_features, dim=-1)  # [B, hidden_dim]
        return self.norm(self.fusion(combined))


class FrequencyBranch(nn.Module):
    """
    频域特征分支
    借鉴传统算法的FFT特征提取
    """

    def __init__(self, seq_len=7500, hidden_dim=256):
        super().__init__()
        # FFT输出维度
        fft_dim = seq_len // 2 + 1

        self.freq_encoder = nn.Sequential(
            nn.Linear(fft_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, x):
        # x: [B, seq_len]
        # 计算FFT
        x_fft = torch.fft.rfft(x, dim=-1)
        x_mag = torch.abs(x_fft)  # 幅度谱

        return self.freq_encoder(x_mag)


class TransformerEncoderBlock(nn.Module):
    """Transformer编码器块"""

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class ECGTransformerClassifier(nn.Module):
    """
    ECG Transformer分类器

    架构:
    1. Patch Embedding: 将7500点信号分割成patches
    2. Transformer Encoder: 学习时序特征
    3. 频域分支: 提取频率特征 (借鉴传统FFT)
    4. 多尺度分支: 多尺度特征 (借鉴传统分块)
    5. 分类头: 多标签分类
    """

    def __init__(
        self,
        seq_len=7500,
        num_classes=40,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        patch_size=50,
        dropout=0.1,
        use_freq_branch=True,
        use_multiscale=True
    ):
        super().__init__()

        self.use_freq_branch = use_freq_branch
        self.use_multiscale = use_multiscale

        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(seq_len, patch_size, hidden_dim)
        num_patches = seq_len // patch_size

        # 2. CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # 3. 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
        self.pos_drop = nn.Dropout(dropout)

        # 4. Transformer Encoder
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # 5. 频域分支 (可选)
        if use_freq_branch:
            self.freq_branch = FrequencyBranch(seq_len, hidden_dim)

        # 6. 多尺度分支 (可选)
        if use_multiscale:
            self.multiscale_branch = MultiScalePatchEmbedding(seq_len, hidden_dim)

        # 7. 特征融合
        fusion_dim = hidden_dim
        if use_freq_branch:
            fusion_dim += hidden_dim
        if use_multiscale:
            fusion_dim += hidden_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 8. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: [B, seq_len] ECG信号

        Returns:
            logits: [B, num_classes] 分类logits
        """
        B = x.shape[0]

        # Patch embedding
        x_patches = self.patch_embed(x)  # [B, num_patches, hidden_dim]

        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_patches = torch.cat([cls_tokens, x_patches], dim=1)  # [B, num_patches+1, hidden_dim]

        # 添加位置编码
        x_patches = x_patches + self.pos_embed
        x_patches = self.pos_drop(x_patches)

        # Transformer encoder
        for block in self.encoder_blocks:
            x_patches = block(x_patches)
        x_patches = self.norm(x_patches)

        # 取CLS token作为全局特征
        transformer_feat = x_patches[:, 0]  # [B, hidden_dim]

        # 收集所有分支特征
        features = [transformer_feat]

        if self.use_freq_branch:
            freq_feat = self.freq_branch(x)
            features.append(freq_feat)

        if self.use_multiscale:
            ms_feat = self.multiscale_branch(x)
            features.append(ms_feat)

        # 融合
        if len(features) > 1:
            combined = torch.cat(features, dim=-1)
            fused = self.fusion(combined)
        else:
            fused = transformer_feat

        # 分类
        logits = self.classifier(fused)

        return logits

    def get_attention_weights(self, x):
        """获取注意力权重（用于可视化）"""
        B = x.shape[0]

        x_patches = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_patches = torch.cat([cls_tokens, x_patches], dim=1)
        x_patches = x_patches + self.pos_embed

        attention_weights = []
        for block in self.encoder_blocks:
            x_norm = block.norm1(x_patches)
            _, attn = block.attn(x_norm, x_norm, x_norm, need_weights=True)
            attention_weights.append(attn)
            x_patches = block(x_patches)

        return attention_weights


class ECGTransformerLite(nn.Module):
    """轻量级版本，适合快速实验"""

    def __init__(self, seq_len=7500, num_classes=40, hidden_dim=128, num_layers=4):
        super().__init__()

        self.patch_embed = PatchEmbedding(seq_len, patch_size=75, hidden_dim=hidden_dim)
        num_patches = seq_len // 75

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        x = self.encoder(x)
        x = x[:, 0]

        return self.classifier(x)


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    model = ECGTransformerClassifier(
        seq_len=7500,
        num_classes=40,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        use_freq_branch=True,
        use_multiscale=True
    )

    print(f"模型参数量: {count_parameters(model):,}")

    # 测试前向传播
    x = torch.randn(4, 7500)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")

    # 测试轻量版
    model_lite = ECGTransformerLite()
    print(f"轻量版参数量: {count_parameters(model_lite):,}")
