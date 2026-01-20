"""
简化版ECG分类模型

五种配置供选择:
1. simple: 纯Transformer (最简单)
2. fft: Transformer + FFT (推荐起步)
3. full: Transformer + FFT + 多尺度
4. se: Transformer + SE-Block通道注意力 (借鉴ECG-with-Deep-learning)
5. se_full: 完整版 + SE-Block (最强配置)
"""

import math
import torch
import torch.nn as nn


# ==================== SE-Block (通道注意力) ====================
class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation Block for 1D signals

    原理:
    1. Squeeze: 全局平均池化，将每个通道压缩为一个标量
    2. Excitation: 两层FC学习通道间的依赖关系
    3. Scale: 将学到的权重乘回原特征

    为何适合ECG:
    - 不同卷积核捕获P波、QRS波群、T波等不同特征
    - SE-Block能动态学习哪些通道(波形特征)更具诊断价值
    - 自动抑制噪声通道，增强诊断相关通道
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, L]
        B, C, L = x.shape
        # Squeeze: [B, C, L] -> [B, C, 1] -> [B, C]
        y = self.squeeze(x).view(B, C)
        # Excitation: [B, C] -> [B, C]
        y = self.excitation(y).view(B, C, 1)
        # Scale: element-wise multiplication
        return x * y.expand_as(x)


class SEResBlock1D(nn.Module):
    """带残差连接的SE-Block"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = SEBlock1D(channels, reduction)

    def forward(self, x):
        return x + self.se(x)


class SimpleECGClassifier(nn.Module):
    """
    简化版: 纯Transformer

    最简单的baseline，先跑通再说
    """

    def __init__(self, seq_len=7500, num_classes=40, hidden_dim=256,
                 num_layers=6, num_heads=8, patch_size=50):
        super().__init__()

        self.num_patches = seq_len // patch_size

        # Patch Embedding: 1D卷积
        self.patch_embed = nn.Conv1d(1, hidden_dim, kernel_size=patch_size, stride=patch_size)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_dim))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: [B, 7500]
        B = x.shape[0]

        # Patch embedding
        x = x.unsqueeze(1)  # [B, 1, 7500]
        x = self.patch_embed(x)  # [B, hidden_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, hidden_dim]

        # 添加CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, num_patches+1, hidden_dim]

        # 位置编码
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # 取CLS token分类
        x = self.norm(x[:, 0])
        return self.classifier(x)


class ECGClassifierWithFFT(nn.Module):
    """
    推荐版: Transformer + FFT

    加入频域信息，通常能提升2-5%
    """

    def __init__(self, seq_len=7500, num_classes=40, hidden_dim=256,
                 num_layers=6, num_heads=8, patch_size=50):
        super().__init__()

        self.num_patches = seq_len // patch_size
        self.seq_len = seq_len

        # 时域分支: Transformer
        self.patch_embed = nn.Conv1d(1, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        # 频域分支: FFT + MLP
        fft_dim = seq_len // 2 + 1  # rfft输出维度
        self.fft_encoder = nn.Sequential(
            nn.Linear(fft_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # 融合分类
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # === 时域分支 ===
        x_time = x.unsqueeze(1)
        x_time = self.patch_embed(x_time).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x_time = torch.cat([cls, x_time], dim=1)
        x_time = x_time + self.pos_embed
        x_time = self.transformer(x_time)
        feat_time = self.norm(x_time[:, 0])  # [B, hidden_dim]

        # === 频域分支 ===
        x_fft = torch.fft.rfft(x, dim=-1)
        x_fft_mag = torch.abs(x_fft)  # 取幅度
        feat_freq = self.fft_encoder(x_fft_mag)  # [B, hidden_dim//2]

        # === 融合 ===
        feat = torch.cat([feat_time, feat_freq], dim=-1)
        return self.classifier(feat)


class ECGClassifierFull(nn.Module):
    """
    完整版: Transformer + FFT + 多尺度

    最强配置，但训练慢，参数多
    """

    def __init__(self, seq_len=7500, num_classes=40, hidden_dim=256,
                 num_layers=6, num_heads=8):
        super().__init__()

        self.seq_len = seq_len

        # === 时域主干 ===
        patch_size = 50
        self.num_patches = seq_len // patch_size
        self.patch_embed = nn.Conv1d(1, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        # === 频域分支 ===
        fft_dim = seq_len // 2 + 1
        self.fft_encoder = nn.Sequential(
            nn.Linear(fft_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # === 多尺度分支 ===
        # 借鉴传统算法: 小/中/大三个尺度
        self.ms_convs = nn.ModuleList([
            nn.Conv1d(1, hidden_dim // 4, kernel_size=25, stride=25),   # 0.1s
            nn.Conv1d(1, hidden_dim // 4, kernel_size=75, stride=75),   # 0.3s
            nn.Conv1d(1, hidden_dim // 4, kernel_size=150, stride=150), # 0.6s
        ])
        self.ms_pool = nn.AdaptiveAvgPool1d(1)

        # === 融合分类 ===
        # hidden_dim + hidden_dim//2 + hidden_dim//4*3
        fusion_dim = hidden_dim + hidden_dim // 2 + (hidden_dim // 4) * 3
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # 时域
        x_time = x.unsqueeze(1)
        x_time = self.patch_embed(x_time).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x_time = torch.cat([cls, x_time], dim=1) + self.pos_embed
        x_time = self.transformer(x_time)
        feat_time = self.norm(x_time[:, 0])

        # 频域
        x_fft = torch.abs(torch.fft.rfft(x, dim=-1))
        feat_freq = self.fft_encoder(x_fft)

        # 多尺度
        x_ms = x.unsqueeze(1)
        ms_feats = []
        for conv in self.ms_convs:
            feat = conv(x_ms)
            feat = self.ms_pool(feat).squeeze(-1)
            ms_feats.append(feat)
        feat_ms = torch.cat(ms_feats, dim=-1)

        # 融合
        feat = torch.cat([feat_time, feat_freq, feat_ms], dim=-1)
        return self.classifier(feat)


class ECGClassifierWithSE(nn.Module):
    """
    SE增强版: Transformer + FFT + SE-Block

    借鉴ECG-with-Deep-learning项目的SE-Net思想:
    - 在多尺度卷积后添加SE-Block进行通道注意力
    - 动态学习哪些特征通道更重要
    """

    def __init__(self, seq_len=7500, num_classes=40, hidden_dim=256,
                 num_layers=6, num_heads=8, patch_size=50):
        super().__init__()

        self.seq_len = seq_len
        self.num_patches = seq_len // patch_size

        # === 时域Transformer分支 ===
        self.patch_embed = nn.Conv1d(1, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        # === 频域分支 ===
        fft_dim = seq_len // 2 + 1
        self.fft_encoder = nn.Sequential(
            nn.Linear(fft_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # === 多尺度CNN + SE-Block ===
        # 借鉴ECG-with-Deep-learning: 不同尺度捕获不同波形特征
        ms_dim = hidden_dim // 4
        self.ms_conv1 = nn.Sequential(
            nn.Conv1d(1, ms_dim, kernel_size=25, stride=25),  # 0.1s @250Hz
            nn.BatchNorm1d(ms_dim),
            nn.ReLU(),
            SEBlock1D(ms_dim, reduction=4),  # SE-Block增强
        )
        self.ms_conv2 = nn.Sequential(
            nn.Conv1d(1, ms_dim, kernel_size=75, stride=75),  # 0.3s
            nn.BatchNorm1d(ms_dim),
            nn.ReLU(),
            SEBlock1D(ms_dim, reduction=4),
        )
        self.ms_conv3 = nn.Sequential(
            nn.Conv1d(1, ms_dim, kernel_size=150, stride=150),  # 0.6s
            nn.BatchNorm1d(ms_dim),
            nn.ReLU(),
            SEBlock1D(ms_dim, reduction=4),
        )
        self.ms_pool = nn.AdaptiveAvgPool1d(1)

        # === 融合分类 ===
        fusion_dim = hidden_dim + hidden_dim // 2 + ms_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # 时域Transformer
        x_time = x.unsqueeze(1)
        x_time = self.patch_embed(x_time).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x_time = torch.cat([cls, x_time], dim=1) + self.pos_embed
        x_time = self.transformer(x_time)
        feat_time = self.norm(x_time[:, 0])

        # 频域
        x_fft = torch.abs(torch.fft.rfft(x, dim=-1))
        feat_freq = self.fft_encoder(x_fft)

        # 多尺度 + SE注意力
        x_ms = x.unsqueeze(1)
        feat_ms1 = self.ms_pool(self.ms_conv1(x_ms)).squeeze(-1)
        feat_ms2 = self.ms_pool(self.ms_conv2(x_ms)).squeeze(-1)
        feat_ms3 = self.ms_pool(self.ms_conv3(x_ms)).squeeze(-1)
        feat_ms = torch.cat([feat_ms1, feat_ms2, feat_ms3], dim=-1)

        # 融合
        feat = torch.cat([feat_time, feat_freq, feat_ms], dim=-1)
        return self.classifier(feat)


class ECGClassifierSEFull(nn.Module):
    """
    完整配置: Transformer + FFT + 多尺度 + SE-Block + 深度SE增强

    设计理念:
    - Transformer: 全局时序注意力 (已经具备时序建模能力，不需要LSTM)
    - FFT: 频域特征 (心率变异性、频率成分)
    - 多尺度CNN + SE: 不同时间尺度的局部特征 + 通道注意力
    - Patch级SE: 在Transformer之前增强patch特征

    注: 不使用LSTM，因为Transformer已经覆盖了时序建模能力
    """

    def __init__(self, seq_len=7500, num_classes=40, hidden_dim=256,
                 num_layers=6, num_heads=8, patch_size=50):
        super().__init__()

        self.seq_len = seq_len
        self.num_patches = seq_len // patch_size

        # === 时域Transformer + SE增强 ===
        self.patch_embed = nn.Conv1d(1, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_se = SEBlock1D(hidden_dim, reduction=16)  # Patch特征SE增强
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        # === 频域分支 + SE ===
        fft_dim = seq_len // 2 + 1
        fft_hidden = hidden_dim // 2
        self.fft_encoder = nn.Sequential(
            nn.Linear(fft_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, fft_hidden)
        )

        # === 多尺度CNN + SE (借鉴ECG-with-Deep-learning) ===
        ms_dim = hidden_dim // 4
        self.ms_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, ms_dim, kernel_size=k, stride=k),
                nn.BatchNorm1d(ms_dim),
                nn.ReLU(),
                SEBlock1D(ms_dim, reduction=4),  # 每个尺度独立SE
            )
            for k in [25, 75, 150]  # 0.1s, 0.3s, 0.6s @250Hz
        ])
        self.ms_pool = nn.AdaptiveAvgPool1d(1)

        # === 分支级注意力 (Branch Attention) ===
        # 学习三个分支(时域/频域/多尺度)的相对重要性
        # 输入: 3个分支特征 -> 输出: 3个权重
        self.branch_dims = [hidden_dim, fft_hidden, ms_dim * 3]  # 各分支维度
        fusion_dim = sum(self.branch_dims)
        self.branch_attention = nn.Sequential(
            nn.Linear(fusion_dim, 16),  # 压缩
            nn.ReLU(),
            nn.Linear(16, 3),  # 3个分支
            nn.Softmax(dim=-1)  # 归一化权重
        )

        # === 分类头 ===
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # 时域Transformer + SE
        x_time = x.unsqueeze(1)
        x_time = self.patch_embed(x_time)
        x_time = self.patch_se(x_time)  # Patch级SE增强
        x_time = x_time.transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x_time = torch.cat([cls, x_time], dim=1) + self.pos_embed
        x_time = self.transformer(x_time)
        feat_time = self.norm(x_time[:, 0])  # [B, hidden_dim]

        # 频域
        x_fft = torch.abs(torch.fft.rfft(x, dim=-1))
        feat_freq = self.fft_encoder(x_fft)  # [B, hidden_dim//2]

        # 多尺度 + SE
        x_ms = x.unsqueeze(1)
        ms_feats = []
        for conv in self.ms_convs:
            feat = self.ms_pool(conv(x_ms)).squeeze(-1)
            ms_feats.append(feat)
        feat_ms = torch.cat(ms_feats, dim=-1)  # [B, ms_dim*3]

        # 分支级注意力融合
        feat_concat = torch.cat([feat_time, feat_freq, feat_ms], dim=-1)

        # 计算分支权重 [B, 3]
        branch_weights = self.branch_attention(feat_concat)

        # 对各分支加权 (广播到各分支维度)
        w_time = branch_weights[:, 0:1]  # [B, 1]
        w_freq = branch_weights[:, 1:2]
        w_ms = branch_weights[:, 2:3]

        feat_time_weighted = feat_time * w_time  # [B, hidden_dim]
        feat_freq_weighted = feat_freq * w_freq  # [B, fft_hidden]
        feat_ms_weighted = feat_ms * w_ms  # [B, ms_dim*3]

        feat = torch.cat([feat_time_weighted, feat_freq_weighted, feat_ms_weighted], dim=-1)

        return self.classifier(feat)


def get_model(version='simple', **kwargs):
    """
    获取模型

    Args:
        version: 模型版本
            - 'simple': 纯Transformer (基线)
            - 'fft': Transformer + FFT (推荐起步)
            - 'full': Transformer + FFT + 多尺度
            - 'se': Transformer + FFT + 多尺度 + SE-Block (推荐)
            - 'se_full': 完整版 + Patch SE + 分支注意力融合
    """
    if version == 'simple':
        return SimpleECGClassifier(**kwargs)
    elif version == 'fft':
        return ECGClassifierWithFFT(**kwargs)
    elif version == 'full':
        return ECGClassifierFull(**kwargs)
    elif version == 'se':
        return ECGClassifierWithSE(**kwargs)
    elif version == 'se_full':
        return ECGClassifierSEFull(**kwargs)
    else:
        raise ValueError(f"Unknown version: {version}. Choose from: simple, fft, full, se, se_full")


if __name__ == "__main__":
    # 测试所有版本
    x = torch.randn(4, 7500)

    print("=" * 60)
    print("ECG分类模型对比")
    print("=" * 60)

    for version in ['simple', 'fft', 'full', 'se', 'se_full']:
        model = get_model(version, seq_len=7500, num_classes=40)
        params = sum(p.numel() for p in model.parameters()) / 1e6

        out = model(x)
        print(f"{version:10s}: 参数量={params:.2f}M, 输出={out.shape}")

    print("=" * 60)
    print("推荐:")
    print("  - 起步: fft (Transformer + FFT)")
    print("  - 推荐: se (+ SE-Block通道注意力)")
    print("  - 最强: se_full (+ 深度SE融合)")
