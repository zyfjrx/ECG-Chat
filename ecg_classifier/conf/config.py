# ECG分类模型配置文件
# 参考ECG-CoCa配置，针对单导联分类任务优化

class Config:
    # ==================== 数据参数 ====================
    data_dir = "/path/to/your/ecg_data"  # 修改为您的数据目录
    label_map_file = "诊断类型对应表.csv"

    # ECG信号参数
    sample_rate = 250   # Hz (你的数据)
    duration = 30       # 秒 (你的数据)
    seq_len = 7500      # 250 * 30
    num_classes = 40

    # ==================== 模型参数 ====================
    # 对比: ECG-CoCa用 width=768, layers=12, heads=8 (12导联)
    # 单导联，可以适当缩小

    # 配置选项 (根据GPU显存选择):
    # - 小配置: hidden_dim=256, num_layers=6  (~8M参数, 适合8G显存)
    # - 中配置: hidden_dim=384, num_layers=8  (~15M参数, 适合12G显存)
    # - 大配置: hidden_dim=512, num_layers=12 (~30M参数, 适合24G显存)

    hidden_dim = 256    # ECG-CoCa用768，单导联用256-512够了
    num_layers = 6      # ECG-CoCa用12，单导联用6-8够了
    num_heads = 8       # 保持与ECG-CoCa一致
    patch_size = 50     # 保持与ECG-CoCa一致 (0.2秒@250Hz)
    dropout = 0.1

    # ==================== 训练参数 ====================
    # 参考ECG-CoCa: lr=5e-4, wd=0.2, epochs=32, batch=64

    batch_size = 64         # 与ECG-CoCa一致，显存允许可增大到128
    num_epochs = 50         # 你数据量大(75万)，50-100轮
    learning_rate = 3e-4    # 分类任务，比ECG-CoCa稍低
    weight_decay = 0.05     # 分类任务，比ECG-CoCa(0.2)低一些
    warmup_epochs = 5       # 约10%的epochs做warmup

    # 优化器参数 (与ECG-CoCa一致)
    beta1 = 0.9
    beta2 = 0.98            # ViT推荐值
    eps = 1e-6

    # 学习率调度
    lr_scheduler = 'cosine'  # 与ECG-CoCa一致

    # ==================== 多标签分类参数 ====================
    pos_weight = 2.0        # 正样本权重（处理类别不平衡）
    threshold = 0.5         # 推理阈值

    # ==================== 设备与存储 ====================
    device = "cuda"         # 或 "mps" for Mac M系列
    save_dir = "../checkpoints"
    log_dir = "./logs"
    num_workers = 4         # DataLoader worker数


# ==================== 不同规模配置 ====================

class ConfigSmall(Config):
    """小配置: ~8M参数, 8G显存"""
    hidden_dim = 256
    num_layers = 6
    batch_size = 64


class ConfigMedium(Config):
    """中配置: ~15M参数, 12G显存"""
    hidden_dim = 384
    num_layers = 8
    batch_size = 48


class ConfigLarge(Config):
    """大配置: ~30M参数, 24G显存, 接近ECG-CoCa"""
    hidden_dim = 512
    num_layers = 12
    batch_size = 32
