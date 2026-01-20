"""ECG数据集加载模块

支持两种数据源:
1. 原始文件列表 (ECGDataset) - 每次加载时解析文件
2. 预处理后的文件 (ECGPreprocessedDataset) - 直接从npz/hdf5/pkl加载，更快
"""
import json
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class ECGDataset(Dataset):
    """
    单导联ECG数据集

    数据格式:
    - 第3行: 诊断类型1
    - 第4行: 诊断类型2 (可能为空)
    - 第5行: 采样率 (250)
    - 第6行: 时长 (30)
    - 第7行: 分隔符 (32767)
    - 第8行起: ECG时序数据 (7500个点)

    归一化方式:
    - 'zscore': Z-score标准化，每个样本独立 (推荐)
    - 'minmax': Min-Max归一化到[0,1]
    - 'robust': 鲁棒标准化，使用中位数和IQR
    - 'global': 使用全局统计量标准化
    - None: 不做归一化
    """

    def __init__(self, file_list, num_classes=40, transform=None,
                 normalize_method='zscore', global_mean=None, global_std=None):
        self.file_list = file_list
        self.num_classes = num_classes
        self.transform = transform
        self.normalize_method = normalize_method
        self.global_mean = global_mean
        self.global_std = global_std

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        ecg_data, labels = self.parse_ecg_file(file_path)

        # 转换为tensor
        ecg_tensor = torch.tensor(ecg_data, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.float32)

        # 数据增强
        if self.transform:
            ecg_tensor = self.transform(ecg_tensor)

        return ecg_tensor, label_tensor

    def parse_ecg_file(self, file_path):
        """解析ECG文件"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # 解析诊断类型 (从第3行开始，到250采样率之前)
        # 格式: 第1-2行未知, 第3行起为标签, 遇到250停止, 然后是30(时长), 32767(分隔符)
        labels = np.zeros(self.num_classes, dtype=np.float32)

        for i in range(2, len(lines)):  # 从第3行(索引2)开始
            try:
                val = int(lines[i].strip())
                if val == 250:  # 遇到采样率，标签结束
                    break
                if 1 <= val <= self.num_classes:
                    labels[val - 1] = 1.0
            except:
                continue

        # 找到32767(起始)和32763(结束)分隔符之间的数据
        data_start = 0
        data_end = len(lines)

        for i, line in enumerate(lines):
            val = line.strip()
            if val == '32767' and data_start == 0:
                data_start = i + 1  # 32767之后开始
            elif val == '32763' and data_start > 0:
                data_end = i  # 32763之前结束
                break

        # 解析ECG数据 (32767到32763之间)
        # 策略: 保留所有位置，异常值用前一个有效值填充
        ecg_values = []
        last_valid = 0.0  # 上一个有效值

        for i in range(data_start, data_end):
            try:
                value = float(lines[i].strip())
                # 检查是否为异常值 (int32边界值通常表示无效数据)
                if -32768 <= value <= 32767:  # 16位ADC范围内，有效
                    ecg_values.append(value)
                    last_valid = value
                else:
                    # 异常值，用前一个有效值填充，保持位置
                    ecg_values.append(last_valid)
            except:
                # 解析失败，用前一个有效值填充
                ecg_values.append(last_valid)

        # 确保长度为7500
        ecg_data = np.array(ecg_values, dtype=np.float32)
        if len(ecg_data) < 7500:
            ecg_data = np.pad(ecg_data, (0, 7500 - len(ecg_data)), mode='constant')
        elif len(ecg_data) > 7500:
            ecg_data = ecg_data[:7500]

        # 标准化
        ecg_data = self.normalize(ecg_data)

        return ecg_data, labels

    def normalize(self, ecg_data):
        """
        归一化ECG数据

        推荐使用 'zscore' (每个样本独立标准化):
        - 消除不同设备/患者的基线差异
        - 保持波形相对形态
        """
        if self.normalize_method is None:
            return ecg_data

        elif self.normalize_method == 'zscore':
            # Z-score: (x - mean) / std，每个样本独立
            mean = np.mean(ecg_data)
            std = np.std(ecg_data)
            if std > 1e-6:
                ecg_data = (ecg_data - mean) / std
            return ecg_data

        elif self.normalize_method == 'minmax':
            # Min-Max: 归一化到 [0, 1]
            min_val = np.min(ecg_data)
            max_val = np.max(ecg_data)
            if max_val - min_val > 1e-6:
                ecg_data = (ecg_data - min_val) / (max_val - min_val)
            return ecg_data

        elif self.normalize_method == 'robust':
            # 鲁棒标准化: 使用中位数和IQR，对异常值不敏感
            median = np.median(ecg_data)
            q75, q25 = np.percentile(ecg_data, [75, 25])
            iqr = q75 - q25
            if iqr > 1e-6:
                ecg_data = (ecg_data - median) / iqr
            return ecg_data

        elif self.normalize_method == 'global':
            # 全局标准化: 使用预计算的全局mean/std
            if self.global_mean is not None and self.global_std is not None:
                if self.global_std > 1e-6:
                    ecg_data = (ecg_data - self.global_mean) / self.global_std
            return ecg_data

        else:
            return ecg_data


class ECGDataAugmentation:
    """ECG数据增强"""

    def __init__(self, noise_std=0.05, scale_range=(0.9, 1.1), shift_range=(-100, 100)):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.shift_range = shift_range

    def __call__(self, ecg):
        # 随机加噪声
        if np.random.random() < 0.5:
            noise = torch.randn_like(ecg) * self.noise_std
            ecg = ecg + noise

        # 随机缩放
        if np.random.random() < 0.5:
            scale = np.random.uniform(*self.scale_range)
            ecg = ecg * scale

        # 随机平移（时间轴）
        if np.random.random() < 0.3:
            shift = np.random.randint(*self.shift_range)
            ecg = torch.roll(ecg, shifts=shift, dims=0)

        return ecg


# ==================== 预处理数据集 (更快) ====================

class ECGPreprocessedDataset(Dataset):
    """
    从预处理后的文件加载数据集

    支持格式: npz, hdf5, pkl
    比原始文件加载快10-100倍
    """

    def __init__(self, X, Y, transform=None):
        """
        Args:
            X: np.ndarray [N, seq_len] ECG数据
            Y: np.ndarray [N, num_classes] 标签
            transform: 数据增强
        """
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        ecg = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.Y[idx], dtype=torch.float32)

        if self.transform:
            ecg = self.transform(ecg)

        return ecg, label


def load_preprocessed(file_path):
    """
    加载预处理后的数据文件

    Args:
        file_path: npz/hdf5/pkl文件路径

    Returns:
        dict with 'train', 'val', 'test' keys, each containing 'X' and 'Y'
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.npz':
        data = np.load(file_path, allow_pickle=True)
        result = {
            'train': {'X': data['X_train'], 'Y': data['Y_train']},
            'val': {'X': data['X_val'], 'Y': data['Y_val']},
            'test': {'X': data['X_test'], 'Y': data['Y_test']},
        }
        # 加载元数据
        if 'metadata' in data:
            result['metadata'] = json.loads(str(data['metadata'][0]))

    elif ext in ['.h5', '.hdf5']:
        try:
            import h5py
        except ImportError:
            raise ImportError("需要安装h5py: pip install h5py")

        with h5py.File(file_path, 'r') as f:
            result = {
                'train': {'X': f['train/X'][:], 'Y': f['train/Y'][:]},
                'val': {'X': f['val/X'][:], 'Y': f['val/Y'][:]},
                'test': {'X': f['test/X'][:], 'Y': f['test/Y'][:]},
            }
            if 'metadata' in f.attrs:
                result['metadata'] = json.loads(f.attrs['metadata'])

    elif ext == '.pkl':
        import pickle
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        result = {
            'train': {'X': data['train']['X'], 'Y': data['train']['Y']},
            'val': {'X': data['val']['X'], 'Y': data['val']['Y']},
            'test': {'X': data['test']['X'], 'Y': data['test']['Y']},
            'metadata': data.get('metadata', {})
        }

    else:
        raise ValueError(f"不支持的格式: {ext}")

    print(f"加载预处理数据: {file_path}")
    print(f"  训练集: {result['train']['X'].shape[0]} 样本")
    print(f"  验证集: {result['val']['X'].shape[0]} 样本")
    print(f"  测试集: {result['test']['X'].shape[0]} 样本")

    return result


def create_dataloaders_from_preprocessed(file_path, batch_size=64, num_workers=4,
                                          use_augmentation=True):
    """
    从预处理文件创建DataLoader

    Args:
        file_path: 预处理数据文件路径
        batch_size: 批大小
        num_workers: DataLoader进程数
        use_augmentation: 是否对训练集使用数据增强

    Returns:
        train_loader, val_loader, test_loader
    """
    data = load_preprocessed(file_path)

    train_transform = ECGDataAugmentation() if use_augmentation else None

    train_dataset = ECGPreprocessedDataset(
        data['train']['X'], data['train']['Y'], transform=train_transform
    )
    val_dataset = ECGPreprocessedDataset(
        data['val']['X'], data['val']['Y'], transform=None
    )
    test_dataset = ECGPreprocessedDataset(
        data['test']['X'], data['test']['Y'], transform=None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ==================== 原始文件加载 ====================

def load_data(data_dir, test_size=0.1, val_size=0.1, seed=42):
    """
    加载并划分数据集

    Returns:
        train_files, val_files, test_files
    """
    # 获取所有ECG文件
    file_patterns = [
        os.path.join(data_dir, "*.txt"),
        os.path.join(data_dir, "**/*.txt"),
    ]

    all_files = []
    for pattern in file_patterns:
        all_files.extend(glob.glob(pattern, recursive=True))

    all_files = list(set(all_files))
    print(f"找到 {len(all_files)} 个ECG文件")

    # 划分数据集
    train_val_files, test_files = train_test_split(
        all_files, test_size=test_size, random_state=seed
    )
    train_files, val_files = train_test_split(
        train_val_files, test_size=val_size / (1 - test_size), random_state=seed
    )

    print(f"训练集: {len(train_files)}, 验证集: {len(val_files)}, 测试集: {len(test_files)}")

    return train_files, val_files, test_files

def load_data2(data_dir, test_size=0.1, val_size=0.1, seed=42):
    """
    加载并划分数据集

    Returns:
        train_files, val_files, test_files
    """
    # 获取所有ECG文件
    with open(data_dir, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    train_files = splits['train']
    val_files = splits['val']
    test_files = splits['test']

    print(f"训练集: {len(train_files)}, 验证集: {len(val_files)}, 测试集: {len(test_files)}")

    return train_files, val_files, test_files

def create_dataloaders(train_files, val_files, test_files, batch_size=64, num_workers=4):
    """创建DataLoader"""
    train_transform = ECGDataAugmentation()

    train_dataset = ECGDataset(train_files, transform=train_transform)
    val_dataset = ECGDataset(val_files, transform=None)
    test_dataset = ECGDataset(test_files, transform=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def compute_class_weights(file_list, num_classes=40):
    """计算类别权重（处理类别不平衡）"""
    class_counts = np.zeros(num_classes)

    for file_path in tqdm(file_list, desc="统计类别分布"):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        if len(lines) > 2:
            try:
                diag1 = int(lines[2].strip())
                if 1 <= diag1 <= num_classes:
                    class_counts[diag1 - 1] += 1
            except:
                pass

        if len(lines) > 3:
            try:
                diag2 = int(lines[3].strip())
                if 1 <= diag2 <= num_classes:
                    class_counts[diag2 - 1] += 1
            except:
                pass

    # 计算权重: 总样本数 / (类别数 * 类别样本数)
    total = np.sum(class_counts)
    weights = total / (num_classes * (class_counts + 1))  # +1避免除零

    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    # 测试数据加载
    # data_dir = "/Users/zhangyf/Documents/cfel/HLW/HLW_BZ_ddj_001"
    # data_dir = "/Users/zhangyf/PycharmProjects/cfel/plus/ECG-Chat/ecg_classifier/data_splits.json"
    # with open(data_dir, 'r', encoding='utf-8') as f:
    #     splits = json.load(f)
    # train_files = splits['train']
    # print(train_files)
    data = load_preprocessed("ecg_data.pkl")
    print(data['train']['X'][0])

    # train_files, val_files, test_files = load_data(data_dir)
    #
    # # 创建数据集测试
    # dataset = ECGDataset(train_files[:10])
    # ecg, label = dataset[0]
    # print(f"ECG shape: {ecg.shape}, Label shape: {label.shape}")
    # print(f"Labels: {torch.where(label > 0)[0].tolist()}")
