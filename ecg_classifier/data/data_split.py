"""
数据集划分模块

支持:
1. 多标签分层抽样
2. 类别平衡采样
3. 交叉验证划分

用法:
    python data_split.py --data_dirs /path/to/dir1 /path/to/dir2 --output splits.json
"""

import os
import glob
import json
import argparse
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def parse_labels(file_path):
    """快速解析文件标签

    数据格式:
    - 第1-2行: 其他信息
    - 第3行起: 标签 (1-40的整数)
    - 遇到250 (采样率) 停止
    - 然后是30 (时长), 32767 (分隔符), ECG数据...
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        labels = []
        for i in range(2, len(lines)):  # 从第3行(索引2)开始
            try:
                val = int(lines[i].strip())
                if val == 250:  # 遇到采样率，标签结束
                    break
                if 1 <= val <= 40:
                    labels.append(val)
            except:
                continue

        return file_path, tuple(sorted(labels)) if labels else None

    except Exception as e:
        return file_path, None


def collect_files_with_labels(data_dirs, num_workers=8):
    """收集所有文件及其标签"""

    # 收集所有文件
    all_files = []
    for data_dir in data_dirs:
        patterns = [
            os.path.join(data_dir, "*.txt"),
            os.path.join(data_dir, "**/*.txt"),
        ]
        for pattern in patterns:
            all_files.extend(glob.glob(pattern, recursive=True))

    all_files = list(set(all_files))
    print(f"找到 {len(all_files)} 个文件")

    # 并行解析标签
    file_labels = {}
    print(f"使用 {num_workers} 个进程解析标签...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(parse_labels, f): f for f in all_files}

        for future in tqdm(as_completed(futures), total=len(futures)):
            file_path, labels = future.result()
            if labels is not None:
                file_labels[file_path] = labels

    print(f"成功解析: {len(file_labels)} 个文件")
    return file_labels


def stratified_split_multilabel(file_labels, test_size=0.1, val_size=0.1, seed=42):
    """
    多标签分层抽样

    策略:
    1. 按标签组合分组 (如 (1,14) 表示同时有标签1和14)
    2. 每个组内按比例划分
    3. 确保每个类别在train/val/test中都有样本

    Args:
        file_labels: {file_path: (label1, label2, ...)}
        test_size: 测试集比例
        val_size: 验证集比例
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)

    # 按标签组合分组
    label_groups = defaultdict(list)
    for file_path, labels in file_labels.items():
        label_groups[labels].append(file_path)

    print(f"标签组合数: {len(label_groups)}")

    train_files = []
    val_files = []
    test_files = []

    # 统计每个类别的最终分布
    train_label_counts = Counter()
    val_label_counts = Counter()
    test_label_counts = Counter()

    for labels, files in tqdm(label_groups.items(), desc="分层划分"):
        n = len(files)

        if n < 3:
            # 样本太少，全部放入训练集
            train_files.extend(files)
            for label in labels:
                train_label_counts[label] += n
            continue

        # 打乱
        random.shuffle(files)

        # 计算划分数量
        n_test = max(1, int(n * test_size))
        n_val = max(1, int(n * val_size))
        n_train = n - n_test - n_val

        if n_train < 1:
            # 样本不够分三份，优先保证训练集
            n_train = max(1, n - 2)
            n_val = min(1, n - n_train)
            n_test = n - n_train - n_val

        # 划分
        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:n_train + n_val])
        test_files.extend(files[n_train + n_val:])

        # 统计
        for label in labels:
            train_label_counts[label] += n_train
            val_label_counts[label] += n_val
            test_label_counts[label] += n_test

    # 打乱最终列表
    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)

    # 验证分布
    print(f"\n划分结果:")
    print(f"  训练集: {len(train_files)} ({len(train_files)/len(file_labels)*100:.1f}%)")
    print(f"  验证集: {len(val_files)} ({len(val_files)/len(file_labels)*100:.1f}%)")
    print(f"  测试集: {len(test_files)} ({len(test_files)/len(file_labels)*100:.1f}%)")

    return train_files, val_files, test_files, {
        'train_label_counts': dict(train_label_counts),
        'val_label_counts': dict(val_label_counts),
        'test_label_counts': dict(test_label_counts),
    }


def verify_split_quality(train_files, val_files, test_files, label_stats):
    """验证划分质量"""

    print("\n【划分质量验证】")

    # 1. 检查类别覆盖
    train_labels = set(label_stats['train_label_counts'].keys())
    val_labels = set(label_stats['val_label_counts'].keys())
    test_labels = set(label_stats['test_label_counts'].keys())

    all_labels = train_labels | val_labels | test_labels

    missing_in_train = all_labels - train_labels
    missing_in_val = all_labels - val_labels
    missing_in_test = all_labels - test_labels

    if missing_in_train:
        print(f"  ⚠️ 训练集缺少类别: {missing_in_train}")
    if missing_in_val:
        print(f"  ⚠️ 验证集缺少类别: {missing_in_val}")
    if missing_in_test:
        print(f"  ⚠️ 测试集缺少类别: {missing_in_test}")

    if not missing_in_train and not missing_in_val and not missing_in_test:
        print("  ✓ 所有类别在三个集合中都有样本")

    # 2. 检查比例一致性
    print("\n  各类别在三个集合中的比例:")
    print(f"  {'类别':<6} {'训练集':>10} {'验证集':>10} {'测试集':>10} {'比例偏差'}")
    print("-" * 50)

    max_deviation = 0
    for label in sorted(all_labels):
        train_c = label_stats['train_label_counts'].get(label, 0)
        val_c = label_stats['val_label_counts'].get(label, 0)
        test_c = label_stats['test_label_counts'].get(label, 0)
        total = train_c + val_c + test_c

        if total == 0:
            continue

        train_ratio = train_c / total
        val_ratio = val_c / total
        test_ratio = test_c / total

        # 期望比例 (假设80/10/10)
        expected_train = 0.8
        expected_val = 0.1
        expected_test = 0.1

        deviation = abs(train_ratio - expected_train) + abs(val_ratio - expected_val) + abs(test_ratio - expected_test)
        max_deviation = max(max_deviation, deviation)

        status = "✓" if deviation < 0.15 else "⚠️"
        print(f"  {label:<6} {train_ratio*100:>9.1f}% {val_ratio*100:>9.1f}% {test_ratio*100:>9.1f}% {status}")

    print(f"\n  最大比例偏差: {max_deviation:.2f}")
    if max_deviation < 0.2:
        print("  ✓ 分层抽样质量良好")
    else:
        print("  ⚠️ 部分类别比例偏差较大（可能是样本太少导致）")


def create_kfold_splits(file_labels, k=5, seed=42):
    """
    创建K折交叉验证划分

    Args:
        file_labels: {file_path: (label1, label2, ...)}
        k: 折数
        seed: 随机种子

    Returns:
        list of (train_files, val_files) for each fold
    """
    random.seed(seed)
    np.random.seed(seed)

    # 按标签组合分组
    label_groups = defaultdict(list)
    for file_path, labels in file_labels.items():
        label_groups[labels].append(file_path)

    # 为每个组创建K折索引
    folds = [[] for _ in range(k)]

    for labels, files in label_groups.items():
        random.shuffle(files)
        # 轮流分配到各折
        for i, f in enumerate(files):
            folds[i % k].append(f)

    # 生成K折划分
    splits = []
    for i in range(k):
        val_files = folds[i]
        train_files = []
        for j in range(k):
            if j != i:
                train_files.extend(folds[j])
        random.shuffle(train_files)
        splits.append((train_files, val_files))

    return splits


def save_splits(train_files, val_files, test_files, output_path):
    """保存划分结果"""
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files,
        'stats': {
            'train_size': len(train_files),
            'val_size': len(val_files),
            'test_size': len(test_files),
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

    print(f"\n划分结果已保存: {output_path}")


def load_splits(splits_path):
    """加载划分结果"""
    with open(splits_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    return splits['train'], splits['val'], splits['test']


def main():
    parser = argparse.ArgumentParser(description='数据集分层划分')
    parser.add_argument('--data_dirs', nargs='+', required=True, help='数据目录列表')
    parser.add_argument('--output', type=str, default='data_splits.json', help='输出文件')
    parser.add_argument('--test_size', type=float, default=0.05, help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.05, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--workers', type=int, default=32, help='并行进程数')
    args = parser.parse_args()

    # 收集文件和标签
    file_labels = collect_files_with_labels(args.data_dirs, args.workers)

    # 分层划分
    train_files, val_files, test_files, label_stats = stratified_split_multilabel(
        file_labels,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed
    )

    # 验证质量
    verify_split_quality(train_files, val_files, test_files, label_stats)

    # 保存结果
    save_splits(train_files, val_files, test_files, args.output)

    # 同时保存详细统计
    stats_path = args.output.replace('.json', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(label_stats, f, ensure_ascii=False, indent=2)
    print(f"标签统计已保存: {stats_path}")


if __name__ == "__main__":
    main()
