"""
ECG数据预处理脚本

将分散在多个文件夹的ECG数据预处理后存储为统一格式，加速训练加载。

支持输出格式:
- NPZ: numpy压缩格式 (推荐，快速且压缩)
- HDF5: 适合超大数据集，支持分块读取
- PKL: pickle格式

用法:
    # 预处理并保存
    python preprocess.py --data_dirs /path/to/dir1 /path/to/dir2 --output data.npz

    # 指定格式
    python preprocess.py --data_dirs /path/to/dir1 --output data.h5 --format hdf5

    # 使用已有的splits文件
    python preprocess.py --splits data_splits.json --output data.npz
"""

import os
import glob
import json
import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm


def parse_ecg_file(file_path, num_classes=40, seq_len=7500, normalize='zscore'):
    """
    解析单个ECG文件

    Returns:
        dict with 'ecg', 'labels', 'file_path' or None if failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # 解析标签 (第3、4行)
        labels = np.zeros(num_classes, dtype=np.float32)


        for i in range(2, len(lines)):  # 从第3行(索引2)开始
            try:
                val = int(lines[i].strip())
                if val == 250:  # 遇到采样率，标签结束
                    break
                if 1 <= val <= 40:
                    labels[val - 1] = 1.0
            except:
                continue


        # if len(lines) > 2:
        #     try:
        #         diag1 = int(lines[2].strip())
        #         if 1 <= diag1 <= num_classes:
        #             labels[diag1 - 1] = 1.0
        #     except:
        #         pass
        #
        # if len(lines) > 3:
        #     try:
        #         diag2 = int(lines[3].strip())
        #         if 1 <= diag2 <= num_classes:
        #             labels[diag2 - 1] = 1.0
        #     except:
        #         pass

        # 找到32767(起始)和32763(结束)分隔符之间的数据
        data_start = 0
        data_end = len(lines)

        for i, line in enumerate(lines):
            val = line.strip()
            if val == '32767' and data_start == 0:
                data_start = i + 1
            elif val == '32763' and data_start > 0:
                data_end = i
                break

        # 解析ECG数据
        # 策略: 保留所有位置，异常值用前一个有效值填充
        ecg_values = []
        last_valid = 0.0
        invalid_count = 0

        for i in range(data_start, data_end):
            try:
                value = float(lines[i].strip())
                if -32768 <= value <= 32767:  # 有效值
                    ecg_values.append(value)
                    last_valid = value
                else:
                    ecg_values.append(last_valid)  # 用前值填充
                    invalid_count += 1
            except:
                ecg_values.append(last_valid)
                invalid_count += 1

        # 如果异常值超过10%，丢弃该样本
        if len(ecg_values) == 0 or invalid_count / max(len(ecg_values), 1) > 0.1:
            return None

        # 转换为numpy数组并调整长度
        ecg_data = np.array(ecg_values, dtype=np.float32)

        if len(ecg_data) < seq_len:
            ecg_data = np.pad(ecg_data, (0, seq_len - len(ecg_data)), mode='constant')
        elif len(ecg_data) > seq_len:
            ecg_data = ecg_data[:seq_len]

        # 归一化
        if normalize == 'zscore':
            mean = np.mean(ecg_data)
            std = np.std(ecg_data)
            if std > 1e-6:
                ecg_data = (ecg_data - mean) / std
            else:
                # std=0说明信号是平的，异常数据
                return None
        elif normalize == 'minmax':
            min_val = np.min(ecg_data)
            max_val = np.max(ecg_data)
            if max_val - min_val > 1e-6:
                ecg_data = (ecg_data - min_val) / (max_val - min_val)
            else:
                return None

        # 检查归一化后是否有nan/inf
        if np.isnan(ecg_data).any() or np.isinf(ecg_data).any():
            return None

        # 检查归一化后的值范围是否合理 (z-score后应该在[-10, 10]范围内)
        if np.abs(ecg_data).max() > 100:
            return None

        # 过滤全0标签样本 (没有任何诊断标签的样本)
        if labels.sum() == 0:
            return None

        return {
            'ecg': ecg_data,
            'labels': labels,
            'file_path': file_path,
            'ecg_values':ecg_values
        }

    except Exception as e:
        return None


def collect_files(data_dirs):
    """收集所有ECG文件"""
    all_files = []
    for data_dir in data_dirs:
        patterns = [
            os.path.join(data_dir, "*.txt"),
            os.path.join(data_dir, "**/*.txt"),
        ]
        for pattern in patterns:
            all_files.extend(glob.glob(pattern, recursive=True))

    return list(set(all_files))


def preprocess_files(file_list, num_classes=40, seq_len=7500, normalize='zscore',
                     num_workers=8, desc="预处理"):
    """
    并行预处理文件列表

    Returns:
        X: np.ndarray [N, seq_len]
        Y: np.ndarray [N, num_classes]
        paths: list of file paths
    """
    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(parse_ecg_file, f, num_classes, seq_len, normalize): f
            for f in file_list
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            result = future.result()
            if result is not None:
                results.append(result)

    if not results:
        return None, None, None

    # 合并结果
    X = np.stack([r['ecg'] for r in results], axis=0)
    Y = np.stack([r['labels'] for r in results], axis=0)
    paths = [r['file_path'] for r in results]

    return X, Y, paths


def save_npz(output_path, train_data, val_data, test_data, metadata):
    """保存为NPZ格式"""
    np.savez_compressed(
        output_path,
        # 训练集
        X_train=train_data[0],
        Y_train=train_data[1],
        # 验证集
        X_val=val_data[0],
        Y_val=val_data[1],
        # 测试集
        X_test=test_data[0],
        Y_test=test_data[1],
        # 元数据
        train_paths=np.array(train_data[2], dtype=object),
        val_paths=np.array(val_data[2], dtype=object),
        test_paths=np.array(test_data[2], dtype=object),
        metadata=np.array([json.dumps(metadata)], dtype=object)
    )
    print(f"已保存: {output_path}")


def save_hdf5(output_path, train_data, val_data, test_data, metadata):
    """保存为HDF5格式 (适合超大数据集)"""
    try:
        import h5py
    except ImportError:
        print("错误: 需要安装h5py: pip install h5py")
        return

    with h5py.File(output_path, 'w') as f:
        # 训练集
        train_grp = f.create_group('train')
        train_grp.create_dataset('X', data=train_data[0], compression='gzip')
        train_grp.create_dataset('Y', data=train_data[1], compression='gzip')

        # 验证集
        val_grp = f.create_group('val')
        val_grp.create_dataset('X', data=val_data[0], compression='gzip')
        val_grp.create_dataset('Y', data=val_data[1], compression='gzip')

        # 测试集
        test_grp = f.create_group('test')
        test_grp.create_dataset('X', data=test_data[0], compression='gzip')
        test_grp.create_dataset('Y', data=test_data[1], compression='gzip')

        # 元数据
        f.attrs['metadata'] = json.dumps(metadata)

    print(f"已保存: {output_path}")


def save_pkl(output_path, train_data, val_data, test_data, metadata):
    """保存为Pickle格式"""
    data = {
        'train': {'X': train_data[0], 'Y': train_data[1], 'paths': train_data[2]},
        'val': {'X': val_data[0], 'Y': val_data[1], 'paths': val_data[2]},
        'test': {'X': test_data[0], 'Y': test_data[1], 'paths': test_data[2]},
        'metadata': metadata
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='ECG数据预处理')

    # 数据源 (三选一)
    parser.add_argument('--data_dirs', nargs='+', help='数据目录列表')
    parser.add_argument('--splits', type=str, help='已有的splits JSON文件')
    parser.add_argument('--file_list', type=str, help='去重后的文件列表JSON (会自动划分)')

    # 输出
    parser.add_argument('--output', type=str, default='ecg_data.npz', help='输出文件路径')
    parser.add_argument('--format', type=str, default='auto',
                        choices=['auto', 'npz', 'hdf5', 'pkl'],
                        help='输出格式 (auto根据扩展名判断)')

    # 预处理参数
    parser.add_argument('--num_classes', type=int, default=40, help='类别数')
    parser.add_argument('--seq_len', type=int, default=7500, help='序列长度')
    parser.add_argument('--normalize', type=str, default='zscore',
                        choices=['zscore', 'minmax', 'none'], help='归一化方式')
    parser.add_argument('--workers', type=int, default=32, help='并行进程数')

    # 数据划分 (仅当使用--data_dirs时)
    parser.add_argument('--test_size', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    # 确定输出格式
    if args.format == 'auto':
        ext = os.path.splitext(args.output)[1].lower()
        if ext == '.npz':
            output_format = 'npz'
        elif ext in ['.h5', '.hdf5']:
            output_format = 'hdf5'
        elif ext == '.pkl':
            output_format = 'pkl'
        else:
            output_format = 'npz'
            args.output = args.output + '.npz'
    else:
        output_format = args.format

    # 获取文件列表
    if args.splits:
        # 使用已有的splits文件
        print(f"从splits文件加载: {args.splits}")
        with open(args.splits, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        train_files = splits['train']
        val_files = splits['val']
        test_files = splits['test']
    elif args.file_list:
        # 从去重后的文件列表加载并自动划分
        print(f"从文件列表加载: {args.file_list}")
        with open(args.file_list, 'r', encoding='utf-8') as f:
            all_files = json.load(f)
        print(f"加载 {len(all_files)} 个去重文件")

        # 划分数据集
        from sklearn.model_selection import train_test_split
        np.random.seed(args.seed)

        train_val_files, test_files = train_test_split(
            all_files, test_size=args.test_size, random_state=args.seed
        )
        train_files, val_files = train_test_split(
            train_val_files, test_size=args.val_size/(1-args.test_size), random_state=args.seed
        )
    elif args.data_dirs:
        # 从目录收集并划分
        print(f"从目录收集文件: {args.data_dirs}")
        all_files = collect_files(args.data_dirs)
        print(f"找到 {len(all_files)} 个文件")

        # 划分数据集
        from sklearn.model_selection import train_test_split
        np.random.seed(args.seed)

        train_val_files, test_files = train_test_split(
            all_files, test_size=args.test_size, random_state=args.seed
        )
        train_files, val_files = train_test_split(
            train_val_files, test_size=args.val_size/(1-args.test_size), random_state=args.seed
        )
    else:
        print("错误: 请指定 --data_dirs, --file_list 或 --splits")
        return

    print(f"训练集: {len(train_files)}, 验证集: {len(val_files)}, 测试集: {len(test_files)}")

    # 预处理
    print(f"\n开始预处理 (归一化: {args.normalize})...")

    X_train, Y_train, paths_train = preprocess_files(
        train_files, args.num_classes, args.seq_len, args.normalize,
        args.workers, "处理训练集"
    )

    X_val, Y_val, paths_val = preprocess_files(
        val_files, args.num_classes, args.seq_len, args.normalize,
        args.workers, "处理验证集"
    )

    X_test, Y_test, paths_test = preprocess_files(
        test_files, args.num_classes, args.seq_len, args.normalize,
        args.workers, "处理测试集"
    )

    # 统计信息
    print(f"\n预处理完成:")
    print(f"  训练集: {X_train.shape[0]} 样本")
    print(f"  验证集: {X_val.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")
    print(f"  ECG形状: {X_train.shape[1]}")
    print(f"  标签数: {Y_train.shape[1]}")

    # 标签分布
    train_label_counts = Y_train.sum(axis=0)
    print(f"\n训练集标签分布 (top 10):")
    top_labels = np.argsort(train_label_counts)[::-1][:10]
    for idx in top_labels:
        print(f"  类别 {idx+1}: {int(train_label_counts[idx])} 样本")

    # 元数据
    metadata = {
        'num_classes': args.num_classes,
        'seq_len': args.seq_len,
        'normalize': args.normalize,
        'train_size': X_train.shape[0],
        'val_size': X_val.shape[0],
        'test_size': X_test.shape[0],
        'label_counts': train_label_counts.tolist()
    }

    # 保存
    train_data = (X_train, Y_train, paths_train)
    val_data = (X_val, Y_val, paths_val)
    test_data = (X_test, Y_test, paths_test)

    if output_format == 'npz':
        save_npz(args.output, train_data, val_data, test_data, metadata)
    elif output_format == 'hdf5':
        save_hdf5(args.output, train_data, val_data, test_data, metadata)
    elif output_format == 'pkl':
        save_pkl(args.output, train_data, val_data, test_data, metadata)

    # 估算文件大小
    total_samples = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    estimated_size_mb = (total_samples * args.seq_len * 4 + total_samples * args.num_classes * 4) / 1024 / 1024
    print(f"\n预估未压缩大小: {estimated_size_mb:.1f} MB")

    if os.path.exists(args.output):
        actual_size_mb = os.path.getsize(args.output) / 1024 / 1024
        print(f"实际文件大小: {actual_size_mb:.1f} MB")
        print(f"压缩率: {actual_size_mb/estimated_size_mb*100:.1f}%")


if __name__ == "__main__":
    main()
    # out1 = parse_ecg_file("/Users/zhangyf/Documents/cfel/HLW/精标数据备份（心电）/标注数据细分类/HLW_BZ_ddj_040/HLWddj5072932151.txt")
    # out2 = parse_ecg_file("/Users/zhangyf/Documents/cfel/HLW/精标数据备份（心电）/标注数据细分类/HLW_BZ_ddj_034/HLWddj4472932151.txt")
    # print(ecg_values)
