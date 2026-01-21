"""
数据去重脚本 - 稳定哈希版本

修复哈希不一致问题，使用MD5确保稳定性
"""

import os
import glob
import json
import argparse
import hashlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm


def parse_ecg_signal_stable(file_path):
    """
    解析ECG文件，返回信号数据用于去重检查

    关键改进：使用稳定的MD5哈希算法
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # 找到数据区间
        data_start = 0
        data_end = len(lines)

        for i, line in enumerate(lines):
            val = line.strip()
            if val == '32767' and data_start == 0:
                data_start = i + 1
            elif val == '32763' and data_start > 0:
                data_end = i
                break

        # 解析数据
        ecg_values = []
        last_valid = 0.0

        for i in range(data_start, data_end):
            try:
                value = float(lines[i].strip())
                if -32768 <= value <= 32767:
                    ecg_values.append(value)
                    last_valid = value
                else:
                    ecg_values.append(last_valid)
            except:
                ecg_values.append(last_valid)

        if len(ecg_values) == 0:
            return None

        # 关键改进：使用MD5哈希，确保稳定性
        ecg_array = np.array(ecg_values, dtype=np.float32)

        # 使用MD5而不是Python内置的hash()
        signal_hash = hashlib.md5(ecg_array.tobytes()).hexdigest()

        file_size = os.path.getsize(file_path)

        return (file_path, signal_hash, file_size)

    except Exception as e:
        print(f"解析失败: {file_path}, 错误: {e}")
        return None


def deduplicate_files_stable(file_list, num_workers=32):
    """
    去重文件列表 - 稳定版本
    """
    print(f"\n处理 {len(file_list)} 个文件...")

    # 并行解析文件
    results = []
    failed_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(parse_ecg_signal_stable, f): f for f in file_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc="解析文件"):
            result = future.result()
            if result:
                results.append(result)
            else:
                failed_count += 1

    print(f"成功解析: {len(results)} 个文件")
    if failed_count > 0:
        print(f"失败文件: {failed_count} 个")

    # 按哈希分组
    hash_groups = defaultdict(list)
    for file_path, signal_hash, file_size in results:
        hash_groups[signal_hash].append((file_path, file_size))

    # 统计
    unique_count = len(hash_groups)
    duplicate_groups = {h: files for h, files in hash_groups.items() if len(files) > 1}
    total_duplicates = sum(len(files) - 1 for files in duplicate_groups.values())

    print(f"\n去重结果:")
    print(f"  解析成功: {len(results):,} 个文件")
    print(f"  唯一信号数: {unique_count:,}")
    print(f"  重复组数: {len(duplicate_groups):,}")
    print(f"  重复文件数: {total_duplicates:,}")
    if len(results) > 0:
        print(f"  去重率: {total_duplicates/len(results)*100:.2f}%")

    # 从每组中选择一个文件（选择文件名最短的，通常是原始文件）
    unique_files = []
    for signal_hash, files in hash_groups.items():
        # 按文件名长度排序，选择最短的
        files.sort(key=lambda x: (len(x[0]), x[0]))
        unique_files.append(files[0][0])

    return unique_files, duplicate_groups

def parse_ecg_signal_with_labels(file_path, num_classes=40):
    """
    解析ECG文件，返回信号数据和标签用于去重检查

    关键改进：
    1. 使用稳定的MD5哈希算法
    2. 同时解析ECG信号和标签
    3. 组合信号+标签作为唯一标识
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # ===== 解析标签（与preprocess.py相同逻辑） =====
        labels = []

        for i in range(2, len(lines)):  # 从第3行(索引2)开始
            try:
                val = int(lines[i].strip())
                if val == 250:  # 遇到采样率，标签结束
                    break
                if 1 <= val <= num_classes:
                    labels.append(val)
            except:
                continue

        # ===== 解析ECG信号 =====
        # 找到数据区间
        data_start = 0
        data_end = len(lines)

        for i, line in enumerate(lines):
            val = line.strip()
            if val == '32767' and data_start == 0:
                data_start = i + 1
            elif val == '32763' and data_start > 0:
                data_end = i
                break

        # 解析数据
        ecg_values = []
        last_valid = 0.0

        for i in range(data_start, data_end):
            try:
                value = float(lines[i].strip())
                if -32768 <= value <= 32767:
                    ecg_values.append(value)
                    last_valid = value
                else:
                    ecg_values.append(last_valid)
            except:
                ecg_values.append(last_valid)

        if len(ecg_values) == 0:
            return None

        # ===== 创建组合哈希（信号+标签） =====
        ecg_array = np.array(ecg_values, dtype=np.float32)

        # 将标签转换为数组
        labels_array = np.array(labels, dtype=np.int32)

        # 组合信号和标签数据
        combined_data = np.concatenate([ecg_array, labels_array.astype(np.float32)])

        # 使用MD5计算组合哈希
        content_hash = hashlib.md5(combined_data.tobytes()).hexdigest()

        file_size = os.path.getsize(file_path)

        return (file_path, content_hash, file_size, labels)

    except Exception as e:
        print(f"解析失败: {file_path}, 错误: {e}")
        return None


def deduplicate_files_with_labels(file_list, num_classes=40, num_workers=32):
    """
    去重文件列表 - ECG+标签组合版本

    关键改进：
    1. 基于ECG信号+标签组合进行去重
    2. 只有信号和标签都相同才认为是重复
    """
    print(f"\n处理 {len(file_list)} 个文件...")
    print(f"  类别数: {num_classes}")

    # 并行解析文件（包含标签）
    results = []
    failed_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(parse_ecg_signal_with_labels, f, num_classes): f
            for f in file_list
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="解析文件（含标签）"):
            result = future.result()
            if result:
                results.append(result)
            else:
                failed_count += 1

    print(f"成功解析: {len(results)} 个文件")
    if failed_count > 0:
        print(f"失败文件: {failed_count} 个")

    # 按组合哈希分组
    hash_groups = defaultdict(list)
    labels_info = {}  # 存储每个哈希对应的标签信息
    for file_path, content_hash, file_size, labels in results:
        hash_groups[content_hash].append((file_path, file_size))
        if content_hash not in labels_info:
            labels_info[content_hash] = labels

    # 统计
    unique_count = len(hash_groups)
    duplicate_groups = {h: files for h, files in hash_groups.items() if len(files) > 1}
    total_duplicates = sum(len(files) - 1 for files in duplicate_groups.values())

    print(f"\n去重结果:")
    print(f"  解析成功: {len(results):,} 个文件")
    print(f"  唯一组合数: {unique_count:,}")
    print(f"  重复组数: {len(duplicate_groups):,}")
    print(f"  重复文件数: {total_duplicates:,}")
    if len(results) > 0:
        print(f"  去重率: {total_duplicates/len(results)*100:.2f}%")

    # 打印标签分布示例
    print(f"\n标签分布示例（前10个唯一组合）:")
    for i, (content_hash, files) in enumerate(list(hash_groups.items())[:10], 1):
        labels = labels_info[content_hash]
        print(f"  组合{i}: 标签={labels} 文件数={len(files)}")

    # 从每组中选择一个文件（选择文件名最短的，通常是原始文件）
    unique_files = []
    for content_hash, files in hash_groups.items():
        # 按文件名长度排序，选择最短的
        files.sort(key=lambda x: (len(x[0]), x[0]))
        unique_files.append(files[0][0])

    return unique_files, duplicate_groups, labels_info


def verify_stable_hash():
    """验证稳定哈希的一致性"""
    import numpy as np

    # 创建测试数据
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    # 多次计算哈希
    hashes = []
    for _ in range(5):
        h = hashlib.md5(test_data.tobytes()).hexdigest()
        hashes.append(h)

    print("稳定哈希测试:")
    print(f"测试数据: {test_data}")
    print(f"5次MD5哈希结果:")
    for i, h in enumerate(hashes):
        print(f"  {i+1}. {h}")

    all_same = len(set(hashes)) == 1
    print(f"所有哈希相同: {'✅' if all_same else '❌'}")

    return all_same


def main():
    parser = argparse.ArgumentParser(description='ECG数据去重（ECG+标签组合版）')
    parser.add_argument('--data_dirs', nargs='+', required=True, help='数据目录列表')
    parser.add_argument('--output', type=str, default='unique_files_with_labels.json', help='输出文件列表')
    parser.add_argument('--workers', type=int, default=32, help='并行进程数')
    parser.add_argument('--save_duplicates', action='store_true', help='保存重复文件信息')
    parser.add_argument('--verify_hash', action='store_true', help='验证哈希稳定性')
    parser.add_argument('--with_labels', action='store_true', help='使用ECG+标签组合去重')
    parser.add_argument('--num_classes', type=int, default=40, help='类别数')

    args = parser.parse_args()

    if args.verify_hash:
        verify_stable_hash()
        return

    print("=" * 70)
    if args.with_labels:
        print("🚀 ECG数据去重（ECG+标签组合版）")
        print("关键改进：基于ECG信号+标签组合进行去重")
        print("只有信号和标签都相同才认为是重复")
    else:
        print("🚀 ECG数据去重（稳定哈希版）")
        print("关键改进：使用MD5替代Python内置hash()，确保一致性")
    print("=" * 70)

    # 收集文件
    def collect_files(data_dirs):
        all_files = []
        for data_dir in data_dirs:
            patterns = [
                os.path.join(data_dir, "*.txt"),
                os.path.join(data_dir, "**/*.txt"),
            ]
            for pattern in patterns:
                all_files.extend(glob.glob(pattern, recursive=True))
        return list(set(all_files))

    print("\n收集文件...")
    all_files = collect_files(args.data_dirs)
    print(f"找到 {len(all_files)} 个文件")

    # 去重
    if args.with_labels:
        unique_files, duplicate_groups, labels_info = deduplicate_files_with_labels(
            all_files, args.num_classes, args.workers
        )
    else:
        unique_files, duplicate_groups = deduplicate_files_stable(all_files, args.workers)

    # 保存唯一文件列表
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(unique_files, f, ensure_ascii=False, indent=2)

    print(f"\n唯一文件列表已保存: {args.output}")
    print(f"  包含 {len(unique_files):,} 个文件")

    # 保存重复信息
    if args.save_duplicates and duplicate_groups:
        dup_output = args.output.replace('.json', '_duplicates.json')

        # 转换为可序列化的格式
        dup_info = []
        if args.with_labels:
            # ECG+标签版本
            for content_hash, files in list(duplicate_groups.items()):  # 只保存前100组
                labels = labels_info[content_hash]
                dup_info.append({
                    'hash': content_hash,
                    'labels': labels,
                    'count': len(files),
                    'files': [f[0] for f in files],
                    'sizes': [f[1] for f in files]
                })
        else:
            # 仅ECG版本
            for signal_hash, files in list(duplicate_groups.items()):  # 只保存前100组
                dup_info.append({
                    'hash': signal_hash,
                    'count': len(files),
                    'files': [f[0] for f in files],
                    'sizes': [f[1] for f in files]
                })

        with open(dup_output, 'w', encoding='utf-8') as f:
            json.dump(dup_info, f, ensure_ascii=False, indent=2)

        print(f"重复文件信息已保存: {dup_output}")

    # 打印重复示例
    if duplicate_groups:
        print(f"\n重复文件示例 (前5组):")
        if args.with_labels:
            # ECG+标签版本
            for i, (content_hash, files) in enumerate(list(duplicate_groups.items())[:5], 1):
                labels = labels_info[content_hash]
                print(f"\n  组{i}: 标签={labels} 哈希={content_hash[:16]}... {len(files)} 个相同文件")
                for f, size in files[:3]:
                    print(f"    - {f} ({size} bytes)")
                if len(files) > 3:
                    print(f"    ... 还有 {len(files)-3} 个文件")
        else:
            # 仅ECG版本
            for i, (signal_hash, files) in enumerate(list(duplicate_groups.items())[:5], 1):
                print(f"\n  组{i}: 哈希={signal_hash[:16]}... {len(files)} 个相同文件")
                for f, size in files[:3]:
                    print(f"    - {f} ({size} bytes)")
                if len(files) > 3:
                    print(f"    ... 还有 {len(files)-3} 个文件")



if __name__ == "__main__":
    main()