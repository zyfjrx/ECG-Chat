"""
ECG数据分析脚本

功能:
1. 统计类别分布
2. 分析ECG信号数值分布
3. 检查数据质量
4. 生成分析报告

用法:
    python data_analysis.py --data_dirs /path/to/dir1 /path/to/dir2 ...
"""

import os
import glob
import argparse
import json
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'STKaiti', 'Arial Unicode MS']  # 优先使用苹方，其次是楷体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 诊断类型映射
DIAGNOSIS_NAMES = {
    1: "窦性心律", 2: "心电图未见异常", 3: "窦性心动过速", 4: "窦性心动过缓",
    5: "窦性停搏", 6: "心房颤动", 7: "房性早搏", 8: "偶发房性早搏",
    9: "频发房性早搏", 10: "房性早搏二联律", 11: "房性早搏三联律", 12: "成对房性早搏",
    13: "短阵房性心动过速", 14: "室性早搏", 15: "偶发室性早搏", 16: "频发室性早搏",
    17: "室性早搏二联律", 18: "室性早搏三联律", 19: "成对室性早搏", 20: "短阵室性心动过速",
    21: "室上性心动过速", 22: "一度房室阻滞", 23: "ST段抬高", 24: "ST段压低",
    25: "QT/QTc间期延长", 26: "RR长间歇", 27: "心室内差异传导", 28: "干扰波",
    29: "导联脱落", 30: "心房扑动", 31: "短PR间期", 32: "二度Ⅱ型房室阻滞",
    33: "P波增高", 34: "P波增宽", 35: "疑似左右手反接心电图", 36: "R波高电压",
    37: "室内阻滞", 38: "T波改变", 39: "短QT/QTc间期", 40: "心电图未见明显异常",
}


def parse_single_file(file_path):
    """解析单个文件，返回标签和信号统计信息"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # 解析标签
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

        # if len(lines) > 2:
        #     try:
        #         diag1 = int(lines[2].strip())
        #         if 1 <= diag1 <= 40:
        #             labels.append(diag1)
        #     except:
        #         pass
        #
        # if len(lines) > 3:
        #     try:
        #         diag2 = int(lines[3].strip())
        #         if 1 <= diag2 <= 40:
        #             labels.append(diag2)
        #     except:
        #         pass

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
        last_valid = 0.0
        invalid_count = 0
        total_points = 0

        for i in range(data_start, data_end):
            total_points += 1
            try:
                value = float(lines[i].strip())
                if -32768 <= value <= 32767:  # 有效值
                    ecg_values.append(value)
                    last_valid = value
                else:
                    ecg_values.append(last_valid)
                    invalid_count += 1
            except:
                ecg_values.append(last_valid)
                invalid_count += 1

        if len(ecg_values) == 0:
            return None

        ecg_array = np.array(ecg_values, dtype=np.float32)

        # 信号统计
        signal_stats = {
            'length': len(ecg_array),
            'min': float(np.min(ecg_array)),
            'max': float(np.max(ecg_array)),
            'mean': float(np.mean(ecg_array)),
            'std': float(np.std(ecg_array)),
            'has_nan': bool(np.isnan(ecg_array).any()),
            'has_inf': bool(np.isinf(ecg_array).any()),
            'invalid_count': invalid_count,  # 异常值数量
            'invalid_ratio': invalid_count / total_points if total_points > 0 else 0,  # 异常值比例
        }

        return {
            'file': file_path,
            'labels': labels,
            'signal_stats': signal_stats
        }

    except Exception as e:
        return {'file': file_path, 'error': str(e)}


def analyze_data(data_dirs, num_workers=8, sample_ratio=1.0):
    """
    分析数据分布

    Args:
        data_dirs: 数据目录列表
        num_workers: 并行处理进程数
        sample_ratio: 采样比例 (1.0表示全部分析)
    """
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

    # 采样
    if sample_ratio < 1.0:
        np.random.shuffle(all_files)
        sample_size = int(len(all_files) * sample_ratio)
        all_files = all_files[:sample_size]
        print(f"采样 {sample_size} 个文件进行分析")

    # 并行解析
    results = []
    errors = []

    print(f"使用 {num_workers} 个进程并行分析...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(parse_single_file, f): f for f in all_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="分析进度"):
            result = future.result()
            if result is None:
                continue
            if 'error' in result:
                errors.append(result)
            else:
                results.append(result)

    print(f"\n成功解析: {len(results)} 个文件")
    print(f"解析失败: {len(errors)} 个文件")

    return results, errors


def compute_statistics(results):
    """计算统计信息"""

    # ==================== 类别分布统计 ====================
    label_counts = Counter()
    label_cooccurrence = defaultdict(Counter)  # 标签共现
    label_combinations = Counter()  # 所有标签组合
    multi_label_count = 0
    single_label_count = 0
    zero_label_count = 0
    label_count_distribution = Counter()  # 标签数量分布

    for r in results:
        labels = r['labels']

        # 统计标签数量分布
        label_count_distribution[len(labels)] += 1

        if len(labels) == 0:
            zero_label_count += 1
            continue

        if len(labels) > 1:
            multi_label_count += 1
        else:
            single_label_count += 1

        for label in labels:
            label_counts[label] += 1

        # 标签组合统计 - 将标签排序后作为key
        combo_key = tuple(sorted(labels))
        label_combinations[combo_key] += 1

        # 共现统计 (仅双标签)
        if len(labels) == 2:
            label_cooccurrence[labels[0]][labels[1]] += 1
            label_cooccurrence[labels[1]][labels[0]] += 1

    # ==================== 信号分布统计 ====================
    lengths = []
    mins = []
    maxs = []
    means = []
    stds = []
    nan_count = 0
    inf_count = 0
    invalid_counts = []  # 每个样本的异常值数量
    invalid_ratios = []  # 每个样本的异常值比例

    for r in results:
        stats = r['signal_stats']
        lengths.append(stats['length'])
        mins.append(stats['min'])
        maxs.append(stats['max'])
        means.append(stats['mean'])
        stds.append(stats['std'])
        if stats['has_nan']:
            nan_count += 1
        if stats['has_inf']:
            inf_count += 1
        invalid_counts.append(stats.get('invalid_count', 0))
        invalid_ratios.append(stats.get('invalid_ratio', 0))

    signal_distribution = {
        'length': {
            'min': int(np.min(lengths)),
            'max': int(np.max(lengths)),
            'mean': float(np.mean(lengths)),
            'std': float(np.std(lengths)),
            'expected': 7500,
            'correct_ratio': sum(1 for l in lengths if l == 7500) / len(lengths)
        },
        'value_range': {
            'global_min': float(np.min(mins)),
            'global_max': float(np.max(maxs)),
            'mean_of_means': float(np.mean(means)),
            'mean_of_stds': float(np.mean(stds)),
            'percentile_1': float(np.percentile(mins, 1)),
            'percentile_99': float(np.percentile(maxs, 99)),
            'percentile_5': float(np.percentile(mins, 5)),
            'percentile_95': float(np.percentile(maxs, 95)),
        },
        'quality': {
            'nan_count': nan_count,
            'inf_count': inf_count,
            'nan_ratio': nan_count / len(results),
            'inf_ratio': inf_count / len(results),
            'invalid_total': int(np.sum(invalid_counts)),  # 总异常点数
            'invalid_samples': sum(1 for c in invalid_counts if c > 0),  # 有异常值的样本数
            'invalid_mean_ratio': float(np.mean(invalid_ratios)),  # 平均异常值比例
            'invalid_max_ratio': float(np.max(invalid_ratios)) if invalid_ratios else 0,  # 最大异常值比例
        }
    }

    return {
        'total_samples': len(results),
        'single_label_count': single_label_count,
        'multi_label_count': multi_label_count,
        'zero_label_count': zero_label_count,
        'multi_label_ratio': multi_label_count / len(results) if results else 0,
        'label_counts': dict(label_counts),
        'label_cooccurrence': {k: dict(v) for k, v in label_cooccurrence.items()},
        'label_combinations': {str(k): v for k, v in label_combinations.items()},  # 所有标签组合
        'label_count_distribution': dict(label_count_distribution),  # 标签数量分布
        'signal_distribution': signal_distribution,
    }


def print_report(stats):
    """打印分析报告"""

    print("\n" + "=" * 70)
    print("ECG数据分析报告")
    print("=" * 70)

    # 基本信息
    print(f"\n【基本信息】")
    print(f"  总样本数: {stats['total_samples']:,}")
    print(f"  单标签样本: {stats['single_label_count']:,} ({stats['single_label_count']/stats['total_samples']*100:.1f}%)")
    print(f"  多标签样本: {stats['multi_label_count']:,} ({stats['multi_label_ratio']*100:.1f}%)")

    # 类别分布
    print(f"\n【类别分布】")
    print(f"  {'ID':<4} {'诊断名称':<20} {'样本数':>10} {'占比':>8} {'状态'}")
    print("-" * 60)

    label_counts = stats['label_counts']
    total = stats['total_samples']
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

    for label_id, count in sorted_labels:
        name = DIAGNOSIS_NAMES.get(label_id, f"未知{label_id}")
        ratio = count / total * 100

        # 状态判断
        if count < 100:
            status = "⚠️ 极少"
        elif count < 1000:
            status = "⚠️ 较少"
        elif ratio > 30:
            status = "📊 主导"
        else:
            status = "✓ 正常"

        print(f"  {label_id:<4} {name:<20} {count:>10,} {ratio:>7.2f}% {status}")

    # 未出现的类别
    missing_labels = set(range(1, 41)) - set(label_counts.keys())
    if missing_labels:
        print(f"\n  ⚠️ 未出现的类别: {sorted(missing_labels)}")

    # 类别不平衡分析
    if label_counts:
        max_count = max(label_counts.values())
        min_count = min(label_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"\n  类别不平衡比: {imbalance_ratio:.1f}:1 (最多/最少)")

    # 标签数量分布
    print(f"\n【每样本标签数量分布】")
    label_count_dist = stats.get('label_count_distribution', {})
    for num_labels in sorted(label_count_dist.keys()):
        count = label_count_dist[num_labels]
        ratio = count / stats['total_samples'] * 100
        print(f"  {num_labels}个标签: {count:,} 样本 ({ratio:.2f}%)")

    if stats.get('zero_label_count', 0) > 0:
        print(f"  ⚠️ 无标签样本: {stats['zero_label_count']:,}")

    # 所有标签组合 (完整统计)
    print(f"\n【所有标签组合 (完整列表)】")
    label_combinations = stats.get('label_combinations', {})

    # 按出现次数排序
    sorted_combos = sorted(label_combinations.items(), key=lambda x: x[1], reverse=True)

    # 分组显示：单标签、双标签、多标签
    single_combos = [(k, v) for k, v in sorted_combos if len(eval(k)) == 1]
    double_combos = [(k, v) for k, v in sorted_combos if len(eval(k)) == 2]
    multi_combos = [(k, v) for k, v in sorted_combos if len(eval(k)) > 2]

    print(f"\n  --- 单标签组合 ({len(single_combos)}种) ---")
    for combo_str, count in single_combos:
        combo = eval(combo_str)
        names = [DIAGNOSIS_NAMES.get(l, f"类别{l}") for l in combo]
        print(f"  [{combo[0]:2d}] {names[0]}: {count:,}")

    print(f"\n  --- 双标签组合 ({len(double_combos)}种) ---")
    for combo_str, count in double_combos:
        combo = eval(combo_str)
        names = [DIAGNOSIS_NAMES.get(l, f"类别{l}")[:12] for l in combo]
        print(f"  [{combo[0]:2d},{combo[1]:2d}] {names[0]} + {names[1]}: {count:,}")

    if multi_combos:
        print(f"\n  --- 三标签及以上组合 ({len(multi_combos)}种) ---")
        for combo_str, count in multi_combos:
            combo = eval(combo_str)
            label_ids = ','.join(str(l) for l in combo)
            names = ' + '.join(DIAGNOSIS_NAMES.get(l, f"类别{l}")[:8] for l in combo)
            print(f"  [{label_ids}] {names}: {count:,}")

    print(f"\n  共 {len(sorted_combos)} 种不同的标签组合")

    # 信号分布
    print(f"\n【ECG信号分布】")
    sig = stats['signal_distribution']

    print(f"  序列长度:")
    print(f"    预期: {sig['length']['expected']}")
    print(f"    实际: {sig['length']['min']} ~ {sig['length']['max']}")
    print(f"    正确率: {sig['length']['correct_ratio']*100:.1f}%")

    print(f"\n  ECG信号数值范围 (基于分位数，排除异常值):")
    print(f"    1%-99%分位: [{sig['value_range']['percentile_1']:.2f}, {sig['value_range']['percentile_99']:.2f}]")
    print(f"    5%-95%分位: [{sig['value_range']['percentile_5']:.2f}, {sig['value_range']['percentile_95']:.2f}]")
    print(f"    各样本均值的均值: {sig['value_range']['mean_of_means']:.4f}")
    print(f"    各样本标准差的均值: {sig['value_range']['mean_of_stds']:.4f}")
    print(f"    (全局极值: [{sig['value_range']['global_min']:.2f}, {sig['value_range']['global_max']:.2f}] - 可能含边界值)")

    print(f"\n  数据质量:")
    print(f"    含NaN: {sig['quality']['nan_count']} ({sig['quality']['nan_ratio']*100:.2f}%)")
    print(f"    含Inf: {sig['quality']['inf_count']} ({sig['quality']['inf_ratio']*100:.2f}%)")
    print(f"    异常值点位: {sig['quality']['invalid_total']:,} (已用前值填充)")
    print(f"    含异常值的样本: {sig['quality']['invalid_samples']} ({sig['quality']['invalid_samples']/stats['total_samples']*100:.2f}%)")
    print(f"    平均异常值比例: {sig['quality']['invalid_mean_ratio']*100:.4f}%")
    if sig['quality']['invalid_max_ratio'] > 0.01:
        print(f"    ⚠️ 最大异常值比例: {sig['quality']['invalid_max_ratio']*100:.2f}% (>1%需关注)")

    # 归一化建议
    print(f"\n【归一化建议】")
    if abs(sig['value_range']['mean_of_means']) < 0.1 and 0.5 < sig['value_range']['mean_of_stds'] < 2:
        print("  ✓ 数据已接近标准化，可以直接使用或做轻微调整")
    else:
        print("  ⚠️ 建议进行Z-score标准化:")
        print(f"     normalized = (x - mean) / std")
        print(f"     每个样本独立标准化，使均值≈0，标准差≈1")

    print("\n" + "=" * 70)


def plot_distribution(stats, save_path='data_distribution.png'):
    """绘制分布图"""

    fig = plt.figure(figsize=(40, 30))

    # 创建子图布局: 3行2列
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1], hspace=0.3, wspace=0.3)

    label_counts = stats['label_counts']
    sorted_items = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

    # ==================== 1. 全部40类别样本数分布 (横向条形图) ====================
    # ax1 = fig.add_subplot(gs[0, :])  # 占据第一行两列
    #
    # # 按类别ID排序显示所有40个类别
    all_class_ids = list(range(1, 41))
    all_counts = [label_counts.get(i, 0) for i in all_class_ids]
    all_names = [f"{i}.{DIAGNOSIS_NAMES.get(i, f'类别{i}')[:6]}" for i in all_class_ids]
    #
    # # 根据样本数量设置颜色
    # colors = []
    # for c in all_counts:
    #     if c == 0:
    #         colors.append('#d62728')  # 红色 - 无样本
    #     elif c < 100:
    #         colors.append('#ff7f0e')  # 橙色 - 极少 (<100)
    #     elif c < 1000:
    #         colors.append('#ffbb78')  # 浅橙 - 较少 (<1000)
    #     elif c < 5000:
    #         colors.append('#98df8a')  # 浅绿 - 中等
    #     else:
    #         colors.append('#2ca02c')  # 深绿 - 充足 (>5000)
    #
    # y_pos = np.arange(len(all_class_ids))
    # bars = ax1.barh(y_pos, all_counts, color=colors, edgecolor='white', height=0.8)
    #
    # ax1.set_yticks(y_pos)
    # ax1.set_yticklabels(all_names, fontsize=8)
    # ax1.set_xlabel('样本数', fontsize=10)
    # ax1.set_title('全部40类别样本数分布 (按类别ID排序)', fontsize=12, fontweight='bold')
    # ax1.invert_yaxis()
    #
    # # 在条形上显示数值
    # for bar, count in zip(bars, all_counts):
    #     if count > 0:
    #         ax1.text(bar.get_width() + max(all_counts) * 0.01, bar.get_y() + bar.get_height()/2,
    #                 f'{count:,}', va='center', fontsize=7)
    #     else:
    #         ax1.text(max(all_counts) * 0.01, bar.get_y() + bar.get_height()/2,
    #                 '无样本', va='center', fontsize=7, color='red')
    #
    # # 添加图例
    # from matplotlib.patches import Patch
    # legend_elements = [
    #     Patch(facecolor='#2ca02c', label='充足 (>5000)'),
    #     Patch(facecolor='#98df8a', label='中等 (1000-5000)'),
    #     Patch(facecolor='#ffbb78', label='较少 (100-1000)'),
    #     Patch(facecolor='#ff7f0e', label='极少 (<100)'),
    #     Patch(facecolor='#d62728', label='无样本 (0)'),
    # ]
    # ax1.legend(handles=legend_elements, loc='lower right', fontsize=8)

    # ==================== 1. Top 10 类别样本数分布 (按样本数排序) ====================
    ax1 = fig.add_subplot(gs[0, :])  # 占据第一行两列

    # 1. 数据准备：按样本数降序排序，并取前10
    # label_counts 是 {id: count}
    sorted_items = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:20]

    # 解包数据
    top_ids = [item[0] for item in sorted_items]
    top_counts = [item[1] for item in sorted_items]
    # 生成显示名称 "ID.名称"
    top_names = [f"{DIAGNOSIS_NAMES.get(int(i), f'类别{i}')}" for i in top_ids]

    # 2. 根据样本数量设置颜色
    colors = []
    for c in top_counts:
        if c == 0:
            colors.append('#d62728')  # 红色
        elif c < 100:
            colors.append('#ff7f0e')  # 橙色
        elif c < 1000:
            colors.append('#ffbb78')  # 浅橙
        elif c < 5000:
            colors.append('#98df8a')  # 浅绿
        else:
            colors.append('#2ca02c')  # 深绿

    # 3. 绘图
    y_pos = np.arange(len(top_ids))
    # 注意：barh默认从下往上画(0在下)，为了让Top1在最上面，我们后面会invert_yaxis
    bars = ax1.barh(y_pos, top_counts, color=colors, edgecolor='white', height=0.7)

    # 4. 设置轴和标签
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_names, fontsize=10)  # 只有10个，字体可以稍大
    ax1.set_xlabel('样本数', fontsize=10)
    ax1.set_title(f'样本数 Top {len(top_ids)} 类别分布', fontsize=12, fontweight='bold')

    # 反转Y轴，使第一个元素（Top 1）显示在最上方
    ax1.invert_yaxis()

    # 5. 在条形上显示数值 (增加占比显示)
    total_samples = stats['total_samples']
    max_val = max(top_counts) if top_counts else 0

    for bar, count in zip(bars, top_counts):
        # 计算显示位置
        width = bar.get_width()
        # 文本内容：数量 + (占比)
        ratio = (count / total_samples * 100) if total_samples > 0 else 0
        label_text = f'{count:,} ({ratio:.1f}%)'

        ax1.text(width + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                 label_text, va='center', fontsize=9)

    # 6. 添加图例 (保持原有逻辑以解释颜色含义)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', label='充足 (>5000)'),
        Patch(facecolor='#98df8a', label='中等 (1000-5000)'),
        Patch(facecolor='#ffbb78', label='较少 (100-1000)'),
        Patch(facecolor='#ff7f0e', label='极少 (<100)'),
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=8)

    # ==================== 2. 类别样本数对数分布 ====================
    ax2 = fig.add_subplot(gs[1, 0])

    # 按样本数排序
    sorted_counts = sorted(all_counts, reverse=True)
    x_pos = np.arange(len(sorted_counts))

    ax2.bar(x_pos, sorted_counts, color='steelblue', edgecolor='white')
    ax2.set_yscale('log')  # 对数刻度更容易看出差异
    ax2.set_xlabel('类别排名 (按样本数降序)', fontsize=10)
    ax2.set_ylabel('样本数 (对数刻度)', fontsize=10)
    ax2.set_title('类别样本数分布 (对数刻度)', fontsize=11, fontweight='bold')

    # 添加参考线
    ax2.axhline(y=1000, color='orange', linestyle='--', alpha=0.7, label='1000样本线')
    ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100样本线')
    ax2.legend(fontsize=8)

    # 标注统计信息
    non_zero_counts = [c for c in all_counts if c > 0]
    if non_zero_counts:
        median_val = np.median(non_zero_counts)
        ax2.text(0.95, 0.95, f'有样本类别: {len(non_zero_counts)}/40\n'
                            f'中位数: {median_val:,.0f}\n'
                            f'最大: {max(non_zero_counts):,}\n'
                            f'最小(非0): {min(non_zero_counts):,}',
                transform=ax2.transAxes, fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ==================== 3. 单标签 vs 多标签 ====================
    ax3 = fig.add_subplot(gs[1, 1])

    single = stats['single_label_count']
    multi = stats['multi_label_count']
    zero = stats.get('zero_label_count', 0)

    if zero > 0:
        categories = ['单标签', '多标签', '无标签']
        values = [single, multi, zero]
        bar_colors = ['#2ca02c', '#1f77b4', '#d62728']
    else:
        categories = ['单标签', '多标签']
        values = [single, multi]
        bar_colors = ['#2ca02c', '#1f77b4']

    bars = ax3.bar(categories, values, color=bar_colors, edgecolor='white')
    ax3.set_ylabel('样本数', fontsize=10)
    ax3.set_title(f'标签数量分布 (多标签占比: {stats["multi_label_ratio"]*100:.1f}%)',
                  fontsize=11, fontweight='bold')

    # 在柱状图上显示数值和百分比
    total = sum(values)
    for bar, val in zip(bars, values):
        pct = val / total * 100 if total > 0 else 0
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total * 0.01,
                f'{val:,},({pct:.1f}%)', ha='center', va='bottom', fontsize=9)

    # ==================== 4. 每样本标签数量分布 ====================
    ax4 = fig.add_subplot(gs[2, 0])

    label_count_dist = stats.get('label_count_distribution', {})
    if label_count_dist:
        x_labels = sorted(label_count_dist.keys())
        y_values = [label_count_dist[k] for k in x_labels]

        bars = ax4.bar([str(x) for x in x_labels], y_values, color='#9467bd', edgecolor='white')
        ax4.set_xlabel('每样本的标签数量', fontsize=10)
        ax4.set_ylabel('样本数', fontsize=10)
        ax4.set_title('每样本标签数量分布', fontsize=11, fontweight='bold')

        # 显示数值
        for bar, val in zip(bars, y_values):
            if val > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(y_values) * 0.01,
                        f'{val:,}', ha='center', va='bottom', fontsize=8)

    # ==================== 5. Top 15 类别占比饼图 ====================
    ax5 = fig.add_subplot(gs[2, 1])

    top_n = 10
    top_items = sorted_items[:top_n]
    others_count = sum(v for k, v in sorted_items[top_n:])

    pie_labels = [DIAGNOSIS_NAMES.get(int(k), f"类别{k}") for k, v in top_items]
    pie_values = [v for k, v in top_items]

    if others_count > 0:
        pie_labels.append(f'其他{len(sorted_items)-top_n}类')
        pie_values.append(others_count)

    # 使用更好看的颜色
    cmap = plt.cm.Set3
    pie_colors = [cmap(i / len(pie_values)) for i in range(len(pie_values))]

    wedges, texts, autotexts = ax5.pie(pie_values, labels=pie_labels, autopct='%1.1f%%',
                                        colors=pie_colors, pctdistance=0.75,
                                        wedgeprops=dict(width=0.5, edgecolor='white'))

    ax5.set_title(f'Top {top_n} 类别占比', fontsize=11, fontweight='bold')

    # 调整字体大小
    for text in texts:
        text.set_fontsize(15)
    for autotext in autotexts:
        autotext.set_fontsize(8)

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n分布图已保存: {save_path}")





def main():
    parser = argparse.ArgumentParser(description='ECG数据分析')
    parser.add_argument('--data_dirs', nargs='+', required=True, help='数据目录列表')
    parser.add_argument('--output', type=str, default='data_analysis_report.json', help='输出报告文件')
    parser.add_argument('--workers', type=int, default=32, help='并行进程数')
    parser.add_argument('--sample_ratio', type=float, default=1.0, help='采样比例')
    parser.add_argument('--plot', action='store_true', help='生成分布图',default=True)
    args = parser.parse_args()

    # 分析数据
    results, errors = analyze_data(args.data_dirs, args.workers, args.sample_ratio)

    # 计算统计
    stats = compute_statistics(results)

    # 打印报告
    print_report(stats)

    # 保存报告
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {args.output}")

    # 绘图
    if args.plot:
        plot_distribution(stats)

    # 保存错误文件列表
    if errors:
        error_file = args.output.replace('.json', '_errors.json')
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"错误文件列表: {error_file}")


if __name__ == "__main__":
    main()
    # with open('data_analysis_report.json', 'r', encoding='utf-8') as f:
    #     stats = json.load(f)
    # plot_distribution(stats)
