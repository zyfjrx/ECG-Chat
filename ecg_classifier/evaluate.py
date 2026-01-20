"""
ECG分类模型评估脚本

用法:
    # 使用预处理后的数据文件 (推荐)
    python evaluate.py --model_path checkpoints/best_model.pth --data_file ecg_data.npz

    # 使用原始数据目录 (旧方式)
    python evaluate.py --model_path checkpoints/best_model.pth --data_dir /path/to/test_data
"""

import os
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score
)
import matplotlib.pyplot as plt
from tqdm import tqdm

from ecg_classifier.model.model_simple import get_model
from ecg_classifier.data.dataset import (
    load_preprocessed,
    ECGPreprocessedDataset
)
from inference import DIAGNOSIS_NAMES


def compute_specificity(y_true, y_pred):
    """计算特异度 Specificity = TN / (TN + FP)"""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if tn + fp == 0:
        return 0.0
    return tn / (tn + fp)


def compute_npv(y_true, y_pred):
    """计算阴性预测值 NPV = TN / (TN + FN)"""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tn + fn == 0:
        return 0.0
    return tn / (tn + fn)


# 高危诊断类型（漏诊代价高，需要特别关注检出率）
HIGH_RISK_IDS = {5, 6, 20, 21, 30, 32}  # 窦性停搏, 房颤, 室速, 室上速, 房扑, 二度房室阻滞


def evaluate_model(model, test_loader, device, thresholds=None):
    """
    全面评估模型（医学场景优化版）

    评估指标说明:
    - Sensitivity(Sen/召回率): 有病的能查出多少 = TP/(TP+FN)
    - Specificity(Spe/特异度): 没病的能排除多少 = TN/(TN+FP)
    - Precision(PPV/阳性预测值): 预测有病的真有病比例 = TP/(TP+FP)
    - NPV(阴性预测值): 预测没病的真没病比例 = TN/(TN+FN)
    - AUC: 模型区分能力
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for ecg, labels in tqdm(test_loader, desc="评估中"):
            ecg = ecg.to(device)
            logits = model(ecg)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    probs = torch.sigmoid(all_logits).numpy()
    labels = all_labels.numpy()

    if thresholds is None:
        thresholds = np.ones(labels.shape[1]) * 0.5

    preds = (probs >= thresholds).astype(int)

    # ==================== 统计有效类别 ====================
    # 有效类别 = 在测试集中有样本的类别
    valid_class_indices = [i for i in range(labels.shape[1]) if labels[:, i].sum() > 0]
    zero_sample_classes = [i + 1 for i in range(labels.shape[1]) if labels[:, i].sum() == 0]
    num_valid_classes = len(valid_class_indices)
    num_total_classes = labels.shape[1]

    print("\n" + "=" * 70)
    print("类别统计")
    print("=" * 70)
    print(f"  总类别数: {num_total_classes}")
    print(f"  有样本类别: {num_valid_classes}")
    print(f"  无样本类别: {num_total_classes - num_valid_classes}")
    if zero_sample_classes:
        zero_names = [DIAGNOSIS_NAMES.get(i, f"类别{i}") for i in zero_sample_classes]
        print(f"  无样本类别ID: {zero_sample_classes}")
        print(f"  无样本类别名: {', '.join(zero_names)}")

    # ==================== 整体指标 ====================
    print("\n" + "=" * 70)
    print("整体评估指标")
    print("=" * 70)

    metrics = {}

    # --- Micro 指标 (按样本计算，推荐) ---
    metrics['f1_micro'] = f1_score(labels, preds, average='micro', zero_division=0)
    metrics['precision_micro'] = precision_score(labels, preds, average='micro', zero_division=0)
    metrics['recall_micro'] = recall_score(labels, preds, average='micro', zero_division=0)

    # --- Macro 指标 (按类别平均) ---
    # 标准Macro: 包含所有类别，无样本类别F1=0，会拉低平均值
    metrics['f1_macro_all'] = f1_score(labels, preds, average='macro', zero_division=0)

    # 客观Macro: 仅计算有样本的类别 (推荐用于评估真实能力)
    if valid_class_indices:
        valid_labels = labels[:, valid_class_indices]
        valid_preds = preds[:, valid_class_indices]
        metrics['f1_macro_valid'] = f1_score(valid_labels, valid_preds, average='macro', zero_division=0)
        metrics['precision_macro_valid'] = precision_score(valid_labels, valid_preds, average='macro', zero_division=0)
        metrics['recall_macro_valid'] = recall_score(valid_labels, valid_preds, average='macro', zero_division=0)
    else:
        metrics['f1_macro_valid'] = 0
        metrics['precision_macro_valid'] = 0
        metrics['recall_macro_valid'] = 0

    # --- Weighted 指标 (按样本数加权) ---
    metrics['f1_weighted'] = f1_score(labels, preds, average='weighted', zero_division=0)

    # --- 其他指标 ---
    metrics['exact_match'] = np.mean(np.all(preds == labels, axis=1))
    metrics['hamming_loss'] = np.mean(preds != labels)
    metrics['num_valid_classes'] = num_valid_classes
    metrics['num_total_classes'] = num_total_classes
    metrics['zero_sample_classes'] = zero_sample_classes

    print(f"\n【按样本计算 (Micro) - 反映整体表现】")
    print(f"  F1 Micro:        {metrics['f1_micro']:.4f}")
    print(f"  Precision Micro: {metrics['precision_micro']:.4f}")
    print(f"  Recall Micro:    {metrics['recall_micro']:.4f}")

    print(f"\n【按类别平均 (Macro) - 对比说明】")
    print(f"  F1 Macro (全部{num_total_classes}类): {metrics['f1_macro_all']:.4f}  ← 包含{num_total_classes - num_valid_classes}个无样本类别(F1=0)，被拉低")
    print(f"  F1 Macro (有效{num_valid_classes}类): {metrics['f1_macro_valid']:.4f}  ← 仅计算有样本类别，客观反映模型能力")

    print(f"\n【加权平均 (Weighted) - 考虑类别样本量】")
    print(f"  F1 Weighted:     {metrics['f1_weighted']:.4f}")

    print(f"\n【其他指标】")
    print(f"  Exact Match:     {metrics['exact_match']:.4f}  (仅供参考)")
    print(f"  Hamming Loss:    {metrics['hamming_loss']:.4f}")

    # ==================== 按类别评估 ====================
    print("\n" + "=" * 70)
    print("按类别评估 (医学核心指标)")
    print("=" * 70)
    print(f"{'ID':>3} {'诊断名称':<18} {'样本数':>6} {'Sen':>6} {'Spe':>6} {'PPV':>6} {'NPV':>6} {'AUC':>6} {'F1':>6}")
    print("-" * 70)

    class_metrics = []
    high_risk_metrics = []

    for i in range(labels.shape[1]):
        class_name = DIAGNOSIS_NAMES.get(i + 1, f"类别{i+1}")
        support = int(labels[:, i].sum())

        # 计算所有指标
        sensitivity = recall_score(labels[:, i], preds[:, i], zero_division=0)  # 检出率 = TP/(TP+FN)
        specificity = compute_specificity(labels[:, i], preds[:, i])  # 特异度 = TN/(TN+FP)
        ppv = precision_score(labels[:, i], preds[:, i], zero_division=0)  # 阳性预测值
        npv = compute_npv(labels[:, i], preds[:, i])  # 阴性预测值
        f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)

        try:
            auc = roc_auc_score(labels[:, i], probs[:, i]) if support > 0 and support < len(labels) else 0
        except:
            auc = 0

        class_metric = {
            'id': i + 1,
            'name': class_name,
            'support': support,
            'sensitivity': sensitivity,  # 检出率
            'specificity': specificity,  # 特异度
            'ppv': ppv,  # 阳性预测值
            'npv': npv,  # 阴性预测值
            'f1': f1,
            'auc': auc,
            'threshold': thresholds[i]
        }
        class_metrics.append(class_metric)

        # 标记高危诊断
        is_high_risk = (i + 1) in HIGH_RISK_IDS
        risk_marker = "⚠️" if is_high_risk else "  "

        if support > 0:  # 只打印有样本的类别
            print(f"{i+1:>3} {class_name:<18} {support:>6} {sensitivity:>6.1%} {specificity:>6.1%} "
                  f"{ppv:>6.1%} {npv:>6.1%} {auc:>6.3f} {f1:>6.3f} {risk_marker}")

        if is_high_risk and support > 0:
            high_risk_metrics.append(class_metric)

    # ==================== 高危诊断汇总 ====================
    if high_risk_metrics:
        print("\n" + "=" * 70)
        print("⚠️  高危诊断汇总 (这些诊断漏诊代价高，重点关注Sensitivity)")
        print("=" * 70)
        for m in high_risk_metrics:
            status = "✓" if m['sensitivity'] >= 0.9 else "⚠️" if m['sensitivity'] >= 0.8 else "❌"
            print(f"  {status} {m['name']}: 检出率={m['sensitivity']:.1%}, 特异度={m['specificity']:.1%}, AUC={m['auc']:.3f}")

        avg_sen = np.mean([m['sensitivity'] for m in high_risk_metrics])
        print(f"\n  高危诊断平均检出率: {avg_sen:.1%}")
        metrics['high_risk_avg_sensitivity'] = avg_sen

    # 计算AUC
    print("\n" + "=" * 70)
    print("AUC 评估")
    print("=" * 70)

    try:
        auc_scores = []
        valid_auc_count = 0
        for i in range(labels.shape[1]):
            if labels[:, i].sum() > 0 and labels[:, i].sum() < len(labels):
                auc = roc_auc_score(labels[:, i], probs[:, i])
                auc_scores.append(auc)
                valid_auc_count += 1

        metrics['auc_macro'] = np.mean(auc_scores) if auc_scores else 0
        metrics['auc_valid_classes'] = valid_auc_count

        print(f"  可计算AUC的类别: {valid_auc_count}/{num_total_classes}")
        print(f"  Macro AUC (有效类别): {metrics['auc_macro']:.4f}")
        print(f"  说明: AUC只能对有正负样本的类别计算")
    except Exception as e:
        print(f"  AUC计算失败: {e}")
        metrics['auc_macro'] = 0

    # ==================== 评估总结 ====================
    print("\n" + "=" * 70)
    print("评估总结 - 如何客观看待指标")
    print("=" * 70)
    print(f"""
  推荐关注的指标 (按优先级):
  1. AUC Macro: {metrics['auc_macro']:.4f} - 模型区分能力，不受阈值影响
  2. F1 Micro: {metrics['f1_micro']:.4f} - 整体分类准确率
  3. F1 Macro (有效类): {metrics['f1_macro_valid']:.4f} - 各类别平均表现

  关于 F1 Macro 的说明:
  - 全部类别 Macro: {metrics['f1_macro_all']:.4f} (被{num_total_classes - num_valid_classes}个无样本类拉低)
  - 有效类别 Macro: {metrics['f1_macro_valid']:.4f} (客观反映模型能力)

  为什么排除无样本类别更客观?
  - 无样本类别无法评估，强制计入会产生F1=0
  - 这不是模型能力问题，而是数据覆盖问题
  - 应在数据收集阶段解决，而非惩罚模型
""")

    return metrics, class_metrics, probs, labels, preds


def plot_metrics(class_metrics, save_path='evaluation_plots.png'):
    """绘制评估图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 按F1排序
    sorted_metrics = sorted(class_metrics, key=lambda x: x['f1'], reverse=True)

    # 1. F1 Score per class
    ax1 = axes[0, 0]
    names = [m['name'][:10] for m in sorted_metrics]
    f1_scores = [m['f1'] for m in sorted_metrics]
    colors = ['green' if f > 0.7 else 'orange' if f > 0.5 else 'red' for f in f1_scores]
    ax1.barh(names, f1_scores, color=colors)
    ax1.set_xlabel('F1 Score')
    ax1.set_title('F1 Score per Class')
    ax1.axvline(x=0.7, color='green', linestyle='--', alpha=0.5)
    ax1.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5)

    # 2. PPV vs Sensitivity (医学场景)
    ax2 = axes[0, 1]
    ppvs = [m['ppv'] for m in class_metrics]
    sensitivities = [m['sensitivity'] for m in class_metrics]
    ax2.scatter(sensitivities, ppvs, alpha=0.6)
    for m in class_metrics:
        ax2.annotate(str(m['id']), (m['sensitivity'], m['ppv']), fontsize=8)
    ax2.set_xlabel('Sensitivity (检出率)')
    ax2.set_ylabel('PPV (阳性预测值)')
    ax2.set_title('PPV vs Sensitivity per Class')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)

    # 3. AUC per class
    ax3 = axes[1, 0]
    sorted_by_auc = sorted(class_metrics, key=lambda x: x['auc'], reverse=True)
    names = [m['name'][:10] for m in sorted_by_auc]
    aucs = [m['auc'] for m in sorted_by_auc]
    colors = ['green' if a > 0.8 else 'orange' if a > 0.6 else 'red' for a in aucs]
    ax3.barh(names, aucs, color=colors)
    ax3.set_xlabel('AUC')
    ax3.set_title('AUC per Class')
    ax3.axvline(x=0.8, color='green', linestyle='--', alpha=0.5)

    # 4. Support distribution
    ax4 = axes[1, 1]
    sorted_by_support = sorted(class_metrics, key=lambda x: x['support'], reverse=True)
    names = [m['name'][:10] for m in sorted_by_support]
    supports = [m['support'] for m in sorted_by_support]
    ax4.barh(names, supports, color='steelblue')
    ax4.set_xlabel('样本数')
    ax4.set_title('类别样本分布')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n评估图表已保存: {save_path}")


def analyze_errors(probs, labels, preds, save_path='error_analysis.json'):
    """错误分析"""
    errors = {
        'false_positives': {},  # 误诊
        'false_negatives': {}   # 漏诊
    }

    for i in range(labels.shape[1]):
        class_name = DIAGNOSIS_NAMES.get(i + 1, f"类别{i+1}")

        # 假阳性 (预测为正但实际为负)
        fp_mask = (preds[:, i] == 1) & (labels[:, i] == 0)
        fp_count = fp_mask.sum()

        # 假阴性 (预测为负但实际为正)
        fn_mask = (preds[:, i] == 0) & (labels[:, i] == 1)
        fn_count = fn_mask.sum()

        if fp_count > 0:
            errors['false_positives'][class_name] = {
                'count': int(fp_count),
                'avg_prob': float(probs[fp_mask, i].mean())
            }

        if fn_count > 0:
            errors['false_negatives'][class_name] = {
                'count': int(fn_count),
                'avg_prob': float(probs[fn_mask, i].mean())
            }

    # 保存分析结果
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    print(f"\n错误分析已保存: {save_path}")

    # 打印主要错误
    print("\n主要误诊 (False Positives):")
    fp_sorted = sorted(errors['false_positives'].items(), key=lambda x: x[1]['count'], reverse=True)
    for name, info in fp_sorted[:10]:
        print(f"  {name}: {info['count']} 次, 平均概率: {info['avg_prob']:.3f}")

    print("\n主要漏诊 (False Negatives):")
    fn_sorted = sorted(errors['false_negatives'].items(), key=lambda x: x[1]['count'], reverse=True)
    for name, info in fn_sorted[:10]:
        print(f"  {name}: {info['count']} 次, 平均概率: {info['avg_prob']:.3f}")

    return errors


def create_test_loader_from_preprocessed(data_file, batch_size=1, num_workers=4):
    """
    从预处理文件创建测试集DataLoader

    Args:
        data_file: 预处理数据文件路径 (.npz, .pkl, .h5)
        batch_size: 批次大小
        num_workers: 数据加载进程数

    Returns:
        test_loader, test_size
    """
    data = load_preprocessed(data_file)

    # 创建测试集Dataset
    test_dataset = ECGPreprocessedDataset(
        data['test']['X'],
        data['test']['Y'],
        transform=None  # 评估时不做数据增强
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return test_loader, len(test_dataset)


def main():
    parser = argparse.ArgumentParser(description='ECG模型评估')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')

    # 数据源 (二选一)
    parser.add_argument('--data_file', type=str, help='预处理数据文件 (.npz/.pkl/.h5)')
    parser.add_argument('--data_dir', type=str, help='测试数据目录 (旧方式)')

    parser.add_argument('--model_version', type=str, default='auto',
                        choices=['auto', 'simple', 'fft', 'full', 'se', 'se_full'],
                        help='模型版本: auto(从checkpoint读取)/simple/fft/full/se/se_full')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载进程数')
    parser.add_argument('--use_optimal_thresh', action='store_true', help='使用最优阈值')
    parser.add_argument('--output_dir', type=str, default='./evaluation', help='输出目录')
    parser.add_argument('--device', type=str, default='auto', help='设备 (auto/cuda/cpu/mps)')
    args = parser.parse_args()

    # 检查数据源
    if not args.data_file and not args.data_dir:
        print("错误: 请指定 --data_file 或 --data_dir")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # 设备选择
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 加载模型
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    # 获取配置
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {}

    # 确定模型版本
    if args.model_version == 'auto':
        model_version = config.get('model_type', 'se')
    else:
        model_version = args.model_version

    # 创建模型
    model = get_model(
        version=model_version,
        seq_len=config.get('seq_len', 7500),
        num_classes=config.get('num_classes', 40),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
        patch_size=config.get('patch_size', 50)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"加载模型: {args.model_path}")
    print(f"  模型版本: {model_version}")
    print(f"  序列长度: {config.get('seq_len', 7500)}")
    print(f"  类别数: {config.get('num_classes', 40)}")

    # 加载阈值
    if args.use_optimal_thresh:
        thresh_path = os.path.join(os.path.dirname(args.model_path), 'optimal_thresholds.npy')
        if os.path.exists(thresh_path):
            thresholds = np.load(thresh_path)
            print(f"使用最优阈值 (从 {thresh_path})")
        else:
            print(f"警告: 未找到最优阈值文件，使用默认阈值0.5")
            thresholds = None
    else:
        thresholds = None

    # 加载数据
    if args.data_file:
        # 使用预处理数据文件 (推荐)
        print(f"\n从预处理文件加载数据: {args.data_file}")
        test_loader, test_size = create_test_loader_from_preprocessed(
            args.data_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        print(f"测试集大小: {test_size}")
    else:
        # 使用原始数据目录 (旧方式，保持向后兼容)
        print(f"\n从目录加载数据: {args.data_dir}")
        print("警告: 建议使用预处理后的数据文件 (--data_file)")

        # 动态导入旧的加载函数
        from ecg_classifier.data.dataset import load_data2, create_dataloaders
        _, _, test_files = load_data2(args.data_dir, test_size=1.0, val_size=0.0)
        _, _, test_loader = create_dataloaders([], [], test_files, batch_size=args.batch_size)
        test_size = len(test_files)
        print(f"测试集大小: {test_size}")

    # 评估
    metrics, class_metrics, probs, labels, preds = evaluate_model(
        model, test_loader, device, thresholds
    )

    # 保存结果
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(args.output_dir, 'class_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(class_metrics, f, ensure_ascii=False, indent=2)

    # 绘图
    plot_metrics(class_metrics, os.path.join(args.output_dir, 'evaluation_plots.png'))

    # 错误分析
    analyze_errors(probs, labels, preds, os.path.join(args.output_dir, 'error_analysis.json'))

    print(f"\n评估完成! 结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
