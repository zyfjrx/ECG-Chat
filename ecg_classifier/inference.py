"""
ECG分类模型推理脚本

用法:
    # 单文件推理
    python inference.py --model_path checkpoints/best_model.pth --ecg_file data/test.txt

    # 批量推理
    python inference.py --model_path checkpoints/best_model.pth --data_dir data/test/

    # 使用最优阈值
    python inference.py --model_path checkpoints/best_model.pth --ecg_file data/test.txt --use_optimal_thresh
"""

import os
import argparse
import glob
import json

import numpy as np
import torch
from tqdm import tqdm

from ecg_classifier.model.model_simple import get_model


# 诊断类型映射 (来自 诊断类型对应表.csv)
DIAGNOSIS_NAMES = {
    1: "窦性心律",
    2: "心电图未见异常",
    3: "窦性心动过速",
    4: "窦性心动过缓",
    5: "窦性停搏",
    6: "心房颤动",
    7: "房性早搏",
    8: "偶发房性早搏",
    9: "频发房性早搏",
    10: "房性早搏二联律",
    11: "房性早搏三联律",
    12: "成对房性早搏",
    13: "短阵房性心动过速",
    14: "室性早搏",
    15: "偶发室性早搏",
    16: "频发室性早搏",
    17: "室性早搏二联律",
    18: "室性早搏三联律",
    19: "成对室性早搏",
    20: "短阵室性心动过速",
    21: "室上性心动过速",
    22: "一度房室阻滞",
    23: "ST段抬高",
    24: "ST段压低",
    25: "QT/QTc间期延长",
    26: "RR长间歇",
    27: "心室内差异传导",
    28: "干扰波",
    29: "导联脱落",
    30: "心房扑动",
    31: "短PR间期",
    32: "二度Ⅱ型房室阻滞",
    33: "P波增高",
    34: "P波增宽",
    35: "疑似左右手反接心电图",
    36: "R波高电压",
    37: "室内阻滞",
    38: "T波改变",
    39: "短QT/QTc间期",
    40: "心电图未见明显异常",
}

# 医学风险等级分类
HIGH_RISK_DIAGNOSES = {5, 6, 20, 21, 30, 32}  # 窦性停搏, 房颤, 室速等
MEDIUM_RISK_DIAGNOSES = {3, 4, 14, 22, 23, 24, 25, 26}  # 窦速, 窦缓, 室早等
LOW_RISK_DIAGNOSES = {1, 2, 40}  # 正常


class ECGClassifier:
    """ECG分类器封装类"""

    def __init__(self, model_path, device='auto', use_optimal_thresh=False):
        # 设备选择
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # 获取配置
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            config = {}

        # 确定模型版本
        model_type = config.get('model_type', 'se')
        self.seq_len = config.get('seq_len', 7500)
        self.num_classes = config.get('num_classes', 40)

        # 创建模型
        self.model = get_model(
            version=model_type,
            seq_len=self.seq_len,
            num_classes=self.num_classes,
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 6),
            num_heads=config.get('num_heads', 8),
            patch_size=config.get('patch_size', 50)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"模型加载成功: {model_path}")
        print(f"  设备: {self.device}")
        print(f"  模型版本: {model_type}")

        # 加载最优阈值
        self.use_optimal_thresh = use_optimal_thresh
        if use_optimal_thresh:
            thresh_path = os.path.join(os.path.dirname(model_path), 'optimal_thresholds.npy')
            if os.path.exists(thresh_path):
                self.thresholds = np.load(thresh_path)
                print(f"  使用最优阈值")
            else:
                print("  未找到最优阈值文件，使用默认阈值0.5")
                self.thresholds = np.ones(self.num_classes) * 0.5
        else:
            self.thresholds = np.ones(self.num_classes) * 0.5

    def parse_ecg_file(self, file_path):
        """
        解析ECG文件

        数据格式:
        - 32767 为数据开始标记
        - 32763 为数据结束标记
        - 异常值使用前一个有效值填充
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

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

        # 解析ECG数据，异常值用前值填充
        ecg_values = []
        last_valid = 0.0

        for i in range(data_start, data_end):
            try:
                value = float(lines[i].strip())
                if -32768 <= value <= 32767:  # 有效值范围
                    ecg_values.append(value)
                    last_valid = value
                else:
                    ecg_values.append(last_valid)  # 用前值填充
            except:
                ecg_values.append(last_valid)

        ecg_data = np.array(ecg_values, dtype=np.float32)

        # 确保长度为seq_len
        if len(ecg_data) < self.seq_len:
            ecg_data = np.pad(ecg_data, (0, self.seq_len - len(ecg_data)), mode='constant')
        elif len(ecg_data) > self.seq_len:
            ecg_data = ecg_data[:self.seq_len]

        # Z-score标准化
        mean = np.mean(ecg_data)
        std = np.std(ecg_data)
        if std > 1e-6:
            ecg_data = (ecg_data - mean) / std

        return ecg_data

    @torch.no_grad()
    def predict(self, ecg_data):
        """
        预测单条ECG数据

        Args:
            ecg_data: numpy array, shape (7500,)

        Returns:
            dict: 包含预测结果的字典
        """
        # 转换为tensor
        ecg_tensor = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 预测
        logits = self.model(ecg_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

        # 应用阈值
        predictions = (probs >= self.thresholds).astype(int)

        # 构建结果
        result = {
            'probabilities': probs.tolist(),
            'predictions': predictions.tolist(),
            'diagnoses': [],
            'risk_level': 'low'
        }

        # 获取预测的诊断名称
        for i, pred in enumerate(predictions):
            if pred == 1:
                diag_id = i + 1
                result['diagnoses'].append({
                    'id': diag_id,
                    'name': DIAGNOSIS_NAMES.get(diag_id, f"未知诊断{diag_id}"),
                    'probability': float(probs[i])
                })

        # 按概率排序
        result['diagnoses'].sort(key=lambda x: x['probability'], reverse=True)

        # 计算风险等级
        predicted_ids = set(i + 1 for i, p in enumerate(predictions) if p == 1)
        if predicted_ids & HIGH_RISK_DIAGNOSES:
            result['risk_level'] = 'high'
        elif predicted_ids & MEDIUM_RISK_DIAGNOSES:
            result['risk_level'] = 'medium'
        else:
            result['risk_level'] = 'low'

        return result

    def predict_file(self, file_path):
        """预测单个文件"""
        ecg_data = self.parse_ecg_file(file_path)
        result = self.predict(ecg_data)
        result['file_path'] = file_path
        return result

    def predict_batch(self, file_list):
        """批量预测"""
        results = []
        for file_path in tqdm(file_list, desc="预测中"):
            try:
                result = self.predict_file(file_path)
                results.append(result)
            except Exception as e:
                print(f"预测失败: {file_path}, 错误: {e}")
                results.append({'file_path': file_path, 'error': str(e)})
        return results

    def predict_top_k(self, ecg_data, k=5):
        """
        返回概率最高的k个诊断

        医学场景中，可以返回Top-K而不是用阈值
        """
        ecg_tensor = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(ecg_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # 获取Top-K
        top_k_indices = np.argsort(probs)[::-1][:k]

        results = []
        for idx in top_k_indices:
            diag_id = idx + 1
            results.append({
                'id': diag_id,
                'name': DIAGNOSIS_NAMES.get(diag_id, f"未知诊断{diag_id}"),
                'probability': float(probs[idx])
            })

        return results


def format_report(result):
    """格式化诊断报告"""
    report = []
    report.append("=" * 50)
    report.append("心电图诊断报告")
    report.append("=" * 50)

    if 'error' in result:
        report.append(f"错误: {result['error']}")
        return "\n".join(report)

    # 风险等级
    risk_map = {'low': '低风险 ✓', 'medium': '中风险 ⚠', 'high': '高风险 ⚠⚠'}
    report.append(f"\n风险等级: {risk_map.get(result['risk_level'], '未知')}")

    # 诊断结果
    report.append("\n诊断结果:")
    if result['diagnoses']:
        for diag in result['diagnoses']:
            confidence = "高" if diag['probability'] > 0.8 else "中" if diag['probability'] > 0.5 else "低"
            report.append(f"  • {diag['name']} (置信度: {confidence}, {diag['probability']:.1%})")
    else:
        report.append("  未检测到明显异常")

    report.append("=" * 50)
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='ECG分类模型推理')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--ecg_file', type=str, help='单个ECG文件路径')
    parser.add_argument('--data_dir', type=str, help='批量预测的数据目录')
    parser.add_argument('--output', type=str, default='predictions.json', help='输出文件')
    parser.add_argument('--use_optimal_thresh', action='store_true', help='使用最优阈值')
    parser.add_argument('--top_k', type=int, default=0, help='返回Top-K结果(0表示使用阈值)')
    parser.add_argument('--device', type=str, default='auto', help='设备 (auto/cuda/cpu/mps)')
    args = parser.parse_args()

    # 初始化分类器
    classifier = ECGClassifier(
        args.model_path,
        device=args.device,
        use_optimal_thresh=args.use_optimal_thresh
    )

    # 单文件预测
    if args.ecg_file:
        if args.top_k > 0:
            ecg_data = classifier.parse_ecg_file(args.ecg_file)
            result = classifier.predict_top_k(ecg_data, k=args.top_k)
            print(f"\nTop-{args.top_k} 诊断结果:")
            for r in result:
                print(f"  {r['id']:2d}. {r['name']}: {r['probability']:.1%}")
        else:
            result = classifier.predict_file(args.ecg_file)
            print(format_report(result))

    # 批量预测
    elif args.data_dir:
        file_list = glob.glob(os.path.join(args.data_dir, "*.txt"))
        print(f"找到 {len(file_list)} 个文件")

        results = classifier.predict_batch(file_list)

        # 保存结果
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")

        # 统计
        risk_counts = {'low': 0, 'medium': 0, 'high': 0}
        for r in results:
            if 'risk_level' in r:
                risk_counts[r['risk_level']] += 1

        print(f"\n风险统计:")
        print(f"  低风险: {risk_counts['low']}")
        print(f"  中风险: {risk_counts['medium']}")
        print(f"  高风险: {risk_counts['high']}")

    else:
        print("请指定 --ecg_file 或 --data_dir")


if __name__ == "__main__":
    main()
