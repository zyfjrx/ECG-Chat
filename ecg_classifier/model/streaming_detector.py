"""
ECG滑动窗口实时检测模块

支持流式ECG数据的实时分析和预警。

主要功能:
1. 滑动窗口缓冲区管理
2. 实时推理和预测平滑
3. 高危心律异常预警
4. 检测结果记录和导出

用法:
    # 基础用法
    detector = StreamingECGDetector(
        model_path='checkpoints/best_model.pth',
        window_size=7500,  # 30秒 @ 250Hz
        step_size=250,     # 1秒步长
    )

    # 模拟实时数据流
    for chunk in ecg_stream:
        results = detector.process(chunk)
        if results and results['alerts']:
            print(f"警报: {results['alerts']}")

    # 获取完整报告
    report = detector.get_report()
"""

import json
import time
from datetime import datetime
from collections import deque
from typing import Optional, List, Dict, Any, Tuple, Callable

import numpy as np
import torch

# 诊断标签映射 (示例，根据实际情况修改)
DIAGNOSIS_NAMES = {
    0: "窦性心律",
    1: "窦性心动过速",
    2: "窦性心动过缓",
    3: "窦性心律不齐",
    4: "房性早搏",
    5: "室性早搏",
    6: "房颤",
    7: "房扑",
    8: "室上性心动过速",
    9: "室性心动过速",
    10: "一度房室阻滞",
    11: "二度房室阻滞",
    12: "三度房室阻滞",
    13: "左束支阻滞",
    14: "右束支阻滞",
    15: "ST段抬高",
    16: "ST段压低",
    17: "T波倒置",
    18: "T波高尖",
    19: "Q波异常",
    # ... 可继续添加到40个类别
}

# 高危诊断类别 (需要立即报警)
HIGH_RISK_DIAGNOSES = {
    6,   # 房颤
    9,   # 室性心动过速
    12,  # 三度房室阻滞
    15,  # ST段抬高 (可能是心梗)
}

# 中危诊断类别 (需要关注)
MEDIUM_RISK_DIAGNOSES = {
    5,   # 室性早搏
    7,   # 房扑
    8,   # 室上性心动过速
    11,  # 二度房室阻滞
    16,  # ST段压低
}


class CircularBuffer:
    """环形缓冲区，用于高效管理滑动窗口数据"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.write_pos = 0

    def append(self, data: np.ndarray):
        """追加数据到缓冲区"""
        data = np.asarray(data, dtype=np.float32).flatten()
        n = len(data)

        if n >= self.capacity:
            # 数据超过容量，只保留最后capacity个点
            self.buffer[:] = data[-self.capacity:]
            self.size = self.capacity
            self.write_pos = 0
        else:
            # 计算写入位置
            end_pos = self.write_pos + n

            if end_pos <= self.capacity:
                self.buffer[self.write_pos:end_pos] = data
            else:
                # 需要环绕
                first_part = self.capacity - self.write_pos
                self.buffer[self.write_pos:] = data[:first_part]
                self.buffer[:n - first_part] = data[first_part:]

            self.write_pos = end_pos % self.capacity
            self.size = min(self.size + n, self.capacity)

    def get_data(self) -> np.ndarray:
        """获取缓冲区中的有效数据（按时间顺序）"""
        if self.size < self.capacity:
            return self.buffer[:self.size].copy()
        else:
            # 环形缓冲区已满，需要重新排序
            return np.concatenate([
                self.buffer[self.write_pos:],
                self.buffer[:self.write_pos]
            ])

    def is_full(self) -> bool:
        return self.size >= self.capacity

    def clear(self):
        self.buffer.fill(0)
        self.size = 0
        self.write_pos = 0


class PredictionSmoother:
    """预测平滑器，使用指数移动平均减少抖动"""

    def __init__(self, num_classes: int, alpha: float = 0.3,
                 window_size: int = 5):
        """
        Args:
            num_classes: 类别数
            alpha: EMA平滑系数 (越大越敏感)
            window_size: 滑动窗口大小用于多数投票
        """
        self.num_classes = num_classes
        self.alpha = alpha
        self.window_size = window_size

        self.ema_probs = None
        self.history = deque(maxlen=window_size)

    def update(self, probs: np.ndarray) -> np.ndarray:
        """更新并返回平滑后的概率"""
        probs = np.asarray(probs, dtype=np.float32)

        # 更新EMA
        if self.ema_probs is None:
            self.ema_probs = probs.copy()
        else:
            self.ema_probs = self.alpha * probs + (1 - self.alpha) * self.ema_probs

        # 更新历史用于多数投票
        self.history.append((probs > 0.5).astype(np.int32))

        return self.ema_probs

    def get_stable_predictions(self, threshold: float = 0.5) -> np.ndarray:
        """获取稳定预测（多数投票）"""
        if len(self.history) < self.window_size:
            return (self.ema_probs > threshold).astype(np.int32) if self.ema_probs is not None else None

        # 多数投票
        votes = np.stack(list(self.history), axis=0)
        return (votes.mean(axis=0) > 0.5).astype(np.int32)

    def reset(self):
        self.ema_probs = None
        self.history.clear()


class AlertManager:
    """警报管理器，处理去重和冷却"""

    def __init__(self, cooldown_seconds: float = 30.0):
        """
        Args:
            cooldown_seconds: 同类警报冷却时间（秒）
        """
        self.cooldown = cooldown_seconds
        self.last_alert_time: Dict[int, float] = {}
        self.alert_history: List[Dict] = []

    def check_alert(self, diagnosis_idx: int, probability: float,
                    timestamp: float) -> Optional[Dict]:
        """检查是否应该触发警报"""

        # 检查冷却
        last_time = self.last_alert_time.get(diagnosis_idx, 0)
        if timestamp - last_time < self.cooldown:
            return None

        # 确定警报级别
        if diagnosis_idx in HIGH_RISK_DIAGNOSES:
            level = "HIGH"
        elif diagnosis_idx in MEDIUM_RISK_DIAGNOSES:
            level = "MEDIUM"
        else:
            level = "LOW"

        # 创建警报
        alert = {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'diagnosis_idx': diagnosis_idx,
            'diagnosis_name': DIAGNOSIS_NAMES.get(diagnosis_idx, f"类别{diagnosis_idx+1}"),
            'probability': float(probability),
            'level': level
        }

        # 更新状态
        self.last_alert_time[diagnosis_idx] = timestamp
        self.alert_history.append(alert)

        return alert

    def get_history(self) -> List[Dict]:
        return self.alert_history.copy()

    def clear(self):
        self.last_alert_time.clear()
        self.alert_history.clear()


class StreamingECGDetector:
    """
    流式ECG检测器

    支持实时ECG数据流的滑动窗口检测。
    """

    def __init__(
        self,
        model_path: str,
        window_size: int = 7500,      # 30秒 @ 250Hz
        step_size: int = 250,          # 1秒步长
        sample_rate: int = 250,
        num_classes: int = 40,
        threshold: float = 0.5,
        device: str = 'auto',
        # 平滑参数
        smooth_alpha: float = 0.3,
        smooth_window: int = 5,
        # 警报参数
        alert_cooldown: float = 30.0,
        alert_threshold: float = 0.7,  # 警报触发阈值
        # 回调
        on_alert: Optional[Callable[[Dict], None]] = None,
    ):
        """
        Args:
            model_path: 模型检查点路径
            window_size: 滑动窗口大小（采样点数）
            step_size: 滑动步长（采样点数）
            sample_rate: 采样率
            num_classes: 类别数
            threshold: 分类阈值
            device: 推理设备 ('auto', 'cuda', 'cpu', 'mps')
            smooth_alpha: 预测平滑系数
            smooth_window: 平滑窗口大小
            alert_cooldown: 警报冷却时间（秒）
            alert_threshold: 警报触发阈值
            on_alert: 警报回调函数
        """
        self.window_size = window_size
        self.step_size = step_size
        self.sample_rate = sample_rate
        self.num_classes = num_classes
        self.threshold = threshold
        self.alert_threshold = alert_threshold
        self.on_alert = on_alert

        # 设置设备
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
        self.model = self._load_model(model_path)

        # 初始化组件
        self.buffer = CircularBuffer(window_size)
        self.smoother = PredictionSmoother(num_classes, smooth_alpha, smooth_window)
        self.alert_manager = AlertManager(alert_cooldown)

        # 统计
        self.samples_processed = 0
        self.windows_processed = 0
        self.start_time = None
        self.pending_samples = 0  # 自上次推理后累积的样本数

        # 结果历史
        self.prediction_history: List[Dict] = []

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """加载模型"""
        from ecg_classifier.model.model_simple import get_model

        # 从检查点加载
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # 获取模型配置
        if 'config' in checkpoint:
            config = checkpoint['config']
            model_type = config.get('model_type', 'se')
            seq_len = config.get('seq_len', self.window_size)
            num_classes = config.get('num_classes', self.num_classes)
        else:
            model_type = 'se'
            seq_len = self.window_size
            num_classes = self.num_classes

        # 创建模型
        model = get_model(model_type, seq_len=seq_len, num_classes=num_classes)

        # 加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()

        print(f"模型加载成功: {model_path}")
        print(f"  设备: {self.device}")
        print(f"  类型: {model_type}")

        return model

    def _normalize(self, ecg: np.ndarray) -> np.ndarray:
        """Z-score归一化"""
        mean = ecg.mean()
        std = ecg.std()
        if std > 1e-6:
            return (ecg - mean) / std
        return ecg - mean

    def _predict(self, ecg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对单个窗口进行预测

        Returns:
            probs: 概率 [num_classes]
            preds: 二值预测 [num_classes]
        """
        # 归一化
        ecg_norm = self._normalize(ecg)

        # 转换为tensor
        x = torch.from_numpy(ecg_norm).float().unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        preds = (probs > self.threshold).astype(np.int32)

        return probs, preds

    def process(self, data: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        处理新到达的ECG数据

        Args:
            data: 新的ECG数据点 (可以是单点或数组)

        Returns:
            如果进行了推理，返回结果字典，否则返回None
        """
        if self.start_time is None:
            self.start_time = time.time()

        # 添加到缓冲区
        data = np.asarray(data, dtype=np.float32).flatten()
        self.buffer.append(data)
        self.samples_processed += len(data)
        self.pending_samples += len(data)

        # 检查是否需要进行推理
        if not self.buffer.is_full():
            return None

        if self.pending_samples < self.step_size:
            return None

        # 重置pending计数
        self.pending_samples = self.pending_samples % self.step_size

        # 获取当前窗口数据
        window_data = self.buffer.get_data()

        # 预测
        probs, preds = self._predict(window_data)

        # 平滑
        smoothed_probs = self.smoother.update(probs)
        stable_preds = self.smoother.get_stable_predictions(self.threshold)

        self.windows_processed += 1
        current_time = time.time()

        # 检查警报
        alerts = []
        for idx in range(self.num_classes):
            if smoothed_probs[idx] > self.alert_threshold:
                alert = self.alert_manager.check_alert(
                    idx, smoothed_probs[idx], current_time
                )
                if alert:
                    alerts.append(alert)
                    if self.on_alert:
                        self.on_alert(alert)

        # 构建结果
        result = {
            'timestamp': current_time,
            'window_idx': self.windows_processed,
            'raw_probs': probs.tolist(),
            'smoothed_probs': smoothed_probs.tolist(),
            'predictions': preds.tolist(),
            'stable_predictions': stable_preds.tolist() if stable_preds is not None else None,
            'alerts': alerts,
            'active_diagnoses': self._get_active_diagnoses(smoothed_probs),
        }

        self.prediction_history.append(result)

        return result

    def _get_active_diagnoses(self, probs: np.ndarray) -> List[Dict]:
        """获取当前激活的诊断"""
        active = []
        for idx in range(self.num_classes):
            if probs[idx] > self.threshold:
                active.append({
                    'idx': idx,
                    'name': DIAGNOSIS_NAMES.get(idx, f"类别{idx+1}"),
                    'probability': float(probs[idx]),
                    'risk_level': 'HIGH' if idx in HIGH_RISK_DIAGNOSES
                                 else 'MEDIUM' if idx in MEDIUM_RISK_DIAGNOSES
                                 else 'LOW'
                })
        return sorted(active, key=lambda x: x['probability'], reverse=True)

    def process_file(self, file_path: str, chunk_size: int = 250) -> List[Dict]:
        """
        处理ECG文件（模拟实时流）

        Args:
            file_path: ECG文件路径
            chunk_size: 每次处理的数据块大小

        Returns:
            所有检测结果列表
        """
        # 解析文件
        ecg_data = self._parse_ecg_file(file_path)
        if ecg_data is None:
            return []

        # 重置状态
        self.reset()

        # 模拟流式处理
        results = []
        for i in range(0, len(ecg_data), chunk_size):
            chunk = ecg_data[i:i+chunk_size]
            result = self.process(chunk)
            if result:
                results.append(result)

        return results

    def _parse_ecg_file(self, file_path: str) -> Optional[np.ndarray]:
        """解析ECG文件"""
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

            return np.array(ecg_values, dtype=np.float32)

        except Exception as e:
            print(f"解析文件失败: {e}")
            return None

    def get_report(self) -> Dict[str, Any]:
        """获取完整检测报告"""
        if not self.prediction_history:
            return {'status': 'no_data'}

        # 汇总统计
        all_alerts = self.alert_manager.get_history()

        # 诊断频率统计
        diagnosis_counts = {}
        for result in self.prediction_history:
            for diag in result['active_diagnoses']:
                idx = diag['idx']
                if idx not in diagnosis_counts:
                    diagnosis_counts[idx] = {
                        'name': diag['name'],
                        'count': 0,
                        'max_prob': 0,
                        'avg_prob': 0,
                        'probs': []
                    }
                diagnosis_counts[idx]['count'] += 1
                diagnosis_counts[idx]['probs'].append(diag['probability'])

        for idx, stats in diagnosis_counts.items():
            if stats['probs']:
                stats['max_prob'] = max(stats['probs'])
                stats['avg_prob'] = sum(stats['probs']) / len(stats['probs'])
            del stats['probs']  # 删除临时数据

        # 生成报告
        elapsed_time = time.time() - self.start_time if self.start_time else 0

        report = {
            'summary': {
                'samples_processed': self.samples_processed,
                'windows_processed': self.windows_processed,
                'duration_seconds': elapsed_time,
                'ecg_duration_seconds': self.samples_processed / self.sample_rate,
                'total_alerts': len(all_alerts),
                'high_risk_alerts': len([a for a in all_alerts if a['level'] == 'HIGH']),
                'medium_risk_alerts': len([a for a in all_alerts if a['level'] == 'MEDIUM']),
            },
            'diagnosis_statistics': diagnosis_counts,
            'alerts': all_alerts,
            'timestamp': datetime.now().isoformat(),
        }

        return report

    def reset(self):
        """重置检测器状态"""
        self.buffer.clear()
        self.smoother.reset()
        self.alert_manager.clear()
        self.samples_processed = 0
        self.windows_processed = 0
        self.pending_samples = 0
        self.start_time = None
        self.prediction_history.clear()

    def save_report(self, output_path: str):
        """保存检测报告到文件"""
        report = self.get_report()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"报告已保存: {output_path}")


class MultiLeadStreamingDetector:
    """
    多导联流式检测器

    支持1-12导联的实时检测
    """

    def __init__(
        self,
        model_path: str,
        num_leads: int = 12,
        window_size: int = 7500,
        step_size: int = 250,
        sample_rate: int = 250,
        num_classes: int = 40,
        device: str = 'auto',
    ):
        self.num_leads = num_leads
        self.window_size = window_size
        self.step_size = step_size
        self.sample_rate = sample_rate
        self.num_classes = num_classes

        # 每个导联一个缓冲区
        self.buffers = [CircularBuffer(window_size) for _ in range(num_leads)]

        # 设备
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        # 加载多导联模型
        self.model = self._load_model(model_path)

        self.pending_samples = 0
        self.windows_processed = 0

    def _load_model(self, model_path: str):
        """加载多导联模型"""
        # 这里需要根据实际的多导联模型架构来加载
        # 目前使用占位符
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        # model = MultiLeadECGClassifier(...)  # 需要实现多导联模型
        # model.load_state_dict(checkpoint['model_state_dict'])
        # return model.to(self.device)
        raise NotImplementedError("多导联模型需要单独实现")

    def process(self, data: np.ndarray) -> Optional[Dict]:
        """
        处理多导联数据

        Args:
            data: shape [num_leads, samples] 或 [samples] (单导联)
        """
        data = np.asarray(data)

        # 处理单导联输入
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # 添加到各导联缓冲区
        for i, buffer in enumerate(self.buffers):
            if i < data.shape[0]:
                buffer.append(data[i])

        self.pending_samples += data.shape[1]

        # 检查是否可以推理
        if not all(b.is_full() for b in self.buffers):
            return None

        if self.pending_samples < self.step_size:
            return None

        self.pending_samples = 0
        self.windows_processed += 1

        # 获取所有导联数据
        window_data = np.stack([b.get_data() for b in self.buffers], axis=0)

        # 归一化
        window_data = (window_data - window_data.mean(axis=1, keepdims=True)) / \
                      (window_data.std(axis=1, keepdims=True) + 1e-6)

        # 推理
        x = torch.from_numpy(window_data).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        return {
            'window_idx': self.windows_processed,
            'probs': probs.tolist(),
            'predictions': (probs > 0.5).astype(int).tolist()
        }


def demo_streaming_detection():
    """演示流式检测"""
    print("=" * 60)
    print("ECG流式检测演示")
    print("=" * 60)

    # 模拟ECG数据
    print("\n生成模拟ECG数据...")
    duration_seconds = 120  # 2分钟
    sample_rate = 250
    num_samples = duration_seconds * sample_rate

    # 生成模拟信号 (带周期性峰值)
    t = np.linspace(0, duration_seconds, num_samples)
    ecg_signal = np.sin(2 * np.pi * 1.2 * t)  # 基础心跳频率 ~72bpm
    ecg_signal += 0.5 * np.sin(2 * np.pi * 0.2 * t)  # 呼吸调制
    ecg_signal += 0.1 * np.random.randn(num_samples)  # 噪声

    # 添加一些"异常"事件
    ecg_signal[15000:15250] *= 2  # 模拟异常
    ecg_signal[22500:22750] *= 1.5

    print(f"  时长: {duration_seconds}秒")
    print(f"  采样率: {sample_rate}Hz")
    print(f"  样本数: {num_samples}")

    # 创建检测器 (使用假模型路径，实际使用时需要真实模型)
    print("\n注意: 此演示需要训练好的模型文件")
    print("      请将 model_path 修改为实际模型路径")

    # 以下代码展示如何使用（实际运行需要模型文件）
    demo_code = '''
# 实际使用示例:
def on_alert(alert):
    level = alert['level']
    name = alert['diagnosis_name']
    prob = alert['probability']
    print(f"[{level}] 检测到: {name} (置信度: {prob:.2%})")

detector = StreamingECGDetector(
    model_path='checkpoints/best_model.pth',
    window_size=7500,
    step_size=250,
    alert_threshold=0.7,
    on_alert=on_alert
)

# 模拟实时数据到达 (每秒250个点)
for i in range(0, len(ecg_signal), 250):
    chunk = ecg_signal[i:i+250]
    result = detector.process(chunk)

    if result and result['active_diagnoses']:
        print(f"窗口 {result['window_idx']}: {result['active_diagnoses']}")

# 获取报告
report = detector.get_report()
print(f"总共处理: {report['summary']['samples_processed']} 样本")
print(f"检测窗口: {report['summary']['windows_processed']} 个")
print(f"触发警报: {report['summary']['total_alerts']} 次")

# 保存报告
detector.save_report('detection_report.json')
'''
    print(demo_code)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='ECG流式检测')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--input', type=str, help='ECG文件路径 (可选)')
    parser.add_argument('--window', type=int, default=7500, help='窗口大小')
    parser.add_argument('--step', type=int, default=250, help='滑动步长')
    parser.add_argument('--threshold', type=float, default=0.5, help='分类阈值')
    parser.add_argument('--alert_threshold', type=float, default=0.7, help='警报阈值')
    parser.add_argument('--output', type=str, help='报告输出路径')
    parser.add_argument('--demo', action='store_true', help='运行演示')

    args = parser.parse_args()

    if args.demo:
        demo_streaming_detection()
        return

    # 警报回调
    def alert_callback(alert):
        level = alert['level']
        name = alert['diagnosis_name']
        prob = alert['probability']
        print(f"[{level}警报] {name} - 置信度: {prob:.2%}")

    # 创建检测器
    detector = StreamingECGDetector(
        model_path=args.model,
        window_size=args.window,
        step_size=args.step,
        threshold=args.threshold,
        alert_threshold=args.alert_threshold,
        on_alert=alert_callback
    )

    if args.input:
        # 处理文件
        print(f"\n处理文件: {args.input}")
        results = detector.process_file(args.input)
        print(f"处理完成，共 {len(results)} 个检测窗口")

        # 输出报告
        report = detector.get_report()
        print(f"\n=== 检测报告 ===")
        print(f"ECG时长: {report['summary']['ecg_duration_seconds']:.1f}秒")
        print(f"检测窗口: {report['summary']['windows_processed']}个")
        print(f"警报总数: {report['summary']['total_alerts']}")
        print(f"  - 高危: {report['summary']['high_risk_alerts']}")
        print(f"  - 中危: {report['summary']['medium_risk_alerts']}")

        if report['diagnosis_statistics']:
            print(f"\n检测到的诊断:")
            for idx, stats in sorted(report['diagnosis_statistics'].items(),
                                     key=lambda x: x[1]['count'], reverse=True):
                print(f"  {stats['name']}: {stats['count']}次 "
                      f"(最高: {stats['max_prob']:.2%}, 平均: {stats['avg_prob']:.2%})")

        if args.output:
            detector.save_report(args.output)
    else:
        # 交互模式
        print("\n进入交互模式，等待数据输入...")
        print("(实际应用中，这里会接收来自硬件/网络的实时数据)")
        print("演示: 使用 --demo 参数查看使用示例")


if __name__ == "__main__":
    main()
