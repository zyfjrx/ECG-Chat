"""
ECG分类模型训练脚本

用法:
    python train.py --data_dir /path/to/data --epochs 100
"""

import os
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)
from tqdm import tqdm
import matplotlib.pyplot as plt

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'STKaiti', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

from ecg_classifier.conf.config import Config
from ecg_classifier.data.dataset import (load_data, load_data2, create_dataloaders, create_dataloaders_from_preprocessed)
from ecg_classifier.model.model_simple import get_model


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification
    处理类别不平衡问题
    """

    def __init__(self, alpha=1, gamma=2, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        loss = self.alpha * focal_weight * bce_loss
        return loss.mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification
    对正负样本使用不同的gamma值
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits, targets):
        # Sigmoid
        probs = torch.sigmoid(logits)

        # 正样本损失
        pos_loss = targets * torch.log(probs.clamp(min=1e-8))
        pos_loss = pos_loss * ((1 - probs) ** self.gamma_pos)

        # 负样本损失 (with clipping)
        probs_neg = (probs + self.clip).clamp(max=1)
        neg_loss = (1 - targets) * torch.log((1 - probs_neg).clamp(min=1e-8))
        neg_loss = neg_loss * (probs_neg ** self.gamma_neg)

        loss = -pos_loss - neg_loss
        return loss.mean()


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


class Trainer:
    """训练器"""

    def __init__(self, config, model, train_loader, val_loader, device):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # 计算类别权重
        self.pos_weight = torch.ones(config.num_classes).to(device) * config.pos_weight

        # 损失函数 (选择一种)
        # self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        # self.criterion = FocalLoss(gamma=2, pos_weight=self.pos_weight)
        self.criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1)

        # 优化器 (参考ECG-CoCa的参数设置)
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(getattr(config, 'beta1', 0.9), getattr(config, 'beta2', 0.98)),
            eps=getattr(config, 'eps', 1e-6)
        )

        # 学习率调度器
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )

        # 混合精度训练 (可通过 --no-amp 禁用)
        self.use_amp = getattr(config, 'use_amp', False) and device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        print(f"混合精度训练: {'启用' if self.use_amp else '禁用'}")

        # 记录
        self.best_f1 = 0
        self.history = {
            'train_loss': [], 'val_loss': [],
            'val_f1': [], 'val_auc': [],
            'val_sensitivity': [], 'val_specificity': []
        }

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for ecg, labels in pbar:
            ecg = ecg.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # 混合精度训练
            # 检查输入数据是否有nan
            if torch.isnan(ecg).any() or torch.isinf(ecg).any():
                print(f"\n⚠️ 跳过含nan/inf的batch")
                continue

            if self.use_amp and self.scaler is not None:
                with autocast():
                    logits = self.model(ecg)
                    loss = self.criterion(logits, labels)

                # 检查loss是否为nan
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n⚠️ loss=nan/inf! 输入范围:[{ecg.min():.2f}, {ecg.max():.2f}], 跳过此batch")
                    continue

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(ecg)
                loss = self.criterion(logits, labels)

                # 检查loss是否为nan
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n⚠️ loss=nan/inf! 输入范围:[{ecg.min():.2f}, {ecg.max():.2f}], 跳过此batch")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        all_logits = []
        all_labels = []
        total_loss = 0
        num_batches = 0

        for ecg, labels in tqdm(self.val_loader, desc="Validating"):
            ecg = ecg.to(self.device)
            labels = labels.to(self.device)

            if self.use_amp and self.scaler is not None:
                with autocast():
                    logits = self.model(ecg)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(ecg)
                loss = self.criterion(logits, labels)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item()
            num_batches += 1

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # 计算指标
        metrics = self.compute_metrics(all_logits, all_labels)
        metrics['loss'] = total_loss / num_batches

        return metrics

    def compute_metrics(self, logits, labels, threshold=0.5):
        """计算多标签分类指标 (与evaluate.py一致的医学指标)"""
        probs = torch.sigmoid(logits).numpy()
        labels = labels.numpy()
        preds = (probs >= threshold).astype(int)

        # 计算各种指标
        metrics = {}

        # Micro/Macro F1
        metrics['f1_micro'] = f1_score(labels, preds, average='micro', zero_division=0)
        metrics['f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(labels, preds, average='weighted', zero_division=0)

        # Precision/Recall (Sensitivity)
        metrics['precision_micro'] = precision_score(labels, preds, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(labels, preds, average='micro', zero_division=0)
        metrics['sensitivity_micro'] = metrics['recall_micro']  # 医学别名

        # Specificity (macro average)
        specificities = []
        for i in range(labels.shape[1]):
            if labels[:, i].sum() < len(labels):  # 有负样本才能算
                spec = compute_specificity(labels[:, i], preds[:, i])
                specificities.append(spec)
        metrics['specificity_macro'] = np.mean(specificities) if specificities else 0

        # AUC (per class then average)
        try:
            auc_scores = []
            for i in range(labels.shape[1]):
                if labels[:, i].sum() > 0 and labels[:, i].sum() < len(labels):
                    auc = roc_auc_score(labels[:, i], probs[:, i])
                    auc_scores.append(auc)
            metrics['auc_macro'] = np.mean(auc_scores) if auc_scores else 0
        except:
            metrics['auc_macro'] = 0

        # mAP (mean Average Precision)
        try:
            ap_scores = []
            for i in range(labels.shape[1]):
                if labels[:, i].sum() > 0:
                    ap = average_precision_score(labels[:, i], probs[:, i])
                    ap_scores.append(ap)
            metrics['mAP'] = np.mean(ap_scores) if ap_scores else 0
        except:
            metrics['mAP'] = 0

        # Exact Match Ratio (完全匹配，医学场景不重要)
        metrics['exact_match'] = np.mean(np.all(preds == labels, axis=1))

        # Hamming Loss
        metrics['hamming_loss'] = np.mean(preds != labels)

        return metrics

    def train(self):
        """完整训练流程"""
        print(f"\n开始训练...")
        print(f"设备: {self.device}")
        print(f"模型参数量: {count_parameters(self.model):,}")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Epochs: {self.config.num_epochs}")
        print("-" * 50)

        for epoch in range(1, self.config.num_epochs + 1):
            # 训练
            train_loss = self.train_epoch(epoch)

            # 验证
            val_metrics = self.validate()

            # 记录
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_f1'].append(val_metrics['f1_macro'])
            self.history['val_auc'].append(val_metrics['auc_macro'])
            self.history['val_sensitivity'].append(val_metrics['sensitivity_micro'])
            self.history['val_specificity'].append(val_metrics['specificity_macro'])

            # 打印 (医学核心指标)
            print(f"\nEpoch {epoch}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val F1 (macro): {val_metrics['f1_macro']:.4f}")
            print(f"  Val F1 (micro): {val_metrics['f1_micro']:.4f}")
            print(f"  Val Sensitivity (micro): {val_metrics['sensitivity_micro']:.4f}  (检出率)")
            print(f"  Val Specificity (macro): {val_metrics['specificity_macro']:.4f}  (特异度)")
            print(f"  Val AUC: {val_metrics['auc_macro']:.4f}")
            print(f"  Val mAP: {val_metrics['mAP']:.4f}")

            # 保存最佳模型
            if val_metrics['f1_macro'] > self.best_f1:
                self.best_f1 = val_metrics['f1_macro']
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"  ★ 新最佳模型! F1: {self.best_f1:.4f}")

            # 定期保存
            if epoch % 50 == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)

        print(f"\n训练完成! 最佳F1: {self.best_f1:.4f}")
        return self.history

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """保存检查点"""
        os.makedirs(self.config.save_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': vars(self.config)
        }

        if is_best:
            path = os.path.join(self.config.save_dir, 'best_model.pt')
        else:
            path = os.path.join(self.config.save_dir, f'checkpoint_epoch_{epoch}.pt')

        torch.save(checkpoint, path)
        print(f"  保存模型: {path}")


def find_optimal_thresholds(model, val_loader, device, num_classes=40):
    """
    为每个类别寻找最优阈值

    医学场景建议:
    - 对于漏诊代价高的疾病，降低阈值（提高召回率）
    - 对于误诊代价高的疾病，提高阈值（提高精确率）
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for ecg, labels in val_loader:
            ecg = ecg.to(device)
            logits = model(ecg)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    probs = torch.sigmoid(all_logits).numpy()
    labels = all_labels.numpy()

    # 为每个类别寻找最优阈值
    optimal_thresholds = []
    for i in range(num_classes):
        if labels[:, i].sum() == 0:
            optimal_thresholds.append(0.5)
            continue

        best_f1 = 0
        best_thresh = 0.5

        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (probs[:, i] >= thresh).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        optimal_thresholds.append(best_thresh)
        print(f"类别 {i+1}: 最优阈值={best_thresh:.2f}, F1={best_f1:.4f}")

    return np.array(optimal_thresholds)


def plot_training_history(history, save_path='training_curves.png'):
    """
    绘制训练曲线

    Args:
        history: dict with keys like 'train_loss', 'val_loss', 'val_f1', etc.
        save_path: 保存路径
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 标注最低val loss
    min_val_loss_idx = np.argmin(history['val_loss'])
    ax1.axvline(x=min_val_loss_idx + 1, color='r', linestyle='--', alpha=0.5)
    ax1.annotate(f'Min: {history["val_loss"][min_val_loss_idx]:.4f}',
                 xy=(min_val_loss_idx + 1, history['val_loss'][min_val_loss_idx]),
                 xytext=(10, 10), textcoords='offset points')

    # 2. F1曲线
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['val_f1'], 'g-', label='Val F1 (macro)', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Validation F1 Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 标注最高F1
    max_f1_idx = np.argmax(history['val_f1'])
    ax2.axvline(x=max_f1_idx + 1, color='g', linestyle='--', alpha=0.5)
    ax2.annotate(f'Max: {history["val_f1"][max_f1_idx]:.4f}',
                 xy=(max_f1_idx + 1, history['val_f1'][max_f1_idx]),
                 xytext=(10, -10), textcoords='offset points')

    # 3. AUC曲线
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['val_auc'], 'm-', label='Val AUC (macro)', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AUC')
    ax3.set_title('Validation AUC')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Sensitivity & Specificity
    ax4 = axes[1, 1]
    if 'val_sensitivity' in history and 'val_specificity' in history:
        ax4.plot(epochs, history['val_sensitivity'], 'c-', label='Sensitivity (检出率)', linewidth=2)
        ax4.plot(epochs, history['val_specificity'], 'orange', label='Specificity (特异度)', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Score')
        ax4.set_title('Sensitivity & Specificity')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No Sensitivity/Specificity data', ha='center', va='center')
        ax4.set_title('Sensitivity & Specificity (N/A)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n训练曲线已保存: {save_path}")

    # 打印最佳指标
    print(f"\n【训练摘要】")
    print(f"  最佳 Val Loss: {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})")
    print(f"  最佳 Val F1:   {max(history['val_f1']):.4f} (Epoch {np.argmax(history['val_f1']) + 1})")
    print(f"  最佳 Val AUC:  {max(history['val_auc']):.4f} (Epoch {np.argmax(history['val_auc']) + 1})")


def main():
    parser = argparse.ArgumentParser(description='ECG分类模型训练')

    # 数据源 (三选一)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--data', type=str, help='预处理后的数据文件 (npz/hdf5/pkl，推荐)')
    data_group.add_argument('--data_dir', type=str, help='原始数据目录')
    data_group.add_argument('--splits', type=str, help='数据划分JSON文件')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=6, help='Transformer层数')
    parser.add_argument('--model_version', type=str, default='se',
                        choices=['simple', 'fft', 'full', 'se', 'se_full'],
                        help='模型版本: simple/fft/full/se/se_full (推荐se)')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader进程数')
    parser.add_argument('--no-amp', action='store_true', help='禁用混合精度训练(解决nan问题)')
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 配置
    config = Config()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.hidden_dim = args.hidden_dim
    config.num_layers = args.num_layers
    config.use_amp = not getattr(args, 'no_amp', False)  # 混合精度开关

    # 设备
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"使用设备: {device}")

    # 加载数据
    if args.data:
        # 方式1: 从预处理文件加载 (最快)
        print(f"从预处理文件加载: {args.data}")
        train_loader, val_loader, test_loader = create_dataloaders_from_preprocessed(
            args.data,
            batch_size=config.batch_size,
            num_workers=args.num_workers
        )
    elif args.splits:
        # 方式2: 从splits JSON加载
        print(f"从splits文件加载: {args.splits}")
        train_files, val_files, test_files = load_data2(args.splits)
        train_loader, val_loader, test_loader = create_dataloaders(
            train_files, val_files, test_files,
            batch_size=config.batch_size,
            num_workers=args.num_workers
        )
    else:
        # 方式3: 从原始目录加载并自动划分
        print(f"从目录加载: {args.data_dir}")
        train_files, val_files, test_files = load_data(args.data_dir)
        train_loader, val_loader, test_loader = create_dataloaders(
            train_files, val_files, test_files,
            batch_size=config.batch_size,
            num_workers=args.num_workers
        )

    # 创建模型 (使用model_simple.py)
    model = get_model(
        version=args.model_version,
        seq_len=config.seq_len,
        num_classes=config.num_classes,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        patch_size=config.patch_size if args.model_version != 'full' else 50  # full版本patch_size固定
    )
    print(f"模型版本: {args.model_version}")

    # 训练
    trainer = Trainer(config, model, train_loader, val_loader, device)
    history = trainer.train()

    # 保存训练历史
    history_path = os.path.join(config.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)

    # 绘制训练曲线
    plot_path = os.path.join(config.save_dir, 'training_curves.png')
    plot_training_history(history, save_path=plot_path)

    # 寻找最优阈值
    print("\n寻找每个类别的最优阈值...")
    model.load_state_dict(torch.load(os.path.join(config.save_dir, 'best_model.pt'),weights_only=False))
    optimal_thresholds = find_optimal_thresholds(model, val_loader, device)
    np.save(os.path.join(config.save_dir, 'optimal_thresholds.npy'), optimal_thresholds)

    print("\n训练完成!")


if __name__ == "__main__":
    main()
