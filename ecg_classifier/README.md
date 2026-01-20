# ECG Transformer 分类模型

基于Transformer架构的单导联ECG多标签分类模型，融合传统算法ECGBrain505的设计思想。

## 特性

- **多尺度Patch Embedding**: 借鉴传统算法的分块策略
- **频域特征分支**: 融合FFT频谱特征
- **多标签分类**: 支持40种诊断类型的多标签预测
- **医学级评估**: 完整的评估指标和错误分析
- **最优阈值搜索**: 为每个类别自动寻找最优分类阈值

## 快速开始

### 1. 安装依赖

```bash
cd ecg_classifier
pip install -r requirements.txt
```

### 2. 准备数据

数据格式要求:
```
第1行: 文件信息(可选)
第2行: 文件信息(可选)
第3行: 诊断类型1 (1-40的整数)
第4行: 诊断类型2 (1-40的整数，可为空)
第5行: 采样率 (250)
第6行: 时长 (30)
第7行: 分隔符 (32767)
第8行起: ECG时序数据 (7500个数据点)
```

### 3. 训练模型

```bash
# 基础训练
python train.py --data_dir /path/to/your/ecg_data --epochs 100

# 完整参数
python train.py \
    --data_dir /path/to/your/ecg_data \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --hidden_dim 256 \
    --num_layers 6 \
    --device cuda

# 使用轻量版模型 (更快训练)
python train.py --data_dir /path/to/data --lite
```

### 4. 模型推理

```bash
# 单文件预测
python inference.py \
    --model_path checkpoints/best_model.pt \
    --ecg_file /path/to/test.txt

# 使用最优阈值
python inference.py \
    --model_path checkpoints/best_model.pt \
    --ecg_file /path/to/test.txt \
    --use_optimal_thresh

# Top-K预测 (医学场景推荐)
python inference.py \
    --model_path checkpoints/best_model.pt \
    --ecg_file /path/to/test.txt \
    --top_k 5

# 批量预测
python inference.py \
    --model_path checkpoints/best_model.pt \
    --data_dir /path/to/test_folder \
    --output predictions.json
```

### 5. 模型评估

```bash
python evaluate.py \
    --model_path checkpoints/best_model.pt \
    --data_dir /path/to/test_data \
    --use_optimal_thresh \
    --output_dir ./evaluation
```

## 模型架构

```
ECGTransformerClassifier
├── PatchEmbedding          # 信号分割 (7500 -> 150 patches)
├── PositionalEncoding      # 位置编码
├── TransformerEncoder      # 6层Transformer
├── FrequencyBranch         # FFT频域特征
├── MultiScalePatchEmbed    # 多尺度特征 (25/50/150)
├── FeatureFusion           # 特征融合
└── ClassificationHead      # 40类输出
```

## 诊断类型

| ID | 诊断名称 | ID | 诊断名称 |
|----|---------|-----|---------|
| 1 | 窦性心律 | 21 | 室上性心动过速 |
| 2 | 心电图未见异常 | 22 | 一度房室阻滞 |
| 3 | 窦性心动过速 | 23 | ST段抬高 |
| 4 | 窦性心动过缓 | 24 | ST段压低 |
| 5 | 窦性停搏 | 25 | QT/QTc间期延长 |
| 6 | 心房颤动 | 26 | RR长间歇 |
| 7-13 | 房性早搏系列 | 27-29 | 传导/干扰/脱落 |
| 14-20 | 室性早搏系列 | 30-40 | 其他异常 |

## 医学场景建议

### 阈值选择策略

1. **高敏感度场景** (漏诊代价高): 降低阈值到0.3-0.4
   - 心房颤动、室性心动过速等高危诊断

2. **高特异度场景** (误诊代价高): 提高阈值到0.6-0.7
   - 一般性异常诊断

3. **平衡场景**: 使用`--use_optimal_thresh`自动选择

### Top-K vs 阈值

```python
# 医学场景推荐: 使用Top-K而非固定阈值
# 返回概率最高的5个诊断供医生参考
python inference.py --top_k 5
```

## 训练技巧

1. **类别不平衡处理**
   - 使用`AsymmetricLoss`对正负样本不同加权
   - 或使用`FocalLoss`关注难分类样本

2. **数据增强**
   - 随机噪声注入
   - 随机幅度缩放
   - 随机时间平移

3. **混合精度训练**
   - 自动启用FP16加速(CUDA设备)

## 文件结构

```
ecg_classifier/
├── config.py        # 配置文件
├── dataset.py       # 数据加载
├── model.py         # 模型定义
├── train.py         # 训练脚本
├── inference.py     # 推理脚本
├── evaluate.py      # 评估脚本
├── requirements.txt # 依赖
└── README.md        # 说明文档
```

## 输出示例

```
==================================================
心电图诊断报告
==================================================

风险等级: 中风险 ⚠

诊断结果:
  • 窦性心律 (置信度: 高, 95.2%)
  • 偶发室性早搏 (置信度: 中, 67.3%)
  • T波改变 (置信度: 低, 45.1%)
==================================================
```

## 性能参考

| 模型 | 参数量 | F1 (macro) | AUC | 推理速度 |
|-----|--------|-----------|-----|---------|
| ECGTransformer | 12M | ~0.75 | ~0.85 | 100 samples/s |
| ECGTransformerLite | 3M | ~0.70 | ~0.82 | 300 samples/s |

*实际性能取决于数据质量和分布*
