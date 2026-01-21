"""完整数据泄露检查 - 使用哈希加速"""
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm

print("=" * 70)
print("完整数据泄露检查（基于哈希）")
print("=" * 70)

# 加载数据
print("\n[1/4] 加载数据...")
# with open('ecg_data_clean.pkl', 'rb') as f:
with open('/ecg_classifier/data/ecg_data_dedup.pkl', 'rb') as f:
    data = pickle.load(f)

train_X = data['train']['X']
test_X = data['test']['X']
train_paths = data['train']['paths']
test_paths = data['test']['paths']


print(f"  训练集: {len(train_X):,} 样本")
print(f"  测试集: {len(test_X):,} 样本")

# 计算训练集哈希
print("\n[2/4] 计算训练集哈希...")
train_hashes = {}
for i, x in enumerate(tqdm(train_X, desc="训练集")):
    h = hash(x.tobytes())
    if h not in train_hashes:
        train_hashes[h] = []
    train_hashes[h].append(i)

print(f"  训练集唯一哈希数: {len(train_hashes):,}")
print(f"  训练集重复组数: {len([v for v in train_hashes.values() if len(v) > 1])}")

# 检查测试集重复
print("\n[3/4] 检查测试集与训练集重叠...")
leak_count = 0
leak_details = []

for i, x in enumerate(tqdm(test_X, desc="测试集")):
    h = hash(x.tobytes())
    if h in train_hashes:
        # 找到重复
        leak_count += 1
        if len(leak_details) < 10:  # 只记录前10个示例
            train_indices = train_hashes[h]
            leak_details.append({
                'test_idx': i,
                'test_path': test_paths[i],
                'train_indices': train_indices[:3],  # 只显示前3个
                'train_paths': [train_paths[idx] for idx in train_indices[:3]]
            })

# 结果
print("\n" + "=" * 70)
print("检查结果")
print("=" * 70)

print(f"\n测试集总样本数: {len(test_X):,}")
print(f"泄露样本数: {leak_count:,}")
print(f"泄露比例: {leak_count/len(test_X)*100:.2f}%")

if leak_count > 0:
    print(f"\n⚠️⚠️⚠️ 严重数据泄露！")
    print(f"  {leak_count:,} 个测试样本在训练集中出现过！")
    print(f"  这解释了为什么F1高达99.91%！")

    print(f"\n示例泄露数据（前10个）:")
    for i, detail in enumerate(leak_details, 1):
        print(f"\n  [{i}] 测试样本 {detail['test_idx']}:")
        print(f"      测试文件: {detail['test_path']}")
        print(f"      在训练集中的位置: {detail['train_indices']}")
        print(f"      训练文件: {detail['train_paths'][0]}")
else:
    print(f"\n✓ 未发现数据泄露")

# 检查训练集内部重复
print("\n" + "=" * 70)
print("训练集内部重复检查")
print("=" * 70)

dup_groups = [v for v in train_hashes.values() if len(v) > 1]
total_dups = sum(len(v) - 1 for v in dup_groups)

print(f"重复组数: {len(dup_groups)}")
print(f"重复样本数: {total_dups:,}")
print(f"重复比例: {total_dups/len(train_X)*100:.2f}%")

if len(dup_groups) > 0:
    print(f"\n前5个重复组:")
    for i, group in enumerate(dup_groups[:5], 1):
        print(f"  组{i}: {len(group)} 个相同样本")
        print(f"    索引: {group[:5]}")
        print(f"    文件: {[train_paths[idx] for idx in group[:2]]}")

print("\n" + "=" * 70)
