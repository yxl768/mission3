"""
数据准备模块：
1) 合成革兰氏阴性菌活性AMP数据集（模拟DBAASP/DRAMP风格）
2) 数据集清洗、过滤、划分（train/val/test）
3) 提供Dataset类供训练使用
"""
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    AMINO_ACIDS, MIN_SEQ_LEN, MAX_SEQ_LEN, SYNTHETIC_DATA_SIZE,
    DATA_DIR, TARGET_PROPERTY_RANGES
)

# 常见AMP核心motif（革兰氏阴性菌活性常见模式）
COMMON_MOTIFS = [
    "KL", "LK", "KLK", "LKL", "KLAK", "AKLK",
    "KW", "KR", "RK", "FR", "RW", "WR",
    "KK", "RR", "KAK", "KAAK", "KLLK",
    "WKK", "KKW", "RWR", "WRW",
]

# 非AMP负样本倾向的残基模式（偏极性/中性、低电荷）
NEGATIVE_BIAS_RESIDUES = list("ACDGNPSTQE")

# 高疏水残基
HYDROPHOBIC_RESIDUES = set("AILMFWV")
# 正电荷残基
POSITIVE_RESIDUES = set("KRH")
# 芳香残基
AROMATIC_RESIDUES = set("FWY")


def _biased_amino_choice(rng, gram_active=True):
    """根据是否为AMP阳性样本生成带偏置的氨基酸采样"""
    if gram_active:
        # AMP阳性：增加K/R、疏水残基、芳香残基概率
        weights = []
        for aa in AMINO_ACIDS:
            w = 1.0
            if aa in POSITIVE_RESIDUES:
                w = 3.5
            elif aa in HYDROPHOBIC_RESIDUES:
                w = 2.5
            elif aa in AROMATIC_RESIDUES:
                w = 2.0
            elif aa in "DE":
                w = 0.4
            weights.append(w)
    else:
        # 负样本：偏极性/中性、低K/R
        weights = []
        for aa in AMINO_ACIDS:
            w = 1.0
            if aa in NEGATIVE_BIAS_RESIDUES:
                w = 2.5
            elif aa in POSITIVE_RESIDUES:
                w = 0.5
            weights.append(w)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    return rng.choice(AMINO_ACIDS, p=weights)


def _insert_motifs(seq_list, rng, num_motifs=1):
    """在序列中插入常见AMP motif片段"""
    seq_len = len(seq_list)
    for _ in range(num_motifs):
        motif = rng.choice(COMMON_MOTIFS)
        if len(motif) >= seq_len - 2:
            continue
        pos = rng.randint(0, max(1, seq_len - len(motif)))
        seq_list[pos:pos + len(motif)] = list(motif)
    return seq_list


def generate_synthetic_amp_dataset(n_positive=None, n_negative=None, seed=42):
    """
    生成模拟革兰氏阴性菌活性AMP数据集
    正样本：高正电荷、适中疏水、常见motif
    负样本：低电荷、偏极性、长度相近
    """
    rng = np.random.RandomState(seed)
    random.seed(seed)

    if n_positive is None:
        n_positive = int(SYNTHETIC_DATA_SIZE * 0.7)
    if n_negative is None:
        n_negative = SYNTHETIC_DATA_SIZE - n_positive

    records = []
    seq_id = 1

    # ===== 正样本 =====
    for _ in range(n_positive):
        seq_len = int(rng.randint(MIN_SEQ_LEN, MAX_SEQ_LEN + 1))
        seq_list = [_biased_amino_choice(rng, gram_active=True) for _ in range(seq_len)]
        # 插入motif（概率性）
        if rng.random() < 0.7:
            n_m = int(rng.randint(1, 3))
            seq_list = _insert_motifs(seq_list, rng, num_motifs=n_m)
        # 保证N端偏向K/R（AMP常见模式）
        if rng.random() < 0.4 and seq_list[0] not in POSITIVE_RESIDUES:
            candidates = ["K", "R"]
            seq_list[0] = rng.choice(candidates)

        sequence = "".join(seq_list)
        records.append({
            "seq_id": f"AMP_POS_{seq_id:05d}",
            "sequence": sequence,
            "label": 1,  # Gram-negative active
            "source": "synthetic_DBAASP_like",
        })
        seq_id += 1

    # ===== 负样本 =====
    for _ in range(n_negative):
        seq_len = int(rng.randint(MIN_SEQ_LEN, MAX_SEQ_LEN + 1))
        seq_list = [_biased_amino_choice(rng, gram_active=False) for _ in range(seq_len)]
        sequence = "".join(seq_list)
        records.append({
            "seq_id": f"AMP_NEG_{seq_id:05d}",
            "sequence": sequence,
            "label": 0,
            "source": "synthetic_UniProt_like",
        })
        seq_id += 1

    df = pd.DataFrame(records)
    # 简单去重
    df = df.drop_duplicates(subset=["sequence"]).reset_index(drop=True)
    return df


def filter_dataset(df):
    """
    序列过滤：
    - 只保留标准20种氨基酸
    - 长度在[MIN_SEQ_LEN, MAX_SEQ_LEN]
    """
    standard = set(AMINO_ACIDS)

    def _is_valid(seq):
        if not isinstance(seq, str):
            return False
        if not (MIN_SEQ_LEN <= len(seq) <= MAX_SEQ_LEN):
            return False
        return all(c in standard for c in seq)

    mask = df["sequence"].apply(_is_valid)
    df_filtered = df[mask].reset_index(drop=True)
    return df_filtered


def _seq_identity(s1: str, s2: str) -> float:
    """
    计算两条序列的相似度（基于全局对齐的简化版：长度归一化匹配率）
    采用局部对齐思想：较短的序列在较长序列上滑动找最佳匹配位置
    """
    if not s1 or not s2:
        return 0.0
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    L1, L2 = len(s1), len(s2)
    # 如果长度差异大，用滑动窗口找最佳匹配
    if L2 > int(L1 * 1.5):
        best = 0
        for start in range(0, L2 - L1 + 1):
            match = sum(1 for a, b in zip(s1, s2[start:start + L1]) if a == b)
            score = match / L1
            if score > best:
                best = score
            if best >= 1.0:
                break
        return best
    # 长度相近：逐位对齐
    match = sum(1 for a, b in zip(s1, s2) if a == b)
    return match / max(L1, L2)


def cdhit_cluster_sequences(df, identity_threshold=0.8, seed=42):
    """
    CD-HIT风格的贪心序列聚类
    1. 按序列长度降序排序
    2. 依次将每条序列与已有簇的代表序列比较
    3. 若相似度 >= threshold，归入该簇；否则新建簇
    返回 cluster_id 列表，同 cluster_id 的序列属于同一相似性簇

    参数：
        identity_threshold: 序列相似度阈值，默认0.8（即80% identity聚类）
    """
    rng = np.random.RandomState(seed)
    n = len(df)
    if n == 0:
        return []

    # 按长度降序排序（CD-HIT风格，长序列优先做代表）
    order = df["sequence"].str.len().sort_values(ascending=False).index.tolist()
    # 加入随机性避免完全确定性
    rng.shuffle(order)

    sequences = df["sequence"].values
    cluster_ids = [-1] * n
    representatives = []  # [(cluster_id, seq_str)]
    next_cluster_id = 0

    for idx in order:
        seq = sequences[idx]
        assigned = False
        for cid, rep_seq in representatives:
            if _seq_identity(seq, rep_seq) >= identity_threshold:
                cluster_ids[idx] = cid
                assigned = True
                break
        if not assigned:
            cluster_ids[idx] = next_cluster_id
            representatives.append((next_cluster_id, seq))
            next_cluster_id += 1

    return cluster_ids


def split_dataset(df, val_ratio=0.15, test_ratio=0.15, lead_ratio=0.10,
                  seed=42, identity_threshold=0.8):
    """
    CD-HIT风格聚类划分 train/val/test/lead：
    1. 按 sequence identity 阈值聚类，同簇序列整体划分到同一split
    2. 避免高度相似序列跨集泄漏
    3. 按 label 分层，保证正负样本比例
    4. lead/optimization split 与训练、测试完全独立
    """
    # A sequence must belong to one split even if DBAASP has duplicate records.
    # Keep a deterministic representative before label-wise clustering.
    df = df.drop_duplicates(subset=["sequence"], keep="first").reset_index(drop=True)
    rng = np.random.RandomState(seed)

    df_pos = df[df["label"] == 1].copy().reset_index(drop=True)
    df_neg = df[df["label"] == 0].copy().reset_index(drop=True)

    def _split_one(d):
        if len(d) == 0:
            return d, d, d

        # CD-HIT聚类
        cluster_ids = cdhit_cluster_sequences(d, identity_threshold=identity_threshold, seed=seed)
        d = d.copy()
        d["cluster_id"] = cluster_ids

        # 按簇划分：将簇随机分配到train/val/test
        unique_clusters = sorted(d["cluster_id"].unique())
        rng.shuffle(unique_clusters)

        n_clusters = len(unique_clusters)
        n_test = max(1, int(n_clusters * test_ratio))
        n_val = max(1, int(n_clusters * val_ratio))
        n_lead = max(1, int(n_clusters * lead_ratio))

        test_clusters = set(unique_clusters[:n_test])
        val_clusters = set(unique_clusters[n_test:n_test + n_val])
        lead_clusters = set(unique_clusters[n_test + n_val:n_test + n_val + n_lead])
        train_clusters = set(unique_clusters[n_test + n_val + n_lead:])

        train = d[d["cluster_id"].isin(train_clusters)].drop(columns=["cluster_id"])
        val = d[d["cluster_id"].isin(val_clusters)].drop(columns=["cluster_id"])
        test = d[d["cluster_id"].isin(test_clusters)].drop(columns=["cluster_id"])
        lead = d[d["cluster_id"].isin(lead_clusters)].drop(columns=["cluster_id"])

        # 如果某个split为空，回退到随机划分
        if len(train) == 0 or len(val) == 0 or len(test) == 0 or len(lead) == 0:
            rest, test = train_test_split(d, test_size=test_ratio, random_state=seed)
            rest, lead = train_test_split(rest, test_size=lead_ratio / (1 - test_ratio), random_state=seed)
            train, val = train_test_split(rest, test_size=val_ratio / (1 - test_ratio - lead_ratio), random_state=seed)
            train = train.drop(columns=["cluster_id"]) if "cluster_id" in train.columns else train
            val = val.drop(columns=["cluster_id"]) if "cluster_id" in val.columns else val
            test = test.drop(columns=["cluster_id"]) if "cluster_id" in test.columns else test
            lead = lead.drop(columns=["cluster_id"]) if "cluster_id" in lead.columns else lead

        return (train.reset_index(drop=True), val.reset_index(drop=True),
                test.reset_index(drop=True), lead.reset_index(drop=True))

    tr_p, va_p, te_p, le_p = _split_one(df_pos)
    tr_n, va_n, te_n, le_n = _split_one(df_neg)

    train = pd.concat([tr_p, tr_n]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val = pd.concat([va_p, va_n]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = pd.concat([te_p, te_n]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    lead = pd.concat([le_p, le_n]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"[DATA] CD-HIT聚类划分(identity_threshold={identity_threshold}): "
            f"train={len(train)}, val={len(val)}, test={len(test)}, lead={len(lead)}")
    return train, val, test, lead


def load_dbaasp_dataset():
    """加载DBAASP真实数据集，如果不存在则触发下载"""
    dbaasp_path = os.path.join(DATA_DIR, "dbaasp_dataset.csv")
    if os.path.exists(dbaasp_path):
        df = pd.read_csv(dbaasp_path)
        print(f"[DATA] 已加载DBAASP真实数据集: {len(df)} 条")
        return df
    # 尝试下载
    print("[DATA] 未找到DBAASP数据集，开始从DBAASP下载...")
    try:
        from data.download_dbaasp import download_dbaasp_dataset, build_train_dataset_from_dbaasp
        df_raw = download_dbaasp_dataset()
        df = build_train_dataset_from_dbaasp(df_raw)
        return df
    except Exception as e:
        print(f"[DATA] DBAASP下载失败: {e}")
        return None


def build_and_save_dataset():
    """主入口：优先使用DBAASP真实数据，否则回退到合成数据"""
    # 优先尝试加载DBAASP真实数据
    df = load_dbaasp_dataset()

    if df is None or len(df) == 0:
        print("[DATA] DBAASP数据不可用，回退到合成数据集...")
        df = generate_synthetic_amp_dataset()
        print(f"[DATA] 生成样本数: {len(df)} (正样本={(df.label==1).sum()}, 负样本={(df.label==0).sum()})")
        df = filter_dataset(df)
    else:
        # DBAASP真实数据：统一列名
        # 确保有 seq_id, sequence, label, source 列
        if "seq_id" not in df.columns:
            df["seq_id"] = df.get("dbaasp_id", df.index.astype(str))
        # 过滤
        df = filter_dataset(df)
        print(f"[DATA] DBAASP过滤后样本数: {len(df)} (正样本={(df.label==1).sum()}, 负样本={(df.label==0).sum()})")
        # 统计MIC信息
        if "best_gn_mic" in df.columns:
            mic_vals = df[df["label"] == 1]["best_gn_mic"].dropna()
            if len(mic_vals) > 0:
                print(f"[DATA] G-阳性样本MIC分布: min={mic_vals.min():.2f}, "
                      f"median={mic_vals.median():.2f}, max={mic_vals.max():.2f}")
                print(f"[DATA] MIC ≤ 16: {(mic_vals <= 16).sum()} 条; MIC ≤ 8: {(mic_vals <= 8).sum()} 条")

    print(f"[DATA] 过滤后样本数: {len(df)}")

    train, val, test, lead = split_dataset(df)
    print(f"[DATA] 划分: train={len(train)}, val={len(val)}, test={len(test)}, lead={len(lead)}")

    # 保存CSV
    train.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False, encoding="utf-8")
    val.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False, encoding="utf-8")
    test.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False, encoding="utf-8")
    lead.to_csv(os.path.join(DATA_DIR, "lead.csv"), index=False, encoding="utf-8")

    leads = lead[lead["label"] == 1].head(50).copy()
    if "seq_id" in leads.columns:
        leads = leads.rename(columns={"seq_id": "lead_id"})
    elif "dbaasp_id" in leads.columns:
        leads = leads.rename(columns={"dbaasp_id": "lead_id"})
    leads.to_csv(os.path.join(DATA_DIR, "lead_peptides.csv"), index=False, encoding="utf-8")
    print(f"[DATA] Lead peptides数量: {len(leads)}")
    print("[DATA] 数据集已保存到", DATA_DIR)

    return train, val, test, lead


def load_saved_dataset():
    """从已保存CSV加载数据集"""
    paths = [
        os.path.join(DATA_DIR, "train.csv"),
        os.path.join(DATA_DIR, "val.csv"),
        os.path.join(DATA_DIR, "test.csv"),
        os.path.join(DATA_DIR, "lead.csv"),
    ]
    if all(os.path.exists(p) for p in paths):
        train = pd.read_csv(paths[0])
        val = pd.read_csv(paths[1])
        test = pd.read_csv(paths[2])
        lead = pd.read_csv(paths[3])
        return train, val, test, lead
    return None, None, None


if __name__ == "__main__":
    build_and_save_dataset()
