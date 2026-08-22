"""
机制相关描述符计算模块
围绕抗革兰氏阴性菌活性AMP的作用机制，构建可解释的描述符
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AMINO_ACIDS, DESCRIPTOR_NAMES

# 基础物理化学参数
# 氨基酸分子量 (Da，平均)
AA_MW = {
    "A": 71.08, "R": 156.19, "N": 114.10, "D": 115.09, "C": 103.14,
    "E": 129.12, "Q": 128.13, "G": 57.05, "H": 137.14, "I": 113.16,
    "L": 113.16, "K": 128.17, "M": 131.20, "F": 147.18, "P": 97.12,
    "S": 87.08, "T": 101.10, "W": 186.21, "Y": 163.18, "V": 99.13,
}

# pKa值 (用于计算pH7下净电荷)
PKA_NTERM = 9.69
PKA_CTERM = 2.34
PKA_SIDECHAIN = {
    "D": 3.65, "E": 4.25,
    "C": 8.18, "Y": 10.07,
    "H": 6.00,
    "K": 10.54, "R": 12.48,
}

# K-D疏水性（GRAVY）
KD_HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# 不稳定性指数参数
II_PARAMS = {
    "A": (-0.38, -0.57), "R": (-0.30, -0.66), "N": (-0.54, -0.92), "D": (-0.76, -1.24),
    "C": (-0.30, -0.01), "Q": (-0.47, -0.71), "E": (-0.75, -1.23), "G": (-0.48, -0.80),
    "H": (-0.49, -0.81), "I": (-0.38, -0.56), "L": (-0.35, -0.51), "K": (-0.47, -0.73),
    "M": (-0.46, -0.71), "F": (-0.49, -0.76), "P": (-0.40, -0.63), "S": (-0.39, -0.65),
    "T": (-0.41, -0.68), "W": (-0.38, -0.56), "Y": (-0.38, -0.56), "V": (-0.39, -0.58),
}

# 二级结构倾向（简化版）
HELIX_PROPENSITY = {
    "A": 1.42, "R": 0.95, "N": 0.67, "D": 1.01, "C": 0.70,
    "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.34,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}
TURN_PROPENSITY = {
    "A": 0.60, "R": 0.95, "N": 1.56, "D": 1.46, "C": 1.19,
    "Q": 0.98, "E": 0.54, "G": 1.64, "H": 0.95, "I": 0.47,
    "L": 0.59, "K": 1.01, "M": 0.60, "F": 0.60, "P": 1.52,
    "S": 1.23, "T": 0.96, "W": 0.96, "Y": 1.14, "V": 0.50,
}

# Boman相互作用潜力指数（估算膜结合自由能）
BOMAN_INDEX = {
    "A": -0.19, "R": 0.81, "N": 0.23, "D": -0.15, "C": -0.18,
    "Q": 0.20, "E": 0.04, "G": -0.14, "H": 0.51, "I": -0.60,
    "L": -0.62, "K": 0.38, "M": -0.41, "F": -0.50, "P": -0.01,
    "S": 0.13, "T": 0.11, "W": -0.40, "Y": -0.14, "V": -0.41,
}

# 疏水矩（Eisenberg，用于计算两亲性）
HYDROPHOBIC_EISENBERG = {
    "A": 0.25, "R": -1.76, "N": -0.64, "D": -0.72, "C": 0.04,
    "Q": -0.69, "E": -0.62, "G": 0.16, "H": -0.40, "I": 0.73,
    "L": 0.53, "K": -1.10, "M": 0.26, "F": 0.61, "P": -0.07,
    "S": -0.26, "T": -0.18, "W": 0.37, "Y": 0.02, "V": 0.54,
}

# 正/负/疏水/芳香残基集合
POSITIVE_AAS = set("KRH")
NEGATIVE_AAS = set("DE")
HYDROPHOBIC_AAS = set("AILMFWV")
AROMATIC_AAS = set("FWY")
ALIPHATIC_AAS = set("AILV")


# 描述符计算函数
def calc_length(seq: str) -> int:
    return len(seq)


def calc_molecular_weight(seq: str) -> float:
    """估算分子量（减去脱水缩合H2O）"""
    mw = sum(AA_MW.get(aa, 110.0) for aa in seq) - 18.015 * (len(seq) - 1)
    return max(0.0, mw)


def calc_net_charge(seq: str, pH: float = 7.0) -> float:
    """pH下净电荷（Henderson-Hasselbalch）"""
    # N端正电
    charge = 1.0 / (10 ** (pH - PKA_NTERM) + 1.0)
    # C端负电
    charge -= 1.0 / (10 ** (PKA_CTERM - pH) + 1.0)
    for aa in seq:
        pka = PKA_SIDECHAIN.get(aa)
        if pka is None:
            continue
        if aa in POSITIVE_AAS:
            charge += 1.0 / (10 ** (pH - pka) + 1.0)
        elif aa in NEGATIVE_AAS:
            charge -= 1.0 / (10 ** (pka - pH) + 1.0)
        elif aa in ("C", "Y"):
            charge -= 1.0 / (10 ** (pka - pH) + 1.0)
    return charge


def calc_KRH_ratio(seq: str) -> float:
    if len(seq) == 0:
        return 0.0
    return sum(1 for aa in seq if aa in POSITIVE_AAS) / len(seq)


def calc_positive_density(seq: str) -> float:
    """正电荷密度 = 净电荷 / 长度（归一化）"""
    L = len(seq)
    if L == 0:
        return 0.0
    return max(0.0, calc_net_charge(seq)) / L


def calc_GRAVY(seq: str) -> float:
    if len(seq) == 0:
        return 0.0
    return sum(KD_HYDROPHOBICITY.get(aa, 0.0) for aa in seq) / len(seq)


def calc_hydrophobic_ratio(seq: str) -> float:
    if len(seq) == 0:
        return 0.0
    return sum(1 for aa in seq if aa in HYDROPHOBIC_AAS) / len(seq)


def calc_aliphatic_index(seq: str) -> float:
    """脂肪族指数 = X(Ala) + a*X(Val) + b*(X(Ile)+X(Leu))"""
    L = len(seq)
    if L == 0:
        return 0.0
    xA = seq.count("A") / L
    xV = seq.count("V") / L
    xIL = (seq.count("I") + seq.count("L")) / L
    # 经验系数 a=2.9, b=3.9
    return xA * 100 + 2.9 * xV * 100 + 3.9 * xIL * 100


def calc_aromatic_ratio(seq: str) -> float:
    if len(seq) == 0:
        return 0.0
    return sum(1 for aa in seq if aa in AROMATIC_AAS) / len(seq)


def calc_helix_propensity(seq: str) -> float:
    if len(seq) == 0:
        return 0.0
    return sum(HELIX_PROPENSITY.get(aa, 1.0) for aa in seq) / len(seq)


def calc_turn_propensity(seq: str) -> float:
    if len(seq) == 0:
        return 0.0
    return sum(TURN_PROPENSITY.get(aa, 1.0) for aa in seq) / len(seq)


def calc_instability_index(seq: str) -> float:
    """基于400种二肽频率的不稳定性指数（简化版）"""
    L = len(seq)
    if L < 2:
        return 0.0
    total = 0.0
    for i in range(L - 1):
        a1, a2 = seq[i], seq[i + 1]
        d1 = II_PARAMS.get(a1, (-0.4, -0.7))
        d2 = II_PARAMS.get(a2, (-0.4, -0.7))
        total += d1[0] * d2[1]
    return (10.0 / L) * total


def calc_Boman_index(seq: str) -> float:
    if len(seq) == 0:
        return 0.0
    return sum(BOMAN_INDEX.get(aa, 0.0) for aa in seq) / len(seq)


# 无序倾向参数（基于IUPred风格的简化估计，高值=更倾向无序）
DISORDER_PROPENSITY = {
    "A": 0.26, "R": 0.45, "N": 0.50, "D": 0.50, "C": 0.12,
    "Q": 0.40, "E": 0.50, "G": 0.40, "H": 0.30, "I": 0.12,
    "L": 0.20, "K": 0.45, "M": 0.22, "F": 0.10, "P": 0.60,
    "S": 0.40, "T": 0.35, "W": 0.10, "Y": 0.25, "V": 0.15,
}


def calc_disorder_tendency(seq: str) -> float:
    """
    序列无序倾向（0-1）：高值表示更柔性的无序区域
    辅助判断是否具有柔性结合片段潜力
    """
    if len(seq) == 0:
        return 0.0
    return sum(DISORDER_PROPENSITY.get(aa, 0.3) for aa in seq) / len(seq)


def calc_hydrophobic_moment(seq: str, window: int = 11) -> float:
    """
    计算螺旋疏水矩（两亲性），简化版
    muH = sqrt(Sum(Hn*cos(delta*n))^2 + Sum(Hn*sin(delta*n))^2) / N
    delta = 100度（完美α-螺旋）
    """
    L = len(seq)
    if L < 5:
        return 0.0
    delta = np.deg2rad(100.0)
    w = min(window, L)
    # 用滑窗平均
    moments = []
    for start in range(0, L - w + 1):
        s = 0.0
        c = 0.0
        for i in range(w):
            aa = seq[start + i]
            h = HYDROPHOBIC_EISENBERG.get(aa, 0.0)
            angle = delta * i
            s += h * np.sin(angle)
            c += h * np.cos(angle)
        mu = np.sqrt(s * s + c * c) / w
        moments.append(mu)
    if not moments:
        return 0.0
    return float(np.mean(moments))


def calc_repeat_ratio(seq: str) -> float:
    """
    重复片段比例：统计长度>=3的重复二肽或三肽频率占比
    """
    L = len(seq)
    if L < 6:
        return 0.0
    k3_count = {}
    for i in range(L - 2):
        kmer = seq[i:i + 3]
        k3_count[kmer] = k3_count.get(kmer, 0) + 1
    repeat_count = sum(c - 1 for c in k3_count.values() if c > 1)
    return min(1.0, repeat_count * 3 / L)


# ML 模型加载（延迟加载，首次调用时初始化）
import pickle as _pickle

_ML_MODELS = {}
_ML_LOADED = False


def _load_ml_models():
    """首次调用时加载ML预测模型"""
    global _ML_MODELS, _ML_LOADED
    if _ML_LOADED:
        return
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")

    for name in ["hemolysis_regressor", "hemolysis_classifier", "activity_classifier"]:
        path = os.path.join(model_dir, f"{name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                _ML_MODELS[name] = _pickle.load(f)

    _ML_LOADED = True


def _extract_feature_vector(seq: str) -> np.ndarray:
    """提取ML模型所需的17维基础特征向量（不含依赖风险预测的特征，与训练一致）"""
    L = len(seq)
    features = {
        "length": float(L),
        "molecular_weight": calc_molecular_weight(seq),
        "net_charge": calc_net_charge(seq),
        "KRH_ratio": calc_KRH_ratio(seq),
        "positive_density": calc_positive_density(seq),
        "GRAVY": calc_GRAVY(seq),
        "hydrophobic_ratio": calc_hydrophobic_ratio(seq),
        "aliphatic_index": calc_aliphatic_index(seq),
        "hydrophobic_moment": calc_hydrophobic_moment(seq),
        "aromatic_ratio": calc_aromatic_ratio(seq),
        "helix_propensity": calc_helix_propensity(seq),
        "turn_propensity": calc_turn_propensity(seq),
        "disorder_tendency": calc_disorder_tendency(seq),
        "instability_index": calc_instability_index(seq),
        "Boman_index": calc_Boman_index(seq),
        "repeat_ratio": calc_repeat_ratio(seq),
        "aggregation_propensity": calc_aggregation_propensity(seq),
    }
    # 注意：不含 hemolysis_risk, toxicity_risk, gram_negative_active（与训练时的EXCLUDE_COLS一致）
    exclude = {"hemolysis_risk", "toxicity_risk", "gram_negative_active"}
    feature_cols = [n for n in DESCRIPTOR_NAMES if n not in exclude]
    vec = np.array([features.get(n, 0.0) for n in feature_cols], dtype=np.float64)
    return vec.reshape(1, -1)


# 风险预测（ML 模型 + 规则回退）
def predict_hemolysis_risk_rule(seq: str) -> float:
    """纯规则公式溶血风险预测（用于ML模型对比基准）"""
    L = len(seq)
    if L == 0:
        return 0.5
    net_charge = calc_net_charge(seq)
    gravy = calc_GRAVY(seq)
    hratio = calc_hydrophobic_ratio(seq)
    arom = calc_aromatic_ratio(seq)
    score = 0.0
    score += min(1.0, max(0.0, (gravy + 0.5) / 2.0)) * 0.35
    score += min(1.0, max(0.0, (net_charge - 3) / 8.0)) * 0.15
    score += min(1.0, max(0.0, (hratio - 0.55) / 0.3)) * 0.25
    score += min(1.0, max(0.0, (arom - 0.15) / 0.2)) * 0.25
    return float(max(0.0, min(1.0, score)))


def predict_hemolysis_risk(seq: str) -> float:
    """
    溶血风险预测：优先使用ML分类器（AUC=0.9122），规则公式作为回退
    HC50≤50 µM → 高风险（输出1.0），HC50≥200 → 低风险（输出0.0）
    """
    L = len(seq)
    if L == 0:
        return 0.5

    try:
        _load_ml_models()
        if "hemolysis_classifier" in _ML_MODELS:
            bundle = _ML_MODELS["hemolysis_classifier"]
            model = bundle["model"]
            scaler = bundle["scaler"]
            feat = _extract_feature_vector(seq)
            feat_s = scaler.transform(feat)
            prob = model.predict_proba(feat_s)[0, 1]
            return float(max(0.0, min(1.0, prob)))
    except Exception:
        pass

    return predict_hemolysis_risk_rule(seq)


def predict_hemolysis_hc50(seq: str) -> float:
    """
    预测HC50（µM），基于ML回归模型
    返回log10尺度的预测值，可通过10**val转换为µM
    """
    L = len(seq)
    if L == 0:
        return 2.0  # 默认100µM
    try:
        _load_ml_models()
        if "hemolysis_regressor" in _ML_MODELS:
            bundle = _ML_MODELS["hemolysis_regressor"]
            model = bundle["model"]
            scaler = bundle["scaler"]
            feat = _extract_feature_vector(seq)
            feat_s = scaler.transform(feat)
            log_hc50 = model.predict(feat_s)[0]
            return float(max(-2.0, min(5.0, log_hc50)))
    except Exception:
        pass
    return 2.0


def predict_toxicity_risk(seq: str) -> float:
    """
    细胞毒性风险预测：
    无独立ML毒性模型（DBAASP无细胞毒性实验数据），
    用ML溶血风险作为主要输入（溶血与细胞毒性高度相关，r>0.7），
    结合聚集倾向和不稳定指数综合评估
    """
    L = len(seq)
    if L == 0:
        return 0.5

    hemol = predict_hemolysis_risk(seq)
    agg = calc_aggregation_propensity(seq)
    instab = calc_instability_index(seq)
    score = 0.4 * hemol + 0.35 * agg + 0.25 * min(1.0, max(0.0, (instab - 20) / 60))
    return float(max(0.0, min(1.0, score)))


def predict_activity_score(seq: str) -> float:
    """
    革兰氏阴性菌活性预测：使用已保存的ML分类器；AUC需结合当前评估划分读取，不能硬编码
    MIC≤16 µg/mL为Active，MIC≥64为Inactive
    返回活性概率（0-1）
    """
    L = len(seq)
    if L == 0:
        return 0.5
    try:
        _load_ml_models()
        if "activity_classifier" in _ML_MODELS:
            bundle = _ML_MODELS["activity_classifier"]
            model = bundle["model"]
            scaler = bundle["scaler"]
            feat = _extract_feature_vector(seq)
            feat_s = scaler.transform(feat)
            prob = model.predict_proba(feat_s)[0, 1]
            return float(max(0.0, min(1.0, prob)))
    except Exception:
        pass

    # 回退：规则公式
    charge = calc_net_charge(seq)
    gravy = calc_GRAVY(seq)
    krh = calc_KRH_ratio(seq)
    score = 0.0
    score += min(1.0, max(0.0, (charge - 1) / 6.0)) * 0.4
    score += min(1.0, max(0.0, (1.0 - abs(gravy)) / 1.5)) * 0.3
    score += min(1.0, max(0.0, krh / 0.5)) * 0.3
    return float(max(0.0, min(1.0, score)))


def predict_activity_probability(seq: str) -> float:
    """Return the continuous G-negative activity probability used for ranking."""
    return predict_activity_score(seq)


def calc_aggregation_propensity(seq: str) -> float:
    """聚集倾向：高疏水+长序列+重复片段 容易聚集"""
    L = len(seq)
    if L == 0:
        return 0.0
    hratio = calc_hydrophobic_ratio(seq)
    repeat = calc_repeat_ratio(seq)
    score = 0.4 * min(1.0, max(0.0, (hratio - 0.5) / 0.3))
    score += 0.3 * min(1.0, max(0.0, (L - 20) / 15))
    score += 0.3 * repeat
    return float(max(0.0, min(1.0, score)))


def calc_synthesis_feasibility(seq: str) -> float:
    """Heuristic 0-1 synthesis score; lower length, instability and aggregation are favored."""
    if not seq:
        return 0.0
    length_score = 1.0 if 8 <= len(seq) <= 25 else max(0.0, 1.0 - abs(len(seq) - 16) / 20)
    instability_score = 1.0 - min(1.0, max(0.0, (calc_instability_index(seq) - 10) / 60))
    aggregation_score = 1.0 - calc_aggregation_propensity(seq)
    cysteine_penalty = min(0.3, seq.count("C") * 0.05)
    return float(max(0.0, min(1.0,
        0.45 * length_score + 0.30 * instability_score +
        0.25 * aggregation_score - cysteine_penalty)))


# 主入口：批量计算
def compute_descriptors_for_sequence(seq: str, label: int = None) -> Dict[str, float]:
    """对单条序列计算全部描述符"""
    d = {
        "length": calc_length(seq),
        "molecular_weight": calc_molecular_weight(seq),
        "net_charge": calc_net_charge(seq),
        "KRH_ratio": calc_KRH_ratio(seq),
        "positive_density": calc_positive_density(seq),
        "GRAVY": calc_GRAVY(seq),
        "hydrophobic_ratio": calc_hydrophobic_ratio(seq),
        "aliphatic_index": calc_aliphatic_index(seq),
        "hydrophobic_moment": calc_hydrophobic_moment(seq),
        "aromatic_ratio": calc_aromatic_ratio(seq),
        "helix_propensity": calc_helix_propensity(seq),
        "turn_propensity": calc_turn_propensity(seq),
        "disorder_tendency": calc_disorder_tendency(seq),
        "instability_index": calc_instability_index(seq),
        "Boman_index": calc_Boman_index(seq),
        "repeat_ratio": calc_repeat_ratio(seq),
        "hemolysis_risk": predict_hemolysis_risk(seq),
        "toxicity_risk": predict_toxicity_risk(seq),
        "aggregation_propensity": calc_aggregation_propensity(seq),
        "synthesis_feasibility": calc_synthesis_feasibility(seq),
        "gram_negative_active": float(label) if label is not None else (
            # 使用ML活性预测器，阈值0.5；具体性能见当前评估结果
            1.0 if predict_activity_score(seq) >= 0.5 else 0.0
        ),
    }
    return d


def compute_descriptor_array(seq: str, label: int = None) -> np.ndarray:
    """返回与DESCRIPTOR_NAMES对齐的numpy数组"""
    d = compute_descriptors_for_sequence(seq, label=label)
    arr = np.array([d[name] for name in DESCRIPTOR_NAMES], dtype=np.float32)
    return arr


def compute_descriptors_dataframe(df: pd.DataFrame,
                                  seq_col: str = "sequence",
                                  label_col: str = "label") -> pd.DataFrame:
    """
    批量计算描述符，返回带描述符列的dataframe
    如果df中有真实实验数据列（hemo_activity_value, best_gn_mic等），优先使用
    """
    rows = []
    for _, row in df.iterrows():
        seq = row[seq_col]
        label = row[label_col] if label_col in df.columns else None
        des = compute_descriptors_for_sequence(seq, label=label)
        merged = {**row.to_dict(), **des}
        # 如果有真实溶血实验数据，覆盖规则预测值
        if "hemo_activity_value" in df.columns and pd.notna(row.get("hemo_activity_value")):
            real_hemo = float(row["hemo_activity_value"])
            # DBAASP溶血活性值通常是HC50(μM)，值越大表示越安全（需要更高浓度才溶血）
            # 归一化到0-1风险：低HC50=高风险，高HC50=低风险
            # 经验阈值：HC50 < 50 μM = 高风险，> 500 μM = 低风险
            hemo_risk = max(0.0, min(1.0, 1.0 - (real_hemo - 10) / 490.0))
            merged["hemolysis_risk"] = hemo_risk
            merged["has_real_hemo_data"] = 1.0
        else:
            merged["has_real_hemo_data"] = 0.0
        # 如果有真实MIC数据，记录
        if "best_gn_mic" in df.columns and pd.notna(row.get("best_gn_mic")):
            merged["real_gn_mic"] = float(row["best_gn_mic"])
        rows.append(merged)
    return pd.DataFrame(rows)


def save_descriptors(train_df, val_df, test_df):
    """保存带描述符的数据集"""
    from config import DATA_DIR
    train_des = compute_descriptors_dataframe(train_df)
    val_des = compute_descriptors_dataframe(val_df)
    test_des = compute_descriptors_dataframe(test_df)

    train_des.to_csv(os.path.join(DATA_DIR, "train_with_descriptors.csv"), index=False, encoding="utf-8")
    val_des.to_csv(os.path.join(DATA_DIR, "val_with_descriptors.csv"), index=False, encoding="utf-8")
    test_des.to_csv(os.path.join(DATA_DIR, "test_with_descriptors.csv"), index=False, encoding="utf-8")
    print("[DESCRIPTOR] 描述符数据集已保存到", DATA_DIR)
    return train_des, val_des, test_des


if __name__ == "__main__":
    # 简单测试
    sample_seq = "KLKLLKLAAKK"
    des = compute_descriptors_for_sequence(sample_seq, label=1)
    print("[TEST] 示例肽:", sample_seq)
    for k, v in des.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
