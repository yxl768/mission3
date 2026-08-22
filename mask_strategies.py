"""
Mask策略模块
实现多种遮盖策略，模拟不同的AMP研发场景：
- random mask: 一般序列上下文学习
- contiguous mask: 局部片段替换
- motif_preserving mask: 核心motif保留、可变位点遮盖（lead优化）
- property_guided mask: 优先遮盖风险位点（优化方向引导）
"""
import os
import sys
import numpy as np
from typing import List, Tuple, Optional

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
from config import (
    MASK_STRATEGIES, MOTIF_RESIDUES, CONTIGUOUS_MASK_LEN_RANGE, MASK_RATIOS,
    AA_TO_IDX, AMINO_ACIDS, DESCRIPTOR_NAMES
)
from descriptor_calculator import (
    calc_GRAVY, calc_hydrophobic_ratio, calc_net_charge,
    predict_hemolysis_risk, predict_toxicity_risk,
)

MASK_TOKEN = "[MASK]"


def _random_mask_indices(seq_len: int, mask_ratio: float, rng: np.random.RandomState) -> List[int]:
    """生成随机mask索引（保证至少1个）"""
    n_mask = max(1, int(round(seq_len * mask_ratio)))
    n_mask = min(n_mask, seq_len)
    indices = rng.choice(seq_len, size=n_mask, replace=False)
    return sorted(indices.tolist())


def apply_random_mask(sequence: str, mask_ratio: float, rng: np.random.RandomState) -> Tuple[str, List[int]]:
    """随机遮盖 15%-40% 氨基酸"""
    seq_list = list(sequence)
    seq_len = len(seq_list)
    if seq_len == 0:
        return sequence, []
    indices = _random_mask_indices(seq_len, mask_ratio, rng)
    for idx in indices:
        seq_list[idx] = MASK_TOKEN
    return "".join(seq_list), indices


def apply_contiguous_mask(sequence: str, mask_ratio: float, rng: np.random.RandomState) -> Tuple[str, List[int]]:
    """
    连续遮盖3-8个氨基酸（局部片段替换场景）
    连续长度优先落在CONTIGUOUS_MASK_LEN_RANGE，实际也受mask_ratio约束
    """
    seq_list = list(sequence)
    seq_len = len(seq_list)
    if seq_len == 0:
        return sequence, []

    # 连续mask的目标长度
    target_total = max(1, int(round(seq_len * mask_ratio)))
    max_clen = min(CONTIGUOUS_MASK_LEN_RANGE[1], seq_len, target_total + 2)
    min_clen = min(CONTIGUOUS_MASK_LEN_RANGE[0], seq_len, max_clen)

    # 分若干段连续mask（通常1-2段）
    remaining = target_total
    indices = []
    available = np.ones(seq_len, dtype=bool)
    n_segments = max(1, min(3, target_total // min_clen if min_clen > 0 else 1))

    for seg in range(n_segments):
        if remaining <= 0 or not available.any():
            break
        seg_len = rng.randint(min_clen, max_clen + 1)
        seg_len = min(seg_len, remaining, available.sum())
        if seg_len <= 0:
            continue
        # 在可用位置中找连续seg_len个位置
        cand_starts = []
        for s in range(seq_len - seg_len + 1):
            if available[s:s + seg_len].all():
                cand_starts.append(s)
        if not cand_starts:
            seg_len = 1
            cand_starts = np.where(available)[0].tolist()
            if not cand_starts:
                break
        start = cand_starts[rng.randint(len(cand_starts))]
        seg_indices = list(range(start, start + seg_len))
        indices.extend(seg_indices)
        available[seg_indices] = False
        remaining -= seg_len

    indices = sorted(set(indices))
    for idx in indices:
        seq_list[idx] = MASK_TOKEN
    return "".join(seq_list), indices


def apply_motif_preserving_mask(sequence: str, mask_ratio: float, rng: np.random.RandomState) -> Tuple[str, List[int]]:
    """
    保留核心motif残基（K/R/F/W/Y，已知抗菌活性核心）
    优先遮盖周围的可变位点
    """
    seq_list = list(sequence)
    seq_len = len(seq_list)
    if seq_len == 0:
        return sequence, []

    # 区分 motif位点 vs 可变位点
    motif_indices = [i for i, aa in enumerate(seq_list) if aa in MOTIF_RESIDUES]
    variable_indices = [i for i, aa in enumerate(seq_list) if aa not in MOTIF_RESIDUES]

    target_total = max(1, int(round(seq_len * mask_ratio)))

    # 优先遮盖可变位点（90%），少量允许遮盖motif边缘（10%）
    n_var_mask = min(len(variable_indices), int(target_total * 0.95))
    n_motif_mask = min(len(motif_indices), target_total - n_var_mask)

    mask_indices = []
    if n_var_mask > 0 and variable_indices:
        picks = rng.choice(variable_indices, size=n_var_mask, replace=False).tolist()
        mask_indices.extend(picks)
    if n_motif_mask > 0 and motif_indices:
        picks = rng.choice(motif_indices, size=n_motif_mask, replace=False).tolist()
        mask_indices.extend(picks)

    # 如果不足，补全
    while len(mask_indices) < target_total and len(mask_indices) < seq_len:
        extra = [i for i in range(seq_len) if i not in mask_indices]
        if not extra:
            break
        mask_indices.append(extra[rng.randint(len(extra))])

    mask_indices = sorted(set(mask_indices))
    for idx in mask_indices:
        seq_list[idx] = MASK_TOKEN
    return "".join(seq_list), mask_indices


def apply_property_guided_mask(sequence: str, mask_ratio: float, rng: np.random.RandomState) -> Tuple[str, List[int]]:
    """
    进阶策略：优先遮盖风险位点
    - 高疏水的位置（可能导致溶血）
    - 高溶血风险相关的位置
    - 低电荷的区域（需要增加正电荷的位点）
    """
    seq_list = list(sequence)
    seq_len = len(seq_list)
    if seq_len == 0:
        return sequence, []

    target_total = max(1, int(round(seq_len * mask_ratio)))

    # 逐位点计算“风险/可优化”得分
    risk_scores = np.zeros(seq_len, dtype=np.float32)
    HYDRO = set("AILMFWV")
    AROM = set("FWY")
    POS = set("KRH")
    NEG = set("DE")

    # 计算全局参考
    overall_charge = calc_net_charge(sequence)
    overall_hemo = predict_hemolysis_risk(sequence)

    for i, aa in enumerate(seq_list):
        s = 0.0
        # 疏水位点（如果全局已经偏疏水，则提高这些位点的遮盖优先度）
        if aa in HYDRO:
            s += 0.4 + (calc_GRAVY(sequence) - 0.0) * 0.3
        # 芳香位点（溶血相关）
        if aa in AROM:
            s += 0.3 + overall_hemo * 0.2
        # 负电荷位点
        if aa in NEG:
            s += 0.25
        # 全局电荷不足，且该位点是中性，可以换成正电荷
        if overall_charge < 5 and aa not in POS and aa not in NEG:
            s += 0.25
        # 重复位点（如果是重复残基）
        if i > 0 and seq_list[i] == seq_list[i - 1]:
            s += 0.15
        risk_scores[i] = max(0.05, s)

    # 归一化概率
    p = risk_scores / risk_scores.sum()
    # 采样（带点随机性，不完全贪婪）
    try:
        picks = rng.choice(seq_len, size=min(target_total, seq_len), replace=False, p=p)
    except Exception:
        picks = rng.choice(seq_len, size=min(target_total, seq_len), replace=False)
    indices = sorted(picks.tolist())

    for idx in indices:
        seq_list[idx] = MASK_TOKEN
    return "".join(seq_list), indices


# ===================== 调度器 =====================
STRATEGY_FN = {
    "random": apply_random_mask,
    "contiguous": apply_contiguous_mask,
    "motif_preserving": apply_motif_preserving_mask,
    "property_guided": apply_property_guided_mask,
}


def apply_mask(sequence: str,
               strategy: str,
               mask_ratio: Optional[float] = None,
               seed: int = None) -> Tuple[str, List[int]]:
    """
    统一调度：根据策略执行mask
    返回 masked_sequence, mask_indices
    """
    if strategy not in STRATEGY_FN:
        raise ValueError(f"Unknown mask strategy: {strategy}, valid: {list(STRATEGY_FN.keys())}")
    if mask_ratio is None:
        mask_ratio = MASK_RATIOS[np.random.randint(len(MASK_RATIOS))]
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    return STRATEGY_FN[strategy](sequence, mask_ratio, rng)


def sample_strategy(rng: np.random.RandomState = None) -> str:
    """训练时随机选择策略（让模型学习所有场景）"""
    all_strategies = list(STRATEGY_FN.keys())
    if rng is None:
        return all_strategies[np.random.randint(len(all_strategies))]
    return all_strategies[rng.randint(len(all_strategies))]


def sample_mask_ratio(rng: np.random.RandomState = None) -> float:
    """随机采样一个mask ratio"""
    if rng is None:
        return MASK_RATIOS[np.random.randint(len(MASK_RATIOS))]
    return MASK_RATIOS[rng.randint(len(MASK_RATIOS))]


# ===================== 用于推理阶段的 lead peptide mask =====================
def mask_lead_peptide_for_optimization(sequence: str,
                                       preserve_regions: List[Tuple[int, int]] = None,
                                       mask_ratio: float = 0.3,
                                       seed: int = 42) -> Tuple[str, List[int]]:
    """
    推理阶段：对lead peptide构造mask，用于引导条件优化生成
    preserve_regions: [(start,end), ...] 区间索引（含头不含尾）保留不mask
    """
    rng = np.random.RandomState(seed)
    seq_list = list(sequence)
    seq_len = len(seq_list)
    target = max(1, int(round(seq_len * mask_ratio)))

    available = np.ones(seq_len, dtype=bool)
    if preserve_regions:
        for s, e in preserve_regions:
            s = max(0, min(seq_len, s))
            e = max(0, min(seq_len, e))
            available[s:e] = False

    candidates = np.where(available)[0].tolist()
    if len(candidates) == 0:
        return sequence, []

    n = min(target, len(candidates))
    picks = rng.choice(candidates, size=n, replace=False).tolist()
    picks = sorted(picks)
    for idx in picks:
        seq_list[idx] = MASK_TOKEN
    return "".join(seq_list), picks


if __name__ == "__main__":
    # 测试
    sample = "KLKLLKLAAKKWWR"
    print("[TEST] 原始序列:", sample)
    for strat in list(STRATEGY_FN.keys()):
        masked, indices = apply_mask(sample, strat, mask_ratio=0.3, seed=42)
        print(f"  {strat:20s}: {masked}  mask位置={indices}")
