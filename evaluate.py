"""
评估与可视化模块
覆盖任务要求的所有评估指标：
1. 重构能力：masked token accuracy / full seq recovery / edit distance / mask ratio曲线
2. 条件利用能力：改变目标条件后生成性质变化
3. 生成质量：valid rate / unique rate / novel rate / nearest-neighbor similarity
4. 抗菌相关性质：长度、电荷、疏水等分布是否接近已知活性AMP
5. 综合优化效果：相对lead的改善
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
from config import (
    RESULTS_DIR, PLOTS_DIR, DATA_DIR, MODEL_DIR,
    DESCRIPTOR_NAMES, EVAL_NEAREST_NEIGHBOR_K, AMINO_ACIDS
)

# 绘图（在无GUI环境可能agg backend）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import roc_auc_score

sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


# ===================== 1. 基础生成质量指标 =====================
def _get_sequence_series(df: pd.DataFrame) -> pd.Series:
    """兼容 raw generation 与 final export 两种列名"""
    if "generated_sequence" in df.columns:
        return df["generated_sequence"]
    if "sequence" in df.columns:
        return df["sequence"]
    return pd.Series(["" for _ in range(len(df))], index=df.index, dtype=object)


def _get_metric_series(df: pd.DataFrame, metric: str, prefix: str = "gen_") -> pd.Series:
    """兼容 gen_<metric> 与 <metric> 两种列名"""
    col = f"{prefix}{metric}"
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    if metric in df.columns:
        return pd.to_numeric(df[metric], errors="coerce")
    return pd.Series([np.nan] * len(df), index=df.index, dtype=float)


def _safe_float(value) -> float:
    """谨慎把字符串/缺失值转换为浮点数，避免报告格式化崩溃"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _format_float(value, default: str = "N/A") -> str:
    """安全地格式化数值字段"""
    value = _safe_float(value)
    if pd.isna(value):
        return default
    return f"{value:.3f}"


def _is_valid_peptide(seq: str) -> bool:
    """只保留标准20种氨基酸、长度合理"""
    if not isinstance(seq, str) or len(seq) < 8 or len(seq) > 40:
        return False
    return all(c in set(AMINO_ACIDS) for c in seq)


def _compute_nearest_neighbor_similarity(candidates: List[str], train_seqs: List[str]) -> np.ndarray:
    """
    最近邻相似度（归一化最长公共子序列/编辑距离）
    对每条候选，找与训练集最近的K条，取平均相似度
    """
    def _norm_edit(a, b):
        L = max(len(a), len(b), 1)
        d = abs(len(a) - len(b))
        for i in range(min(len(a), len(b))):
            if a[i] != b[i]:
                d += 1
        return 1.0 - d / L

    scores = np.zeros(len(candidates), dtype=np.float32)
    if not train_seqs:
        return scores
    for i, cand in enumerate(candidates):
        sims = []
        for ts in train_seqs:
            sims.append(_norm_edit(cand, ts))
        sims = np.sort(sims)[-EVAL_NEAREST_NEIGHBOR_K:]
        scores[i] = float(np.mean(sims))
    return scores


def compute_generation_quality_metrics(candidates_df: pd.DataFrame,
                                      train_df: pd.DataFrame) -> Dict[str, float]:
    """
    计算生成质量：valid rate, unique rate, novel rate, avg NN similarity
    返回字典
    """
    if candidates_df.empty:
        return {"valid_rate": 0.0, "unique_rate": 0.0, "novel_rate": 0.0,
                "avg_nn_similarity": 0.0, "total": 0}
    gen_seqs = _get_sequence_series(candidates_df).tolist()
    total = len(gen_seqs)
    valid_mask = [_is_valid_peptide(s) for s in gen_seqs]
    valid_seqs = [s for s, v in zip(gen_seqs, valid_mask) if v]
    valid_rate = sum(valid_mask) / max(1, total)

    # Calculate before any downstream deduplication: this is the true sampling rate.
    unique_rate = len(set(valid_seqs)) / max(1, len(valid_seqs))

    # novel rate：不在训练集中
    train_set = set(train_df["sequence"].tolist()) if train_df is not None else set()
    n_novel = sum(1 for s in valid_seqs if s not in train_set)
    novel_rate = n_novel / max(1, len(valid_seqs))

    # nearest neighbor similarity（采样部分，避免O(N*M)过大）
    sample_n = min(200, len(valid_seqs))
    if sample_n > 0 and train_df is not None:
        idx = np.linspace(0, len(valid_seqs) - 1, sample_n).astype(int)
        sample_cands = [valid_seqs[i] for i in idx]
        train_sample = train_df["sequence"].tolist()
        if len(train_sample) > 1000:
            train_sample = list(pd.Series(train_sample).sample(1000, random_state=42))
        nn_sim = _compute_nearest_neighbor_similarity(sample_cands, train_sample)
        avg_nn = float(np.mean(nn_sim))
    else:
        avg_nn = 0.0

    return {
        "total": total,
        "valid_rate": valid_rate,
        "unique_rate": unique_rate,
        "novel_rate": novel_rate,
        "avg_nn_similarity": avg_nn,
    }


def analyze_mic_stratification(train_df: pd.DataFrame) -> Dict[str, object]:
    """Evaluate continuous activity probability against MIC threshold strata."""
    raw_path = os.path.join(DATA_DIR, "dbaasp_raw.csv")
    if not os.path.exists(raw_path):
        return {"available": False, "reason": "dbaasp_raw.csv not found"}
    raw = pd.read_csv(raw_path)
    if "best_gn_mic" not in raw.columns:
        return {"available": False, "reason": "best_gn_mic column not found"}
    raw["best_gn_mic"] = pd.to_numeric(raw["best_gn_mic"], errors="coerce")
    raw = raw.dropna(subset=["best_gn_mic", "sequence"])
    active = raw[raw["best_gn_mic"] <= 16].copy()
    inactive = raw[raw["best_gn_mic"] >= 64].copy()
    if active.empty or inactive.empty:
        return {"available": False, "reason": "insufficient MIC strata"}
    from descriptor_calculator import predict_activity_probability
    active_prob = np.array([predict_activity_probability(str(s)) for s in active["sequence"]])
    inactive_prob = np.array([predict_activity_probability(str(s)) for s in inactive["sequence"]])
    labels = np.r_[np.ones(len(active_prob)), np.zeros(len(inactive_prob))]
    probs = np.r_[active_prob, inactive_prob]
    return {
        "available": True,
        "active_threshold_ug_ml": 16,
        "inactive_threshold_ug_ml": 64,
        "active_n": int(len(active_prob)),
        "inactive_n": int(len(inactive_prob)),
        "active_probability_mean": float(active_prob.mean()),
        "inactive_probability_mean": float(inactive_prob.mean()),
        "roc_auc": float(roc_auc_score(labels, probs)),
    }


# ===================== 2. 重构评估 =====================
def analyze_reconstruction(recon_df: pd.DataFrame) -> pd.DataFrame:
    """
    对重构结果做策略级聚合，返回每策略的表现
    """
    if recon_df is None or recon_df.empty:
        return pd.DataFrame()
    g = recon_df.groupby("strategy").agg(
        n=("mask_acc", "size"),
        avg_mask_acc=("mask_acc", "mean"),
        median_mask_acc=("mask_acc", "median"),
        seq_recovery_rate=("full_recovery", "mean"),
        avg_edit_distance=("edit_distance", "mean"),
        avg_n_mask=("n_mask", "mean"),
    ).reset_index()
    return g


# ===================== 2.5 不同mask ratio下的性能曲线 =====================
def analyze_mask_ratio_performance(recon_df: pd.DataFrame) -> pd.DataFrame:
    """
    按mask_ratio分组，计算每个ratio下的mask_acc/seq_recovery/edit_distance
    用于绘制"不同mask ratio下的性能曲线"
    """
    if recon_df is None or recon_df.empty or "mask_ratio" not in recon_df.columns:
        return pd.DataFrame()
    g = recon_df.groupby("mask_ratio").agg(
        n=("mask_acc", "size"),
        avg_mask_acc=("mask_acc", "mean"),
        median_mask_acc=("mask_acc", "median"),
        seq_recovery_rate=("full_recovery", "mean"),
        avg_edit_distance=("edit_distance", "mean"),
        avg_n_mask=("n_mask", "mean"),
    ).reset_index()
    return g


def plot_mask_ratio_curve(recon_df: pd.DataFrame):
    """
    绘制不同mask ratio下的性能曲线（任务要求：不同mask ratio下的性能曲线）
    包含两条子图：(a) mask_acc vs ratio, (b) seq_recovery & edit_distance vs ratio
    并按策略分组展示
    """
    if recon_df is None or recon_df.empty or "mask_ratio" not in recon_df.columns:
        return

    ratios = sorted(recon_df["mask_ratio"].unique())

    # 按ratio汇总（全体策略平均）
    summary = analyze_mask_ratio_performance(recon_df)

    # 按策略分组
    strategies = sorted(recon_df["strategy"].unique()) if "strategy" in recon_df.columns else ["all"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = sns.color_palette("Set2", len(strategies))

    # 子图1: mask_acc vs ratio（按策略）
    ax = axes[0]
    for i, strat in enumerate(strategies):
        sub = recon_df[recon_df["strategy"] == strat] if "strategy" in recon_df.columns else recon_df
        g = sub.groupby("mask_ratio")["mask_acc"].mean().reset_index()
        ax.plot(g["mask_ratio"], g["mask_acc"], marker="o", label=strat, color=colors[i])
    ax.set_xlabel("Mask Ratio"); ax.set_ylabel("Masked Token Accuracy")
    ax.set_title("Mask Token Acc vs Mask Ratio (by Strategy)")
    ax.legend(fontsize=8)

    # 子图2: seq_recovery vs ratio（按策略）
    ax = axes[1]
    for i, strat in enumerate(strategies):
        sub = recon_df[recon_df["strategy"] == strat] if "strategy" in recon_df.columns else recon_df
        g = sub.groupby("mask_ratio")["full_recovery"].mean().reset_index()
        ax.plot(g["mask_ratio"], g["full_recovery"], marker="s", label=strat, color=colors[i])
    ax.set_xlabel("Mask Ratio"); ax.set_ylabel("Full Sequence Recovery Rate")
    ax.set_title("Seq Recovery vs Mask Ratio (by Strategy)")
    ax.legend(fontsize=8)

    # 子图3: edit_distance vs ratio（全体平均）
    ax = axes[2]
    ax.plot(summary["mask_ratio"], summary["avg_edit_distance"], marker="^", color="#d62728", label="Avg Edit Distance")
    ax2 = ax.twinx()
    ax2.plot(summary["mask_ratio"], summary["avg_mask_acc"], marker="o", color="#1f77b4", label="Mask Acc")
    ax.set_xlabel("Mask Ratio"); ax.set_ylabel("Edit Distance", color="#d62728")
    ax2.set_ylabel("Mask Token Accuracy", color="#1f77b4")
    ax.set_title("Edit Distance & Accuracy vs Mask Ratio (Overall)")
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle("Performance vs Mask Ratio (Task Requirement: mask ratio curve)", fontsize=13)
    fig.tight_layout()
    p = os.path.join(PLOTS_DIR, "mask_ratio_performance_curve.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {p}")


# ===================== 3. 条件利用能力评估 =====================
def analyze_condition_guidance(candidates_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    比较生成序列 vs lead序列 vs target描述符
    验证条件引导是否生效
    """
    metrics = ["net_charge", "GRAVY", "hydrophobic_ratio", "KRH_ratio",
               "hemolysis_risk", "toxicity_risk", "aromatic_ratio", "length"]
    result = {}
    for m in metrics:
        lead_v = _get_metric_series(candidates_df, f"lead_{m}", prefix="")
        gen_v = _get_metric_series(candidates_df, m, prefix="gen_")
        target_v = _get_metric_series(candidates_df, f"target_{m}", prefix="")

        l_mean = float(lead_v.mean()) if len(lead_v) > 0 else np.nan
        g_mean = float(gen_v.mean()) if len(gen_v) > 0 else np.nan
        t_mean = float(target_v.mean()) if len(target_v) > 0 else np.nan

        if np.isnan(l_mean) or np.isnan(g_mean):
            continue
        if not np.isnan(t_mean) and t_mean != l_mean:
            delta_target = t_mean - l_mean
            delta_actual = g_mean - l_mean
            guidance_ratio = 0.0 if delta_target == 0 else (delta_actual / delta_target)
        else:
            guidance_ratio = np.nan
        result[m] = {
            "lead_mean": float(l_mean),
            "gen_mean": float(g_mean),
            "target_mean": float(t_mean) if not np.isnan(t_mean) else None,
            "delta_lead_to_gen": float(g_mean - l_mean),
            "guidance_ratio": float(guidance_ratio) if not np.isnan(guidance_ratio) else None,
        }
    return result


def analyze_counterfactual_conditions(candidates_df: pd.DataFrame) -> Dict[str, Dict]:
    """Compare candidate properties across named conditions on shared scaffolds."""
    if candidates_df.empty or "condition_name" not in candidates_df.columns:
        return {}
    metrics = ["net_charge", "GRAVY", "hydrophobic_ratio", "hemolysis_risk",
               "toxicity_risk", "activity_probability", "synthesis_feasibility"]
    result = {}
    for condition, group in candidates_df.groupby("condition_name"):
        result[condition] = {"n": int(len(group))}
        for metric in metrics:
            col = f"gen_{metric}" if f"gen_{metric}" in group.columns else metric
            if col in group.columns:
                result[condition][metric] = float(pd.to_numeric(group[col], errors="coerce").mean())
    return result


# ===================== 4. 性质分布对比 vs 训练集阳性AMP =====================
def analyze_property_distribution(candidates_df: pd.DataFrame,
                                  train_pos_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    比较生成候选 vs 训练集阳性样本的性质分布相似度（均值、std、KL-like）
    """
    if candidates_df.empty:
        return {}
    metrics = ["length", "net_charge", "GRAVY", "hydrophobic_ratio",
               "KRH_ratio", "aromatic_ratio", "hemolysis_risk", "toxicity_risk"]
    result = {}
    train_pos_seqs = train_pos_df["sequence"].tolist()
    from descriptor_calculator import compute_descriptors_for_sequence
    train_pos_des_list = [compute_descriptors_for_sequence(s, label=1) for s in train_pos_seqs]

    for m in metrics:
        # 训练集阳性分布
        t_vals = np.array([d[m] for d in train_pos_des_list], dtype=np.float32)
        # 生成候选分布
        g_col = f"gen_{m}"
        if g_col in candidates_df.columns:
            g_vals = candidates_df[g_col].to_numpy(dtype=np.float32)
        elif m in candidates_df.columns:
            g_vals = candidates_df[m].to_numpy(dtype=np.float32)
        else:
            continue
        distances = {
            "ks_statistic": float(ks_2samp(t_vals, g_vals).statistic),
            "ks_pvalue": float(ks_2samp(t_vals, g_vals).pvalue),
            "wasserstein_distance": float(wasserstein_distance(t_vals, g_vals)),
        }
        result[m] = {
            "train_pos_mean": float(t_vals.mean()),
            "train_pos_std": float(t_vals.std()),
            "gen_mean": float(g_vals.mean()),
            "gen_std": float(g_vals.std()),
            "mean_abs_diff": float(abs(g_vals.mean() - t_vals.mean())),
            **distances,
        }
    return result


# ===================== 5. 综合候选肽评分与筛选 =====================
def score_and_rank_candidates(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    为每个候选计算综合得分（可解释加权），排序返回
    """
    if candidates_df.empty:
        return candidates_df

    def _score(row):
        s = 0.0
        # 1. 电荷得分：(3,10)区间
        ch = row.get("gen_net_charge", row.get("net_charge", 0))
        s += 1.2 if 4 <= ch <= 9 else (0.6 if 2 <= ch <= 10 else 0.0)
        # 2. 疏水得分：(-0.5, 0.7)
        gravy = row.get("gen_GRAVY", row.get("GRAVY", 0))
        s += 0.8 if -0.4 <= gravy <= 0.6 else (0.4 if -0.8 <= gravy <= 0.9 else 0.0)
        # 3. KRH比例
        krh = row.get("gen_KRH_ratio", row.get("KRH_ratio", 0))
        s += 1.0 if 0.25 <= krh <= 0.5 else (0.5 if 0.15 <= krh <= 0.6 else 0.0)
        # 4. 溶血风险（越低越好）
        hemo = row.get("gen_hemolysis_risk", row.get("hemolysis_risk", 0.5))
        s += (1.0 - hemo) * 1.5
        # 5. 毒性风险
        tox = row.get("gen_toxicity_risk", row.get("toxicity_risk", 0.5))
        s += (1.0 - tox) * 1.2
        # 6. Gram-negative active 预测
        activity = row.get("activity_probability", row.get("gen_gram_negative_active", 0))
        s += activity * 1.5
        # 7. 长度合适
        L = row.get("gen_length", row.get("length", 0))
        s += 0.8 if 10 <= L <= 25 else (0.4 if 8 <= L <= 30 else 0.0)
        # 8. 与lead差异适中（既不太近也不太远）
        d = row.get("edit_distance_from_lead", 0)
        s += 0.6 if 2 <= d <= 12 else (0.3 if 1 <= d <= 18 else 0.0)
        # 9. 聚集倾向低
        agg = row.get("gen_aggregation_propensity", row.get("aggregation_propensity", 0.5))
        s += (1.0 - agg) * 0.8
        synthesis = row.get("synthesis_feasibility", 0.5)
        s += synthesis * 0.8
        return round(s, 4)

    def _condition_fit(row):
        """Normalized distance to the requested mechanism descriptors."""
        pairs = [
            ("net_charge", 4.0), ("GRAVY", 1.0),
            ("hydrophobic_ratio", 0.25), ("KRH_ratio", 0.25),
            ("hemolysis_risk", 0.25), ("toxicity_risk", 0.25),
        ]
        errors = []
        for name, scale in pairs:
            gen = row.get(f"gen_{name}", row.get(name, np.nan))
            target = row.get(f"target_{name}", np.nan)
            try:
                if np.isfinite(float(gen)) and np.isfinite(float(target)):
                    errors.append(min(1.0, abs(float(gen) - float(target)) / scale))
            except (TypeError, ValueError):
                continue
        return round(1.0 - float(np.mean(errors)), 4) if errors else 0.0

    candidates_df = candidates_df.copy()
    candidates_df["condition_fit_score"] = candidates_df.apply(_condition_fit, axis=1)
    candidates_df["composite_score"] = candidates_df.apply(_score, axis=1)
    # Use condition fit as a tie-breaking/selective pressure while retaining
    # activity, safety, and scaffold-edit terms in the primary score.
    candidates_df["composite_score"] = (
        candidates_df["composite_score"] + 1.0 * candidates_df["condition_fit_score"]
    ).round(4)
    candidates_df = candidates_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return candidates_df


# ===================== 6. 可视化图表 =====================
def plot_training_curves(history: dict = None):
    """训练loss & accuracy曲线"""
    if history is None:
        hist_path = os.path.join(MODEL_DIR, "history.json")
        if not os.path.exists(hist_path):
            return
        with open(hist_path, "r", encoding="utf-8") as f:
            history = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    epochs = list(range(1, len(history["train_total"]) + 1))

    # 总loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_total"], label="Train Total", color="#1f77b4")
    ax.plot(epochs, history["val_total"], label="Val Total", color="#ff7f0e")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Total Loss (Recon + KL)")
    ax.legend()

    # recon loss
    ax = axes[0, 1]
    ax.plot(epochs, history["train_recon"], label="Train Recon", color="#1f77b4")
    ax.plot(epochs, history["val_recon"], label="Val Recon", color="#ff7f0e")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Recon Loss")
    ax.set_title("Reconstruction Loss")
    ax.legend()

    # KL loss
    ax = axes[1, 0]
    ax.plot(epochs, history["train_kl"], label="Train KL", color="#2ca02c")
    ax.plot(epochs, history["val_kl"], label="Val KL", color="#d62728")
    ax.set_xlabel("Epoch"); ax.set_ylabel("KL Divergence")
    ax.set_title("KL Loss (latent regularization)")
    ax.legend()

    # val acc
    ax = axes[1, 1]
    ax.plot(epochs, history["val_mask_acc"], label="Masked Token Acc", color="#9467bd")
    ax.plot(epochs, history["val_seq_recovery"], label="Full Seq Recovery", color="#8c564b")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Rate")
    ax.set_title("Validation Accuracy Metrics")
    ax.legend()

    fig.tight_layout()
    p = os.path.join(PLOTS_DIR, "training_curves.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {p}")


def plot_mask_strategy_comparison(recon_summary: pd.DataFrame):
    """不同mask策略的重构性能柱状图"""
    if recon_summary.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = recon_summary["strategy"]
    axes[0].bar(x, recon_summary["avg_mask_acc"], color=sns.color_palette("Set2", len(x)))
    axes[0].set_title("Masked Token Accuracy")
    axes[0].set_ylabel("Accuracy")
    for tick in axes[0].get_xticklabels():
        tick.set_rotation(25); tick.set_ha("right")

    axes[1].bar(x, recon_summary["seq_recovery_rate"], color=sns.color_palette("Set2", len(x)))
    axes[1].set_title("Full Sequence Recovery Rate")
    for tick in axes[1].get_xticklabels():
        tick.set_rotation(25); tick.set_ha("right")

    axes[2].bar(x, recon_summary["avg_edit_distance"], color=sns.color_palette("Set2", len(x)))
    axes[2].set_title("Avg Edit Distance")
    for tick in axes[2].get_xticklabels():
        tick.set_rotation(25); tick.set_ha("right")

    fig.tight_layout()
    p = os.path.join(PLOTS_DIR, "mask_strategy_comparison.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {p}")


def plot_condition_guidance(cond_analysis: Dict):
    """条件利用能力：散点/条形图 lead vs gen vs target"""
    if not cond_analysis:
        return
    metrics = list(cond_analysis.keys())
    lead_means = [cond_analysis[m]["lead_mean"] for m in metrics]
    gen_means = [cond_analysis[m]["gen_mean"] for m in metrics]
    target_means = [cond_analysis[m]["target_mean"] if cond_analysis[m]["target_mean"] is not None else np.nan for m in metrics]

    x = np.arange(len(metrics))
    width = 0.28
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - width, lead_means, width, label="Lead Peptide (orig)", color="#8da0cb")
    b2 = ax.bar(x,         gen_means,  width, label="Generated Candidate", color="#fc8d62")
    has_target = not np.all(np.isnan(target_means))
    if has_target:
        b3 = ax.bar(x + width, target_means, width, label="Target Condition", color="#66c2a5")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_title("Condition Guidance: Lead → Generated vs Target Descriptor")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(PLOTS_DIR, "condition_guidance.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {p}")


def plot_property_distribution(candidates_df: pd.DataFrame, train_pos_df: pd.DataFrame):
    """生成候选 vs 训练阳性AMP的性质分布小提琴图"""
    if candidates_df.empty or train_pos_df is None or train_pos_df.empty:
        return
    from descriptor_calculator import compute_descriptors_for_sequence
    metrics = ["net_charge", "GRAVY", "hydrophobic_ratio", "hemolysis_risk", "length"]
    rows = []
    for s in train_pos_df["sequence"].tolist():
        d = compute_descriptors_for_sequence(s, label=1)
        row = {"group": "Train G- Active AMP"}
        for m in metrics:
            row[m] = d[m]
        rows.append(row)
    for _, r in candidates_df.iterrows():
        row = {"group": "Generated Candidates"}
        for m in metrics:
            col = f"gen_{m}"
            row[m] = r[col] if col in candidates_df.columns else r.get(m, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for i, m in enumerate(metrics):
        ax = axes[i]
        try:
            sns.violinplot(data=df, x="group", y=m, ax=ax, palette="Set2", inner="quartile")
        except Exception:
            continue
        ax.set_title(m)
    # 去掉最后一个空ax
    if len(axes) > len(metrics):
        for j in range(len(metrics), len(axes)):
            axes[j].axis("off")
    fig.suptitle("Property Distribution: Generated vs Train G- Active AMP", fontsize=13)
    fig.tight_layout()
    p = os.path.join(PLOTS_DIR, "property_distribution.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {p}")


def plot_composite_score_histogram(candidates_df: pd.DataFrame):
    """综合得分直方图"""
    if candidates_df.empty or "composite_score" not in candidates_df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(candidates_df["composite_score"], bins=30, color="#66c2a5", edgecolor="white")
    ax.axvline(candidates_df["composite_score"].median(), color="#fc8d62", linestyle="--", label="Median")
    ax.set_xlabel("Composite Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Candidate Composite Scores")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(PLOTS_DIR, "composite_score_histogram.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {p}")


# ===================== 7. 最终结果输出 =====================
def build_final_candidates_output(candidates_df: pd.DataFrame,
                                  add_predicted_activity: bool = True) -> pd.DataFrame:
    """
    按任务交付物要求生成最终CSV字段：
    sequence, descriptor, predicted activity, hemolysis/toxicity risk,
    novelty, nearest neighbor similarity
    """
    if candidates_df.empty:
        return candidates_df
    # 加载训练集用于novelty和NN
    train_path = os.path.join(DATA_DIR, "train_with_descriptors.csv")
    train_df = pd.read_csv(train_path) if os.path.exists(train_path) else None
    train_set = set(train_df["sequence"].tolist()) if train_df is not None else set()

    seq_series = _get_sequence_series(candidates_df)
    final = pd.DataFrame({
        "sequence": seq_series,
        "lead_id": candidates_df.get("lead_id", ""),
        "lead_sequence": candidates_df.get("lead_sequence", ""),
        "mask_strategy": candidates_df.get("mask_strategy", ""),
        "composite_score": candidates_df.get("composite_score", 0),
        "condition_fit_score": candidates_df.get("condition_fit_score", 0),
        "condition_name": candidates_df.get("condition_name", "default"),
        "edit_distance_from_lead": candidates_df.get("edit_distance_from_lead", 0),
    })
    # Preserve optimization provenance and target descriptors in the public CSV.
    # Without these fields a later eval-only run cannot reproduce guidance metrics.
    for col in ["lead_sequence", "mask_strategy", "temperature"]:
        if col in candidates_df.columns:
            final[col] = candidates_df[col].values
    for prefix in ("lead_", "target_"):
        for name in DESCRIPTOR_NAMES:
            col = f"{prefix}{name}"
            if col in candidates_df.columns:
                final[col] = candidates_df[col].values
    # descriptor 列：拼接为字符串
    des_cols = []
    for n in DESCRIPTOR_NAMES:
        if f"gen_{n}" in candidates_df.columns:
            des_cols.append(f"gen_{n}")
        elif n in candidates_df.columns:
            des_cols.append(n)
    if des_cols:
        final["descriptor"] = candidates_df[des_cols].round(4).astype(str).agg(";".join, axis=1)
    else:
        final["descriptor"] = ""

    # Preserve the calibrated continuous activity probability when available.
    if "activity_probability" in candidates_df.columns:
        final["activity_probability"] = pd.to_numeric(
            candidates_df["activity_probability"], errors="coerce").round(4)
    gact_series = _get_metric_series(candidates_df, "gram_negative_active", prefix="gen_")
    score_series = pd.to_numeric(candidates_df.get("composite_score", 0), errors="coerce")
    activity_prob = pd.to_numeric(
        candidates_df.get("activity_probability", pd.Series(np.nan, index=candidates_df.index)),
        errors="coerce")
    if add_predicted_activity and activity_prob.notna().any():
        final["predicted_activity"] = activity_prob.round(4)
    elif add_predicted_activity and not gact_series.isna().all():
        gact = gact_series.to_numpy(dtype=np.float32)
        score = score_series.to_numpy(dtype=np.float32)
        max_s = max(float(np.nanmax(score)) if np.nanmax(score) == np.nanmax(score) else 1e-6, 1e-6)
        s_norm = np.divide(score, max_s, out=np.zeros_like(score, dtype=np.float32), where=np.isfinite(score))
        activity = 0.5 * gact + 0.5 * s_norm
        final["predicted_activity"] = np.round(activity, 4)
    else:
        final["predicted_activity"] = 0.0

    # 风险
    hemolysis = _get_metric_series(candidates_df, "hemolysis_risk", prefix="gen_")
    toxicity = _get_metric_series(candidates_df, "toxicity_risk", prefix="gen_")
    aggregation = _get_metric_series(candidates_df, "aggregation_propensity", prefix="gen_")
    if not hemolysis.isna().all():
        final["hemolysis_risk"] = hemolysis.round(4)
    if not toxicity.isna().all():
        final["toxicity_risk"] = toxicity.round(4)
    if not aggregation.isna().all():
        final["aggregation_propensity"] = aggregation.round(4)
    if "synthesis_feasibility" in candidates_df.columns:
        final["synthesis_feasibility"] = pd.to_numeric(
            candidates_df["synthesis_feasibility"], errors="coerce").round(4)

    # novelty
    final["novelty"] = final["sequence"].apply(lambda s: 1.0 if s not in train_set else 0.0)

    # nearest neighbor similarity（采样式
    seqs = final["sequence"].tolist()
    if train_df is not None:
        train_sample = train_df["sequence"].tolist()
        if len(train_sample) > 500:
            train_sample = list(pd.Series(train_sample).sample(500, random_state=42))
        nn_sim = _compute_nearest_neighbor_similarity(seqs, train_sample)
    else:
        nn_sim = np.zeros(len(seqs))
    final["nearest_neighbor_similarity"] = np.round(nn_sim, 4)

    # 每个descriptor单列（便于用户查看）
    for n in DESCRIPTOR_NAMES:
        col = f"gen_{n}"
        if col in candidates_df.columns:
            final[n] = candidates_df[col].round(4)
    return final


# ===================== 8. 完整评估管线 =====================
def run_full_evaluation(candidates_df: pd.DataFrame = None,
                        recon_df: pd.DataFrame = None,
                        history: dict = None) -> Dict:
    """
    全评估流程，返回指标字典并绘图
    """
    # 1. 训练曲线
    plot_training_curves(history)

    # 加载数据集
    train_path = os.path.join(DATA_DIR, "train_with_descriptors.csv")
    test_path = os.path.join(DATA_DIR, "test_with_descriptors.csv")
    train_df = pd.read_csv(train_path) if os.path.exists(train_path) else None
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None
    train_pos_df = train_df[train_df["label"] == 1] if train_df is not None else pd.DataFrame()

    report = {}

    # 2. 重构评估
    if recon_df is None:
        recon_path = os.path.join(RESULTS_DIR, "reconstruction_eval.csv")
        if os.path.exists(recon_path):
            recon_df = pd.read_csv(recon_path)
    recon_summary = analyze_reconstruction(recon_df)
    if not recon_summary.empty:
        recon_sum_path = os.path.join(RESULTS_DIR, "reconstruction_summary.csv")
        recon_summary.to_csv(recon_sum_path, index=False, encoding="utf-8")
        print(f"[EVAL] 重构评估汇总:\n{recon_summary.to_string(index=False)}")
        plot_mask_strategy_comparison(recon_summary)
        report["reconstruction"] = recon_summary.to_dict(orient="records")

    # 2.5 不同mask ratio下的性能曲线
    if recon_df is not None and not recon_df.empty:
        ratio_summary = analyze_mask_ratio_performance(recon_df)
        if not ratio_summary.empty:
            ratio_sum_path = os.path.join(RESULTS_DIR, "mask_ratio_summary.csv")
            ratio_summary.to_csv(ratio_sum_path, index=False, encoding="utf-8")
            print(f"[EVAL] mask ratio性能汇总:\n{ratio_summary.to_string(index=False)}")
            plot_mask_ratio_curve(recon_df)
            report["mask_ratio_performance"] = ratio_summary.to_dict(orient="records")

    # 3. 候选评分与排序
    if candidates_df is None:
        cand_path = os.path.join(RESULTS_DIR, "generated_candidates.csv")
        if os.path.exists(cand_path):
            candidates_df = pd.read_csv(cand_path)
    if candidates_df is not None and not candidates_df.empty:
        candidates_df = score_and_rank_candidates(candidates_df)
        # 保存带评分的
        candidates_df.to_csv(os.path.join(RESULTS_DIR, "generated_candidates_scored.csv"),
                            index=False, encoding="utf-8")

        # 4. 生成质量
        quality = compute_generation_quality_metrics(candidates_df, train_df)
        print(f"[EVAL] 生成质量指标: {quality}")
        report["generation_quality"] = quality
        report["mic_stratification"] = analyze_mic_stratification(train_df)

        # 5. 条件利用
        cond = analyze_condition_guidance(candidates_df)
        print("[EVAL] 条件引导分析:")
        for m, v in cond.items():
            if v["target_mean"] is not None:
                gr = v["guidance_ratio"]
                gr_str = f"{gr:.2%}" if gr is not None else "N/A"
                print(f"  {m:20s}: lead={v['lead_mean']:.3f} gen={v['gen_mean']:.3f} "
                      f"target={v['target_mean']:.3f} guidance={gr_str}")
        report["condition_guidance"] = cond
        report["counterfactual_conditions"] = analyze_counterfactual_conditions(candidates_df)
        plot_condition_guidance(cond)

        # 6. 性质分布
        prop = analyze_property_distribution(candidates_df, train_pos_df)
        report["property_distribution"] = prop
        lead_metrics = ["activity_probability", "hemolysis_risk", "toxicity_risk",
                "synthesis_feasibility"]
        optimization = {}
        for metric in lead_metrics:
            gen_col = metric if metric in candidates_df.columns else f"gen_{metric}"
            lead_col = f"lead_{metric}"
            if metric == "activity_probability":
                lead_col = "lead_activity_probability"
            if gen_col in candidates_df.columns and lead_col in candidates_df.columns:
                gen_vals = pd.to_numeric(candidates_df[gen_col], errors="coerce")
                lead_vals = pd.to_numeric(candidates_df[lead_col], errors="coerce")
                optimization[metric] = {
                    "lead_mean": float(lead_vals.mean()),
                    "generated_mean": float(gen_vals.mean()),
                    "delta_generated_minus_lead": float((gen_vals - lead_vals).mean()),
                }
        report["optimization_vs_lead"] = optimization
        plot_property_distribution(candidates_df, train_pos_df)

        # 7. 综合评分分布
        plot_composite_score_histogram(candidates_df)

        # 8. 最终输出CSV（按交付要求字段）
        final_df = build_final_candidates_output(candidates_df)
        out_final = os.path.join(RESULTS_DIR, "generated_candidates.csv")
        final_df.to_csv(out_final, index=False, encoding="utf-8")
        print(f"[EVAL] 最终候选CSV已保存: {out_final}  (共{len(final_df)}条)")
        # Top 10 预览
        print("[EVAL] Top-10 候选:")
        top = final_df.head(10)
        for _, row in top.iterrows():
            print(f"  Score={_format_float(row.get('composite_score', np.nan))} "
                  f"Act={_format_float(row.get('predicted_activity', np.nan))} "
                  f"Hemo={_format_float(row.get('hemolysis_risk', np.nan))} Seq={row['sequence']}")

        report["total_candidates"] = len(final_df)

    # 保存报告JSON
    report_path = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"[EVAL] 评估指标已保存到 {report_path}")
    return report


if __name__ == "__main__":
    run_full_evaluation()
