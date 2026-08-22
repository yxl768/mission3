"""
候选肽有效性验证脚本（in silico validation）

验证内容：
1. 溶血预测公式准确性回测：用DBAASP真实HC50实验值 vs ML预测的hemolysis_risk
2. 候选肽性质与已知活性AMP分布对比
3. 候选肽关键motif检查（K/R-rich, W/F/Y插入）
4. Top候选肽详细活性分析

注意：湿实验（MIC/HC50真实测定）无法在代码中完成，本脚本只做计算验证。
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import DATA_DIR, RESULTS_DIR
from descriptor_calculator import (
    compute_descriptors_for_sequence,
    predict_hemolysis_risk,
    predict_hemolysis_risk_rule,
    predict_toxicity_risk,
    predict_activity_score,
    calc_net_charge,
    calc_GRAVY,
    calc_KRH_ratio,
    calc_hydrophobic_ratio,
    calc_aromatic_ratio,
)


CANDIDATE_COL_MAP = {
    "sequence": "generated_sequence",
    "length": "gen_length",
    "net_charge": "gen_net_charge",
    "GRAVY": "gen_GRAVY",
    "KRH_ratio": "gen_KRH_ratio",
    "hydrophobic_ratio": "gen_hydrophobic_ratio",
    "aromatic_ratio": "gen_aromatic_ratio",
    "hemolysis_risk": "gen_hemolysis_risk",
    "toxicity_risk": "gen_toxicity_risk",
    "predicted_activity": "gen_gram_negative_active",
}


def get_cand_col(df, logical_name):
    col = CANDIDATE_COL_MAP.get(logical_name, logical_name)
    if col in df.columns:
        return df[col]
    if logical_name in df.columns:
        return df[logical_name]
    raise KeyError(f"candidate column not found: {logical_name} (tried {col})")


def get_cand_value(row, logical_name):
    col = CANDIDATE_COL_MAP.get(logical_name, logical_name)
    if col in row.index:
        return row[col]
    if logical_name in row.index:
        return row[logical_name]
    return row[logical_name]


def validate_hemolysis_prediction():
    print("=" * 70)
    print("【验证1】ML溶血预测器准确性回测（DBAASP真实HC50数据）")
    print("=" * 70)

    raw_path = os.path.join(DATA_DIR, "dbaasp_raw.csv")
    if not os.path.exists(raw_path):
        print("  [ERROR] dbaasp_raw.csv 不存在")
        return

    df = pd.read_csv(raw_path)
    hemol_df = df[df["has_hemolytic_data"] == True].copy()
    hemol_df = hemol_df.dropna(subset=["hemo_activity_value", "sequence"])
    hemol_df["hemo_activity_value"] = pd.to_numeric(hemol_df["hemo_activity_value"], errors="coerce")
    hemol_df = hemol_df.dropna(subset=["hemo_activity_value"])
    hemol_df = hemol_df[(hemol_df["length"] >= 4) & (hemol_df["length"] <= 50)]
    hemol_df = hemol_df[hemol_df["hemo_activity_value"] < 1e5].reset_index(drop=True)

    print(f"  DBAASP中有真实溶血实验值的肽: {len(hemol_df)} 条")
    print(f"  HC50范围: {hemol_df['hemo_activity_value'].min():.2f} - {hemol_df['hemo_activity_value'].max():.2f} µM")
    print(f"  HC50中位数: {hemol_df['hemo_activity_value'].median():.2f} µM")

    pred_risks = []
    pred_rule_risks = []
    true_hc50 = []
    true_risk_labels = []
    for _, row in hemol_df.iterrows():
        seq = str(row["sequence"])
        hc50 = float(row["hemo_activity_value"])
        try:
            pred = predict_hemolysis_risk(seq)
            pred_rule = predict_hemolysis_risk_rule(seq)
            pred_risks.append(pred)
            pred_rule_risks.append(pred_rule)
            true_hc50.append(hc50)
            if hc50 <= 50:
                true_risk_labels.append(1)
            elif hc50 >= 200:
                true_risk_labels.append(0)
            else:
                true_risk_labels.append(-1)
        except Exception:
            pass

    pred_risks = np.array(pred_risks)
    pred_rule_risks = np.array(pred_rule_risks)
    true_hc50 = np.array(true_hc50)
    true_risk_labels = np.array(true_risk_labels)

    if len(pred_risks) >= 5:
        log_hc50 = np.log10(np.clip(true_hc50, 1, 1e6))

        corr_pearson, p_pearson = pearsonr(pred_risks, log_hc50)
        corr_spearman, p_spearman = spearmanr(pred_risks, log_hc50)
        print(f"\n  [ML模型] 预测风险 vs log10(HC50):")
        print(f"    Pearson r = {corr_pearson:.4f} (p={p_pearson:.4e})")
        print(f"    Spearman ρ = {corr_spearman:.4f} (p={p_spearman:.4e})")
        if corr_spearman < 0:
            print(f"    ✅ 方向正确：预测风险↑ → HC50↓（溶血性↑）")
        else:
            print(f"    ⚠️ 方向异常：应为负相关")

        corr_pearson_r, _ = pearsonr(pred_rule_risks, log_hc50)
        corr_spearman_r, _ = spearmanr(pred_rule_risks, log_hc50)
        print(f"\n  [规则公式] 预测风险 vs log10(HC50):")
        print(f"    Pearson r = {corr_pearson_r:.4f}")
        print(f"    Spearman ρ = {corr_spearman_r:.4f}")

    mask_valid = true_risk_labels >= 0
    if mask_valid.sum() >= 10:
        from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
        y_true = true_risk_labels[mask_valid]
        y_pred_prob = pred_risks[mask_valid]
        y_pred_rule = pred_rule_risks[mask_valid]

        acc_ml = accuracy_score(y_true, (y_pred_prob >= 0.5).astype(int))
        acc_rule = accuracy_score(y_true, (y_pred_rule >= 0.5).astype(int))
        auc_ml = roc_auc_score(y_true, y_pred_prob)
        auc_rule = roc_auc_score(y_true, y_pred_rule)

        print(f"\n  [分类性能] 高风险(HC50≤50) vs 低风险(HC50≥200):")
        print(f"    样本数: {mask_valid.sum()} (高风险{(y_true==1).sum()} / 低风险{(y_true==0).sum()})")
        print(f"    ML模型:   AUC={auc_ml:.4f}, Accuracy={acc_ml:.4f}")
        print(f"    规则公式: AUC={auc_rule:.4f}, Accuracy={acc_rule:.4f}")
        print(f"    提升:     AUC +{(auc_ml-auc_rule)*100:.1f}%, Accuracy +{(acc_ml-acc_rule)*100:.1f}%")

        print(f"\n  [典型例子] 前15条真实肽的预测vs实际:")
        header = f"  {'sequence':<30} {'HC50(µM)':<12} {'真实':<6} {'ML预测':<10} {'规则预测':<10}"
        print(header)
        print(f"  {'-'*30} {'-'*12} {'-'*6} {'-'*10} {'-'*10}")
        for i in range(min(15, len(pred_risks))):
            seq_short = str(hemol_df.iloc[i]["sequence"])[:28]
            hc50 = true_hc50[i]
            true_lbl = "高" if hc50 <= 50 else ("低" if hc50 >= 200 else "中")
            ml_p = pred_risks[i]
            rule_p = pred_rule_risks[i]
            print(f"  {seq_short:<30} {hc50:<12.2f} {true_lbl:<6} {ml_p:<10.4f} {rule_p:<10.4f}")

    return pred_risks, pred_rule_risks, true_hc50


def validate_candidates_properties():
    print("\n" + "=" * 70)
    print("【验证2】候选肽性质 vs 已知G-活性AMP分布对比")
    print("=" * 70)

    cand_path = os.path.join(RESULTS_DIR, "generated_candidates.csv")
    train_path = os.path.join(DATA_DIR, "train_with_descriptors.csv")

    if not os.path.exists(cand_path):
        print("  [ERROR] generated_candidates.csv 不存在")
        return

    cand = pd.read_csv(cand_path)
    train = pd.read_csv(train_path) if os.path.exists(train_path) else pd.DataFrame()
    train_pos = train[train["label"] == 1] if not train.empty else pd.DataFrame()

    print(f"  候选肽数: {len(cand)}")
    print(f"  训练集G-活性AMP数: {len(train_pos)}")

    props_logical = ["length", "net_charge", "GRAVY", "KRH_ratio", "hydrophobic_ratio",
                     "aromatic_ratio", "hemolysis_risk", "toxicity_risk"]

    print(f"\n  {'性质':<20} {'候选均值':<12} {'AMP均值':<12} {'AMP区间(P5-P95)':<22} {'落入率':<10}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*22} {'-'*10}")

    for p in props_logical:
        train_col = p
        try:
            cand_vals = get_cand_col(cand, p).dropna()
        except KeyError:
            continue
        if train_pos.empty or train_col not in train_pos.columns:
            print(f"  {p:<20} {cand_vals.mean():<12.4f} {'N/A':<12} {'N/A':<22} {'N/A':<10}")
            continue
        train_vals = train_pos[train_col].dropna()
        p5, p95 = np.percentile(train_vals, [5, 95])
        in_range = ((cand_vals >= p5) & (cand_vals <= p95)).mean()
        print(f"  {p:<20} {cand_vals.mean():<12.4f} {train_vals.mean():<12.4f} "
              f"[{p5:.3f}, {p95:.3f}]{'':<8} {in_range*100:.1f}%")


def validate_candidates_motif():
    print("\n" + "=" * 70)
    print("【验证3】候选肽关键AMP motif检查")
    print("=" * 70)

    cand_path = os.path.join(RESULTS_DIR, "generated_candidates.csv")
    cand = pd.read_csv(cand_path)
    seqs = get_cand_col(cand, "sequence").tolist()

    motifs = {
        "K/R-rich (KK/RR连续)": lambda s: ("KK" in s) or ("RR" in s) or ("KR" in s) or ("RK" in s),
        "W插入 (Trp)": lambda s: "W" in s,
        "F插入 (Phe)": lambda s: "F" in s,
        "GKG/GRG motif": lambda s: ("GKG" in s) or ("GRG" in s),
        "KXK motif (K-任意-K)": lambda s: any(s[i]=='K' and s[i+2]=='K' for i in range(len(s)-2)),
        "阳离子性 (charge≥4)": lambda s: calc_net_charge(str(s)) >= 4,
        "两亲性 (GRAVY -0.5~0.3)": lambda s: -0.5 <= calc_GRAVY(str(s)) <= 0.3,
    }

    print(f"  候选肽数: {len(seqs)}")
    print(f"\n  {'Motif/特征':<30} {'含该特征的候选数':<15} {'占比':<8}")
    print(f"  {'-'*30} {'-'*15} {'-'*8}")
    for name, fn in motifs.items():
        count = sum(1 for s in seqs if fn(str(s)))
        print(f"  {name:<30} {count:<15} {count/len(seqs)*100:.1f}%")


def validate_top_candidates_detail():
    print("\n" + "=" * 70)
    print("【验证4】Top-10候选肽详细活性与安全性分析")
    print("=" * 70)

    cand_path = os.path.join(RESULTS_DIR, "generated_candidates.csv")
    cand = pd.read_csv(cand_path)

    # Prefer the pipeline's multi-objective score when available.  Sorting
    # only by hemolysis surfaced very low-activity, low-charge peptides as
    # "top" candidates and contradicted the exported ranking.
    def col(name):
        mapped = CANDIDATE_COL_MAP.get(name, name)
        return mapped if mapped in cand.columns else name
    hemo_col = col("hemolysis_risk")
    act_col = col("predicted_activity")
    charge_col = col("net_charge")
    gravy_col = col("GRAVY")
    krh_col = col("KRH_ratio")
    tox_col = col("toxicity_risk")
    seq_col = col("sequence")

    if "composite_score" in cand.columns:
        cand_sorted = cand.sort_values("composite_score", ascending=False)
    else:
        cand_sorted = cand.sort_values([hemo_col, act_col], ascending=[True, False])
    top10 = cand_sorted.head(10)

    print(f"\n  {'#':<3} {'sequence':<18} {'charge':<8} {'GRAVY':<8} {'KRH':<8} {'hemo':<8} {'tox':<8} {'act':<8} {'评估':<20}")
    print(f"  {'-'*3} {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*20}")

    for i, (_, row) in enumerate(top10.iterrows()):
        seq = row[seq_col]
        charge = row[charge_col]
        gravy = row[gravy_col]
        krh = row[krh_col]
        hemo = row[hemo_col]
        tox = row[tox_col]
        act = row[act_col]

        score = 0
        reasons = []
        if 3 <= charge <= 7:
            score += 1; reasons.append("电荷优")
        else:
            reasons.append("电荷偏")
        if -0.5 <= gravy <= 0.3:
            score += 1; reasons.append("疏水适")
        else:
            reasons.append("疏水偏")
        if hemo < 0.2:
            score += 1; reasons.append("溶血低")
        else:
            reasons.append("溶血高")
        if tox < 0.1:
            score += 1; reasons.append("毒性低")
        else:
            reasons.append("毒性高")
        if krh >= 0.25:
            score += 1; reasons.append("阳离子足")

        verdict = "优" if score >= 4 else ("良" if score >= 3 else "需改")
        print(f"  {i+1:<3} {seq:<18} {charge:<8.2f} {gravy:<8.3f} {krh:<8.3f} {hemo:<8.4f} {tox:<8.4f} {act:<8.3f} {verdict+'('+','.join(reasons)+')':<20}")


def validate_novelty_analysis():
    print("\n" + "=" * 70)
    print("【验证5】候选肽新颖性分析")
    print("=" * 70)

    cand_path = os.path.join(RESULTS_DIR, "generated_candidates.csv")
    raw_path = os.path.join(DATA_DIR, "dbaasp_raw.csv")

    if not os.path.exists(cand_path):
        print("  [ERROR] generated_candidates.csv 不存在")
        return

    cand = pd.read_csv(cand_path)
    cand_seqs = set(get_cand_col(cand, "sequence").unique())

    if os.path.exists(raw_path):
        raw = pd.read_csv(raw_path)
        raw_seqs = set(raw["sequence"].dropna().unique())
        overlap = raw_seqs & cand_seqs
        print(f"  候选肽中在DBAASP已存在: {len(overlap)} 条 ({len(overlap)/len(cand_seqs)*100:.1f}%)")
        print(f"  全新候选肽: {len(cand_seqs)-len(overlap)} 条 ({(len(cand_seqs)-len(overlap))/len(cand_seqs)*100:.1f}%)")
    else:
        print(f"  候选肽数: {len(cand_seqs)}")
        print(f"  未找到DBAASP原始数据，无法做重叠检查")
        print(f"  注意：候选肽由条件VAE从头生成，理论上全部为新颖序列")

    print(f"\n  序列长度分布:")
    lengths = get_cand_col(cand, "length")
    print(f"    min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.1f}, median={lengths.median():.0f}")

    print(f"\n  生成来源分布:")
    if "mask_strategy" in cand.columns:
        for strat, count in cand["mask_strategy"].value_counts().items():
            print(f"    {strat}: {count} 条")


def main():
    print("=" * 70)
    print("  候选肽有效性验证报告（in silico validation）")
    print("  注意：湿实验验证（MIC/HC50真实测定）需实验室完成")
    print("=" * 70)

    validate_hemolysis_prediction()
    validate_candidates_properties()
    validate_candidates_motif()
    validate_top_candidates_detail()
    validate_novelty_analysis()

    print("\n" + "=" * 70)
    print("  验证总结")
    print("=" * 70)
    print("  1. 溶血预测：ML模型 vs 规则公式 AUC对比")
    print("  2. 性质分布：候选肽 vs 已知G-活性AMP分布落入率")
    print("  3. AMP motif：检查K/R-rich、W/F插入等关键特征")
    print("  4. Top候选：综合电荷/疏水/溶血/毒性评估")
    print("  5. 新颖性：候选肽与DBAASP重叠检查")
    print()
    print("  ⚠️ 最终有效性确认需湿实验：")
    print("    - MIC肉汤稀释法（E. coli/P. aeruginosa）")
    print("    - HC50溶血测定（人红细胞）")
    print("    - 细胞毒性MTT（HEK293/HeLa）")


if __name__ == "__main__":
    main()
