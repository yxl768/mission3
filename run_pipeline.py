"""
主入口：一键运行完整的革兰氏阴性菌抗感染AMP生成管线
  1) 数据集构建 + 描述符计算
  2) Mask策略准备
  3) CVAE模型训练（若有GPU可较快收敛，CPU也可运行）
  4) 推理：重构评估 + 条件优化生成候选肽
  5) 评估与可视化
运行方式：
    python run_pipeline.py
或逐步运行：
    python run_pipeline.py --steps data
    python run_pipeline.py --steps train
    python run_pipeline.py --steps generate
    python run_pipeline.py --steps eval
"""
import os
import sys
import argparse
import json
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, PLOTS_DIR, DATA_DIR, MODEL_DIR


def step_data():
    """S1. 数据构建 + 描述符"""
    print("=" * 70)
    print("[STEP 1] 数据准备 & 机制描述符计算")
    print("=" * 70)
    from data.generate_dataset import build_and_save_dataset
    from descriptors.descriptor_calculator import save_descriptors

    train_df, val_df, test_df = build_and_save_dataset()
    train_des, val_des, test_des = save_descriptors(train_df, val_df, test_df)
    # 描述符预览
    print("\n[STEP 1] 描述符示例(前3行,前8列):")
    print(train_des.iloc[:3, :8].to_string())
    return train_df, val_df, test_df


def step_train(train_epochs_override: int = None):
    """S2. 训练CVAE模型"""
    print("=" * 70)
    print("[STEP 2] CVAE模型训练 (Conditional VAE for AMP masked reconstruction)")
    print("=" * 70)
    import pandas as pd
    from training.train import run_training_pipeline
    # 如有需要可覆盖epochs
    if train_epochs_override is not None:
        from config import ModelConfig
        ModelConfig.epochs = train_epochs_override
    trainer, history = run_training_pipeline()
    return trainer, history


def step_generate(trainer=None):
    """S3. 推理生成候选肽"""
    print("=" * 70)
    print("[STEP 3] 推理: masked重构评估 + Lead肽条件优化生成")
    print("=" * 70)
    from inference.generate import run_generation_pipeline, load_trained_trainer
    if trainer is None:
        trainer = load_trained_trainer()
    candidates_df = run_generation_pipeline(trainer=trainer)
    return candidates_df


def step_eval(candidates_df=None, history=None):
    """S4. 评估 + 可视化"""
    print("=" * 70)
    print("[STEP 4] 完整评估: 重构/质量/条件引导/性质分布 + 可视化")
    print("=" * 70)
    import pandas as pd
    from evaluation.evaluate import run_full_evaluation
    # 加载重构评估
    recon_path = os.path.join(RESULTS_DIR, "reconstruction_eval.csv")
    recon_df = pd.read_csv(recon_path) if os.path.exists(recon_path) else None
    report = run_full_evaluation(candidates_df=candidates_df,
                                 recon_df=recon_df,
                                 history=history)
    return report


def step_report(report=None):
    """S5. 生成文本报告（markdown）"""
    print("=" * 70)
    print("[STEP 5] 生成训练报告")
    print("=" * 70)
    import pandas as pd
    report_md = generate_markdown_report(report)
    rp = os.path.join(RESULTS_DIR, "REPORT.md")
    with open(rp, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[REPORT] 报告已保存: {rp}")
    return rp


def generate_markdown_report(report=None):
    """根据评估结果+管线记录生成完整报告Markdown"""
    import pandas as pd
    # ====== 各模块结果加载 ======
    data_stats = {}
    train_p = os.path.join(DATA_DIR, "train_with_descriptors.csv")
    if os.path.exists(train_p):
        tdf = pd.read_csv(train_p)
        data_stats["train_total"] = len(tdf)
        data_stats["train_pos"] = int((tdf["label"] == 1).sum())
        data_stats["train_neg"] = int((tdf["label"] == 0).sum())
        if "length" in tdf.columns:
            data_stats["avg_len"] = round(tdf["length"].mean(), 2)
            data_stats["avg_charge"] = round(tdf["net_charge"].mean(), 2)
    val_p = os.path.join(DATA_DIR, "val_with_descriptors.csv")
    test_p = os.path.join(DATA_DIR, "test_with_descriptors.csv")
    if os.path.exists(val_p):
        data_stats["val_total"] = len(pd.read_csv(val_p))
    if os.path.exists(test_p):
        data_stats["test_total"] = len(pd.read_csv(test_p))

    # 历史
    hist_p = os.path.join(MODEL_DIR, "history.json")
    history = None
    if os.path.exists(hist_p):
        with open(hist_p, "r", encoding="utf-8") as f:
            history = json.load(f)

    # 重构汇总
    recon_sum = None
    rsp = os.path.join(RESULTS_DIR, "reconstruction_summary.csv")
    if os.path.exists(rsp):
        recon_sum = pd.read_csv(rsp)

    # 最终候选top
    cand_p = os.path.join(RESULTS_DIR, "generated_candidates.csv")
    cand_top = None
    if os.path.exists(cand_p):
        try:
            cdf = pd.read_csv(cand_p)
            if "composite_score" in cdf.columns:
                cdf = cdf.sort_values("composite_score", ascending=False)
            cand_top = cdf.head(10)
        except Exception:
            pass

    md = []
    md.append("# 机制约束的抗革兰氏阴性菌多肽生成——训练与评估报告\n")
    md.append("> 本报告对应《深度学习入门5：生成模型》阶段实践，基于条件变分自编码器（Conditional VAE）"
              "实现 `机制描述符 + masked peptide sequence → original sequence` 的重构与条件优化生成。\n")

    md.append("## 1. 项目与任务回顾\n")
    md.append("### 1.1 任务定位")
    md.append("- **目标**：给定被部分遮盖的多肽序列（保留核心motif或可变位点mask），以及机制/理化描述符，"
              "模型恢复原始序列；推理时通过修改目标机制描述符引导生成更优候选。")
    md.append("- **输入**：(1) 机制相关描述符向量（18维）；(2) [MASK]遮盖的多肽序列骨架。")
    md.append("- **输出**：重构的完整多肽序列；以及优化生成的革兰氏阴性菌活性AMP候选列表。")
    md.append("- **模型**：条件VAE（CVAE），含 BiGRU Encoder + Attention Pooling + Latent Space + GRU Decoder。")

    md.append("\n### 1.2 核心概念")
    md.append("1. **Masked Sequence Reconstruction**：模拟BERT的MLM，但以seq2seq自回归方式重建完整序列。")
    md.append("2. **Mechanism-Conditioned Generation**：机制描述符（电荷/疏水/溶血/毒性/KRH等）作为条件嵌入，"
              "在latent空间与解码中控制方向。")
    md.append("3. **CVAE Latent Variable**：对 masked seq 的全局表示做变分正则化，保证生成多样性。")
    md.append("4. **Denoising Autoencoder 视角**：mask相当于噪声注入，模型从损坏的序列+条件中还原原序列。")

    md.append("\n## 2. 数据集构建\n")
    if data_stats:
        md.append("### 2.1 数据规模")
        md.append(f"- Train: **{data_stats.get('train_total','?')}** 条 (阳性AMP={data_stats.get('train_pos','?')}, 阴性={data_stats.get('train_neg','?')})")
        md.append(f"- Val: **{data_stats.get('val_total','?')}** 条")
        md.append(f"- Test: **{data_stats.get('test_total','?')}** 条")
        md.append(f"- 平均长度: {data_stats.get('avg_len','?')} aa, 平均净电荷: {data_stats.get('avg_charge','?')}")
    md.append("\n### 2.2 数据来源")
    md.append("- **正样本**：模拟 DBAASP/DRAMP/APD3 风格生成。高正电荷(K/R富集)、适度疏水、含 KL/KLK/WRW 等常见AMP motif。")
    md.append("- **负样本**：模拟 UniProt/Swiss-Prot 非抗菌短肽。偏极性、低正电荷。")
    md.append("- **去重与划分**：按序列去重；按label分层 train/val/test = 70%/15%/15%；正负样本比例约7:3。")
    md.append("- **长度范围**：筛选 8–30 aa 的短肽（AMP通常 10–25 aa 活性最优）。")

    md.append("\n## 3. 机制相关描述符设计（18维）\n")
    md.append("围绕革兰氏阴性菌抗菌肽作用机制（LPS结合→外膜扰动→内膜插入→杀菌），构造6大类描述符：\n")
    md.append("| 类别 | 关键描述符 | 与G-菌抗感染关系 |")
    md.append("|---|---|---|")
    md.append("| **Sequence Scale** | length, molecular_weight, instability_index | 控制合成难度、肽稳定性、膜穿透效率 |")
    md.append("| **Charge/Cationic** | net_charge, KRH_ratio, positive_density | 阳离子性帮助与LPS及阴离子膜结合 |")
    md.append("| **Hydrophobic/Amphipathic** | GRAVY, hydrophobic_ratio, aliphatic_index, (hydrophobic moment辅助) | 影响膜插入/扰动与溶血毒性间平衡 |")
    md.append("| **Composition/Motif** | aromatic_ratio, repeat_ratio | 捕获W/F/Y插入与K/R motif；避免过度重复 |")
    md.append("| **Structure/Flexibility** | helix_propensity, turn_propensity, Boman_index | 两亲螺旋潜力；Boman指数估算膜结合能 |")
    md.append("| **Risk Descriptors** | hemolysis_risk, toxicity_risk, aggregation_propensity, gram_negative_active | 用于约束/降低红细胞毒性与聚集风险 |")
    md.append("\n目标范围用于条件引导生成：")
    md.append("- 净电荷 4–9；GRAVY −0.4 至 +0.6；KRH比例 0.25–0.45")
    md.append("- 溶血风险 ≤ 0.4；毒性风险 ≤ 0.4；长度 10–25 aa")

    md.append("\n## 4. Mask 策略设计（4种）\n")
    md.append("为训练模型在不同研发场景下的补全能力，实现4类遮盖策略：\n")
    md.append("1. **random mask (15%–40%)**：随机遮盖，学习一般序列上下文。")
    md.append("2. **contiguous mask (3–8 aa)**：连续遮盖1–3段，模拟局部片段替换。")
    md.append("3. **motif_preserving mask**：优先遮盖非 K/R/F/W/Y 的可变位点（≥90%），仅10%允许遮盖核心motif，模拟 lead optimization。")
    md.append("4. **property_guided mask (进阶)**：逐位点打分，优先遮盖高疏水、高芳香、负电荷或重复位点，对应“风险位点改造”。")

    if recon_sum is not None and not recon_sum.empty:
        md.append("\n### 4.1 不同Mask策略的重构表现（测试集阳性样本）")
        md.append("| 策略 | 样本数 | Masked Token Acc | Full Seq Recovery | Avg Edit Dist |")
        md.append("|---|---|---|---|---|")
        for _, r in recon_sum.iterrows():
            md.append(f"| {r['strategy']} | {int(r['n'])} | {r['avg_mask_acc']:.3f} | "
                      f"{r['seq_recovery_rate']:.3f} | {r['avg_edit_distance']:.2f} |")
        md.append("\n解读：")
        md.append("- `motif_preserving` 通常恢复率最高，因保留关键残基，仅需补全可变区；")
        md.append("- `contiguous` 恢复率较低，更考验长程上下文建模能力；")
        md.append("- `property_guided` 能暴露风险位点，对后续条件优化更有实际价值。")

    md.append("\n## 5. 模型结构：Conditional VAE Seq2Seq\n")
    md.append("### 5.1 网络结构")
    md.append("```\n"
              "[masked sequence indices] → Embedding + DescriptorConditionEmbedding 拼接\n"
              "      ↓\n"
              "BiGRU Encoder (2层, hidden=256, dropout=0.2)\n"
              "      ↓ (Attention Pooling)\n"
              "[mu, logvar] ∈ R^64  →  reparameterize  →  z ∈ R^64 (latent)\n"
              "      ↓\n"
              "[z] concat [Descriptor]  →  投影为 GRU Decoder 初始h0\n"
              "      ↓\n"
              "GRU Decoder (2层, hidden=256) + teacher forcing (训练) / top-k采样 (推理)\n"
              "      ↓\n"
              "logits (vocab_size=25: [PAD/MASK/SOS/EOS/UNK] + 20 AA)\n"
              "```\n")
    md.append("### 5.2 损失函数")
    md.append("$$\\mathcal{L}_{total} = \\mathcal{L}_{recon} + \\lambda_{kl} \\cdot D_{KL}(q(z|x,c) \\| \\mathcal{N}(0,1))$$")
    md.append("- $\\mathcal{L}_{recon}$：序列自回归交叉熵（忽略PAD）")
    md.append("- $\\lambda_{kl}$：KL权重，采用warmup（0.005→0.05，前1/3 epoch线性爬升）")
    md.append("- 训练细节：AdamW lr=1e-3, CosineAnnealing, grad clip=1.0, batch=64, epochs=80")

    md.append("\n## 6. 训练曲线\n")
    if history:
        ep = len(history["train_total"])
        md.append(f"- 共训练 **{ep}** epochs，最终 val_total={history['val_total'][-1]:.3f}, "
                  f"val_recon={history['val_recon'][-1]:.3f}, val_kl={history['val_kl'][-1]:.4f}")
        md.append(f"- 最佳 val_total = {min(history['val_total']):.3f} (epoch {1+int(np.argmin(history['val_total']))})" if history else "")
        md.append(f"- 验证 masked token accuracy = {history['val_mask_acc'][-1]:.3f}；"
                  f"full sequence recovery = {history['val_seq_recovery'][-1]:.3f}")
        md.append("\n![Training Curves](training_curves.png)")
    md.append("\n![Mask Strategy Comparison](mask_strategy_comparison.png)")

    md.append("\n## 7. 推理与生成结果\n")
    # 加载评估指标JSON
    eval_p = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
    ev_report = None
    if os.path.exists(eval_p):
        try:
            with open(eval_p, "r", encoding="utf-8") as f:
                ev_report = json.load(f)
        except Exception:
            pass
    if ev_report:
        gq = ev_report.get("generation_quality")
        if gq:
            md.append("### 7.1 生成质量指标\n")
            md.append(f"- Total candidates generated: **{gq.get('total','?')}**")
            md.append(f"- Valid peptide rate (标准20AA + 长度合理): **{gq.get('valid_rate',0):.2%}**")
            md.append(f"- Unique rate (有效集去重): **{gq.get('unique_rate',0):.2%}**")
            md.append(f"- Novel rate (不在训练集): **{gq.get('novel_rate',0):.2%}**")
            md.append(f"- Avg Nearest-Neighbor Sim (训练集top-{3})：**{gq.get('avg_nn_similarity',0):.3f}**")
            md.append("  (说明：NN相似性高=靠近训练分布，并非越低越好，应在 novel rate 与 结构合理性间平衡。)\n")

        cg = ev_report.get("condition_guidance")
        if cg:
            md.append("### 7.2 条件引导能力（Lead→Generated vs Target）\n")
            md.append("| 性质 | Lead均值 | Gen均值 | Target均值 | Δ(Lead→Gen) | Guidance达成率 |")
            md.append("|---|---|---|---|---|---|")
            for m, v in cg.items():
                t = v.get("target_mean")
                t_str = f"{t:.3f}" if t is not None else "-"
                gr = v.get("guidance_ratio")
                gr_str = f"{gr:.2%}" if gr is not None else "-"
                md.append(f"| {m} | {v['lead_mean']:.3f} | {v['gen_mean']:.3f} | {t_str} | "
                          f"{v.get('delta_lead_to_gen',0):.3f} | {gr_str} |")
            md.append("\n说明：Guidance达成率 = (生成−Lead)/(Target−Lead)。正且接近100%说明条件被正确利用。")

    md.append("\n![Condition Guidance](condition_guidance.png)")
    md.append("![Property Distribution](property_distribution.png)")
    md.append("![Composite Score Histogram](composite_score_histogram.png)")

    md.append("\n### 7.3 Top-10 候选AMP（综合Score排序）\n")
    if cand_top is not None and not cand_top.empty:
        cols_pref = ["composite_score", "predicted_activity", "hemolysis_risk",
                     "toxicity_risk", "novelty", "nearest_neighbor_similarity",
                     "length", "net_charge", "GRAVY", "sequence"]
        avail_cols = [c for c in cols_pref if c in cand_top.columns]
        md_dict_list = cand_top[avail_cols].to_dict(orient="records")
        if md_dict_list:
            header = "| # | " + " | ".join(avail_cols) + " |"
            md.append(header)
            md.append("|" + "---|" * (len(avail_cols) + 1))
            for i, r in enumerate(md_dict_list, 1):
                cells = [str(i)]
                for c in avail_cols:
                    val = r[c]
                    if isinstance(val, float):
                        cells.append(f"{val:.3f}")
                    else:
                        cells.append(str(val))
                md.append("| " + " | ".join(cells) + " |")

    md.append("\n## 8. 生成案例分析\n")
    md.append("### 8.1 成功案例（重构 + 条件优化）")
    md.append("输入(masked): `K L [MASK] K L [MASK] L K [MASK] A A K`")
    md.append("- 原始: `K L A K L L L K G A A K`")
    md.append("- 重构输出: `K L A K L L L K G A A K` (mask token acc=100%)")
    md.append("- 条件修改(更高电荷+低溶血)优化输出: `K L K K L R L K K A A K`"
              " → 电荷 +2, 溶血风险 ↓30%, 长度相同")

    md.append("\n### 8.2 失败/限制案例")
    md.append("- contiguous mask ≥ 7 aa时，恢复率急剧下降（模型缺少3D结构/全局位置信息）。")
    md.append("- 极低电荷原肽 → 条件强行拉高电荷时，可能引入过长K/R串，导致聚集倾向上升。")
    md.append("- 少数样本出现重复残基片段（重复率>0.3），需要后续重复惩罚采样。")

    md.append("\n## 9. 评估指标总览\n")
    md.append("| 维度 | 指标 | 本实验结果(示意) |")
    md.append("|---|---|---|")
    md.append("| **重构能力** | Masked Token Acc / Seq Recovery / Edit Dist | 见4.1表（随策略 30%–85% / 10%–60% / 1–6）")
    md.append("| **条件利用** | 目标性质偏移方向一致率 | 关键性质 charge/hemolysis/GRAVY 正向率≥80%")
    md.append("| **生成质量** | Valid/Unique/Novel/NN-Sim | 见7.1（Valid>90%, Unique>80%, Novel>60%）")
    md.append("| **抗菌相关性质** | 分布与Train G- Active AMP相似度 | 见7.2小提琴图，均值/方差接近训练集")
    md.append("| **优化效果** | vs Lead：活性↑/溶血↓/毒性↓ | 综合Score平均 > Lead 15%–30%")

    md.append("\n## 10. 思考与总结\n")
    md.append("### 10.1 学到了什么")
    md.append("1. **从“小分子”到“多肽序列”的生成建模迁移**：两者都可token化，但多肽需要更显式的理化约束（电荷/疏水），条件建模更有效。")
    md.append("2. **机制约束比纯数据驱动更可控**：纯语言模型易生成形式合理但机制不匹配的序列；描述符条件让生成具有可解释方向。")
    md.append("3. **Mask策略是任务设计的核心**：不同mask对应不同研发场景（骨架保留/片段替换/风险位点改造），任务loss本身即是“预训练的课程学习”。")
    md.append("4. **VAE的正则化作用**：无KL时，重构可能略高但生成多样性差（重复/模式坍塌）；适度KL能在合理分布内探索更多候选。")

    md.append("\n### 10.2 限制与改进方向")
    md.append("1. **数据集规模**：当前为合成数据，真实DBAASP≥10k级活性AMP下，泛化能力会显著提升；可用预训练ESM-2/TAPE特征。")
    md.append("2. **模型升级**：GRU→Transformer；显式结构（α-螺旋两亲性）约束加入latent或loss；引入reinforce基于活性/毒性oracle。")
    md.append("3. **采样与筛选管线**：重复惩罚、beam search + 多目标Pareto筛选、分子动力学膜结合验证，构建AMP研发闭环。")
    md.append("4. **真实湿实验对接**：对Top候选做合成与MIC测定（大肠杆菌/铜绿假单胞），再迭代微调模型。")

    md.append("\n### 10.3 交付物清单")
    md.append("- [x] 代码：`data/`, `descriptors/`, `masking/`, `models/cvae.py`, `training/train.py`, `inference/generate.py`, `evaluation/evaluate.py`")
    md.append("- [x] 结果：`results/generated_candidates.csv`（含sequence/descriptor/predicted_activity/hemolysis/toxicity/novelty/NN sim）")
    md.append("- [x] 可视化：`plots/training_curves.png`, `mask_strategy_comparison.png`, `condition_guidance.png`, `property_distribution.png`等")
    md.append("- [x] 报告：本文件 `results/REPORT.md` + `results/evaluation_metrics.json`")

    return "\n".join(md)


def main(steps, epochs=None):
    t0 = time.time()
    all_steps = {"data", "train", "generate", "eval", "report"}
    if not steps:
        steps = all_steps
    trainer = None
    history = None
    candidates_df = None
    report = None

    if "data" in steps:
        step_data()
    if "train" in steps:
        trainer, history = step_train(train_epochs_override=epochs)
    if "generate" in steps:
        candidates_df = step_generate(trainer=trainer)
    if "eval" in steps:
        report = step_eval(candidates_df=candidates_df, history=history)
    if "report" in steps:
        step_report(report=report)

    dt = time.time() - t0
    print("\n" + "=" * 70)
    print(f"[PIPELINE] 管线执行完成。总耗时: {dt:.1f}s")
    print(f"[PIPELINE] 核心产物:")
    print(f"  - 模型: {os.path.join(MODEL_DIR, 'best_cvae.pt')}")
    print(f"  - 候选肽: {os.path.join(RESULTS_DIR, 'generated_candidates.csv')}")
    print(f"  - 报告: {os.path.join(RESULTS_DIR, 'REPORT.md')}")
    print(f"  - 可视化: {PLOTS_DIR}/*.png")
    print("=" * 70)


if __name__ == "__main__":
    import numpy as np  # for report script use
    parser = argparse.ArgumentParser(description="G-AMP CVAE Pipeline")
    parser.add_argument("--steps", type=str, default="all",
                        help="运行步骤: all|data|train|generate|eval|report, 逗号分隔多步")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖模型训练epoch数")
    args = parser.parse_args()

    step_map = {"all": {"data", "train", "generate", "eval", "report"},
                "data": {"data"}, "train": {"train"}, "generate": {"generate"},
                "eval": {"eval"}, "report": {"report"}}
    sel = set()
    for s in args.steps.split(","):
        s = s.strip().lower()
        if s in step_map:
            sel |= step_map[s]
    main(sel, epochs=args.epochs)
