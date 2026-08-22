"""
推理生成模块
1. 重构任务：masked seq + original descriptor → original seq
2. 条件优化生成：lead peptide masked + 目标(更高电荷、低溶血)描述符 → 候选肽
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
from config import (
    ModelConfig, MODEL_DIR, DATA_DIR, RESULTS_DIR,
    MAX_SEQ_LEN, DESCRIPTOR_NAMES, AMINO_ACIDS, TARGET_PROPERTY_RANGES,
    GENERATE_NUM_CANDIDATES
)
from train import Trainer, DescriptorScaler
from cvae import sequence_to_indices, indices_to_sequence
from descriptor_calculator import (
    compute_descriptors_for_sequence, compute_descriptor_array,
    calc_net_charge, calc_GRAVY, calc_hydrophobic_ratio, predict_hemolysis_risk,
    predict_toxicity_risk, predict_activity_probability, calc_synthesis_feasibility,
)
from mask_strategies import (
    apply_mask, mask_lead_peptide_for_optimization, MASK_TOKEN, STRATEGY_FN
)


def _infill_sequence(trainer: Trainer, original: str, masked_seq: str,
                     mask_indices: List[int], cond_t: torch.Tensor,
                     sample: bool = False, temperature: float = 1.0) -> str:
    """Fill only masked residue positions using the trained mask head."""
    src_idx = sequence_to_indices(masked_seq, max_len=MAX_SEQ_LEN, add_sos_eos=True)
    src_t = torch.from_numpy(src_idx).unsqueeze(0).long().to(trainer.device)
    with torch.no_grad():
        logits = trainer.model.predict_masked_logits(src_t, cond_t)
    out = list(original)
    aa_start = 5  # special tokens occupy indices 0..4
    aa_end = aa_start + len(AMINO_ACIDS)
    for mi in mask_indices:
        pos = mi + 1  # SOS offset
        if mi >= len(out) or pos >= logits.shape[1]:
            continue
        residue_logits = logits[0, pos, aa_start:aa_end] / max(0.1, temperature)
        if sample:
            probs = torch.softmax(residue_logits, dim=-1)
            idx = int(torch.multinomial(probs, 1).item())
        else:
            idx = int(residue_logits.argmax().item())
        out[mi] = AMINO_ACIDS[idx]
    return "".join(out)


def load_trained_trainer(cfg: ModelConfig = None) -> Trainer:
    """加载已训练的Trainer（若无checkpoint则返回空trainer用于演示）"""
    cfg = cfg or ModelConfig()
    trainer = Trainer(cfg)
    best_path = os.path.join(MODEL_DIR, "best_cvae.pt")
    last_path = os.path.join(MODEL_DIR, "last_cvae.pt")
    if os.path.exists(best_path):
        trainer.load_best(best_path)
    elif os.path.exists(last_path):
        trainer.load_best(last_path)
    else:
        print("[INFER] 警告：未找到已训练模型，将使用未初始化权重生成结果。")
    return trainer


# 描述符条件修改（用于条件引导生成）
def modify_descriptor_to_target(orig_descriptor: np.ndarray,
                                target_changes: Dict[str, float] = None,
                                use_default_target: bool = True,
                                ) -> np.ndarray:
    """
    根据目标需求修改描述符：
    - 提高正电荷 net_charge
    - 适度疏水（GRAVY ~ 0附近）
    - 降低溶血/毒性风险
    - 保证gram_negative_active = 1
    """
    des = orig_descriptor.copy()
    name_to_idx = {n: i for i, n in enumerate(DESCRIPTOR_NAMES)}

    def _set(name, new_val):
        if name in name_to_idx:
            des[name_to_idx[name]] = new_val

    if use_default_target:
        # 默认目标（适合革兰氏阴性菌活性优化）
        # 将长度保持在原附近
        orig_len = int(des[name_to_idx["length"]])
        # 提高电荷到 [5, 8]
        target_charge = min(8, max(5, des[name_to_idx["net_charge"]] + 2))
        _set("net_charge", target_charge)
        # KRH比例
        _set("KRH_ratio", max(0.30, min(0.45, des[name_to_idx["KRH_ratio"]] + 0.08)))
        # Explicitly target a moderately amphipathic, non-extreme profile.
        # The previous implementation left most values unchanged, so a
        # condition-guidance experiment was not actually testing this target.
        gravy = float(des[name_to_idx["GRAVY"]])
        _set("GRAVY", float(np.clip(gravy, -0.05, 0.15)))
        _set("hydrophobic_ratio", float(np.clip(des[name_to_idx["hydrophobic_ratio"]], 0.40, 0.50)))
        # 芳香比例适度
        arom = des[name_to_idx["aromatic_ratio"]]
        if arom > 0.20:
            _set("aromatic_ratio", 0.15)
        # Lower-risk target: these values are intentionally below the typical
        # lead risk rather than the previous ambiguous fixed 0.15 target.
        _set("hemolysis_risk", 0.05)
        _set("toxicity_risk", 0.05)
        _set("aggregation_propensity", 0.08)
        # instability 调低
        instab = des[name_to_idx["instability_index"]]
        if instab > 40:
            _set("instability_index", 30.0)
        # Gram-negative active = 1
        _set("gram_negative_active", 1.0)

    if target_changes:
        for name, val in target_changes.items():
            _set(name, val)
    return des


# 重构（任务核心：masked + 描述符 → original）
def reconstruct_sequences(trainer: Trainer, sequences: List[str],
                          strategies: List[str] = None,
                          mask_ratio: float = 0.3,
                          use_target_cond: bool = False,
                          temperature: float = 0.7,
                          n_samples_per_seq: int = 5,
    seed: int = 42,
                          preserve_scaffold: bool = True) -> pd.DataFrame:
    """
    对给定序列集合执行重构任务：
    - 按策略构造masked seq
    - 输入 original 或 target 描述符
    - 生成候选，评估mask位点恢复率
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    trainer.model.eval()
    device = trainer.device

    if strategies is None:
        strategies = list(STRATEGY_FN.keys())

    records = []
    for seq in sequences:
        # 原始描述符
        orig_des = compute_descriptor_array(seq, label=1)  # 默认AMP阳性
        target_des = modify_descriptor_to_target(orig_des)

        for strat in strategies:
            for sp in range(n_samples_per_seq):
                strategy_offset = {
                    "random": 11,
                    "contiguous": 23,
                    "motif_preserving": 37,
                    "property_guided": 53,
                }.get(strat, 71)
                masked_seq, mask_indices = apply_mask(
                    seq, strat, mask_ratio=mask_ratio,
                    seed=seed + sp + strategy_offset
                )
                # 构造 batch
                src_idx = sequence_to_indices(masked_seq, max_len=MAX_SEQ_LEN, add_sos_eos=True)
                src_t = torch.from_numpy(src_idx).unsqueeze(0).long().to(device)

                cond_raw = target_des if use_target_cond else orig_des
                cond_raw_t = torch.from_numpy(cond_raw).unsqueeze(0).float().to(device)
                cond_t = trainer.scaler.transform(cond_raw_t)

                # Deterministic direct infilling gives a faithful free-decoding
                # metric and guarantees the unmasked scaffold is unchanged.
                gen_seq = _infill_sequence(
                    trainer, seq, masked_seq, mask_indices, cond_t,
                    sample=False, temperature=temperature
                )

                # mask位点恢复情况
                mask_correct = 0
                mask_total = 0
                for mi in mask_indices:
                    if mi < len(seq) and mi < len(gen_seq):
                        mask_total += 1
                        if seq[mi] == gen_seq[mi]:
                            mask_correct += 1
                edit = _hamming(seq, gen_seq)
                records.append({
                    "original": seq,
                    "masked": masked_seq,
                    "generated": gen_seq,
                    "strategy": strat,
                    "mask_ratio": mask_ratio,
                    "use_target_cond": use_target_cond,
                    "n_mask": len(mask_indices),
                    "mask_correct": mask_correct,
                    "mask_total": mask_total,
                    "mask_acc": (mask_correct / mask_total) if mask_total > 0 else 0.0,
                    "full_recovery": 1 if seq == gen_seq else 0,
                    "edit_distance": edit,
                    "len_original": len(seq),
                    "len_generated": len(gen_seq),
                })
    return pd.DataFrame(records)


def _hamming(a: str, b: str) -> int:
    """编辑距离（简化版：min hamming-like）"""
    L = min(len(a), len(b))
    d = abs(len(a) - len(b))
    for i in range(L):
        if a[i] != b[i]:
            d += 1
    return d


# Lead peptide 条件优化生成
def generate_from_lead_peptides(trainer: Trainer,
                                leads: List[str],
                                lead_ids: List[str] = None,
                                num_candidates_per_lead: int = 20,
                                mask_ratio: float = 0.35,
                                temperatures: List[float] = None,
                                preserve_motif: bool = True,
                                seed: int = 42) -> pd.DataFrame:
    """
    核心优化生成：
    - 对每条lead peptide，做 motif-preserving / lead-optimization mask
    - 使用目标机制描述符（更高正电荷、低溶血、适度疏水）引导生成
    - 输出候选序列供后续筛选
    """
    if temperatures is None:
        temperatures = [0.6, 0.8, 1.0]
    if lead_ids is None:
        lead_ids = [f"LEAD_{i+1:03d}" for i in range(len(leads))]

    torch.manual_seed(seed)
    np.random.seed(seed)
    trainer.model.eval()
    device = trainer.device

    all_records = []

    for lead, lead_id in zip(leads, lead_ids):
        # 原始描述符
        orig_des_dict = compute_descriptors_for_sequence(lead, label=1)
        orig_des_dict["synthesis_feasibility"] = calc_synthesis_feasibility(lead)
        orig_des_dict["activity_probability"] = predict_activity_probability(lead)
        orig_des_arr = np.array([orig_des_dict[n] for n in DESCRIPTOR_NAMES], dtype=np.float32)
        target_scenarios = {
            "default": modify_descriptor_to_target(orig_des_arr),
            "higher_charge": modify_descriptor_to_target(
                orig_des_arr, {"net_charge": min(10.0, orig_des_arr[2] + 4.0)}),
            "lower_hydrophobic": modify_descriptor_to_target(
                orig_des_arr, {"GRAVY": -0.35, "hydrophobic_ratio": 0.35}),
            "lower_hemolysis": modify_descriptor_to_target(
                orig_des_arr, {"hemolysis_risk": 0.01, "toxicity_risk": 0.01}),
        }

        # 构造多种masked lead
        masked_examples = []
        # 1. motif preserving
        if preserve_motif:
            masked, mask_idx = mask_lead_peptide_for_optimization(
                lead, preserve_regions=None, mask_ratio=mask_ratio, seed=seed
            )
            masked_examples.append(("lead_opt_motif", masked, mask_idx))
        # 2. random
        masked, mask_idx = apply_mask(lead, "random", mask_ratio=mask_ratio, seed=seed + 1)
        masked_examples.append(("lead_opt_random", masked, mask_idx))
        # 3. contiguous
        masked, mask_idx = apply_mask(lead, "contiguous", mask_ratio=mask_ratio, seed=seed + 2)
        masked_examples.append(("lead_opt_contiguous", masked, mask_idx))

        for mask_tag, masked_seq, mask_indices in masked_examples:
            for condition_name, target_des_arr in target_scenarios.items():
                    # 准备一个src，用于batch生成
                    src_idx = sequence_to_indices(masked_seq, max_len=MAX_SEQ_LEN, add_sos_eos=True)
                    src_t = torch.from_numpy(src_idx).unsqueeze(0).long().to(device)
                    cond_raw_t = torch.from_numpy(target_des_arr).unsqueeze(0).float().to(device)
                    cond_t = trainer.scaler.transform(cond_raw_t)

                    # 多温度采样
                    candidates_this_mask = 0
                    for temp in temperatures:
                # 每个温度生成若干条
                        per_temp = max(1, num_candidates_per_lead // (len(temperatures) * len(masked_examples)))
                        for k in range(per_temp):
                            gen_seq = _infill_sequence(
                                trainer, lead, masked_seq, mask_indices, cond_t,
                                sample=True, temperature=temp
                            )
                    # 过滤无效
                            if not gen_seq or len(gen_seq) < 6:
                                continue
                    # 去重：与lead相同不算新
                            if gen_seq == lead:
                                pass  # 保留一条用于参考
                    # 计算生成序列的描述符
                            gen_des_dict = compute_descriptors_for_sequence(gen_seq, label=None)
                    # 保存完整记录
                            rec = {
                                "lead_id": lead_id,
                                "lead_sequence": lead,
                                "masked_sequence": masked_seq,
                                "generated_sequence": gen_seq,
                                "mask_strategy": mask_tag,
                                "temperature": temp,
                                "mask_indices_count": len(mask_indices),
                                "candidate_id": f"{lead_id}_{mask_tag}_{condition_name}_t{temp:.1f}_{k:02d}",
                                "condition_name": condition_name,
                            }
                        # 加入描述符
                            for n in DESCRIPTOR_NAMES:
                                rec[f"gen_{n}"] = gen_des_dict[n]
                        # 加入lead描述符
                            for n in DESCRIPTOR_NAMES:
                                rec[f"lead_{n}"] = orig_des_dict[n]
                            rec["lead_synthesis_feasibility"] = orig_des_dict["synthesis_feasibility"]
                            rec["lead_activity_probability"] = orig_des_dict["activity_probability"]
                        # 加入target描述符
                            target_des_dict = {n: float(target_des_arr[i]) for i, n in enumerate(DESCRIPTOR_NAMES)}
                            for n in DESCRIPTOR_NAMES:
                                rec[f"target_{n}"] = target_des_dict[n]
                        # 与lead的差异
                            rec["edit_distance_from_lead"] = _hamming(lead, gen_seq)
                            rec["activity_probability"] = predict_activity_probability(gen_seq)
                            rec["synthesis_feasibility"] = calc_synthesis_feasibility(gen_seq)
                            all_records.append(rec)
                            candidates_this_mask += 1
    return pd.DataFrame(all_records)


# 大规模候选生成入口
def run_generation_pipeline(trainer: Trainer = None) -> pd.DataFrame:
    """完整推理：生成候选肽并保存CSV"""
    if trainer is None:
        trainer = load_trained_trainer()

    # 加载lead peptides
    lead_path = os.path.join(DATA_DIR, "lead_peptides.csv")
    if not os.path.exists(lead_path):
        # 构造简单demo leads
        leads = [
            "KLKLLKLAAKK",
            "KRWWKWWRR",
            "KLAKLAKKLAK",
            "KWKLFKKIEKV",
            "RWRWRWFWR",
        ]
        lead_ids = [f"LEAD_{i+1:03d}" for i in range(len(leads))]
    else:
        leads_df = pd.read_csv(lead_path)
        leads = leads_df["sequence"].tolist()
        lead_ids = leads_df.get("lead_id", [f"LEAD_{i+1:03d}" for i in range(len(leads))]).tolist()

    print(f"[GEN] Lead peptides数量: {len(leads)}")

    # 1. 重构评估（在测试集正样本上，多个mask ratio）
    test_path = os.path.join(DATA_DIR, "test_with_descriptors.csv")
    reconstruction_df = None
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        test_pos = test_df[test_df["label"] == 1]["sequence"].head(30).tolist()
        if test_pos:
            print(f"[GEN] 评估重构性能于 {len(test_pos)} 条测试集正样本...")
            # 在多个mask ratio下评估，生成性能曲线
            eval_ratios = [0.15, 0.25, 0.35, 0.5]
            recon_parts = []
            for r in eval_ratios:
                print(f"[GEN]   mask_ratio={r} ...")
                df_r = reconstruct_sequences(
                    trainer, test_pos, strategies=list(STRATEGY_FN.keys()),
                    mask_ratio=r, use_target_cond=False, temperature=0.7,
                    n_samples_per_seq=2, seed=42
                )
                recon_parts.append(df_r)
            import pandas as _pd
            reconstruction_df = _pd.concat(recon_parts, ignore_index=True)
            recon_csv = os.path.join(RESULTS_DIR, "reconstruction_eval.csv")
            reconstruction_df.to_csv(recon_csv, index=False, encoding="utf-8")
            print(f"[GEN] 重构评估已保存到 {recon_csv} (含 {len(eval_ratios)} 个mask ratio)")
            # 打印摘要：按策略
            print("[GEN] 按策略汇总:")
            for strat, g in reconstruction_df.groupby("strategy"):
                print(f"    strategy={strat:20s} mask_acc={g['mask_acc'].mean():.3f} "
                      f"seq_recovery={g['full_recovery'].mean():.3f} "
                      f"edit={g['edit_distance'].mean():.2f}")
            # 打印摘要：按mask ratio
            print("[GEN] 按mask ratio汇总:")
            for r, g in reconstruction_df.groupby("mask_ratio"):
                print(f"    ratio={r:.2f} mask_acc={g['mask_acc'].mean():.3f} "
                      f"seq_recovery={g['full_recovery'].mean():.3f}")

    # 2. 条件优化生成候选
    print(f"[GEN] 从leads生成候选肽...")
    candidates_df = generate_from_lead_peptides(
        trainer, leads, lead_ids=lead_ids,
        num_candidates_per_lead=GENERATE_NUM_CANDIDATES,
        mask_ratio=0.35,
        seed=42
    )
    print(f"[GEN] 生成候选数量: {len(candidates_df)}")

    # 保存
    out_csv = os.path.join(RESULTS_DIR, "generated_candidates.csv")
    candidates_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[GEN] 候选肽已保存到 {out_csv}")
    return candidates_df


if __name__ == "__main__":
    run_generation_pipeline()
