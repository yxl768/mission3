"""
训练流程：
- 动态mask的Dataset（每个batch动态生成masked sequence
- CVAE训练：recon loss + KL
- 验证：masked token accuracy, full sequence recovery
- 保存checkpoint、训练曲线
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
from config import (
    ModelConfig, DATA_DIR, MODEL_DIR, PLOTS_DIR,
    DESCRIPTOR_NAMES, MAX_SEQ_LEN, AA_TO_IDX, MASK_RATIOS
)
from cvae import ConditionalVAE, sequence_to_indices, indices_to_sequence
from descriptor_calculator import compute_descriptor_array
from mask_strategies import (
    apply_mask, sample_strategy, sample_mask_ratio, STRATEGY_FN
)

MASK_TOKEN_IDX = AA_TO_IDX["[MASK]"]
PAD_IDX = AA_TO_IDX["[PAD]"]
SOS_IDX = AA_TO_IDX["[SOS]"]
EOS_IDX = AA_TO_IDX["[EOS]"]


# ===================== Dataset =====================
class AMPMaskedDataset(Dataset):
    """
    动态构造masked dataset
    每个样本在__getitem__时动态执行mask，
    增加训练时的数据多样性
    """

    def __init__(self, df: pd.DataFrame, descriptor_cols=None,
                 mode: str = "train", use_all_strategies: bool = True):
        """
        df: 包含 sequence, label, 以及descriptor列（若无则在初始化时计算）
        """
        self.mode = mode
        self.use_all_strategies = use_all_strategies
        self.sequences = df["sequence"].tolist()
        self.labels = df["label"].tolist() if "label" in df.columns else [0] * len(df)

        # 预计算描述符（防止重复计算
        if descriptor_cols is None:
            descriptor_cols = DESCRIPTOR_NAMES
        # 若df已有描述符列
        missing = [c for c in descriptor_cols if c not in df.columns]
        if missing:
            # 需要计算
            self.descriptors = np.stack([
                compute_descriptor_array(seq, lab)
                for seq, lab in zip(self.sequences, self.labels)
            ], axis=0).astype(np.float32)
        else:
            self.descriptors = df[descriptor_cols].to_numpy(dtype=np.float32)

        self.n = len(self.sequences)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        cond = self.descriptors[idx]

        if self.mode == "train":
            # 训练：随机策略 + 随机mask比例
            strategy = sample_strategy()
            mask_ratio = sample_mask_ratio()
        else:
            # 验证：固定策略（避免随机性干扰比较），使用random + 中等mask比例
            strategy = "random"
            mask_ratio = MASK_RATIOS[1]

        masked_seq, mask_indices = apply_mask(seq, strategy, mask_ratio=mask_ratio)

        # 转索引（带SOS/EOS）
        src_idx = sequence_to_indices(masked_seq, max_len=MAX_SEQ_LEN, add_sos_eos=True)
        tgt_idx = sequence_to_indices(seq, max_len=MAX_SEQ_LEN, add_sos_eos=True)

        # 记录被MASK的位置（src中），用于mask_only训练
        src_masked_flags = np.zeros_like(src_idx, dtype=np.float32)
        # mask_indices是原始seq的索引，src_idx索引偏移1位（前面加了SOS）
        for mi in mask_indices:
            mi_plus = mi + 1  # SOS之后
            if 0 <= mi_plus < len(src_idx):
                src_masked_flags[mi_plus] = 1.0

        # 实际序列长度（用于pack rnn）
        seq_len = min(MAX_SEQ_LEN + 2, len(seq) + 2)
        return {
            "src_idx": torch.from_numpy(src_idx).long(),
            "tgt_idx": torch.from_numpy(tgt_idx).long(),
            "cond": torch.from_numpy(cond).float(),
            "src_len": torch.tensor(seq_len, dtype=torch.long),
            "src_masked_flags": torch.from_numpy(src_masked_flags).float(),
            "original": seq,
            "masked": masked_seq,
            "strategy": strategy,
            "mask_indices": torch.tensor(mask_indices, dtype=torch.long) if mask_indices else torch.zeros(0, dtype=torch.long),
            "label": label,
        }


def collate_fn(batch):
    keys0 = batch[0]
    out = {}
    for k in ["src_idx", "tgt_idx", "cond", "src_len", "src_masked_flags"]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    # mask_indices list
    out["mask_indices"] = [b["mask_indices"] for b in batch]
    out["original"] = [b["original"] for b in batch]
    out["masked"] = [b["masked"] for b in batch]
    out["strategy"] = [b["strategy"] for b in batch]
    out["label"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return out


# ===================== 描述符归一化 =====================
class DescriptorScaler:
    """对描述符做z-score归一化，确保数值稳定"""

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, descriptors: np.ndarray):
        self.mean = descriptors.mean(axis=0)
        self.std = descriptors.std(axis=0) + 1e-8
        return self

    def transform(self, descriptors):
        if isinstance(descriptors, torch.Tensor):
            mean_t = torch.from_numpy(self.mean).to(descriptors.device).float()
            std_t = torch.from_numpy(self.std).to(descriptors.device).float()
            return (descriptors - mean_t) / std_t
        return (descriptors - self.mean) / self.std

    def inverse_transform(self, descriptors):
        if isinstance(descriptors, torch.Tensor):
            mean_t = torch.from_numpy(self.mean).to(descriptors.device).float()
            std_t = torch.from_numpy(self.std).to(descriptors.device).float()
            return descriptors * std_t + mean_t
        return descriptors * self.std + self.mean


# ===================== 训练器 =====================
class Trainer:
    def __init__(self, cfg: ModelConfig, scaler: DescriptorScaler = None, device: str = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConditionalVAE(cfg).to(self.device)
        self.scaler = scaler or DescriptorScaler()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.epochs)

        self.history = {"train_total": [], "train_recon": [], "train_kl": [],
                        "val_total": [], "val_recon": [], "val_kl": [],
                        "val_mask_acc": [], "val_seq_recovery": [],
                        "lr": []}
        self.best_val_loss = float("inf")
        self.best_model_path = None

    def _preprocess_cond(self, cond: torch.Tensor) -> torch.Tensor:
        return self.scaler.transform(cond)

    def _run_epoch(self, loader, training=True, epoch_num: int = 1):
        self.model.train(training)
        total_total = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        # 用于验证集评估指标
        all_mask_correct = 0
        all_mask_total = 0
        all_seq_correct = 0
        all_seq_total = 0

        pbar = tqdm(loader, desc=("TRAIN" if training else "VAL  "), leave=False, ncols=120)
        for batch in pbar:
            src = batch["src_idx"].to(self.device)
            tgt = batch["tgt_idx"].to(self.device)
            cond_raw = batch["cond"].to(self.device)
            src_len = batch["src_len"].to(self.device)
            cond = self._preprocess_cond(cond_raw)

            if training:
                self.optimizer.zero_grad()

            logits, mu, logvar, z, infill_logits = self.model(src, tgt, cond, src_len)
            # KL warmup
            # Warm up KL by training epoch, not by batch count.  The previous
            # implementation divided the completed-epoch count by the number
            # of batches, so a nominal 1/3-run warmup never reached its target
            # during an 80-epoch run.
            progress = min(1.0, max(0, epoch_num - 1) / max(1, self.cfg.epochs // 3))
            kl_w = self.cfg.kl_weight * (0.1 + 0.9 * progress)

            losses = self.model.compute_loss(
                logits, tgt, mu, logvar, kl_weight=kl_w,
                infill_logits=infill_logits,
                src_masked_flags=batch["src_masked_flags"].to(self.device),
            )
            loss = losses["total"]

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

            total_total += losses["total"].item()
            total_recon += losses["recon"].item()
            total_kl += losses["kl"].item()
            n_batches += 1

            pbar.set_postfix({
                "total": f"{losses['total'].item():.3f}",
                "recon": f"{losses['recon'].item():.3f}",
                "kl": f"{losses['kl'].item():.4f}",
            })

            # 验证：计算恢复率
            if not training:
                self.model.eval()
                with torch.no_grad():
                    # Evaluate the explicit infill head at the masked positions.
                    # This is a true free-decoding metric and does not use
                    # teacher-forced decoder inputs.
                    preds = infill_logits.argmax(dim=-1)
                    # mask token恢复率
                    for b in range(tgt.shape[0]):
                        orig_seq = batch["original"][b]
                        mask_indices_b = batch["mask_indices"][b].tolist()
                        orig_list = list(orig_seq)
                        gen_list = list(orig_list)
                        for mi in mask_indices_b:
                            pos = mi + 1  # SOS offset in token tensors
                            if 0 <= mi < len(orig_list) and pos < preds.shape[1]:
                                pred_idx = int(preds[b, pos].cpu())
                                pred_aa = indices_to_sequence([pred_idx], stop_at_eos=False)
                                if pred_aa:
                                    gen_list[mi] = pred_aa[0]
                                all_mask_total += 1
                                if orig_list[mi] == gen_list[mi]:
                                    all_mask_correct += 1
                        gen_seq = "".join(gen_list)
                        if orig_seq == gen_seq:
                            all_seq_correct += 1
                        all_seq_total += 1

        avg = lambda x: x / max(1, n_batches)
        metrics = {
            "total": avg(total_total),
            "recon": avg(total_recon),
            "kl": avg(total_kl),
            "mask_acc": (all_mask_correct / max(1, all_mask_total)) if not training else None,
            "seq_recovery": (all_seq_correct / max(1, all_seq_total)) if not training else None,
        }
        return metrics

    def fit(self, train_loader, val_loader):
        print(f"[TRAIN] 使用设备: {self.device}")
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            train_m = self._run_epoch(train_loader, training=True, epoch_num=epoch)
            val_m = self._run_epoch(val_loader, training=False, epoch_num=epoch)
            self.scheduler.step()
            t1 = time.time()

            self.history["train_total"].append(train_m["total"])
            self.history["train_recon"].append(train_m["recon"])
            self.history["train_kl"].append(train_m["kl"])
            self.history["val_total"].append(val_m["total"])
            self.history["val_recon"].append(val_m["recon"])
            self.history["val_kl"].append(val_m["kl"])
            self.history["val_mask_acc"].append(val_m["mask_acc"] or 0.0)
            self.history["val_seq_recovery"].append(val_m["seq_recovery"] or 0.0)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # Persist progress after every epoch so a long CPU run can be
            # inspected or recovered without losing completed epochs.
            progress_last_path = os.path.join(MODEL_DIR, "last_cvae.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_mean": self.scaler.mean,
                "scaler_std": self.scaler.std,
                "best_val_loss": self.best_val_loss,
            }, progress_last_path)
            with open(os.path.join(MODEL_DIR, "history.json"), "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)

            print(
                f"Ep {epoch:03d}/{self.cfg.epochs} | time{t1-t0:5.1f}s | "
                f"T total={train_m['total']:.3f} recon={train_m['recon']:.3f} kl={train_m['kl']:.4f} | "
                f"V total={val_m['total']:.3f} recon={val_m['recon']:.3f} kl={val_m['kl']:.4f} | "
                f"mask_acc={val_m['mask_acc']:.3f} seq_rec={val_m['seq_recovery']:.3f}"
            )

            # 保存最优
            if val_m["total"] < self.best_val_loss:
                self.best_val_loss = val_m["total"]
                self.best_model_path = os.path.join(MODEL_DIR, "best_cvae.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scaler_mean": self.scaler.mean,
                    "scaler_std": self.scaler.std,
                    "best_val_loss": self.best_val_loss,
                }, self.best_model_path)
                print(f"  [SAVE] best model saved -> {self.best_model_path}")

        # 保存最后模型 (the per-epoch save above makes this idempotent)
        last_path = os.path.join(MODEL_DIR, "last_cvae.pt")
        torch.save({
            "epoch": self.cfg.epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_mean": self.scaler.mean,
            "scaler_std": self.scaler.std,
        }, last_path)
        # 保存history
        hist_path = os.path.join(MODEL_DIR, "history.json")
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        print(f"[TRAIN] 训练结束。history 已保存到 {hist_path}")
        return self.history

    def load_best(self, path=None):
        path = path or self.best_model_path or os.path.join(MODEL_DIR, "best_cvae.pt")
        if not os.path.exists(path):
            print(f"[WARN] 找不到模型文件: {path}")
            return
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.scaler.mean = ckpt["scaler_mean"]
        self.scaler.std = ckpt["scaler_std"]
        print(f"[LOAD] 已加载模型: {path}")


# ===================== 主入口训练 =====================
def build_dataloaders(train_df, val_df, cfg: ModelConfig, scaler: DescriptorScaler = None):
    """构造 dataloader 并 fit scaler"""
    train_ds = AMPMaskedDataset(train_df, mode="train")
    val_ds = AMPMaskedDataset(val_df, mode="val")

    # fit scaler using training descriptors only
    if scaler is None:
        scaler = DescriptorScaler()
    scaler.fit(train_ds.descriptors)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )
    return train_loader, val_loader, scaler


def run_training_pipeline():
    """完整训练流程入口"""
    # Keep dynamic masking, DataLoader shuffling, and VAE sampling reproducible.
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    # 1. 加载数据
    train_path = os.path.join(DATA_DIR, "train_with_descriptors.csv")
    val_path = os.path.join(DATA_DIR, "val_with_descriptors.csv")
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print("[TRAIN] 找不到带描述符的数据集，先执行数据与描述符准备...")
        from data.generate_dataset import build_and_save_dataset
        from descriptor_calculator import save_descriptors
        train_df, val_df, test_df, lead_df = build_and_save_dataset()
        save_descriptors(train_df, val_df, test_df)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    print(f"[TRAIN] 数据加载完成 train={len(train_df)}, val={len(val_df)}")

    cfg = ModelConfig()
    train_loader, val_loader, scaler = build_dataloaders(train_df, val_df, cfg)

    trainer = Trainer(cfg, scaler=scaler)
    history = trainer.fit(train_loader, val_loader)
    return trainer, history


if __name__ == "__main__":
    run_training_pipeline()
