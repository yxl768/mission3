"""
条件变分自编码器（CVAE）模型
- Encoder: BiGRU 编码 masked sequence + 机制描述符条件
- Latent: mean, logvar (VAE)
- Decoder: GRU 解码，使用 latent + descriptor 作为条件
用于抗革兰氏阴性菌AMP的 masked sequence → original sequence 重构与条件生成
"""
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig, AA_TO_IDX, IDX_TO_AA, MAX_SEQ_LEN, DESCRIPTOR_NAMES

SOS_IDX = AA_TO_IDX["[SOS]"]
EOS_IDX = AA_TO_IDX["[EOS]"]
PAD_IDX = AA_TO_IDX["[PAD]"]
MASK_IDX = AA_TO_IDX["[MASK]"]


# ===================== 工具函数 =====================
def sequence_to_indices(seq: str, max_len: int = MAX_SEQ_LEN,
                        add_sos_eos: bool = True) -> np.ndarray:
    """序列转索引，补PAD到max_len"""
    indices = []
    if add_sos_eos:
        indices.append(SOS_IDX)
    for c in seq:
        indices.append(AA_TO_IDX.get(c, AA_TO_IDX["[UNK]"]))
    if add_sos_eos:
        indices.append(EOS_IDX)
    # padding
    while len(indices) < max_len + 2:  # +2 for SOS/EOS
        indices.append(PAD_IDX)
    indices = indices[:max_len + 2]
    return np.array(indices, dtype=np.int64)


def indices_to_sequence(indices: list, stop_at_eos: bool = True) -> str:
    """索引转序列（跳过SOS/EOS/PAD/MASK）"""
    out = []
    for idx in indices:
        if idx == SOS_IDX:
            continue
        if idx in (EOS_IDX, PAD_IDX) and stop_at_eos:
            break
        aa = IDX_TO_AA.get(int(idx), "")
        if aa and aa not in ("[PAD]", "[MASK]", "[SOS]", "[EOS]", "[UNK]"):
            out.append(aa)
    return "".join(out)


# ===================== Encoder =====================
class Encoder(nn.Module):
    """BiGRU Encoder：编码 masked sequence + descriptor 条件"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(cfg.dropout)

        # descriptor 条件投影
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.descriptor_dim, cfg.embed_dim),
            nn.LayerNorm(cfg.embed_dim),
            nn.GELU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )

        self.rnn = nn.GRU(
            input_size=cfg.embed_dim * 2,  # token emb + cond emb broadcast
            hidden_size=cfg.encoder_hidden,
            num_layers=cfg.encoder_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.encoder_layers > 1 else 0.0,
        )

        enc_out_dim = cfg.encoder_hidden * 2  # BiGRU

        # VAE latent
        self.fc_mu = nn.Linear(enc_out_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, cfg.latent_dim)

        # 使用attention池化得到固定长度向量
        self.attn = nn.Sequential(
            nn.Linear(enc_out_dim, enc_out_dim // 2),
            nn.Tanh(),
            nn.Linear(enc_out_dim // 2, 1, bias=False),
        )

    def forward(self, src_idx: torch.Tensor, cond: torch.Tensor, src_len: torch.Tensor = None):
        """
        src_idx: [B, T]
        cond: [B, D_descriptor]
        """
        B, T = src_idx.shape

        emb = self.embedding(src_idx)  # [B,T,E]
        cond_emb = self.cond_proj(cond).unsqueeze(1).expand(-1, T, -1)  # [B,T,E]
        rnn_input = torch.cat([emb, cond_emb], dim=-1)  # [B,T,2E]
        rnn_input = self.dropout(rnn_input)

        # pack for speed
        if src_len is not None:
            src_len_cpu = src_len.detach().cpu().long()
            packed = nn.utils.rnn.pack_padded_sequence(
                rnn_input, src_len_cpu, batch_first=True, enforce_sorted=False
            )
            out_packed, _ = self.rnn(packed)
            enc_out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=T)
        else:
            enc_out, _ = self.rnn(rnn_input)  # [B,T,2H]

        # attention pooling
        attn_scores = self.attn(enc_out).squeeze(-1)  # [B,T]
        # mask PAD
        pad_mask = (src_idx == PAD_IDX).bool()
        attn_scores = attn_scores.masked_fill(pad_mask, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = torch.sum(enc_out * attn_weights.unsqueeze(-1), dim=1)  # [B,2H]

        mu = self.fc_mu(pooled)  # [B,Z]
        logvar = self.fc_logvar(pooled)  # [B,Z]
        return mu, logvar, enc_out, attn_weights


# ===================== Decoder =====================
class Decoder(nn.Module):
    """Autoregressive GRU Decoder，使用 latent + descriptor 作为条件"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(cfg.dropout)

        # latent + descriptor → decoder init hidden
        self.latent_cond_proj = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.descriptor_dim, cfg.decoder_hidden * cfg.decoder_layers),
            nn.LayerNorm(cfg.decoder_hidden * cfg.decoder_layers),
            nn.GELU(),
        )

        # GRU input: token_emb + latent_broadcast + descriptor_broadcast
        self.rnn = nn.GRU(
            input_size=cfg.embed_dim + cfg.latent_dim + cfg.descriptor_dim,
            hidden_size=cfg.decoder_hidden,
            num_layers=cfg.decoder_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.decoder_layers > 1 else 0.0,
        )

        self.out_proj = nn.Sequential(
            nn.Linear(cfg.decoder_hidden, cfg.decoder_hidden // 2),
            nn.GELU(),
            nn.Linear(cfg.decoder_hidden // 2, cfg.vocab_size),
        )

    def _init_hidden(self, latent_z: torch.Tensor, cond: torch.Tensor):
        """用latent + 条件初始化解码器隐状态"""
        B = latent_z.shape[0]
        h = self.latent_cond_proj(torch.cat([latent_z, cond], dim=-1))  # [B, D*L]
        h = h.view(self.cfg.decoder_layers, B, self.cfg.decoder_hidden).contiguous()
        return h

    def forward(self, tgt_idx: torch.Tensor, latent_z: torch.Tensor, cond: torch.Tensor,
                init_hidden: torch.Tensor = None):
        """
        teacher forcing：接收tgt_idx (带SOS)，输出logits（不含最后一个EOS位置）
        """
        B, T = tgt_idx.shape
        emb = self.embedding(tgt_idx)  # [B,T,E]
        z_b = latent_z.unsqueeze(1).expand(-1, T, -1)
        c_b = cond.unsqueeze(1).expand(-1, T, -1)
        rnn_in = torch.cat([emb, z_b, c_b], dim=-1)
        rnn_in = self.dropout(rnn_in)

        if init_hidden is None:
            init_hidden = self._init_hidden(latent_z, cond)

        dec_out, _ = self.rnn(rnn_in, init_hidden)  # [B,T,H]
        logits = self.out_proj(dec_out)  # [B,T,V]
        return logits

    def generate_step(self, prev_token: torch.Tensor, prev_hidden: torch.Tensor,
                      latent_z: torch.Tensor, cond: torch.Tensor):
        """自回归单步生成（用于推理）"""
        B = prev_token.shape[0]
        emb = self.embedding(prev_token)  # [B,1,E]
        z_b = latent_z.unsqueeze(1)
        c_b = cond.unsqueeze(1)
        rnn_in = torch.cat([emb, z_b, c_b], dim=-1)
        rnn_in = self.dropout(rnn_in)

        out, new_hidden = self.rnn(rnn_in, prev_hidden)
        logits = self.out_proj(out).squeeze(1)  # [B,V]
        return logits, new_hidden


# ===================== CVAE 主模型 =====================
class ConditionalVAE(nn.Module):
    """条件VAE：机制描述符 + masked sequence → original sequence"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, src_idx: torch.Tensor, tgt_idx: torch.Tensor,
                cond: torch.Tensor, src_len: torch.Tensor = None):
        """
        返回：
          logits: [B,T,V]
          mu, logvar: [B,Z]
          latent_z: [B,Z]
        """
        mu, logvar, _, _ = self.encoder(src_idx, cond, src_len)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(tgt_idx, z, cond)
        return logits, mu, logvar, z

    # ===================== 损失函数 =====================
    def compute_loss(self, logits: torch.Tensor, tgt_idx: torch.Tensor,
                     mu: torch.Tensor, logvar: torch.Tensor,
                     kl_weight: float = 0.05, mask_only: bool = False,
                     src_masked_idx: torch.Tensor = None):
        """
        损失 = 重建交叉熵 + KL散度
        如果mask_only=True，只对被MASK的token位置计算重建损失（训练更稳定）
        """
        B, T, V = logits.shape
        # tgt_idx形状 [B, T+1_with_SOS]？这里假设tgt_idx = [SOS x1 x2 ... xn EOS PAD]，长度=T
        # logits对应 [SOS->x1, x1->x2, ..., x_{n-1}->xn, xn->EOS...]，与tgt_idx[:,1:]对齐
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tgt_idx[:, 1:].contiguous()

        # CE
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=PAD_IDX,
            reduction="none",
        ).view(B, T - 1)

        if mask_only and src_masked_idx is not None:
            # src_masked_idx: [B, T_src], 1=被MASK位置；但需要与shift_labels对齐
            # 简化：使用非PAD的token平均损失（因src与tgt长度不一定严格对应）
            loss_mask = (shift_labels != PAD_IDX).float()
            recon_loss = (ce_loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)
        else:
            loss_mask = (shift_labels != PAD_IDX).float()
            recon_loss = (ce_loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)

        # KL(N(mu,var) || N(0,1))
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_loss = kl.mean()

        total_loss = recon_loss + kl_weight * kl_loss
        return {
            "total": total_loss,
            "recon": recon_loss,
            "kl": kl_loss,
        }

    # ===================== 推理生成 =====================
    @torch.no_grad()
    def reconstruct_or_generate(self, src_idx: torch.Tensor, cond: torch.Tensor,
                                max_len: int = None, temperature: float = 0.8,
                                top_k: int = 40, sample_z: bool = True,
                                custom_z: torch.Tensor = None):
        """
        推理：给定 masked sequence + 条件 → 生成完整序列
        支持 custom_z 用于插值/条件改变
        """
        self.eval()
        B = src_idx.shape[0]
        device = next(self.parameters()).device
        if max_len is None:
            max_len = self.cfg.max_generate_len + 2

        # 编码
        mu, logvar, _, _ = self.encoder(src_idx, cond)
        if custom_z is not None:
            z = custom_z.to(device)
        elif sample_z:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        # 解码（自回归）
        hidden = self.decoder._init_hidden(z, cond)
        prev = torch.full((B,), SOS_IDX, dtype=torch.long, device=device).unsqueeze(1)
        generated = [prev.squeeze(1)]

        for step in range(max_len - 1):
            logits, hidden = self.decoder.generate_step(prev, hidden, z, cond)
            # temperature
            logits = logits / max(0.1, temperature)
            # top-k sampling
            if top_k > 0 and top_k < logits.shape[-1]:
                top_v, top_i = torch.topk(logits, top_k, dim=-1)
                probs = F.softmax(top_v, dim=-1)
                sampled_rel = torch.multinomial(probs, num_samples=1).squeeze(-1)
                next_token = torch.gather(top_i, 1, sampled_rel.unsqueeze(1)).squeeze(1)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            generated.append(next_token)
            prev = next_token.unsqueeze(1)

            # 全部EOS则提前终止（可选）
            if (next_token == EOS_IDX).all():
                break

        generated = torch.stack(generated, dim=1)  # [B, T]
        return generated, z


if __name__ == "__main__":
    # 简单前向测试
    cfg = ModelConfig()
    model = ConditionalVAE(cfg)
    B, T = 2, MAX_SEQ_LEN + 2
    src = torch.randint(0, cfg.vocab_size, (B, T))
    tgt = torch.randint(0, cfg.vocab_size, (B, T))
    cond = torch.randn(B, cfg.descriptor_dim)

    logits, mu, logvar, z = model(src, tgt, cond)
    losses = model.compute_loss(logits, tgt, mu, logvar, kl_weight=0.05)
    print("[TEST] CVAE forward:")
    print(f"  logits: {tuple(logits.shape)}")
    print(f"  latent mu: {tuple(mu.shape)}")
    print(f"  total_loss: {losses['total'].item():.4f}  recon: {losses['recon'].item():.4f}  kl: {losses['kl'].item():.4f}")

    # 生成测试
    gen, _ = model.reconstruct_or_generate(src, cond, max_len=20, temperature=0.8)
    print(f"  generated indices shape: {tuple(gen.shape)}")
