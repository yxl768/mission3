"""
全局配置文件：革兰氏阴性菌抗感染AMP生成项目
"""
import os

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODEL_DIR = os.path.join(BASE_DIR, "checkpoints")

for d in [DATA_DIR, RESULTS_DIR, PLOTS_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ==================== 数据配置 ====================
# 标准20种氨基酸 + 特殊token
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["[PAD]", "[MASK]", "[SOS]", "[EOS]", "[UNK]"]
VOCAB = SPECIAL_TOKENS + AMINO_ACIDS

# token映射
AA_TO_IDX = {aa: i for i, aa in enumerate(VOCAB)}
IDX_TO_AA = {i: aa for i, aa in enumerate(VOCAB)}

# 序列长度限制
MIN_SEQ_LEN = 8
MAX_SEQ_LEN = 30
MASK_RATIOS = [0.15, 0.25, 0.35]

# 模拟数据集大小（无真实数据库时使用）
SYNTHETIC_DATA_SIZE = 3000

# ==================== 描述符配置 ====================
DESCRIPTOR_NAMES = [
    "length",
    "molecular_weight",
    "net_charge",
    "KRH_ratio",
    "positive_density",
    "GRAVY",
    "hydrophobic_ratio",
    "aliphatic_index",
    "hydrophobic_moment",
    "aromatic_ratio",
    "helix_propensity",
    "turn_propensity",
    "disorder_tendency",
    "instability_index",
    "Boman_index",
    "repeat_ratio",
    "hemolysis_risk",
    "toxicity_risk",
    "aggregation_propensity",
    "gram_negative_active",
]

# 革兰氏阴性菌活性AMP的目标性质范围（用于条件引导生成）
TARGET_PROPERTY_RANGES = {
    "net_charge": (3, 10),
    "GRAVY": (-1.0, 1.0),
    "hydrophobic_ratio": (0.3, 0.55),
    "KRH_ratio": (0.2, 0.45),
    "hemolysis_risk": (0.0, 0.4),
    "toxicity_risk": (0.0, 0.4),
    "length": (10, 25),
}

# ==================== Mask策略配置 ====================
MASK_STRATEGIES = ["random", "contiguous", "motif_preserving", "property_guided"]
# motif保护的残基类型
MOTIF_RESIDUES = set("KRFWY")
# contiguous mask长度范围
CONTIGUOUS_MASK_LEN_RANGE = (3, 8)

# ==================== 模型配置 ====================
class ModelConfig:
    # embedding
    vocab_size = len(VOCAB)
    embed_dim = 128
    descriptor_dim = len(DESCRIPTOR_NAMES)

    # encoder (BiGRU)
    encoder_hidden = 256
    encoder_layers = 2
    dropout = 0.2

    # latent
    latent_dim = 64

    # decoder (GRU)
    decoder_hidden = 256
    decoder_layers = 2

    # training
    batch_size = 64
    lr = 1e-3
    epochs = 80
    kl_weight = 0.05  # KL散度权重（warmup可调整）
    grad_clip = 1.0

    # generation
    max_generate_len = MAX_SEQ_LEN
    temperature = 0.8
    top_k = 40

# ==================== 评估配置 ====================
EVAL_NEAREST_NEIGHBOR_K = 3
GENERATE_NUM_CANDIDATES = 100
