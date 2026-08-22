# 机制约束的抗革兰氏阴性菌多肽生成

本项目是《深度学习入门 5：生成模型》阶段实践，面向革兰氏阴性菌抗感染 AIDD 场景，实现

```text
机制/理化描述符 + masked peptide sequence -> original peptide sequence
```

模型训练阶段学习抗菌肽的 masked reconstruction；推理阶段保留 lead peptide 的非遮盖骨架，只在 `[MASK]` 位点进行补全或定向替换，并用机制条件、活性预测、溶血/毒性风险和新颖性进行候选筛选。当前交付物是可复现的 in-silico 原型，不能替代 MIC、HC50、细胞毒性或动物实验。

## 1. 项目目标与任务对应

| 任务要求 | 本项目实现 | 主要代码/产物 |
|---|---|---|
| 多肽序列 token 化 | 20 种标准氨基酸 + `[PAD]`、`[MASK]`、`[SOS]`、`[EOS]`、`[UNK]`；`[MASK]` 作为单个 token | [`config.py`](config.py) |
| 机制条件输入 | 20 维可解释描述符，覆盖电荷、疏水性、两亲性、结构倾向和风险 | [`descriptor_calculator.py`](descriptor_calculator.py) |
| masked reconstruction | 动态随机 mask，训练 CVAE 恢复原始序列；额外的 masked-token infill head 直接监督遮盖位点 | [`train.py`](train.py)、[`cvae.py`](cvae.py) |
| 多种研发场景 | random、contiguous、motif-preserving、property-guided 四类 mask | [`mask_strategies.py`](mask_strategies.py) |
| 条件优化生成 | 改变目标净电荷、KRH 比例、溶血/毒性风险等条件，固定 lead scaffold 生成候选 | [`generate.py`](generate.py) |
| 候选筛选 | valid、unique、novel、训练集最近邻相似度、活性/风险预测、condition fit 和综合分数 | [`evaluate.py`](evaluate.py)、[`validate_candidates.py`](validate_candidates.py) |
| 可复现实验 | 数据、训练、生成、评估和报告均可分步或一键执行 | [`run_pipeline.py`](run_pipeline.py) |

## 2. 快速开始

### 2.1 环境

建议 Python 3.8+。依赖见 [`requirements.txt`](requirements.txt)：PyTorch、NumPy、pandas、scikit-learn、SciPy、matplotlib、seaborn 和 tqdm。

```bash
pip install -r requirements.txt
```

### 2.2 一键运行

```bash
python run_pipeline.py
```

完整流程为：数据集构建与描述符计算 -> CVAE 训练 -> masked 重构和条件生成 -> 评估、绘图和报告输出。

### 2.3 分步运行

```bash
python run_pipeline.py --steps data
python run_pipeline.py --steps train
python run_pipeline.py --steps generate
python run_pipeline.py --steps eval
```

训练 epoch 可以临时覆盖：

```bash
python run_pipeline.py --steps train --epochs 80
```

已经生成的最新结果可以直接查看 [`results/REPORT.md`](results/REPORT.md)，但本 README 已包含核心实验报告内容。

## 3. 代码结构与运行产物

```text
mission3/
├── config.py                 # 词表、描述符、mask、模型和训练超参数
├── run_pipeline.py           # 一键/分步运行入口，并生成 results/REPORT.md
├── data/
│   ├── download_dbaasp.py    # DBAASP 数据下载/清洗入口
│   ├── generate_dataset.py   # 数据过滤、标签和相似性划分
│   ├── dbaasp_dataset.csv    # 当前落盘数据
│   ├── train.csv             # 训练序列
│   ├── val.csv               # 验证序列
│   ├── test.csv              # 测试序列
│   └── lead.csv              # 独立 lead/optimization 序列
├── descriptor_calculator.py  # 20 维机制/理化描述符
├── mask_strategies.py        # 四种 mask 策略
├── cvae.py                   # Conditional VAE Seq2Seq
├── train.py                  # Dataset、动态 mask、训练和验证
├── generate.py               # 自由补全、lead 局部优化和候选导出
├── evaluate.py               # 指标、候选排序和绘图
├── train_ml_models.py        # 活性/溶血预测器训练
├── validate_candidates.py    # 候选有效性检查
├── checkpoints/
│   ├── best_cvae.pt          # 最优 CVAE 权重
│   ├── last_cvae.pt          # 最后一个 epoch 权重
│   └── history.json          # 训练曲线数据
├── results/                  # CSV、JSON 和自动生成报告
├── plots/                    # 实验图
└── README.md
```

主要结果文件：

- [`results/evaluation_metrics.json`](results/evaluation_metrics.json)：重构、mask ratio、生成质量和条件引导指标。
- [`results/reconstruction_summary.csv`](results/reconstruction_summary.csv)：四类 mask 的测试结果。
- [`results/mask_ratio_summary.csv`](results/mask_ratio_summary.csv)：不同 mask ratio 的性能曲线数据。
- [`results/generated_candidates.csv`](results/generated_candidates.csv)：候选肽及描述符、预测器分数和来源信息。
- [`checkpoints/history.json`](checkpoints/history.json)：每个 epoch 的 train/validation loss、mask accuracy 和 sequence recovery。

## 4. 数据集构建与处理

### 4.1 数据来源和标签

项目优先读取 [`data/dbaasp_dataset.csv`](data/dbaasp_dataset.csv) 中的 DBAASP-derived 数据；原始文件不可用时，代码提供合成数据回退路径，便于教学环境运行。当前落盘数据共 2094 条记录、2090 条不同序列，其中革兰氏阴性菌活性正样本 1843 条、非活性负样本 251 条；划分前去掉 4 条重复序列。序列过滤范围为 8--30 aa，并统一为标准 20 种氨基酸。

标签处理包括：

1. 优先使用 Gram-negative activity 记录。
2. 具有明确活性记录的序列标为正样本；非活性或弱活性记录作为负样本。
3. 若有 MIC/HC50 信息，保留到数据表用于后续分析；模型中的 activity、hemolysis 和 toxicity 预测仍属于数据驱动的近似值。

### 4.2 数据划分与泄漏控制

最终数据划分如下：

| Split | 样本数 | 说明 |
|---|---:|---|
| Train | 1245 | 阳性 1087，阴性 158 |
| Validation | 334 | 用于 model selection |
| Test | 289 | 用于独立重构和 mask ratio 评估 |
| Lead/optimization | 222 | 阳性 193；只用于局部优化生成，不参与 CVAE 训练 |

划分先按序列去重，再在正、负标签内部各自采用 80% identity 的 CD-HIT 风格贪心聚类，将簇整体分配到 train/validation/test/lead；四个 split 不共享完全相同序列。由于正负标签是分别聚类的，该实现不能严格排除跨标签高相似序列跨 split，正式研究仍应使用全数据统一聚类并做外部 identity 审计。数据处理入口是 [`data/generate_dataset.py`](data/generate_dataset.py)，描述符落盘入口是 [`descriptor_calculator.py`](descriptor_calculator.py)。这是一种教学原型的近似聚类实现，不等同于正式 CD-HIT 命令行结果。

### 4.3 机制相关描述符

模型条件向量共 20 维，按作用机制分组如下：

| 维度类别 | 描述符 |
|---|---|
| Sequence scale | `length`, `molecular_weight`, `instability_index` |
| Charge/cationic | `net_charge`, `KRH_ratio`, `positive_density` |
| Hydrophobic/amphipathic | `GRAVY`, `hydrophobic_ratio`, `aliphatic_index`, `hydrophobic_moment` |
| Composition/motif | `aromatic_ratio`, `repeat_ratio` |
| Structure/flexibility | `helix_propensity`, `turn_propensity`, `disorder_tendency`, `Boman_index` |
| Risk and activity | `hemolysis_risk`, `toxicity_risk`, `aggregation_propensity`, `gram_negative_active` |

这些特征分别对应 LPS 结合、膜表面富集、膜插入/扰动、两亲性平衡以及溶血、毒性和聚集风险。条件生成使用的目标范围包括：净电荷 3--10、疏水残基比例 0.30--0.55、KRH 比例 0.20--0.45、溶血/毒性风险不高于 0.4、长度 10--25 aa。

## 5. Mask 策略

训练时 Dataset 动态生成损坏序列；验证时固定策略和 mask ratio，避免随机性干扰比较。推理阶段只允许修改被 mask 的位置，未遮盖的 scaffold 会被确定性保留。

| 策略 | 研发含义 |
|---|---|
| `random`，15%--40% | 学习一般序列上下文和随机缺失补全 |
| `contiguous`，目标连续 3--8 aa（受 mask ratio/序列长度约束） | 模拟局部片段替换，难度较高 |
| `motif_preserving` | 保留 K/R-rich、W/F/Y 等核心 motif，遮盖周围可变位点 |
| `property_guided` | 优先遮盖高疏水、高芳香、负电荷或重复位点，模拟风险位点改造 |

## 6. 模型结构与训练方法

### 6.1 Conditional VAE Seq2Seq

```text
masked sequence indices
        │
Embedding + descriptor condition embedding
        │
2-layer BiGRU encoder, hidden=256, dropout=0.2
        │ attention pooling
        ├── mu, logvar ∈ R^64 -> reparameterization -> z ∈ R^64
        │
        └── z + descriptor -> 2-layer GRU decoder, hidden=256
                                      │
                         autoregressive sequence logits

encoder states -> masked-token infill head -> [MASK] 位点 logits
```

词表大小为 25（5 个特殊 token + 20 个氨基酸）。自回归 decoder 在训练中使用 teacher forcing；独立重构评估使用 greedy 解码，并将未遮盖 scaffold 直接写回输出。

### 6.2 损失函数

$$
\mathcal{L}=\mathcal{L}_{recon}+1.5\,\mathcal{L}_{masked}+\lambda_{KL}D_{KL}\left(q(z|x,c)\|\mathcal{N}(0,I)\right)
$$

- `recon loss`：完整序列自回归交叉熵，忽略 `[PAD]`。
- `masked loss`：只在真实 `[MASK]` 位点计算，直接优化局部补全能力。
- `KL loss`：约束潜变量分布；权重从 0.005 warm up 到 0.05，前 1/3 个 epoch 逐步增加。
- 优化器：AdamW，初始学习率 `1e-3`，CosineAnnealing 学习率调度，梯度裁剪 1.0，batch size 64，训练 80 epochs。

## 7. 训练曲线与训练结果

![训练曲线](plots/training_curves.png)

当前评估使用 [`checkpoints/best_cvae.pt`](checkpoints/best_cvae.pt)，其中保存的最佳 checkpoint epoch 为 **73**；[`checkpoints/last_cvae.pt`](checkpoints/last_cvae.pt) 为完整训练的 epoch 80，[`checkpoints/history.json`](checkpoints/history.json) 含本次新 split 的完整 80 epoch 历史，训练、重构和生成结果来自同一轮实验。

| 指标 | 结果 |
|---|---:|
| 最优 validation total loss | 4.023，epoch 73 |
| 最终 validation total loss | 4.078 |
| 最终 validation reconstruction loss | 1.944 |
| 最终 validation KL | 0.031 |
| validation masked-token accuracy | 最终 0.540，最高 0.561（epoch 73） |
| validation full sequence recovery | 最终 0.231，最高 0.275（epoch 66） |

训练损失整体下降，validation total loss 在第 73 个 epoch 达到最低后进入平台期；后期 validation reconstruction loss 略有回升。独立测试集的 masked-token infill 结果显示模型能够学习局部恢复，但指标仍不足以支持“完全恢复任意连续片段”的结论。

## 8. 重构评估

测试集评估采用自由 greedy 补全，保留非 mask scaffold，仅统计被遮盖位置。每种策略评估 240 个阳性测试样本。

![Mask 策略重构对比](plots/mask_strategy_comparison.png)

### 8.1 Mask 策略比较

| 策略 | n | Masked token accuracy | Full sequence recovery | 平均 edit distance |
|---|---:|---:|---:|---:|
| `property_guided` | 240 | **0.625** | **0.225** | 2.28 |
| `motif_preserving` | 240 | 0.556 | 0.158 | 2.60 |
| `random` | 240 | 0.566 | 0.171 | 2.53 |
| `contiguous` | 240 | 0.478 | 0.104 | 2.93 |

`property_guided` 在本次测试中最好，说明风险位点往往具有更明确的序列上下文或组成偏差；这不是对所有数据集都成立的普遍结论。`contiguous` 和 `motif_preserving` 的完整序列恢复较低，提示长片段或 motif 周围的补全仍是主要短板。

### 8.2 Mask ratio 曲线

![Mask 比例性能曲线](plots/mask_ratio_performance_curve.png)

| Mask ratio | Masked token accuracy | Full sequence recovery | 平均 edit distance |
|---:|---:|---:|---:|
| 0.15 | **0.641** | **0.400** | 0.98 |
| 0.25 | 0.572 | 0.183 | 1.86 |
| 0.35 | 0.532 | 0.067 | 2.87 |
| 0.50 | 0.480 | 0.008 | 4.63 |

mask ratio 增大后 token accuracy 和完整恢复率均下降，符合上下文信息减少的预期。0.15--0.25 更适合作为 lead peptide 的局部优化强度；0.50 更接近片段重设计，当前模型能力不足。

## 9. 条件引导与候选生成

候选评估包括 default、higher_charge、lower_hydrophobic、lower_hemolysis 四个 counterfactual 条件场景，并比较连续活性概率、风险和合成可行性。unique rate 在去重前计算；MIC<=16 和 MIC>=64 的分层 AUC、KS/Wasserstein 性质分布距离及相对 lead 的变化会写入 `results/evaluation_metrics.json`。

### 9.1 生成质量

本轮使用 50 条独立 lead peptides、多个 mask 位置、4 个 counterfactual 条件和多种温度，共生成 3600 条候选：

| 指标 | 结果 |
|---|---:|
| Valid peptide rate | 100.0% |
| Unique rate（有效候选去重前统计） | 68.75% |
| Novel rate（不在训练集） | 99.86% |
| 平均训练集 top-3 nearest-neighbor similarity | 0.455 |

这里的 valid 只表示标准氨基酸和长度等内部规则通过；novel 只表示不与训练集字符串完全相同，不能等同于真实抗菌活性或实验新颖性。最近邻相似度用于监控分布偏离，过高可能复制训练模式，过低也可能脱离已知 AMP 分布。

### 9.2 条件利用能力

目标条件为：提高阳离子性、保持适度疏水、降低溶血和毒性风险。`guidance ratio=(Generated-Lead)/(Target-Lead)`，正值表示朝目标方向移动。

![条件引导结果](plots/condition_guidance.png)

| 性质 | Lead 均值 | Generated 均值 | Target 均值 | Δ | Guidance ratio |
|---|---:|---:|---:|---:|---:|
| `net_charge` | 4.160 | 5.667 | 6.724 | +1.506 | **58.76%** |
| `KRH_ratio` | 0.283 | 0.362 | 0.368 | +0.080 | **93.55%** |
| `hemolysis_risk` | 0.187 | 0.152 | 0.040 | -0.035 | 24.01% |
| `toxicity_risk` | 0.134 | 0.107 | 0.040 | -0.027 | 28.72% |
| `GRAVY` | 0.058 | -0.308 | -0.054 | -0.365 | 326.07% |
| `hydrophobic_ratio` | 0.475 | 0.429 | 0.430 | -0.046 | 101.18% |
| `aromatic_ratio` | 0.094 | 0.073 | 0.080 | -0.021 | 152.01% |
| `length` | 16.920 | 16.920 | 16.920 | 0.000 | - |

结论是：电荷和 KRH 条件有可解释的正向响应；溶血和毒性风险有下降趋势但达成率有限；GRAVY 明显向更负方向过冲，hydrophobic ratio 也超过目标偏移。因而当前模型属于“部分条件可控”，不能宣称已经实现可靠的多目标性质控制。

### 9.3 MIC 分层活性预测

训练了基于 DBAASP MIC 标签的 GradientBoosting 活性预测器，将 MIC ≤ 16 μg/mL 作为 active、MIC ≥ 64 μg/mL 作为 inactive，中间区间不参与二分类训练。本次分层数据为 active 996 条、inactive 337 条；使用当前保存的全量拟合预测器做分层诊断时，连续 activity probability 均值分别为 0.502 和 0.317，ROC-AUC=0.758。由于预测器在保存前使用了全部 MIC 分层数据重拟合，该 AUC 不是独立测试性能，只能作为标签可分性诊断；它同样不是生成肽的真实 MIC。

### 9.4 Counterfactual 条件对照

| 条件 | n | Net charge | GRAVY | Hydrophobic ratio | Hemo risk | Activity probability | Synthesis feasibility |
|---|---:|---:|---:|---:|---:|---:|---:|
| default | 900 | 5.570 | -0.251 | 0.436 | 0.160 | 0.541 | 0.913 |
| higher_charge | 900 | 5.802 | -0.330 | 0.429 | 0.151 | 0.560 | 0.916 |
| lower_hemolysis | 900 | 5.599 | -0.280 | 0.436 | 0.155 | 0.549 | 0.914 |
| lower_hydrophobic | 900 | 5.696 | -0.369 | 0.416 | 0.142 | 0.566 | 0.916 |

Counterfactual 对照显示目标条件对净电荷、风险和活性代理分数有方向影响，但不同条件之间的差异不大，且 GRAVY 可能过冲。该分析只能说明模型输出对条件输入有响应，不能证明因果控制或真实机制改变。

### 9.5 生成分布与 lead 对照

相对于训练集阳性 AMP，生成集均值为：`length` 16.920（训练 16.944）、`net_charge` 5.667（训练 3.639）、`GRAVY` -0.308（训练 0.051）、`hydrophobic_ratio` 0.429（训练 0.471）、`KRH_ratio` 0.362（训练 0.273）。KS/Wasserstein 统计量见 [`results/evaluation_metrics.json`](results/evaluation_metrics.json)；显著的分布偏移说明生成结果更偏向高阳离子、低 GRAVY 的目标条件，不能简单表述为“完全接近训练分布”。

相对 lead 的模型代理变化为：activity probability 0.474 -> 0.554（+0.080），hemolysis risk 0.187 -> 0.152（-0.035），toxicity risk 0.134 -> 0.107（-0.027），synthesis feasibility 0.895 -> 0.915（+0.020）。这些是预测器或启发式分数的变化，不是实验测量。

### 9.6 性质分布与候选排序

![性质分布对比](plots/property_distribution.png)

![候选综合得分分布](plots/composite_score_histogram.png)

候选综合分数同时考虑预测活性、低溶血/低毒性、条件拟合、新颖性和训练集相似度。示例 Top 候选如下，完整列表见 [`results/generated_candidates.csv`](results/generated_candidates.csv)：

| # | Sequence | Score | Activity | Hemolysis | Toxicity | Length | Net charge | GRAVY |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `KVWKKIASIGKKVLKKL` | 10.512 | 0.816 | 0.136 | 0.054 | 17 | 6.996 | -0.153 |
| 2 | `KKWKKFASIGKKVLKAL` | 10.505 | 0.768 | 0.093 | 0.037 | 17 | 6.996 | -0.394 |
| 3 | `VRLRRIVRVIRK` | 10.493 | 0.817 | 0.129 | 0.052 | 12 | 5.998 | -0.083 |
| 4 | `KKWKKIASIGKKVLKAL` | 10.465 | 0.788 | 0.111 | 0.044 | 17 | 6.996 | -0.294 |
| 5 | `KKWKKIASIGKAVLKKL` | 10.465 | 0.788 | 0.111 | 0.044 | 17 | 6.996 | -0.294 |

## 10. 案例分析

以综合分数 Top-1 的 `KVWKKIASIGKKVLKKL` 为例：

1. 输出长度为 17 aa，落在目标长度区间内。
2. 净电荷约 7，K/R-rich 组成符合革兰氏阴性菌外膜 LPS 静电吸附的机制假设。
3. 模型预测 activity probability 为 0.816，溶血风险 0.136、毒性风险 0.054，综合分数为 10.512。
4. 该序列不是从零随机采样，而是在 lead peptide 的未遮盖 scaffold 上仅改写 mask 位点；因此 edit distance 和 novelty 的含义是局部优化，而非全新蛋白折叠设计。
5. 该结果只是模型排序案例，必须进一步进行 MIC、HC50、细胞毒性、稳定性和合成可行性实验。

失败和限制案例：

- 连续 3--8 aa 的片段补全明显困难，mask ratio 0.50 时完整序列恢复率为 0.008。
- GRAVY 和 hydrophobic ratio 虽然朝目标方向移动，但分别出现明显过冲（guidance ratio 326.07% 和 101.18%），说明潜变量和 decoder 尚未学到稳定、单调且可精确调节的性质控制。
- 活性、溶血和毒性预测器受到训练数据规模和标签质量限制，预测值不能当作实验测量值。
- 训练集来自 DBAASP 子集，分布可能偏向已发表的短肽，外部数据库和独立实验集上的泛化仍未验证。

## 11. 结果复核与任务完成度

| 维度 | 当前判断 |
|---|---|
| 序列 token 化和 masked reconstruction | 已完成；显式 infill loss、greedy scaffold-preserving reconstruction 已加入 |
| 机制描述符 | 已完成 20 维基础版，覆盖任务要求的主要类别 |
| 描述符完整性 | 当前未加入完整 20AA composition 和显式 K/R-rich motif 计数；KRH ratio、positive density 和 aromatic/repeat ratio 是近似替代，属于后续可补强项 |
| 四类 mask 策略 | 已完成，并有策略表和 mask ratio 曲线 |
| Conditional VAE / denoising 思路 | 已完成；BiGRU、attention、latent `mu/logvar`、KL warmup 均实现 |
| 扩散模型 | 本任务采用 CVAE 作为指定主模型，未实现 peptide diffusion；mask 过程体现去噪思想，但不能称为扩散模型结果 |
| 局部优化生成 | 已完成；只改 mask 位点，支持目标条件和候选排序 |
| 重构和生成评估 | 已完成；包括 token accuracy、sequence recovery、edit distance、valid/unique/novel、NN similarity、条件引导和性质分布 |
| 真实药效验证 | 未完成；没有湿实验数据，当前只能报告 in-silico 结果 |
| 研究级泛化和多目标控制 | 部分完成；电荷控制较明显，疏水性控制仍不理想 |

综合判断：课程任务的核心闭环已完成，属于高完成度教学/研究原型；不能把当前结果表述为已经获得可直接开发的抗菌药物候选。

## 12. 思考与总结

### 12.1 主要收获

1. 多肽可以像 SELFIES 一样 token 化，但序列生成必须同时考虑电荷、疏水性、两亲性和毒性等机制约束。
2. mask 是一种去噪过程：random mask 学习一般上下文，contiguous mask 对应片段替换，property-guided mask 更贴近风险位点改造。
3. CVAE 的 latent space 使模型能够在保留 scaffold 的情况下产生多个候选；KL warmup 用于平衡重构和潜变量正则化。
4. 仅有低 reconstruction loss 并不代表条件生成有效，必须单独检查属性偏移方向、条件达成率和训练集相似度。

### 12.2 后续改进

- 扩大 DBAASP、DRAMP、APD3、CAMPR3 的去重合并数据，并用真正的 sequence identity 聚类做外部测试。
- 使用 Transformer/预训练蛋白模型（如 ESM/TAPE）替代或增强 GRU encoder。
- 对 GRAVY、hydrophobic moment、两亲性加入显式 property loss 或 differentiable oracle，并采用 Pareto 前沿进行多目标筛选。
- 增加重复惩罚、beam search、多温度采样和不确定性估计，避免只依赖单一 composite score。
- 对候选开展大肠杆菌、铜绿假单胞菌等目标菌的 MIC 测定，以及 HC50、细胞毒性、蛋白酶稳定性和合成可行性验证。

## 13. 交付物清单

- [x] 数据处理：[`data/`](data/)、描述符计算、标签和相似性划分。
- [x] 模型训练：[`train.py`](train.py)、[`cvae.py`](cvae.py)、[`checkpoints/best_cvae.pt`](checkpoints/best_cvae.pt)。
- [x] 生成与筛选：[`generate.py`](generate.py)、[`validate_candidates.py`](validate_candidates.py)、[`results/generated_candidates.csv`](results/generated_candidates.csv)。
- [x] 评估与可视化：[`evaluate.py`](evaluate.py)、[`results/evaluation_metrics.json`](results/evaluation_metrics.json)、[`plots/`](plots/)。
- [x] 训练报告：[`results/REPORT.md`](results/REPORT.md)。

## 14. 复现注意事项

- 固定随机种子后可复现当前训练流程；不要使用 Python `hash()` 作为随机种子来源。
- 训练集描述符 scaler 只在 train 上拟合，再应用于 validation/test，避免统计泄漏。
- 比较不同 mask 策略时应使用同一测试集和相近 mask 数量。
- 所有活性、溶血和毒性数值均应标注为预测值；报告中不得替代实验结论。
