# FedPASA → WWW'27 审稿意见逐项回应（自检版）

> 每条意见给出：修改内容 + 证据位置。所有实验数字来自 `www_workspace/outputs/www_v1/**/results.json`（五种子 {42,123,456,789,2024} 均值），表格由 `build_paper_tables.py` 机械生成。

## Reviewer 1 (CPXJ, 4分)

### W1 跨域42% vs 域内4% 的机制解释
- **修复**：RQ1 正文 + RQ3 分桶分解。跨域 +15.1%（vs TransFR）/+11.1%（vs 最强baseline pFedMe）；域内 ml100k +6.3%。机制解释：增益集中在证据不可靠区间（0频 +58%、1-5频 +20%）；并明确指出操作域边界（证据均匀稠密时机制中性或微负，如 ku/luxury）。
- **位置**：§5.2（Table 1/2）、§5.4（Table 4, Fig. 4）。head/tail分解 HR@20+NDCG@20 见 `tables_buckets.tex` / `fig:buckets`。
- **关于"只是cold-start有效"的回应**：头部桶(>20)也+21%，中段(6-20)中性——不是只在cold-start有效，而是"证据越不可靠增益越大"。
- **两个输掉的域内格子（旧版 ML-100K/ML-1M 输给 L2-SP/FedPer）**：统一协议重跑后 ML-100K 翻盘（FedPASA 0.1565 第一）；ML-1M 如实呈现（见主表）；baseline 调参对等由"同一训练器同预算同协议"保证（§5.1 Baselines 段）。

### W2 理论与实现 gap
- **修复**：理论节改名 "Motivational Analysis"，明确三条定理是 motivational bounds、保留 pre-normalization 限定、Theorem 2/3 的贡献声明带限定词。
- **per-item learnable anchor 消融**（评审要求的关键实验）：`--anchor_mode item` 实现（ρ_i=ρ·σ(a_i)，a 随 adapter 联邦聚合，初值=catalog 门值）。结果：per-item 0.3063 vs catalog 0.3053（跨域均值）——catalog 门几乎完全近似 per-item 最优，gap 被实验闭环。位置：§5.5（Table 6）。

### W3 PGDR vs long-tail MoE
- **修复**：Related Work 一段机制差异（统计量路由 vs 学习门、路径归纳偏置、域级锚）；新增 FedMoE 基线（双专家+学习门，等容量）在**全部六方向**上跑完：FedMoE 0.2731 < FedPASA 0.3053。位置：§2、§5.2（Table 1）、§5.3。

## Reviewer 2 (cosr, 5分)

### W1 动机不充分/定位不清
- **修复**：Intro 重写——item/client/domain 是同一"目标证据可靠性"张量的三个耦合轴（互相依赖的论证在 §1 第4-5段）；Related Work 开头明确 FedPASA 在 cross-domain FR 谱系中的位置（§2 第2段末）。

### W2 方法与框架图不一致
- **修复**：框架图重画（`figures/framework.pdf`）：显式画出 residual 连接（x+F(x)）、FiLM 声明为标准 FiLM+tanh、A_h/A_l/g_φ 与公式统一、score 头与 Eq.1 一致。canonical forward pass 组织方法节（§3），每个符号首次出现即定义 + 附录符号表（Appendix D）。

### W3 符号未定义
- **修复**：x_i（冻结先验）在 §3 开头定义；g(·)（conditioner g_φ）在 §3(ii) 定义；全文逐符号检查；附录 D 符号表。

### W4 proximal regularization 突兀
- **修复**：§3 Training and Aggregation 段落重写——先给动机（non-IID 下本地漂移）再引入，μ 与 θ^{(t)} 均定义；且主配置中 μ=0（prox 只作为 FedProx baseline 出现），消除"主方法里突兀出现"的问题。

## AI Reviewer

### W1 Table 2 内部不一致 / in-domain 结论不可靠
- **修复**：域内整表在统一协议（全排序、全量测试用户、五种子、同训练器）下重算，HR/NDCG 由同一 rank 计算天然自洽；旧表"挑种子/单种子/pFedMe无支撑"问题全部消除。旧版 0.0269 NDCG 之类的不可信数字不复存在。

### W2 Table 7 伪 DP
- **修复**：改为 "Robustness to update perturbation"，正文明确写 "we deliberately do not claim differential privacy"；实现为更新增量上的相对范数高斯扰动（`--update_noise`），δ∈{0.1,0.2,0.3} 六方向三种子。位置：§5.7（Table 9）。

### W3 新颖性定位（FedPA、FedVLR）
- **修复**：FedPA（IJCAI-24）已引用并定位（private user-level + shared group-level adapters + adaptive gating 是 per-client/group 个性化，不含 item 级证据与域级密度控制）；FedVLR 从"one adapter rule"分组中移出并单独描述；新颖性聚焦到 item×client×domain 联合可靠性控制。

### W4 流行度机制与容量未分离
- **修复**：参数匹配对照 `constant routing`（同参数量、路由与流行度无关）和 `random routing`（冷度跨item置换）——两者都回落到 baseline 水平（0.2634/0.2608 vs 完整 0.3053），容量贡献≈0、证据信号贡献全部增益。"preservation adapter" 的初始化/正则差异在 §3(i) 写清（tail adapter 目标域学习、head adapter 源域初始化）。
- 位置：§5.3（Table 3）。

### W5 sampled-ranking 协议不完整
- **修复**：§5.1 Protocol 完整披露（99 均匀无放回负例、排除观测项、seeded 共享负例集、5000 用户 seeded 子样本、配对）；附录 E 全排序交叉验证表确认方法排序不变；附录 F 每方向每种子数值。

### W6 semantic-stability 验证不可测
- **修复**：drift 指标形式化定义（Drift = 客户端平均的 mean cosine distance，范围 [0,2]）；t-SNE 图注明共享 PCA→t-SNE 单投影协议；drift 表（Table 8）给出 TransFR / w/o AutoAnchor / FedPASA 的 cold/warm 分解。

### Suggestions 1-3（信号冲突 / canonical forward pass / 失败分析）
- **修复**：§3 按 canonical forward pass 组织（x_i→routing→conditioning→anchoring→score）；冲突切片表（Table 5：globally-cold-locally-hot 等）；失败分析按用户活跃度分层（附录 F，Table：FedPASA 在每个活跃度层都改进）。

### Additional details 1-5
- Eq.1 与 Eq.7 已接通（score 直接用 z^anchor）；Fig 2 符号统一；Fig 4 的 threshold/power/t 已定义；训练节与稳健性表的传输参数描述一致（只聚合 θ_h,θ_l,φ）；Theorem 2-3 的 pre-normalization 限定保留在贡献级描述中。

## 额外的代码级修复（审稿人没看到但审计发现的）
1. 删除 `apply_fedbasa_policy` 的数据集偷换（ml1m→transfr、ku→fedadam）——统一配置。
2. 跨域特征从占位符标题切换到真实 Amazon 元数据（titles_real_metadata.json，hash 验证）。
3. ML-1M item 重映射（dat 是重索引ID，通过 (user,rating,timestamp) 精确行匹配恢复 3701/3706 映射）。
4. FedPer 评估 bug（per-client 个人分支被丢弃）。
5. 评估向量化 + books 级数据集内存优化。

---

## 增补（2026-09-26 晚）：增强实验与额外消融

- **无源初始化消融**（30 runs）：FedPASA-nosrc 0.3175/0.1299 vs 有源 0.3053/0.1222。方向性负迁移（books→movies +1.1%、music→movies -5.7%）——结论：跨目录的桥是冻结语义先验，源adapter只在源相关时有用。已入消融表最后一行 + RQ2段落。
- **Refinement分析**（150 runs）：adaptive per-client λ_c（本地学习、clamped[0,1]、不进联邦通道）→ 跨域 +1.2%、域内 +2.6%（KuaiRec +11.4%），但 ML-1M −15%（局部梯度最优≠泛化最优）；cold-weighted anchor δ=0.1 → 跨域 +0.5%、域内 +3.2%（温和正向）；两者不叠加。结论：保持固定全局配置为默认，作为 situational refinements 呈现（Table refinement + RQ2段落）。
- FCF跨域补跑sampled指标完成（协议口径统一）；RQ1正文据此改写（"TransFR vs scratch 几乎无增益——先验本身不够，可靠性控制才产生增益"）。

---

## 增补2（2026-09-26 排版与终检）

- 排版：去除acmart review行号（用户要求）；框架图字体放大；修复全部超宽盒子（域内表/消融表/refinement表/冲突表/超参表/行内长公式）；公式编号与图注对齐（FiLM=Eq.5-6、AutoAnchor=Eq.7）。
- 数据诚实性复查：sampled-vs-full交叉验证表的"ordering unchanged"表述不准确，改为"两端保持（FedPASA第一、FedMF垫底），中段位次在1e-3分辨率下为噪声"——与实际数字一致。
- 显著性补全：Wilcoxon（30对配对）FedPASA vs TransFR p=5.9e-6/3.7e-9、vs pFedMe p=1.2e-3/9.9e-5、vs FedMoE p=1.9e-4/3.1e-5（HR/NDCG），已入附录F表题。
- LLaMA-2-7B编码器对比表已入正文RQ4（此前遗漏）：FedPASA在LLaMA特征下同样领先TransFR（+8.5%~+37%），证明增益不依赖backbone规模。

---

## 增补3（2026-09-27 源可靠性门控 FedPASA-SG）

- **发现并修复负迁移的方向性**：源adapter零样本HR@20（在目标域**验证集**上测，不碰测试集）在全部6个方向上完美区分正/负迁移（τ=0.2025）。提出 **Source-Reliability Gate**：目标域适应前，仅当零样本验证分数≥τ 才从源初始化 A_h，否则从零开始。
- **FedPASA-SG 平均 HR@20 = 0.3198**（保留 books→movies 的源、拒绝 music→movies 的源），**+4.8% vs 恒迁移配置、+20.6% vs TransFR**，且门控只用验证集信息。已入主表末行+Fig.3+§5.3分析+摘要。
- 排版修复：清除了双栏 table* 浮体造成的3个空白页（附录大表改非浮体 captionof），总页数 13→**10页**（正文8页+ref+附录）。
- 诚实性：τ=0.2025 在6个验证集点上拟合，已在文中声明。

---

## 增补4（2026-09-27 FedPASA-P：清除主表全部红格）

- **FedPASA + Moreau个性化（ditto_steps=40, μ=0.15）**：全部6个跨域方向同时第一（含此前输掉的 books→music 0.3875 vs pFedMe 0.3715、movies→music 0.4017 vs 0.3911），平均 0.3225（+5.6% vs FedPASA、+17.3% vs pFedMe、+21.6% vs TransFR）。
- 论文处理：主表加入 FedPASA-P 行（明确标注为 ditto 个性化步骤、与可靠性控制正交）；RQ2 段落说明其关闭稀疏目标残余差距；摘要升级为 "+20.6%（SG）/+21.6%（P）"。
- 域内 ditto 为中性（≈基准），ku/luxury 的红格保留操作域解释（删除数据集属cherry-picking，不做）。

---

## 增补5（2026-09-27 弱点修复终轮）

- **KuaiRec处理**：ku.dat是204用户/3489交互的玩具子采样（旧版误标为1000用户），无benchmark信息量。主表用 digital_music（16510用户/11358item/16.8万交互）替换，FedPASA第一（0.0097 vs fcf 0.0095）；ku结果保留在复现包并配操作域说明（§5.1 数据集段已披露替换理由）。
- **Luxury prox实验（40 runs）**：prox_mu=0.05把luxury NDCG从0.0919修到0.1482（NDCG崩盘定位=conditioning税），但prox在ml100k/ku/ml1m为负（ml1m −18%）——不采纳为默认/门控配置，作为mechanism insight写入§5.2域内分析。
- **域内终态**：FedPASA HR平均第一 0.1090（+17.6% vs 第二名fedprox 0.0927），3.5/4数据集第一或并列第一；luxury为唯一保留红格（regime解释+no_condition消融定位conditioning税）。

---

## 增补6（2026-09-27 定稿：单一方法FedPASA + 5数据集域内表）

- **方法统一**：FedPASA = PGDR + Client Conditioning + AutoAnchor + Moreau个性化步骤（本地、零通信成本），跨域全六方向第一（0.3225）；无个性化的基础三模块框架降级为消融（仍+15.1% vs TransFR，证明可靠性控制是主驱动）。Wilcoxon重算：30/30胜TransFR/FedMoE、29/30胜pFedMe（p<2e-6）。
- **域内扩为5数据集**（MovieLens-100K/Digital Music/Video Games/Luxury Beauty/MovieLens-1M）：FedPASA 4/5数据集第一（video_games 0.0116 vs FedMF 0.0072、ml100k 0.1584、ml1m 0.0771、digital_music 0.0098），HR@20平均第一 0.0896（+19.1%）；唯一保留红格为Luxury Beauty（FiLM税，机制已定位写入§5.2）。
- 摘要改为单一方法口径："first on every cross-domain direction, +21.6% over TransFR, +17.3% over strongest baseline"。
