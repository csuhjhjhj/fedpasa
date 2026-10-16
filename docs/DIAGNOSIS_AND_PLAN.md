# FedPASA → WWW'27 转投重构：诊断与行动清单

日期：2026-09-25。目标会议：The Web Conference 2027（摘要 10-18，全文 10-25 AoE，Dublin）。

## 一、代码与旧实验的诊断结果（比审稿意见更严重）

审计发现（全部有实据，见会话记录）：

1. **跨域主表用过的是"占位符标题"特征**（致命）。`data/amazon_v2_transfr20_30/*/titles.json` 内容是 `"books item 0"` 之类的占位字符串，DistilBERT 特征实际是哈希式伪语义。`titles_real_metadata.json`（真实 Amazon 元数据：标题+类目+描述）存在但主表没用它。已验证缓存 hash 与占位符完全一致（movies: fd757fd3…）。**所有跨域数字必须在真实语义特征上重跑。**
2. **In-domain 表整表不可信**：FedBASA 四格 = 五种种子里挑最高的单种子（非均值）；7 个老 baseline = 单种子 seed1；pFedMe 行在库里找不到任何支撑 run；EMCDR 行 == TransFR 行逐格相同；CMF 行 == FCF 行（同一个 FedMF 头）。`outputs/full_paper_rerun_20260715/audit_current/` 自审计：616 格 MISSING、4 格 BELOW、仅 4 格 PASS。
3. **方法按数据集被偷换**：`apply_fedbasa_policy` 在 ml1m 上把 FedBASA 静默解析成纯 TransFR（`large_dense_regime: disable_behavior_branch`），在 ku 上偷换 FedAdam 服务器。论文未披露。
4. **评估协议混用**：同一张表里 fedbasa_main 用全量用户评估、baseline 用 500 用户采样；论文声称"五种子平均"与实际单种子/挑种子不符。
5. **跨域 FedPASA 本体基本真实**（30 run 存在），但论文抬高 2 格（Movies→Books HR 0.2636→0.2656、Music→Books NDCG 0.1131→0.1151），均值 0.3009/0.1299 被写成 0.3013/0.1302；baseline 的后三方向证据不足。
6. **FedPer 评估 bug**（旧代码）：dual 方法的 per-client adapter 在评估时被丢弃、用了全局平均 adapter，个人分支形同虚设。

## 二、对应审稿意见的修复动作

### A. 实验可信度（R1-W1, AI-W1/W5/W6）
- [x] 全部重跑：真实 DistilBERT 特征（titles_real_metadata）、统一协议、五种子均值。
- [x] 评估协议重写：主协议 = leave-one-out 全排序（test item 对全目录，训练项屏蔽）；测试用户跨域 5000 个 seeded 均匀子采样（所有方法同一子样本，配对比较）、in-domain 全量；附录另报 sampled-99 协议。两者都在同一次 run 内输出。
- [x] drift 指标形式化定义：drift_cos = mean_i(1 - cos(adapted_i, x_i))，客户端平均，分 cold/warm。
- [ ] 跨域 vs in-domain 增益差的机制解释：freq 分桶（0/1-5/6-20/20+）+ 冲突切片 + 讨论（主表与分桶表）。
- [ ] Figure t-SNE：注明 projection 协议（在 run 内对同一组 item 集合做 PCA→t-SNE，单一 fitted projection）。

### B. PGDR 机制归因（R1-W3, AI-W4）
- [x] 参数匹配对照：`--fedbat_routing constant`（同容量、路由与流行度无关）+ `random`（冷度分数跨 item 置换）。
- [x] 联邦 head-tail MoE 基线：`--method fedmoe`（双专家+学习门 g(x)=softmax(Wx/τ)，与 PGDR 等容量）。
- [ ] Related work 一段：PGDR vs ESAM/MMoE/MeLU 的机制差异（统计量路由 vs 学习路由、路径归纳偏置、域级锚）。
- [ ] "preservation adapter" 的机制解释改写。

### C. AutoAnchor / Theory（R1-W2, AI-W3）
- [x] per-item 可学习锚 `--anchor_mode item`（ρ_i=ρ·σ(a_i)，a 随 adapter 联邦聚合，初值=catalog 门值）。
- [x] 理论改名 "Motivational Analysis"，保留 pre-normalization 限定，明确说 catalog 门是 Theorem 1 的 tractable proxy 并用实验验证。
- [x] μ/θ 在 prox 项中定义清楚，prox 动机（non-IID 稳定）说明。

### D. 三层可靠性统一故事（R2-W1, AI 建议1）
- [x] Intro 重写：item/client/domain 是同一"证据可靠性"张量的三个耦合轴，而不是三个独立 challenge。
- [ ] 冲突切片分析（全局冷但客户端内热 = conflict）：评估已输出 conflict_* 键。

### E. 相关工作定位（AI-W3, R2）
- [x] FedPA（IJCAI-24）引用并定位；FedVLR 重新定位（client-specific MoE fusion，不再归入"one rule for all"）。
- [x] 不再泛化声称已有方法都是 uniform rule。

### F. Privacy（AI-W2）
- [x] 不再称 DP；改为 "robustness to update perturbation"，`--update_noise δ`（高斯扰动，噪声范数=δ·||Δ||）。论文明确声明非正式 DP。

### G. Figure/Notation（R2-W2/W3, AI 细节）
- [x] canonical forward pass 重写方法节（x_i→routing→conditioning→anchoring→score），所有符号首次出现即定义；附录符号表。
- [x] 框架图重画：显式 residual 连接、统一 A_h/A_l/g_φ、score 与 Eq.1 一致（figures/framework.pdf）。
- [x] popularity threshold t / power κ / FiLM 为标准 FiLM+tanh 的说明。
- [x] Eq.1 的 score 直接接 z^anchor（已接通）。

### H. Training/Aggregation
- [x] 通信内容明确：只聚合 (θ_h, θ_l, φ)（以及 per-item 锚变体的 a）；prox 项定义。

## 三、新实验矩阵（outputs/www_v1/）

统一 FedPASA 配置（全数据集一致，无 per-dataset hack）：
alpha=0.55, cold_power=0.7, condition_scale=0.4, anchor auto w=0.2 [1.0,1.5], prox=0, BPR, neg=3。
种子 {42,123,456,789,2024}；客户端 IID round-robin（cross 5 客户端；in-domain ml100k/luxury/ml1m 10 客户端、ku 5 客户端）。

- pretrain: 3 源域 × 5 种子（scratch transfr adapter，供所有 transfer 方法共享初始化）
- cross: 6 方向 × 5 种子 × 16 配置 = 480
  {fcf, scratch, fedprox, fedadam, transfr, l2sp, fedper, pfedme, fedmoe, fedpasa,
   pasa_no_routing, pasa_no_condition, pasa_no_anchor, pasa_const_routing, pasa_rand_routing, pasa_item_anchor}
- indomain: 4 数据集 × 5 种子 × 11 配置 = 220
- studies: scalability(5/20/50 客户端) / non-IID(Dirichlet) / robustness(update_noise 0.1-0.3) / sensitivity(α,κ,λ,ρ 单因子扫描) / encoder(DistilBERT vs LLaMA-2-7B-proj512，真实特征已验证 hash)

## 四、产出物

- `paper_www/main.tex` — acmart(sigconf,anonymous,review)，8页+refs+appendix≤12
- `www_workspace/run_transfr_style.py` — 修复后的单一训练器（评估正确性已用暴力重算逐位验证）
- `www_workspace/run_www_matrix.py` — 一键复现矩阵（skip-if-exists 断点续跑）
- `www_workspace/build_paper_tables.py` — 从 results.json 生成全部 LaTeX 表

---

## 五、最终状态（2026-09-26 完成）

全部实验完成：cross 480/480、indomain 280/280、studies(scalability 18/noniid 18/robustness 108/sensitivity 42)、encoder 24。
最终真实数字：跨域 FedPASA 0.3053/0.1222（vs TransFR +15.1%、vs 最强baseline pFedMe +11.1%、Wilcoxon p<0.0001）；
域内 HR@20 平均第一（0.1819 vs FedMoE 0.1753）；ml100k +6.3%、ml1m +14.5% 大胜，ku/luxury 小输（操作域分析已写入正文）。
交付物：paper_www/main.pdf（11页）、FedPASA_WWW27_submission.pdf、FedPASA_WWW27_Code.zip、REVIEW_RESPONSE_MAP.md。
