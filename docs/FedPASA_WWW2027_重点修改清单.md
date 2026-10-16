# FedPASA 投递 WWW 2027：重点修改清单

> 目标：不是继续堆模块，而是把论文主线收紧成 **Evidence/Reliability-Aware Semantic Adaptation for Disjoint-Catalog Federated Recommendation**，让 reviewer 清楚看到：  
> **TransFR 等工作解决“能不能跨目录迁移”，FedPASA 进一步解决“不同 target evidence 不应该使用同一种迁移规则”。**

---

## 一、当前论文最值得保留的核心

目前最能支撑论文创新性的不是“三个模块都很新”，而是下面这一条：

> **Target evidence 在 item、client、catalog 三个粒度上不均匀，因此统一 semantic adapter 并不合理；FedPASA 根据不同粒度的 evidence 对 semantic adaptation 进行控制。**

其中真正最强、最应该作为核心创新的是：

### 1. Popularity/Frequency-Gated Dual-Path Routing（PGDR）

当前实验已经证明：

- FedPASA base：HR@20 = **0.3053**
- w/o routing：**0.2625**
- constant routing：**0.2634**
- random/shuffled routing：**0.2608**

constant/random routing 保留相同双 adapter 容量，因此这个结果说明：

> **提升不是来自“多一个 adapter”，而是来自 item frequency 与 routing assignment 的对应关系。**

这是目前全文最强的实验证据，应该作为论文的第一创新点。

### 2. Client Conditioning 可以保留，但不要包装得过重

当前：

- FedPASA base：0.3053 / 0.1222
- w/o conditioning：0.3029 / 0.1201

说明它有一定作用，但远弱于 routing。

建议定位为：

> **private client-side personalization mechanism**

而不是和 PGDR 同等级的核心 novelty。

### 3. AutoAnchor 建议降级为“semantic preservation control”

当前：

- FedPASA base：0.3053 / 0.1222
- w/o AutoAnchor：0.3064 / 0.1226
- per-item anchor：0.3063 / 0.1225

AutoAnchor 并没有带来稳定的 aggregate accuracy 增益。

因此不要继续写成：

> “第三个核心模块显著提升推荐准确率”。

建议改成：

> **a lightweight semantic preservation control that avoids item-wise communication overhead**

它的贡献重点放在：

- 保留 frozen semantic prior；
- 无需传输 per-item gate；
- 有几何性质分析；
- 可用于控制 representation drift；

而不是主打 accuracy。

---

# 二、P0：投稿前必须优先修改的问题

---

## P0-1：必须补最新、最接近的 published SOTA

这是当前最危险的问题之一。

现在论文里的很多 baseline 属于：

- FedPer-style
- pFedMe-style
- FedMoE
- TransFR-style

这些 **common-backbone controls 非常适合证明机制有效**，但不能完全回答：

> “和当前真正发表的最新系统相比怎么样？”

### 必须形成两类 baseline

#### A. Controlled baselines

保留现有：

- FedAvg-Adapter
- FedProx
- FedAdam
- TransFR-style
- L2-SP
- FedPer-style
- pFedMe-style
- FedMoE
- constant routing
- shuffled/random routing

作用：

> 证明提升不是来自参数量、额外 personalization 或 optimization trick。

#### B. Published SOTA

至少认真讨论/尽量比较：

- **FeDecider（WWW 2026）**
- **TransFR（AAAI 2026）**
- **FFMSR / Federated Semantic Learning**
- **FedCRF**
- **IFedRec（WWW 2024）**

其中 **FeDecider 是 WWW 2026 的直接相关工作，优先级最高。**

### 修改目标

Related Work 不能只说：

> “我们和 TransFR 不一样。”

而应该明确形成一个定位表：

| 方法 | Disjoint catalog | Semantic bridge | Personalized adaptation | Evidence-aware item routing | Source reliability |
|---|---:|---:|---:|---:|---:|
| IFedRec | 部分 | ✓ | ✓ | ✗ | ✗ |
| TransFR | ✓ | ✓ | ✓ | ✗ | ✗ |
| FeDecider | ✓ | LLM/LoRA | ✓ | ✗ | 部分 |
| FedCRF | ✓ | ✓ | ✓ | ✗ | ✗ |
| **FedPASA** | ✓ | ✓ | ✓ | **✓** | 当前不足 |

这样 reviewer 才能一眼看到你的独特位置。

---

## P0-2：必须正面处理 “w/o source initialization 反而更好”

这是目前全文最大的逻辑问题。

当前结果：

- FedPASA base：**0.3053 / 0.1222**
- w/o source initialization：**0.3175 / 0.1299**

也就是说：

> **平均来看，不使用 source adapter initialization，反而更好。**

这会让 reviewer 直接追问：

> 如果 source-domain knowledge 并没有稳定帮助，那为什么这是 cross-domain transfer？

### 最推荐的改法：增加 Source Compatibility / Source Reliability

把这个负结果转化成新的 motivation：

> **Cross-domain transfer itself is unreliable. Source knowledge should not always be trusted equally.**

可以构造：

\[
r_{S\rightarrow T}=f(\text{source-target semantic compatibility})
\]

用于控制：

\[
A_h^{0}
=
r_{S\rightarrow T}A_{\text{src}}
+
(1-r_{S\rightarrow T})A_{\text{target/random}}
\]

或者在最终 routing 中加入：

\[
q_i' = f(\text{item evidence},\text{source compatibility})
\]

### 可以使用的 compatibility signal

优先考虑简单、可解释、不引入大模型额外训练的统计量：

- source / target frozen semantic centroid similarity；
- distribution distance；
- MMD；
- source-target item embedding neighborhood overlap；
- target validation transfer gain proxy。

### 如果时间来不及加新模块

那至少要重写论文定位：

不要说：

> “source knowledge is consistently transferred to target.”

而说：

> “The frozen semantic encoder supplies the cross-catalog bridge, while source-adapter reuse is direction dependent.”

并把 source-free control 提升到正文重点讨论。

但如果条件允许，**最好真的解决这个问题，而不是只解释。**

---

## P0-3：重写论文三项贡献，不要再把三个模块并列

### 当前问题

如果写成：

1. PGDR
2. FiLM
3. AutoAnchor

reviewer 很容易认为：

> “routing + FiLM + heuristic anchor = module stacking。”

### 建议改成下面三项贡献

#### Contribution 1：问题与统一框架

> **Multi-resolution evidence-aware semantic adaptation.**  
> We identify that semantic transfer in disjoint-catalog federated recommendation is constrained by heterogeneous evidence at item, client, and catalog levels, making a uniform adaptation rule suboptimal.

#### Contribution 2：核心方法

> **Frequency-routed dual-path adaptation with private client conditioning.**  
> FedPASA explicitly routes target items between source-initialized and target-specialized residual paths according to target evidence, while local semantic profiles condition the shared transformation without transmitting client profiles.

#### Contribution 3：机制验证

> **Controlled characterization of semantic transfer.**  
> We use exact capacity-matched routing controls, source-free transfer, full-ranking checks, frequency/client-evidence slices, drift measurements, and heterogeneous-client experiments to separate routing gains from extra capacity, local personalization, and semantic preservation.

### 好处

这样就不需要硬说：

> “AutoAnchor 本身是一个很强的新算法。”

而是把论文真正强的部分——**问题定义 + PGDR + controlled evaluation**——放到前面。

---

## P0-4：把 Cross-domain Full Ranking 提升到主文

目前 cross-domain 主结果主要是：

- 1 positive
- 99 sampled negatives
- sampled with replacement

虽然 Appendix 已经补了 full ranking，但 reviewer 很可能仍然会质疑：

> sampled ranking 是否夸大了提升？

当前 full-ranking HR@20：

- FedPASA base：**0.0048**
- TransFR：**0.0025**
- FedMoE：**0.0029**

说明 FedPASA 仍然有优势，但绝对值很低。

### 建议正文至少加入

- Full-ranking HR@20
- Full-ranking NDCG@20
- Recall@20（若定义一致）
- paired statistical significance

### 推荐结构

主表分成两块：

| Method | Sampled HR@20 | Sampled NDCG@20 | Full HR@20 | Full NDCG@20 |
|---|---:|---:|---:|---:|

不要让 full ranking 只存在 Appendix。

### 另外建议

采样负样本改成：

> **without replacement**

避免重复 negative 造成有效 candidate 数减少。

---

## P0-5：Hyperparameter 必须改成 validation-based selection

当前敏感性里：

- \(\alpha=0.55\)：HR@20 = 0.2803
- \(\alpha=0.85\)：HR@20 = 0.3403

以及：

- \(\kappa=0.7\)：0.2803
- \(\kappa=0.5\)：0.3105

差异很大。

这说明：

> 方法对 routing strength 和 coldness exponent 很敏感。

如果当前 sensitivity 是 test-set sweep，reviewer 很容易问：

> 为什么主结果不使用 validation-selected hyperparameters？

### 建议修改

统一使用：

1. validation set 选择 \(\alpha,\kappa,\lambda,\rho\)
2. 参数固定
3. test set 只做最终评估

### 正文写法

不要说：

> “default values are fixed for controlled comparison.”

建议写成：

> Hyperparameters are selected on validation interactions and then fixed for all reported test evaluations. Sensitivity experiments characterize robustness around the selected configuration.

---

## P0-6：第一页明确加强 WWW/Web relevance

WWW 不是因为用了 Amazon 就自然算 Web paper。

第一页要直接讲：

- Web marketplaces；
- streaming/content platforms；
- independent catalogs；
- user histories cannot be centralized；
- source/target users are not aligned；
- newly listed / long-tail Web items have heterogeneous evidence。

### 建议 Introduction 第一段加入一句

> **Web relevance.** Modern Web platforms and marketplaces maintain independently evolving catalogs and isolated user-interaction logs. Recommendation knowledge must therefore be transferred across Web services without relying on shared user identities, aligned item IDs, or centralized behavioral histories.

### Figure 1 也建议优化

当前 conceptual curve 信息量有限。

更建议画：

**Source Web service → shared frozen semantic space → Target Web service**

Target 侧画出三个 evidence level：

- Item：frequency
- Client：local semantic profile
- Catalog：coverage

然后统一指向：

> Evidence-aware adaptation

这样 Figure 1 可以同时回答：

1. 什么问题？
2. 为什么是 Web 问题？
3. 为什么 uniform adapter 不够？
4. 三种 evidence 为什么不同？

---

# 三、P1：非常建议修改的问题

---

## P1-1：说明 global item frequency 的隐私获取方式

PGDR 核心依赖：

\[
freq(i)
\]

论文目前相当于默认：

> server 可以获得 target training frequency table。

但在 federated setting 下，这个统计量本身也是由 client interaction history 派生的。

### reviewer 可能问

- 谁统计 frequency？
- client 是否直接上传 item count？
- 极低频 item 是否可能暴露行为？
- 为什么 profile 是 private，但 frequency 可以 global？

### 推荐最低成本修改

写清楚：

> Aggregate item counts can be collected using secure aggregation before federated training.

如果要再加强：

- clipping；
- minimum support；
- noisy histogram；
- DP frequency statistics。

不一定要把论文变成 DP 论文，但一定要给出 protocol。

---

## P1-2：Client scalability 要加 fixed-compute control

当前：

- 5 clients
- 20 clients
- 50 clients

但是每个 client 的 local steps 固定。

因此：

> client 越多，总 local computation 也越多。

这不能完全证明 scalability。

### 建议增加一个 fixed-total-compute 实验

例如保持总 local updates 近似一致：

| Clients | Steps/client |
|---:|---:|
| 5 | 100 |
| 20 | 25 |
| 50 | 10 |

至少让 reviewer 看见：

> 在总训练预算近似相同时，FedPASA 仍保持优势。

---

## P1-3：补 partial participation / client dropout

目前每轮所有 client 都参加。

但真实 federated Web systems 很少是 100% participation。

### 可以只做一个小实验

例如：

- 100%
- 50%
- 20%

或者固定 10 clients，随机采样：

- 2 clients/round
- 5 clients/round
- 10 clients/round

不用做很大，但会显著提升 federated setting 的可信度。

---

## P1-4：理论部分可以压缩，不要继续加

当前理论主要是：

1. bias–variance tradeoff
2. normalized anchor geometry
3. FiLM continuity

这些理论：

- 合理；
- 能解释设计；
- 但不是论文最强 novelty。

尤其 Proposition 1 并没有证明：

\[
|U_T|/|I_T|
\]

就是 reliability 的最优估计。

### 建议

正文理论压缩到：

- 1 个核心 Proposition；
- 其余 proof 放 Appendix。

释放正文篇幅给：

- published SOTA；
- full ranking；
- Web relevance；
- source compatibility；
- stronger evaluation。

---

## P1-5：In-domain 实验重新定位

当前 in-domain：

- FedPASA HR@20 平均较好；
- NDCG@20 并没有稳定领先；
- Luxury Beauty 上明显较差。

所以不要把论文包装成：

> universal federated recommender。

### 更好的定位

> FedPASA is designed primarily for **disjoint-catalog semantic transfer under heterogeneous target evidence**.

In-domain 作为：

> scope / stress-test analysis

而不是主要 claim。

---

# 四、摘要必须重写的地方

---

## 当前不建议保留的写法

类似：

> “The contribution is their joint use rather than a new conditioning or mixture-of-experts primitive.”

这句话虽然严谨，但放在摘要里会直接削弱 novelty。

这种解释应该放 Related Work。

---

## 摘要应该强化的逻辑

### 第一句：问题

现有 semantic-transfer FedRec 通常对所有 target evidence 使用相对统一的 adaptation rule。

### 第二句：gap

但 target supervision 在：

- item frequency；
- client preference coverage；
- catalog density

上高度不均匀。

### 第三句：方法

FedPASA 使用 multi-resolution evidence-aware semantic adaptation。

### 第四句：核心机制

突出：

> frequency-routed source/target dual-path adaptation

FiLM 和 anchor 放第二层。

### 第五句：实验

强调：

- six cross-domain directions；
- exact capacity-matched routing controls；
- full-ranking validation；
- heterogeneous-client analysis。

---

# 五、Introduction 建议重新组织成 4 段

---

## Paragraph 1：Web 场景

讲：

- Web services / marketplaces；
- catalogs independently evolve；
- user histories isolated；
- no aligned user IDs；
- cold/long-tail items。

---

## Paragraph 2：现有 semantic transfer 的不足

不要只说：

> semantic encoder helps.

要说：

> Existing approaches largely learn one shared adaptation rule, implicitly assuming comparable target evidence.

然后指出：

- rare item；
- frequent item；
- client locally observed item；
- client locally unobserved item；

不能使用完全相同的 adaptation rule。

---

## Paragraph 3：FedPASA

一句话总览：

> FedPASA adapts semantic representations according to evidence at item, client, and catalog resolutions.

然后明确：

- PGDR：核心；
- FiLM：client personalization；
- AutoAnchor：semantic preservation。

---

## Paragraph 4：贡献

用新的三项贡献，不要三个模块并列。

---

# 六、Related Work 最需要补的不是数量，而是“定位”

建议划成三组。

---

## 1. Federated Recommendation

FedAvg/FedProx/FedPer/pFedMe 等只简述。

---

## 2. Federated Cross-Domain / Semantic Recommendation

重点：

- IFedRec
- TransFR
- FeDecider
- FFMSR
- FedCRF

这里必须明确：

> 这些工作解决 semantic bridge、parameter-efficient transfer、non-overlap、personalization，但没有显式按照 target item evidence 对 source/target semantic adaptation path 进行 frequency-conditioned routing。

---

## 3. Personalized Adapter / Mixture-of-Experts

说明：

- FiLM 不是新 primitive；
- MoE 不是新 primitive；
- novelty 是 **evidence signal 与 semantic transfer path 的结合方式**。

---

# 七、实验建议最终形成下面这套结构

---

## RQ1：Overall Cross-domain Performance

### 必须包含

- 6 个 transfer directions；
- sampled ranking；
- full ranking；
- mean + std；
- significance。

---

## RQ2：Does Frequency-Aware Routing Matter Beyond Capacity?

这是最强的一节。

必须保留：

- w/o routing
- constant routing
- shuffled/random routing
- two-expert learned routing
- exact parameter counts

这一节可以成为全文最重要的实验节。

---

## RQ3：What Does Client Conditioning Contribute?

建议不要只给总平均。

进一步分析：

- locally observed；
- locally unobserved；
- client activity；
- client semantic diversity。

如果 conditioning 在某些 client slice 上更有效，这比 overall +0.0024 更有说服力。

---

## RQ4：When Does Semantic Anchoring Help?

不要再问：

> “AutoAnchor 是否提高 overall accuracy？”

而改成：

> “What does semantic anchoring preserve, and when is it useful?”

看：

- representation drift；
- sparse target；
- low coverage catalog；
- cold item；
- communication overhead。

---

## RQ5：When Is Source Transfer Beneficial?

强烈建议新增。

比较：

- random init；
- source init；
- compatibility-aware source init。

如果新方法能解决当前 source-free > source-init 的问题，会明显提升论文完整度。

---

## RQ6：Federated Robustness

包括：

- non-IID；
- partial participation；
- client number；
- fixed-compute；
- update perturbation。

---

# 八、建议增加一个 Source Compatibility 模块（如果还有实验时间）

这是我认为最值得新增的内容。

---

## Motivation

当前实验已经发现：

> Source initialization is not uniformly helpful.

因此可以顺势提出：

> **Source knowledge is itself a heterogeneous and uncertain signal.**

这与论文题目 “Reliability-Aware” 非常匹配。

---

## 简单版本

计算 source-target compatibility：

\[
r_{S,T}
=
\cos(\bar{x}_S,\bar{x}_T)
\]

其中：

\[
\bar{x}_S=\frac{1}{|I_S|}\sum_{i\in I_S}x_i,\qquad
\bar{x}_T=\frac{1}{|I_T|}\sum_{i\in I_T}x_i
\]

再做：

\[
A_h^0 = r_{S,T}A_{\mathrm{src}} + (1-r_{S,T})A_{\mathrm{rand}}
\]

或者将 \(r_{S,T}\) 作为 routing temperature / head-path prior。

---

## 更稳妥版本

不要直接混参数，而是在 representation 层：

\[
h_i =
(1-q_i')r_{S,T}A_h(x_i)
+
\bigl(1-(1-q_i')r_{S,T}\bigr)A_l(x_i)
\]

这样：

- item evidence 决定 item-level trust；
- source compatibility 决定 domain-level trust。

整个论文就真正形成：

> **item reliability + client evidence + source/domain reliability**

故事会更完整。

---

# 九、建议弱化/删除的内容

---

## 1. 不要反复强调 AutoAnchor accuracy gain

因为主实验不支持。

---

## 2. 不要把 FiLM 本身写成 novelty

FiLM 是已知技术。

创新是：

> private semantic profile conditioned adaptation in this federated semantic transfer pipeline.

---

## 3. 不要过度强调理论贡献

理论作为 supporting analysis 即可。

---

## 4. 不要说“universal superiority”

目前：

- in-domain NDCG 不稳定；
- full ranking absolute HR 较低；
- anchor 不稳定；
- source init 不稳定。

应该强调：

> setting-dependent but consistent cross-domain routing benefit。

---

# 十、最终论文应该形成的主线

建议把全文最终逻辑统一成下面这一句话：

> **Existing semantic-transfer federated recommenders mainly ask how to transfer representations across disjoint catalogs; FedPASA asks when and how much each target item, client, and domain should trust transferred semantics.**

中文可以理解为：

> 现有工作主要解决“跨目录怎么迁移”，FedPASA 进一步解决“不同 target evidence 到底应该信多少迁移知识、采用哪条适配路径”。

这比：

> “我们用了 routing + FiLM + anchor”

强很多。

---

# 十一、修改优先级总结

## 第一优先级：必须改

- [ ] 补 FeDecider 等最新 published SOTA
- [ ] 解决 / 强化解释 w/o source init > source init
- [ ] 把 PGDR 提升为唯一核心机制
- [ ] AutoAnchor 降级为 preservation control
- [ ] Cross-domain full-ranking 进入正文
- [ ] 超参数改成 validation selection
- [ ] 第一页明确 Web relevance
- [ ] 重写 Abstract + Introduction + Contributions

## 第二优先级：强烈建议

- [ ] global item frequency 用 secure aggregation / protocol 说明
- [ ] fixed-compute client scaling
- [ ] partial client participation
- [ ] conditioning 做更细 client slice
- [ ] 理论部分压缩
- [ ] in-domain 重新定位为 stress test

## 第三优先级：有时间再做

- [ ] Source Compatibility / Source Reliability Gate
- [ ] 更丰富的 source-target compatibility 分析
- [ ] 更多 encoder robustness
- [ ] DP/noisy frequency histogram
- [ ] communication/computation wall-clock 对比

---

# 十二、如果只允许再做 5 件事

如果投稿时间已经比较紧，我建议只做下面五件：

1. **补 FeDecider + 最新 SOTA，并加入 Published SOTA 对比。**
2. **解决 source-free > source-init，最好加入 source compatibility-aware mechanism。**
3. **重写创新点：PGDR 为核心，FiLM/Anchor 为辅助控制。**
4. **把 Full Ranking + significance 放正文，并规范 validation-based tuning。**
5. **重写第一页，让 Web relevance 和 evidence-aware motivation 一眼就能看懂。**

这五件做完，对论文完整度和 WWW reviewer 的第一印象提升最大。
