# FedPASA AAAI-27 完整审稿意见整理（兼容版）

> **说明：本版专门解决 Markdown 公式乱码问题。**  
> 不再依赖 Markdown/LaTeX 数学渲染器。所有关键公式均改为 **Unicode + 纯文本代码块**，在 Typora、VS Code、Obsidian、GitHub、记事本类 Markdown 阅读器中都更稳定。
>
> 例如：
>
> ```text
> ρ*_(c,i) = σ² / (σ² + r_(c,i) · b_(c,i)²)
> η_T = |U_T| / |I_T|
> A(x) = norm(x + F(x))
> ```
>
> 论文正式 LaTeX 中仍应使用标准 LaTeX 公式；本文件主要用于阅读、对照和修改清单。

---


# Reviewer 1 — Program Committee CPXJ

- **Review type:** Official Review
- **Rating:** **4 — Ok but not good enough - rejection**
- **Confidence:** **4 — The reviewer is confident but not absolutely certain that the evaluation is correct**

## Paper Summary

The paper addresses a concrete difficulty in cross-domain federated recommendation: target-domain evidence reliability varies substantially across items and clients, so a single uniform target adapter tends to under-fit well-supported items and over-fit sparse ones. Existing semantic-transfer methods reuse pretrained item representations and federate a lightweight adapter, but their update rule is largely uniform and cannot account for item popularity, client heterogeneity, or target coverage.

FedPASA is built around an **item / client / domain three-level reliability** framing and combines three components.

1. **Popularity-Gated Dual-Path Routing** mixes a high-popularity adapter A_h with a low-popularity adapter A_l using an inverse-frequency deficit score per item, and a global coefficient α controls the overall strength.
2. **Client Conditioning** pools a semantic profile p_c from local history and feeds it through a shared conditioner to produce FiLM parameters (γ_c, β_c) that modulate the representation.
3. **AutoAnchor** takes a catalog-level coverage
   
```text
η_T = |U_T| / |I_T|
```

   converts it via a clip-gated function to an effective anchor weight ρ_eff, and interpolates the adapted representation with the frozen semantic prior.

A source-domain adapter is pretrained first; on the target domain the adapter, conditioner, and anchor-control parameters are aggregated via FedAvg while user embeddings and interaction histories remain on device. Three theorems characterize the pre-normalization representation drift, score stability, and Lipschitz smoothness of client conditioning.

## Strengths

### S1. Problem framing

Problem framing is precise and hits a real pain point (Sec. 1, Fig. 1). The paper formulates the common-sense observation that target-domain evidence varies substantially across items and clients into a clean three-level reliability structure (item / client / domain), avoiding the vague “cross-domain is hard” framing common in this space. Fig. 1's contrast between uniform and popularity-aware adaptation communicates the intuition well.

### S2. PGDR motivation and ablation

PGDR's motivation matches its ablation reasonably well (Eq. 3 to 5, Table 3). A popularity-deficit score plus dual adapters is a sensible landing of head-tail thinking in federated semantic transfer. Table 3 shows Popularity Routing as the single most influential component:

- cross-domain HR drops from **0.3013** to **0.2577** when it is removed;
- Fig. 4(d) shows the gains concentrated in the low-popularity buckets.

This aligns the empirical trend with the stated motivation.

### S3. Client Conditioning

Client Conditioning's value is visible under stronger non-IID and larger client counts (Fig. 4(c), Table 5).

---

## Weaknesses

### W1. Cross-domain vs. in-domain gain gap

The roughly **14× gap** between the cross-domain **42% gain** and the in-domain **4% gain** is left unexplained at the mechanism level (Table 1 vs. Table 2).

### W2. Theory–implementation gap

Theorem 1 and Theorem 2 rest on assumptions the authors themselves admit do not hold in the deployed system (Sec. Theoretical Analysis and Eq. 8).

Theorem 1 characterizes an optimal per-`(client,item)` anchor


```text
ρ*_(c,i) = σ² / (σ² + r_(c,i) · b_(c,i)²)
```


but AutoAnchor actually uses a catalog-level gate on


```text
η_T = |U_T| / |I_T|
```


The authors explicitly acknowledge that:

> “the domain-level ratio is a coarse practical proxy rather than an item-specific reliability estimate.”

Theorem 2's pre-normalization Cauchy–Schwarz bound also states that:

> “extending the bounds through normalization additionally requires a positive lower bound on the input norm,”

while the deployed representation does apply `norm(·)`.

Moreover, the three theorems themselves — bias-variance shrinkage, Cauchy–Schwarz, and Lipschitz continuity — are generic mathematical tools with no FR / cross-domain / semantic-transfer structure.

Presenting them as a **“theoretical foundation”** when the assumptions and implementation do not match is, in the reviewer's view, weaker than not presenting theory at all.

### W3. PGDR vs. long-tail MoE

The relationship between PGDR and long-tail mixture-of-experts literature is not explained.

---

## Justification of Recommendation

W1 is central: the roughly 14× gap between the cross-domain 42% gain and the in-domain 4% gain is not explained at the mechanism level, and FedPASA actually loses to baselines on at least two in-domain cells:

- MovieLens-100K HR@20;
- MovieLens-1M HR@20.

This puts a substantial question mark on the headline claim that the “three-level reliability yields 42% cross-domain gain.”

The reviewer argues that three alternative explanations cannot be ruled out from what is reported:

1. weak baseline tuning;
2. a benchmark artifact;
3. a cold-start-specific effect.

W2 and W3 further destabilize the overall story:

- Theorem 1 and Theorem 2 have assumption–implementation gaps that the authors themselves acknowledge;
- presenting them under **Theoretical Foundation** weakens rather than strengthens the paper;
- PGDR's relation to long-tail MoE literature is not clarified;
- therefore, the “three-level reliability” framing remains underdetermined.

---

## Specific Points of Feedback for Rebuttal

### 1. Cross-domain vs. in-domain gain gap

Can the authors re-report Table 1's HR@20 and NDCG@20 decomposed by **head vs. tail items**?

If gains concentrate on tail items, how do the authors respond to the reading that PGDR is effective mainly in cold-start regimes?

For the two in-domain cells where FedPASA loses to L2-SP or FedPer:

- MovieLens-100K HR@20;
- MovieLens-1M HR@20;

can the gap be explained by baseline hyperparameter parity?

### 2. Theorem–implementation gap

Theorem 1's optimal anchor is per-`(client,item)`, yet AutoAnchor uses a catalog-level η_T.

Can the authors provide one ablation point for a **per-item** or **per-client learnable anchor** and compare it directly to the catalog-level gate?

If not feasible, would the authors reframe the three theorems as **motivational bounds** rather than as a theoretical foundation?

### 3. PGDR vs. long-tail MoE

Can the authors clarify in one paragraph how PGDR mechanistically differs from head-tail expert routing in:

- ESAM;
- MELU;
- MoE-Rec;
- similar work?

Would the authors add one **federated head-tail MoE baseline** on a single cross-domain direction?

---

# Reviewer 2 — Program Committee cosr

- **Review type:** Official Review
- **Rating:** **5 — Marginally below acceptance threshold**
- **Confidence:** **4 — The reviewer is confident but not absolutely certain that the evaluation is correct**

## Paper Summary

The manuscript studies cross-domain federated recommendation under disjoint item catalogs and heterogeneous target-domain evidence, and proposes FedPASA, a popularity-aware semantic adaptation framework.

The method integrates:

- popularity-gated dual-path routing;
- client-conditioned representation adaptation;
- adaptive semantic anchoring;

to jointly account for:

- item-level popularity;
- client heterogeneity;
- target-domain sparsity.

The proposed components are supported by extensive experiments covering:

- cross-domain recommendation;
- in-domain recommendation;
- ablation studies;
- sensitivity analysis.

---

## Strengths

1. The manuscript is motivated by a relatively clear problem in cross-domain federated recommendation.

2. The proposed framework is supported by corresponding theoretical analyses.

3. The authors conduct a relatively comprehensive set of experiments, including:
   - overall performance comparisons;
   - ablation studies;
   - sensitivity analyses;
   - scalability evaluations;
   - semantic preservation analysis.

These together provide solid empirical support for the effectiveness of the proposed method.

---

## Weaknesses

### W1. Motivation is not sufficiently established

The manuscript only enumerates three challenges:

1. item-level evidence imbalance;
2. client heterogeneity;
3. representation drift;

and directly introduces modules according to the above issues.

However, it does not explain:

- why these three issues constitute the key unresolved challenges in cross-domain federated recommendation;
- how these three issues are related to each other.

Meanwhile, the Related Work section does not clearly position FedPASA within the research of cross-domain federated recommendation.

### W2. Inconsistencies between method description and framework figure

There are several inconsistencies between the method description and the framework figure.

#### (1) Residual adapter is not reflected in the figure

The manuscript states that:

> “source clients learn a residual adapter in the frozen semantic space”.

However, the residual connection is not reflected in the framework figure.

#### (2) FiLM mechanism is unclear

Eq. (6) explains how the client semantic profile is used to generate client-specific modulation parameters.

However, the corresponding **Client Conditioning** module in Fig. 2 explicitly includes a **FiLM Modulation** component.

FiLM is introduced in the Related Work section as an existing conditioning technique. It is unclear whether FedPASA:

- directly adopts the standard FiLM mechanism; or
- develops a modified variant.

### W3. Important notation is insufficiently defined

For example, in


```text
A(x) = norm(x + F(x))
```


the notation x is not explicitly defined.

Similar notation issues also appear elsewhere.

For instance, the function


```text
g()
```


introduced before Eq. (6) is not explained.

The reviewer suggests carefully checking the entire manuscript.

### W4. Proximal regularization is introduced abruptly

The proximal regularization term in the **Training and Aggregation** section is introduced rather abruptly, because neither:

- the need for such a regularization term; nor
- its motivation

is discussed in the preceding description.

Meanwhile, in the proximal regularization term, neither μ nor θ is clearly defined in this context.

---

## Justification of Recommendation

The manuscript is generally well structured, with:

- clear research questions;
- corresponding methodological designs;
- supporting theoretical analysis.

However:

1. the motivation is not sufficiently established;
2. the positioning of this work within cross-domain federated recommendation remains unclear;
3. the methodology contains several unclear or questionable aspects that require further refinement;
4. the experiments lack sufficient comparisons with representative cross-domain federated recommendation baselines.

The reviewer states that if the authors can:

- better clarify the research motivation and positioning;
- address the methodological issues;

the paper would receive a higher rating.

---

## Specific Points of Feedback for Rebuttal

> Please refer to the Weaknesses.

---

# AI Reviewer

## Title

**FedPASA: Popularity-Aware Semantic Adaptation for Cross-Domain Federated Recommendation**

## Synopsis of the Paper

FedPASA addresses cross-domain federated recommendation when source and target item catalogs are disjoint.

It combines:

1. popularity-gated dual-path item adaptation;
2. client-conditioned semantic transformations;
3. a target-density-dependent anchor to pretrained item representations.

The paper derives reliability and drift bounds and evaluates:

- transfer accuracy;
- in-domain accuracy;
- component contributions;
- efficiency;
- scalability;
- robustness to noisy updates.

---

## Summary of Review

The paper presents a coherent integration of item-, client-, and domain-level controls for adapting pretrained semantic representations, and Table 1 reports large cross-domain margins over the included baselines.

However:

- the in-domain evidence is internally incompatible with its stated protocol and does not substantiate consistent per-dataset gains;
- Table 7 incorrectly characterizes heuristic update perturbation as differential privacy;
- the novelty is narrower than presented given prior personalized, gated federated-adaptation methods;
- mechanism attribution is insufficient;
- sampled-ranking specification is insufficient;
- semantic-stability validation is insufficient.

---

## Strengths

### S1. Design matches the motivating hypotheses

Equations 3–8 assign:

- item popularity to routing;
- local semantic profiles to transformation direction;
- target coverage to anchoring strength.

This makes the proposed item-, client-, and domain-level controls independently interpretable.

### S2. Cross-domain margin is substantial

In Table 1, FedPASA exceeds the strongest average baseline by:

- **0.0897 HR@20**;
- **0.0557 NDCG@20**.

These margins are considerably larger than the differences among the nondegenerate baselines.

### S3. Theorem 1 provides a bias–variance rationale

Theorem 1 provides an explicit bias–variance rationale for evidence-dependent anchoring.

Its optimum depends jointly on:

- evidence reliability;
- prior bias.

This clarifies why the appropriate preservation strength should decrease as target evidence becomes more reliable.

### S4. Diagnostic evaluation targets the stated mechanisms

Table 3 and Figure 4 separately examine:

- component removal;
- client heterogeneity;
- low-popularity items;

instead of relying exclusively on aggregate recommendation accuracy.

---

## Weaknesses

### W1. Table 2 does not support the claimed consistent in-domain gains

Under leave-one-out full ranking, EMCDR's Luxury Beauty HR@20 of **0.1820** requires NDCG@20 of at least **0.0414**, but the table reports **0.0269**.

TransFR likewise requires at least **0.0413**, but reports **0.0257**.

FedPASA also:

- loses five of eight dataset/metric comparisons;
- leads the strongest average baseline by only **0.0077 / 0.0019**.

Therefore, Table 2 requires audit before its in-domain conclusion is reliable.

### W2. Table 7 does not establish differential privacy

The perturbation uses a fixed, update-dependent radius and provides no:

- clipping;
- sensitivity bound;
- adjacency definition;
- privacy accountant;
- (ε, δ) values.

For neighboring updates, the resulting distributions generally lie on distinct spheres and need not satisfy finite differential privacy.

Therefore, the experiment supports:

> robustness to heuristic noise

rather than the stated:

> privacy protection

conclusion.

### W3. Novelty positioning overlooks direct precedents

The novelty positioning overlooks direct precedents for personalized, gated federated adaptation.

#### FedPA

FedPA uses:

- private user-level adapters;
- shared group-level adapters;
- adaptive gating.

#### FedVLR

The paper cites FedVLR in Related Work but then groups it under approaches that:

> “usually optimize one adapter rule for all items and clients”

despite FedVLR's client-specific mixture-of-experts fusion.

These methods do not duplicate popularity-aware anchoring, but they narrow the contribution to the specific joint:

- item;
- client;
- domain

control.

### W4. Popularity mechanism is not isolated from added capacity

Equations 4–5 add a second adapter, yet Table 3 lacks a **parameter-matched dual-adapter control with popularity-independent routing**.

Therefore, the current evidence does not establish whether the gain comes from:

- popularity-aware routing; or
- simply increased model capacity.

In addition, the Cross-Domain Transfer Protocol:

- initializes the high-popularity path from the source adapter;
- calls the target-learned low-popularity path a “preservation adapter”;

without stating an architectural or regularization distinction.

Consequently, the evidence does not identify popularity-dependent preservation as the cause of the gain.

### W5. Sampled-ranking protocol is underspecified

Table 1's large margins cannot be interpreted on an absolute scale because the sampled-ranking protocol is underspecified.

The Evaluation and Training Protocol sections omit:

- candidate-set size;
- negative-sampling distribution;
- filtering;
- client construction;
- source–target user overlap.

An HR@20 of **0.3013** has very different meaning under different candidate sets.

These details are necessary to assess:

- comparability;
- practical significance of the headline result.

### W6. Semantic-stability claim lacks an assessable validation measure

Table 4 reports drift decreasing from:


```text
0.9521 → 0.9059
```


without defining:

- the metric;
- its range;
- its reference.

Figure 3 also omits:

- the projection algorithm;
- whether all panels share one fitted projection.

Therefore, neither:

- the absolute stability level; nor
- the visual neighborhood comparison

can be evaluated.

---

## Suggestions for Improvement

### 1. Conflict-focused analysis

A conflict-focused analysis could establish whether the three signals are genuinely complementary.

Examples include:

- global item popularity disagrees with a client's niche preference;
- catalog-level density conflicts with item-level sparsity.

Such cases would test the value of coordinating the controls rather than applying them independently.

### 2. Rewrite around one canonical forward pass

The method would be easier to follow if organized around one canonical forward pass:


```text
Frozen item vector → Routing → Conditioning → Anchoring → Final scoring
```


Introducing each:

- frozen variable;
- local variable;
- communicated variable

once in that sequence would remove the need to reconcile the equations with Figure 2.

### 3. Failure analysis

A failure analysis stratified jointly by:

- item frequency;
- client activity;
- source–target semantic mismatch

would clarify the method's operating regime.

In particular, it could identify whether anchoring harms sparse items whose pretrained semantic representations are systematically biased for the target domain.

---

## Additional Details and Minor Issues

### 1. Equation 1 and Equation 7 are not reconnected

Equation 1 scores


```text
z_c,i,
```


whereas Equation 7 introduces


```text
z^anchor_c,i
```


as the final representation without reconnecting it to the score.

### 2. Figure 2 notation is inconsistent with the equations

Figure 2 uses:


```text
A_w, A_c, g_θ,
```


while the equations use:


```text
A_h, A_l, g_φ.
```


Its ranking head is also not reconciled with Equation 1.

### 3. Figure 4 introduces undefined hyperparameters

Figure 4 introduces:

- popularity threshold;
- popularity power;
- low-popularity parameter t;

without definitions in the method or experimental setup.

### 4. Training/communication description is inconsistent

The training section says only:

- adapters;
- conditioner

are aggregated.

However, Table 7's setup includes transmitted:

> “anchor-control parameters”.

### 5. Theorems 2–3 only concern pre-normalization vectors

Theorems 2–3 concern pre-normalization vectors, as the paper acknowledges.

Therefore, contribution-level descriptions of score stability should retain this qualifier.

---

# AI Reviewer — References

1. **Li, Z., Long, G., Jiang, J., Zhang, C., & Yang, Q. (2026).**  
   *Federated vision-language-recommendation with personalized fusion.*  
   Proceedings of the AAAI Conference on Artificial Intelligence, 40(28), 23337–23345.

2. **Zhang, C., Long, G., Guo, H., Fang, X., Song, Y., Liu, Z., Zhou, G., Zhang, Z., Liu, Y., & Yang, B. (2024).**  
   *Federated adaptation for foundation model-based recommendations.*  
   Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, 5453–5461.

---

# 合并后的问题索引

下面将三份 review 的问题按修改主题归并，方便后续逐项检查。

## A. 核心实验可信度

- [ ] Cross-domain 42% vs. in-domain 4% 的机制差距需要解释。
- [ ] Table 2 的 HR@20 / NDCG@20 需要重新审计。
- [ ] “consistent in-domain gains” 是否需要弱化或修改。
- [ ] 需要排除 baseline tuning / benchmark artifact 的可能性。
- [ ] sampled-ranking protocol 需要补完整。
- [ ] semantic drift metric 需要正式定义。
- [ ] Figure 3 的 projection protocol 需要说明。

## B. PGDR 机制归因

- [ ] 增加 parameter-matched dual-adapter control。
- [ ] 区分 popularity-aware routing 与额外模型容量。
- [ ] 明确 PGDR 与 long-tail MoE / head-tail routing 的机制差异。
- [ ] 最好补一个 federated head-tail MoE baseline。
- [ ] 重新说明 “preservation adapter” 的真正机制含义。

## C. AutoAnchor / Theory

- [ ] 解决 per-(c, i) theoretical anchor 与 domain-level η_T 的 gap。
- [ ] 考虑 per-item / per-client learnable anchor ablation。
- [ ] 重新处理 normalization 前后理论成立范围。
- [ ] 将 “Theoretical Foundation” 考虑改为 “Theoretical Motivation” 或类似表述。
- [ ] 所有 stability claim 保留 pre-normalization 限定。

## D. 三层 Reliability 统一故事

- [ ] 解释为什么 item / client / domain 是三个关键 unresolved dimensions。
- [ ] 解释三者之间如何关联，而不是三个独立 challenge。
- [ ] 更清楚地定位 FedPASA 在 cross-domain federated recommendation 中的位置。
- [ ] 可增加 signal-conflict analysis。

## E. Recent Related Work / Novelty

- [ ] 补充 FedPA。
- [ ] 重新定位 FedVLR。
- [ ] 补充 long-tail MoE / head-tail expert routing。
- [ ] 补充 representative cross-domain federated recommendation baselines。
- [ ] 不再泛化声称已有方法都是 “one adapter rule for all items and clients”。
- [ ] 将 novelty 聚焦到 joint item/client/domain reliability control。

## F. Privacy / Robustness

- [ ] Table 7 不应直接称为 formal differential privacy。
- [ ] 若不补完整 DP，需要改成 robustness to update perturbation。
- [ ] 若坚持 DP，需要补 clipping、sensitivity、adjacency、accountant、(ε, δ)。

## G. Figure / Formula / Notation

- [ ] 在 Figure 2 中画出 residual connection。
- [ ] 说明 FiLM 是 standard FiLM 还是 modified FiLM。
- [ ] 定义 x。
- [ ] 定义 `g(·)`。
- [ ] 统一 `A_w,A_c` 与 `A_h,A_l`。
- [ ] 统一 g_θ 与 g_φ。
- [ ] 重新连接 Eq. 1 的 score 与 Eq. 7 的 z_anchor_(c,i)。
- [ ] 定义 popularity threshold。
- [ ] 定义 popularity power。
- [ ] 定义 low-popularity parameter t。

## H. Training / Aggregation

- [ ] 解释 proximal regularization 的必要性。
- [ ] 定义 proximal term 中的 μ 与 θ。
- [ ] 统一正文与 Table 7 对 aggregated/transmitted parameters 的描述。
- [ ] 明确 anchor-control parameters 是否参与通信与聚合。

---

# 最优先需要解决的四类问题

1. **Cross-domain 与 In-domain 巨大增益差距的机制解释。**
2. **Theory 与实际 AutoAnchor implementation 的 gap。**
3. **PGDR 是否只是额外容量 / long-tail MoE 的替代解释。**
4. **Table 2、Privacy claim、Evaluation Protocol 等实验可信度问题。**

