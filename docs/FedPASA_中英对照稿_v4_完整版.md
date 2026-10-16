# FedPASA 中英对照稿（v4 完整版）

每段中文下面紧跟英文译稿，表格与图完整保留。`[待补引用]` 与图1核验事项见文末清单。

---

## 摘要

跨域联邦推荐在物品目录互不重叠的服务之间迁移推荐知识，同时将用户交互保留在本地。文本表示能够连接不同目录中的物品，但目标域的监督信号在物品之间和客户端之间分布不均。

Cross-domain federated recommendation transfers knowledge between services with disjoint item catalogs while keeping user interactions local. Text representations connect these catalogs, but target supervision is uneven across items and clients.

本文提出联邦语义适配框架 FedPASA：按物品交互频率将表示路由到两条残差适配路径之间；由私有客户端画像生成 FiLM 参数，对共享变换进行条件化；由目录级门控控制适配表示与冻结语义先验之间的插值。

We propose FedPASA, a federated semantic adaptation framework: item frequency routes each representation between two residual adapters; a private client profile conditions the shared transformation through FiLM; and a catalog-level gate controls interpolation with the frozen semantic prior.

在 Amazon 六个迁移方向和五个域内数据集上，使用相同编码器时，FedPASA 的平均采样 HR@20 比 TransFR 风格基线高 26.8%，比 FedMoE 高 23.7%，并在全目录排序下取得最高平均准确率。等参数量对照表明，基于频率的路由优于固定分配和打乱分配。按物品频率与客户端接触情况的分组分析刻画了条件化与锚定各自有效和失效的范围。

On six Amazon transfer directions and five in-domain datasets with a shared encoder, FedPASA improves mean sampled HR@20 by 26.8% over the TransFR-style baseline and by 23.7% over FedMoE, and achieves the highest mean accuracy under full-catalog ranking. With identical parameter counts, frequency-based routing outperforms constant and shuffled assignments. Group analyses by item frequency and client exposure characterize where conditioning and anchoring help and where their gains vanish.

**关键词：** 联邦推荐；跨域推荐；语义迁移；长尾推荐。

Keywords: Federated Recommendation; Cross-Domain Recommendation; Semantic Transfer; Long-tail Recommendation

**CCS 分类：** 信息系统——推荐系统；计算方法——分布式计算方法。

CCS: Information systems—Recommender systems; Computing methodologies—Distributed computing methodologies



**Federated recommendation (FR) trains ranking models across clients while keeping user interactions on local devices. However, transferring such models across services is difficult: item catalogs are disjoint, and target supervision is uneven—frequently observed items provide abundant evidence, whereas rare items and client-specific preferences are poorly served by a single shared transformation. Prior approaches adapt pretrained text representations with one uniform update rule, which under-fits well-supported items while over-fitting sparse ones. In response, we propose FedPASA, a reliability-aware semantic adaptation framework that adjusts item representations according to the strength of target evidence while keeping user data local. FedPASA routes each item between two residual adapters according to its interaction frequency, modulates the shared transformation with feature-wise scales and shifts generated from a private client profile, and interpolates the adapted representation with the frozen semantic prior under a catalog-level gate. With the parameter count held fixed, shuffling the correspondence between item frequency and routing paths substantially degrades accuracy, indicating that the gains stem from the frequency-based design itself rather than increased capacity. Experiments on cross-domain and in-domain benchmarks show that FedPASA consistently outperforms baselines sharing the same encoder under both sampled and full-catalog ranking, improving mean sampled HR@20 by 23.7% over the strongest baseline and by up to 44.8% on individual transfer directions.**

联邦推荐（FR）在多个客户端上协同训练排序模型，同时将用户交互保留在本地设备。然而，将此类模型迁移到其他服务十分困难：物品目录互不重叠，且目标域的监督信号分布不均——高频物品提供充足证据，而低频物品和各客户端的局部偏好难以被单一共享变换同时适配。已有方法用同一套更新规则适配预训练文本表示，对证据充足的物品拟合不足，对稀疏物品又过度拟合。为此，我们提出 FedPASA，一个可靠性感知的语义适配框架，在用户数据不出本地的前提下，根据目标证据的强度调整物品表示。FedPASA 按物品交互频率将表示路由到两条残差适配路径之间，由私有客户端画像生成逐特征的缩放与平移参数以调节共享变换，并在目录级门控下将适配表示与冻结语义先验插值。在参数量保持不变的对照中，打乱物品频率与路径的对应关系会使性能明显回落，表明收益来自按频率分配路径这一设计本身，而非模型容量的增加。跨域与域内基准上的实验表明，FedPASA 在采样排序与全目录排序下均稳定优于使用相同编码器的基线，平均采样 HR@20 较最强基线提升 23.7%，单个迁移方向上最高提升 44.8%。







## 1 引言

联邦推荐（Federated Recommendation，FR）通过多个客户端协同训练排序模型，同时将原始交互历史保留在本地 [1,17]。对 Web 交易平台而言，一个核心需求是把推荐模型迁移到另一个物品目录：物品标识随平台变化，新上架物品缺少行为证据，参与服务也无法直接汇集用户历史。本文研究目录互不重叠的设定，不要求源域与目标域之间存在可关联的用户。

Federated recommendation (FR) trains ranking models across clients while retaining raw interaction histories locally [1,17]. A central challenge for Web marketplaces is transferring a recommendation model to another catalog: item identifiers change across platforms, newly listed items lack behavioral evidence, and participating services cannot simply pool user histories. We study this disjoint-catalog setting without requiring linked users between source and target domains.

预训练文本编码器将不同目录的物品映射到共同的表示空间，无需匹配物品标识即可完成语义迁移 [24]。剩下的问题是如何把这个空间适配到不均衡的目标监督上。高频物品提供大量排序约束，低频物品几乎不提供适配证据；不同客户端通过各自历史观测到语义空间的不同区域，一个全局低频物品可能对某个客户端很熟悉、对另一个完全未见。适配规则需要同时考虑物品级证据的数量，以及证据被观测时所处的客户端上下文。

Pretrained text encoders map items from different catalogs into a common representation space, enabling semantic transfer without matching identifiers [24]. The remaining problem is adapting this space to uneven target supervision. Frequently observed items supply many ranking constraints, whereas rare items provide little evidence for adaptation. Clients observe different semantic regions through their interaction histories: a globally rare item may be familiar to one client and unseen by another. An adaptation rule needs to account for both the amount of item-level evidence and the client context in which it is observed.

个性化适配器和专家混合可以处理客户端间的差异 [14,23]，但客户端级个性化本身并不规定不同训练频率的物品应如何共享适配路径：单一共享变换对所有物品使用同一参数化，再叠加一个适配器只是增加容量，没有规定容量如何分配。本文将物品频率作为目标监督量的可观测代理，把每个物品路由到两条残差路径之间，再通过客户端条件化将路由后的表示调整到本地交互画像上。

Personalized adapters and expert mixtures address variation across clients [14,23], but client-level personalization does not by itself specify how items with different training frequencies should share adaptation paths: a single shared transformation applies the same parameterization to all items, and an additional adapter introduces capacity without determining how to allocate it. We use item frequency as an observable proxy for the available target supervision, route each item between two residual paths, and then adjust the routed representation to the local interaction profile through client conditioning.

我们提出 **FedPASA**，在一个共享语义适配器中组合三个操作。流行度门控双路径路由（Popularity-Gated Dual-path Routing，PGDR）用频率导出的权重混合源域初始化路径和目标域初始化路径；客户端条件化用本地聚合的语义画像生成 FiLM 参数；AutoAnchor 将所得表示与冻结文本表示插值，锚定强度由目录级用户—物品数量比确定。两条残差路径均在目标域上更新，源域初始化只提供优化起点，不固定任何一条路径的表示。

We propose FedPASA, which combines three operations in a shared semantic adapter. Popularity-Gated Dual-path Routing (PGDR) mixes a source-initialized path and a target-initialized path using a frequency-derived weight; client conditioning produces FiLM parameters from a locally pooled semantic profile; and AutoAnchor interpolates the resulting representation with the frozen text embedding, with the anchor strength set by a catalog-level user-to-item ratio. Both residual paths are updated on the target domain; source initialization provides a starting point rather than a fixed representation.

本文将这三个机制的作用与最终本地个性化分开评估。固定路由和打乱路由保持与 FedPASA 相同的参数量，用于区分频率分配与模型容量；无源初始化变体用于区分预训练语义表示本身与源适配器初始化的贡献。

We separate these effects from final local personalization. Constant and shuffled routing retain FedPASA's parameter count to isolate frequency assignment from capacity; a source-free variant separates the benefit of the pretrained semantic representation from that of source-adapter initialization.

我们在六个迁移方向、五个域内数据集上评估该设计，并用对照实验分离频率分配、模型容量和最终个性化的影响。本文的主要贡献如下：

We evaluate this design on six transfer directions and five in-domain datasets, with controls that separate frequency assignment, model capacity, and final personalization. The principal contributions are summarized as follows:

- 提出面向不重叠物品目录的联邦语义适配框架 FedPASA，将频率路由的残差路径、私有客户端条件化和目录级语义锚定组合在一个共享适配器中。

  We propose FedPASA for semantic adaptation across disjoint catalogs, combining frequency-routed residual paths with private client conditioning and catalog-level anchoring in one shared adapter.

- 在采样排序与全目录排序下开展系统实验。六个迁移方向上，FedPASA 的平均采样 HR@20 比同编码器下平均最强的基线 FedMoE 高 23.7%。

  We conduct systematic experiments under sampled and full-catalog ranking. On six transfer directions, FedPASA improves mean sampled HR@20 by 23.7% over FedMoE, the strongest baseline on average with the same encoder.

- 等参数量对照表明频率路由优于固定和打乱分配；模块移除和频率—接触情况联合分组进一步刻画了条件化与锚定各自不同、依赖分组的作用。

  We show that routing gains persist against constant and shuffled assignments with identical parameter counts. Component removals and joint frequency–exposure groups further characterize the distinct, group-dependent roles of conditioning and anchoring.





**With the growth of e-commerce and digital services, recommender systems have become essential infrastructure for online platforms. However, traditional recommender systems typically consolidate vast amounts of user data centrally, leading to privacy and regulatory concerns. Federated learning allows multiple clients to collaboratively train a model without sharing raw data; federated recommendation (FR) applies this idea to recommendation, where interaction histories remain on local devices and only model parameters are aggregated [1,17].**

**Meanwhile, cross-domain transfer is another longstanding need: platforms wish to reuse a mature catalog to bootstrap a new service, where newly listed items lack behavioral evidence. However, existing cross-domain recommenders typically assume overlapping users or alignable item identifiers, and neither premise holds under federated constraints—participating services cannot pool user histories, and disjoint catalogs share no identifiers to align. We study this disjoint-catalog setting without requiring linked users between source and target domains.**

**A viable route is offered by pretrained text encoders: items from different catalogs are mapped into a common semantic space, enabling transfer without matching identifiers [24]. However, the semantic space alone does not resolve the uneven supervision of the target domain. On the one hand, frequently observed items supply many ranking constraints, whereas rare items provide little evidence for adaptation; on the other hand, clients observe different semantic regions through their own histories, so a globally rare item may be familiar to one client yet unseen by another. Moreover, excessive adaptation drifts representations away from the pretrained semantics, harming items with scarce evidence. An adaptation rule therefore needs to account for the amount of item-level evidence, the client context in which it is observed, and how much of the pretrained prior should be preserved.**

**Personalized adapters and expert mixtures address variation across clients [14,23], but client-level personalization does not by itself specify how items with different training frequencies should share adaptation paths: a single shared transformation applies the same parameterization to all items, and an additional adapter introduces capacity without determining how to allocate it. In this light, we use item frequency as an observable proxy for the available target supervision, route each item between two residual paths, and adjust the routed representation to the local interaction profile through client conditioning.**

**To address the challenges mentioned above, we propose FedPASA, which combines three operations in a shared semantic adapter. Specifically, Popularity-Gated Dual-path Routing (PGDR) mixes a source-initialized path and a target-initialized path using a frequency-derived weight; client conditioning produces feature-wise scales and shifts from a locally pooled semantic profile; in addition, AutoAnchor interpolates the resulting representation with the frozen text embedding, with the anchor strength set by a catalog-level user-to-item ratio. Both residual paths are updated on the target domain, while source initialization provides a starting point rather than a fixed representation.**

**We evaluate this design on multiple real-world datasets, with controls that separate frequency assignment, model capacity, and final personalization. Our contributions are summarized as follows:**

- **We propose FedPASA, a reliability-aware semantic adaptation framework that adjusts item representations according to evidence strength at the item, client, and catalog levels. Our framework is motivated by an analysis of the failure of uniform adaptation in existing federated semantic transfer methods under uneven target supervision: a single shared transformation underfits well-supported frequent items while overfitting sparsely observed items. The proposed framework applies to different backbone encoders.**
- **We design PGDR to combine two trainable residual adaptation paths using frequency-derived weights, without an additional learned routing network. Client conditioning adjusts the combined representations using local semantic profiles, while AutoAnchor controls their mixture with the frozen semantic prior through a catalog-level rule.**
- **We conduct extensive experiments on multiple real-world datasets. With the same text encoder, FedPASA achieves higher average cross-domain recommendation accuracy than the evaluated baselines. Comparisons that retain the same network structure and parameter count further show that frequency-derived mixing weights outperform using a common pair of weights for all items or shuffling the item–weight assignments. Ablation studies further characterize the effects of individual components.**



**随着电子商务与数字服务的发展，推荐系统已成为在线平台的核心基础设施。然而，传统推荐系统通常将大量用户数据集中收集，带来隐私与合规压力。联邦学习允许多个客户端在不共享原始数据的前提下协同训练模型；联邦推荐（Federated Recommendation，FR）将这一思想引入推荐场景，用户交互历史保留在本地，仅上传模型参数参与聚合 [1,17]。**

**与此同时，跨域迁移是推荐系统的另一长期需求：平台希望把成熟业务上的推荐能力迁移到物品目录不同的新业务，而新上架物品往往缺少行为证据。然而，已有跨域推荐方法大多假设存在重叠用户或可对齐的物品标识；在联邦约束下，这两个前提均不成立——参与服务无法汇集用户历史，互不重叠的目录之间也没有可对齐的标识。本文研究的正是这一目录互不重叠、且不要求源域与目标域存在可关联用户的设定。**

**一条可行路线是利用预训练文本编码器：不同目录的物品被映射到同一语义空间，无需匹配标识即可完成迁移 [24]。然而，语义空间本身并不能解决目标域监督不均的问题。一方面，高频物品提供大量排序约束，而低频物品几乎不提供适配证据；另一方面，不同客户端通过各自历史观测到语义空间的不同区域，一个全局低频物品可能对某个客户端很熟悉、对另一个完全未见。此外，过度适配会使表示偏离预训练语义，损害本就缺乏证据的物品。因此，适配规则需要同时考虑物品级证据的数量、证据被观测时所处的客户端上下文，以及预训练先验应保留的程度。**

**个性化适配器与专家混合可以处理客户端之间的差异 [14,23]，但客户端级个性化本身并不规定不同训练频率的物品应如何共享适配路径：单一共享变换对所有物品使用同一参数化，再叠加一个适配器只是增加容量，并未规定容量如何分配。鉴于此，本文将物品频率作为目标监督量的可观测代理，把每个物品路由到两条残差路径之间，再通过客户端条件化将路由后的表示调整到本地交互画像上。**

**为解决上述问题，我们提出 FedPASA，在一个共享语义适配器中组合三个操作。具体而言，流行度门控双路径路由（Popularity-Gated Dual-path Routing，PGDR）用频率导出的权重混合源域初始化路径和目标域初始化路径；客户端条件化用本地聚合的语义画像生成逐特征的缩放与平移参数；此外，AutoAnchor 将所得表示与冻结文本表示插值，锚定强度由目录级用户—物品数量比确定。两条残差路径均在目标域上更新，而源域初始化只提供优化起点，不固定任何一条路径的表示。**

**我们在多个真实世界数据集上评估该设计，并用对照实验分离频率分配、模型容量和最终个性化的影响。本文的主要贡献如下：**

- **我们分析了现有联邦语义迁移方法的均匀适配规则在目标监督不均场景下的失效问题：单一共享变换对证据充足的高频物品拟合不足，同时对缺乏证据的低频物品过度适配。为此，我们提出可靠性感知的语义适配框架 FedPASA，根据物品、客户端与目录三个层面的证据强度调整物品表示，且该范式适用于不同的骨干编码器。**
- **我们引入流行度门控双路径路由（PGDR），以物品交互频率导出的确定性权重在源域初始化路径与目标域初始化路径之间分配监督，并与基于私有客户端画像的条件化和目录级语义锚定相结合，使适配强度随证据多少与客户端上下文变化，而不依赖可学习的门控参数。**
- **我们在多个真实世界数据集上进行了广泛实验，涵盖多种迁移方向与排序评估设置。FedPASA 的平均采样 HR@20 较同编码器下最强基线提升 23.7%，等参数量对照进一步验证了收益来自频率分配设计而非模型容量的增加。**

![fedpasa_innovation_concept](E:\小论文\小论文4-联邦推荐\小论文aaai-www\FedPASA_WWW2027_最终版_LaTeX\figures\fedpasa_innovation_concept.png)

## 2 相关工作

### 2.1 联邦语义迁移

联邦推荐用共享参数从分散的交互数据中学习 [1,17]。近端优化和本地模型组件用于处理统计异质性 [2,4,10,11]。基于物品标识的模型需要对齐关系或内容信息才能迁移到不重叠的目录。TransFR 将预训练语言表示与适配器调优、本地个性化结合 [24]；FedDCSR 解耦共享与领域特定的序列因子 [25]；PFCR 和 FFMSR 迁移文本内容表示 [5,15]；FedCRF 引入语义图与对比融合 [6]；FeDecider 通过客户端特定的方向权重个性化联邦 LoRA 聚合 [8]。这些方法分别侧重序列建模、语义融合或客户端级聚合。FedPASA 关注的是共享语义适配器内部的物品级监督分配。

FR learns from decentralized interactions using shared parameters [1,17]. Proximal optimization and local model components address statistical heterogeneity [2,4,10,11]. Item-ID models require alignment or content information to transfer to a disjoint catalog. TransFR combines pretrained language representations with adapter tuning and local personalization [24]; FedDCSR disentangles shared and domain-specific sequential factors [25]; PFCR and FFMSR transfer text-derived content representations [5,15]; FedCRF adds semantic graphs and contrastive fusion [6]; FeDecider personalizes federated LoRA aggregation through client-specific directional weights [8]. These methods emphasize sequential modeling, semantic fusion, or client-level aggregation. FedPASA focuses on assigning item-level supervision within a shared semantic adapter.

### 2.2 个性化适配器与专家路由

FedPA 组合私有用户适配器、共享群组适配器和自适应门控 [23]；FedVLR 用客户端特定的专家混合进行多模态融合 [14]。长尾不均衡方面，ESAM 对未曝光物品做判别式域适配 [3]，MeLU 通过元学习实现快速偏好适配 [9]；推荐与多任务场景中的专家混合模型（多门控 MoE [16]、MoE-Rec 类方法 [待补引用]）按输入学习专家组合权重。PGDR 与这些工作的机制差异在于：路由权重不学习，由目标物品频率确定性地给出，并与客户端条件化组合，使监督分配规则显式可读。我们与可学习路由、以及保持模型容量不变的固定和打乱分配做对照。

FedPA combines private user adapters, shared group adapters, and adaptive gating [23]; FedVLR uses client-specific expert mixtures for multimodal fusion [14]. For long-tail imbalance, ESAM performs discriminative domain adaptation with non-displayed items [3] and MeLU achieves rapid preference adaptation through meta-learning [9]; mixture-of-experts models in recommendation and multi-task settings [16, to-be-added] learn input-dependent expert weights. PGDR differs mechanistically: its routing weight is not learned but deterministically assigned from target item frequency, and it is combined with client conditioning, making the allocation of supervision explicit. We compare it with learned routing and with constant or shuffled assignments that preserve model capacity.

### 2.3 条件化与先验保留

FiLM 用逐特征的缩放和平移调节中间表示 [19]。FedPASA 在共享适配器中用私有客户端画像生成这些调节参数，并对乘性项施加有界约束（见第 3.3 节）。L2-SP 和 DELTA 分别通过参数或特征惩罚保留源域信息 [12,13]；AutoAnchor 直接作用于归一化后的排序表示，将其与冻结语义先验插值，其目录级门控是固定的覆盖度启发式规则。第 4 节给出偏差—方差动机与插值几何，第 5 节评估其对排序的影响。

FiLM modulates intermediate features with feature-wise scales and shifts [19]. FedPASA generates these parameters from a private client profile within the shared adapter, with a bounded multiplicative term (§3.3). L2-SP and DELTA preserve source information through parameter or feature penalties [12,13]; AutoAnchor instead acts directly on the normalized ranking representation by interpolating it with the frozen semantic prior, and its catalog-level gate is a fixed coverage heuristic. Section 4 gives the bias–variance motivation and the interpolation geometry; Section 5 evaluates the effect on ranking.











## 3 FedPASA

### 3.1 问题定义与整体流程

记客户端集合为 $\mathcal C$，用户集合为 $\mathcal U$，目标物品目录为 $\mathcal I$。冻结文本编码器将物品 $i$ 映射为经过 $\ell_2$ 归一化的向量 $x_i\in\mathbb R^d$，本文将其称为**语义先验**。用户 $u$ 的私有表示为 $e_u\in\mathbb R^d$，存储于其所属客户端。

Let $\mathcal C$ be the clients, $\mathcal U$ their users, and $\mathcal I$ the target item catalog. A frozen text encoder maps each item $i$ to an $\ell_2$-normalized vector $x_i\in\mathbb R^d$, which we call the *semantic prior*. A user $u$ holds a private embedding $e_u\in\mathbb R^d$ stored on its client.

源域训练得到一个残差适配器。目标域训练时，一条路径使用该适配器初始化，另一条路径随机初始化，随后两条路径均利用目标域交互进行优化。对于客户端 $c$ 上的用户，推荐得分为

Source-domain training produces one residual adapter. In target-domain training, one path is initialized from this adapter and the other at random; both paths are then optimized on target interactions. For a user on client $c$, the recommendation score is

$$
s(u,i)=e_u^\top z^{\mathrm{anchor}}_{c,i}.
\tag{1}
$$

训练采用 BPR 损失 [21]：

Training minimizes the BPR loss [21]:

$$
\mathcal L_{\mathrm{BPR}}
=
-\mathbb E_{(u,i,j)}
\log\sigma\left(\tau[s(u,i)-s(u,j)]\right),
$$

其中，$i$ 为已交互的正样本，$j$ 为采样负样本，$\tau=10$ 为得分缩放系数。

where $i$ is an observed positive, $j$ a sampled negative, and $\tau=10$ the logit scale.

图1展示 FedPASA 的前向计算流程，表7汇总主要符号：

Figure 1 shows the forward pass of FedPASA and Table 7 collects the notation:

$$
x_i
\xrightarrow{\mathrm{routing}}
h_i
\xrightarrow{\mathrm{FiLM\ and\ residual}}
z_{c,i}
\xrightarrow{\mathrm{anchoring}}
z^{\mathrm{anchor}}_{c,i}.
\tag{2}
$$

![图1 FedPASA整体框架](E:/小论文/小论文4-联邦推荐/小论文aaai-www/FedPASA_WWW2027_最终版_LaTeX/figures/framework_ppt.png)

![FedPASA_framework_with_BPR](E:\小论文\小论文4-联邦推荐\output\FedPASA_framework\FedPASA_framework_with_BPR.png)

**图1：FedPASA 整体框架。**（a）源域预训练用于初始化 $A_h$，$A_l$ 在目标域上初始化。（b）服务器交换共享适配器及条件化网络的参数。（c）频率路由、基于私有画像的 FiLM，以及归一化先验插值共同构成一次前向计算。底部展示适配器中的两个残差块和最终本地个性化阶段。实线表示计算过程，虚线表示参数交换。

Figure 1: FedPASA. (a) Source pretraining initializes $A_h$; $A_l$ is initialized on the target. (b) The server exchanges shared adapter and conditioner parameters. (c) Frequency routing, private-profile FiLM, and normalized prior interpolation form one forward pass. The bottom panels show the two residual blocks and final local personalization. Solid arrows show computation; dashed arrows exchange parameters.

### 3.2 流行度门控双路径路由

记 $\mathrm{freq}(i)$ 为目标域训练交互中物品 $i$ 的出现次数。我们假设初始化阶段能够获得聚合后的训练频次和目录规模，这些统计量与私有客户端画像相区别。归一化低频程度得分定义为

Let $\mathrm{freq}(i)$ count target training interactions containing item $i$. We assume aggregate training counts and catalog sizes are available at setup; they are distinct from private client profiles. The normalized coldness score is

$$
\bar q_i=\frac{1}{\sqrt{\mathrm{freq}(i)+1}},
\qquad
q_i=
\frac{\bar q_i-\min_j\bar q_j}
{\max_j\bar q_j-\min_j\bar q_j+\epsilon}.
\tag{3}
$$

利用阈值 $t$ 和指数 $\kappa$ 调整路由得分：

A threshold $t$ and exponent $\kappa$ reshape the routing score:

$$
q_i'
=
\mathrm{clip}\left(
\frac{q_i-t}{1-t},0,1
\right)^\kappa.
$$

默认设置为 $t=0$、$\kappa=0.5$，敏感性分析见第 5.7 节。

We use $t{=}0$ and $\kappa{=}0.5$ (sensitivity in §5.7).

路由在头部适配器 $A_h$ 与尾部适配器 $A_l$ 之间进行混合。其中，$A_h$ 使用源域适配器初始化，$A_l$ 在目标域随机初始化。每个适配器包含两个顺序连接的残差块，

Routing mixes a head adapter $A_h$, initialized from the source adapter, and a tail adapter $A_l$, randomly initialized on the target. Each adapter comprises two sequential residual blocks,

$$
v\leftarrow v+F_\ell(v),
$$

随后进行 $\ell_2$ 归一化。$F_\ell$ 为瓶颈多层感知机。路由结果为

followed by $\ell_2$ normalization; each $F_\ell$ is a bottleneck MLP. The routing step is

$$
h_i=(1-q_i')A_h(x_i)+q_i'A_l(x_i).
\tag{4}
$$

混合表示 $h_i$ 随后进入客户端条件化模块。

The mixed representation $h_i$ then enters client conditioning.

### 3.3 客户端条件化

客户端 $c$ 根据本地交互物品的多重集合 $\mathcal I_c$ 构建画像，重复交互按实际出现次数参与聚合。共享条件化网络 $g_\phi:\mathbb R^d\rightarrow\mathbb R^{2d}$ 为瓶颈多层感知机，根据该画像生成 FiLM 参数：

Client $c$ forms a profile over its multiset $\mathcal I_c$ of locally interacted items, so repeated interactions retain their multiplicity. The shared conditioner $g_\phi:\mathbb R^d\rightarrow\mathbb R^{2d}$, a bottleneck MLP, produces FiLM parameters from it:

$$
p_c=
\mathrm{norm}\left(
\frac{1}{|\mathcal I_c|}
\sum_{i\in\mathcal I_c}x_i
\right),
\qquad
(\gamma_c,\beta_c)=g_\phi(p_c).
\tag{5}
$$

调节采用 FiLM 的有界变体：乘性项经 $\tanh$ 压缩、两项统一由强度系数 $\lambda$ 缩放，随后与头部路径残差混合：

The modulation is a bounded variant of FiLM: the multiplicative term is compressed by $\tanh$ and both terms are scaled by a strength coefficient $\lambda$, followed by a residual mixture with the head path:

$$
\begin{aligned}
m_{c,i}
&=
h_i\odot\left(1+\lambda\tanh\gamma_c\right)
+\lambda\beta_c,\\
z_{c,i}
&=
\mathrm{norm}\left(
(1-\alpha)A_h(x_i)+\alpha m_{c,i}
\right).
\end{aligned}
\tag{6}
$$

有界调制避免个别客户端画像极端时调节幅度失控。$\lambda$ 默认值为 $0.2$；$\alpha\in[0,1]$ 默认值为 $0.85$，控制条件化分支相对头部路径的配比。仅共享参数 $\phi$ 参与通信，画像 $p_c$ 不离开客户端。

The bounded modulation prevents extreme client profiles from producing unbounded modulation. We use $\lambda=0.2$; $\alpha\in[0,1]$, set to $0.85$, controls the conditioned mixture relative to the head path. Only $\phi$ is communicated; the profile $p_c$ never leaves the client.

### 3.4 目录级语义锚定

定义目标域用户—物品数量比为

Define the target user-to-item ratio as

$$
\eta_T=\frac{|\mathcal U_T|}{|\mathcal I_T|},
$$

并设置阈值区间 $[\tau_l,\tau_h]=[1.0,1.5]$。锚定权重及最终物品表示为

with an interval $[\tau_l,\tau_h]=[1.0,1.5]$. The anchor weight and the final item representation are

$$
\begin{aligned}
\rho_{\mathrm{eff}}
&=
\rho\cdot
\mathrm{clip}\left(
\frac{\tau_h-\eta_T}{\tau_h-\tau_l},0,1
\right),\\
z^{\mathrm{anchor}}_{c,i}
&=
\mathrm{norm}\left(
(1-\rho_{\mathrm{eff}})z_{c,i}
+\rho_{\mathrm{eff}}x_i
\right).
\end{aligned}
\tag{7}
$$

其中，$\rho=0.3$ 为最大锚定权重。

where $\rho=0.3$ is the maximum anchor weight.

用户—物品数量比用于粗略刻画目录级监督覆盖程度。当 $\eta_T\leq\tau_l$ 时，使用最大先验权重；当 $\eta_T\geq\tau_h$ 时，不进行锚定。这里的 $|\mathcal U_T|$ 表示实际观察到的用户数量，而非用户标识空间的大小。

The user-to-item ratio is a coarse catalog-level proxy for supervision coverage. Targets with $\eta_T\leq\tau_l$ use the maximum prior weight, and targets with $\eta_T\geq\tau_h$ use no anchor. Here $|\mathcal U_T|$ counts observed users, not the size of the user-ID space.

我们还评估了逐物品可学习门控：

We also evaluate a per-item learnable gate:

$$
\rho_i=\rho\,\sigma(a_i),
$$

其中，$a\in\mathbb R^{|\mathcal I|}$ 参与联邦聚合。相关结果见第 5.3 节和表20。

with $a\in\mathbb R^{|\mathcal I|}$ federated. Results appear in §5.3 and Table 20.

### 3.5 训练与联邦聚合

服务器广播共享参数

The server broadcasts the shared parameters

$$
\theta=\{\theta_h,\theta_l,\phi\}.
$$

客户端通过 BPR 损失更新共享模型副本和本地用户表示，服务器使用 FedAvg 聚合模型参数。目录级门控不包含可训练参数；逐物品门控变体还会聚合物品对应的参数 $a$。

Clients update their copies of these parameters and their local user embeddings with BPR, and the server applies FedAvg. The catalog gate has no trainable parameters; the per-item variant additionally federates its item logits $a$.

本地更新可加入近端约束：

Local updates can be regularized by a proximal term:

$$
\frac{\mu}{2}\|\theta-\theta_{\mathrm{ref}}\|_2^2.
$$

对于 FedProx 风格训练，参考参数为本轮广播模型；对于最终个性化阶段，参考参数为最终全局模型 [10,11]。该约束限制本地更新偏离参考模型；个性化阶段防止私有数据覆盖共享适配结果。

For FedProx-style training the reference is the broadcast model, and for final personalization it is the final global model [10,11]. The penalty keeps local updates close to the reference and, during personalization, prevents private data from overwriting the shared adaptation.

默认 FedPASA 在联邦训练期间设置 $\mu=0$，随后进行 40 步本地个性化，近端系数为 $\mu_p=0.15$。个性化后的参数不再上传。本文将未进行最终个性化的模型单独记为 **FedPASA base**。

Default FedPASA uses $\mu=0$ during aggregation, followed by 40 local personalization steps with $\mu_p=0.15$. Personalized parameters are never uploaded. We report the model without final personalization separately as **FedPASA base**.

共享模型包含 553,280 个 fp32 参数，每份模型副本约为 2.11 MiB。交互历史、客户端画像和用户表示均保留在本地；仅共享模型参数本身并不构成差分隐私保证。

The shared payload is 553,280 fp32 parameters (2.11 MiB per model copy). Histories, profiles, and user embeddings remain local; parameter sharing alone does not establish differential privacy.

**算法1：FedPASA 训练与本地个性化**

Algorithm 1: FedPASA training and local personalization

**输入：** 冻结物品特征、训练交互历史、客户端集合 $\mathcal C$、通信轮数 $R$、每轮本地更新步数 $K$。

Require: frozen item features, training histories, clients $\mathcal C$, rounds $R$, local steps $K$.

1. 对于跨域任务，使用源域 BPR 和客户端等权 FedAvg 预训练 $\theta^{\mathrm{src}}$。

   For cross-domain runs, pretrain $\theta^{\mathrm{src}}$ by source BPR and equal-client FedAvg.

2. 初始化 $\theta_h$、$\theta_l$ 和 $\phi$；$\theta_h$ 使用源域参数或随机初始化。

   Initialize $\theta_h$ (source or random), $\theta_l$, and $\phi$.

3. 缓存目标域路由得分 $q_i$ 和客户端画像 $p_c$，固定 $\rho_{\mathrm{eff}}$。

   Cache target $q_i$ and local $p_c$; fix $\rho_{\mathrm{eff}}$.

4. 初始化各客户端私有用户表示表 $E_c=\{e_u:u\in\mathcal U_c\}$。

   Initialize private user tables $E_c=\{e_u:u\in\mathcal U_c\}$.

5. 设置 $\theta^{(0)}=\{\theta_h,\theta_l,\phi\}$。

   Set $\theta^{(0)}\gets\{\theta_h,\theta_l,\phi\}$.

6. 对于每一轮 $t=0,\ldots,R-1$，向所有客户端广播 $\theta^{(t)}$。每个客户端设置 $\theta_c\leftarrow\theta^{(t)}$，保留上一轮的 $E_c$，并对 $\mathcal L_{\mathrm{BPR}}$ 执行 $K$ 步 AdamW 更新，同时更新 $\theta_c$ 和 $E_c$。客户端将 $E_c$ 保留在本地，仅上传 $\theta_c$。服务器进行等权聚合：

   For each round $t=0,\ldots,R-1$: broadcast $\theta^{(t)}$ to every client. Each client sets $\theta_c\gets\theta^{(t)}$, retains $E_c$ from the preceding round, and takes $K$ AdamW steps on $\mathcal L_{\mathrm{BPR}}$, updating $(\theta_c,E_c)$. Clients keep $E_c$ local and upload only $\theta_c$. The server aggregates with equal weights:

$$
\theta^{(t+1)}
=
\frac{1}{|\mathcal C|}
\sum_{c\in\mathcal C}\theta_c.
$$

7. 对每个客户端进行最终个性化：设置 $\theta_c\leftarrow\theta^{(R)}$，保留最终用户表示 $E_c$，构建本地目标：

   For final personalization, each client sets $\theta_c\gets\theta^{(R)}$, retains the final $E_c$, and constructs the local objective

$$
\ell_c=
\mathcal L_{\mathrm{BPR}}
+
\frac{0.15}{2}
\|\theta_c-\theta^{(R)}\|_2^2.
$$

8. 对该目标执行 40 步 AdamW 更新，同时更新 $\theta_c$ 和 $E_c$。

   Take 40 AdamW steps on this objective, updating $(\theta_c,E_c)$.

9. 返回各客户端本地保存的个性化参数 $(\theta_c,E_c)$。

   Return the personalized $(\theta_c,E_c)$ stored on each client.

### 3.6 初始化与本地优化细节

源域预训练使用单个残差适配器和源域用户表示，其适配器参数用于初始化目标域头部路径。尾部路径随机初始化，条件化网络最后一个线性层的权重和偏置初始化为零。域内训练不使用源域预训练，两条适配路径均随机初始化。

Source pretraining uses a single residual adapter with source user embeddings, and its adapter parameters initialize the target head path. The tail path is randomly initialized, and the conditioner's final linear layer has zero weights and biases. In-domain training initializes both adapters at random without source pretraining.

目标物品频率由训练交互历史的并集计算。客户端画像由本地训练物品的多重集合平均得到，保留重复交互的影响。这些统计量和目录级门控在联邦训练、本地个性化和评估期间保持不变，验证集与测试集交互不参与频率或画像计算。

Target item frequencies are computed from the union of training histories. Each client profile averages its own training-item multiset, retaining repeated interactions. These statistics and the catalog gate stay fixed through all rounds, personalization, and evaluation; validation and test interactions do not enter the frequency or profile computation.

默认每轮所有客户端均参与训练，聚合权重均为 $1/|\mathcal C|$，不随客户端交互量变化。私有用户表示在通信轮之间保留，AdamW 优化器状态在每次本地训练阶段开始时重置。

Every round includes all clients, and aggregation assigns each client weight $1/|\mathcal C|$, regardless of its interaction count. Private user embeddings persist across rounds, and AdamW states are reset at the start of each local training phase.

训练时，先在客户端内均匀采样用户，再从该用户训练历史中均匀采样正物品，每个正样本对应三个负样本，负样本排除该用户已经出现的训练物品。BPR 得分差使用缩放系数 10。Dropout 在训练更新时启用，在评估时关闭。最终个性化同时更新共享模型副本和私有用户表示，但近端约束仅作用于模型参数。个性化状态保留在客户端本地。

Training samples users uniformly within a client, positives uniformly from their training histories, and three negatives excluding that user's training items; the BPR score difference uses logit scale 10. Dropout is active during updates and disabled during evaluation. Final personalization updates both the shared-model copy and private user embeddings, while its proximal penalty applies only to model parameters. These personalized states remain local.

## 4 适配机制分析

本节分析先验加权的偏差—方差权衡、归一化锚定的几何性质，以及路由和条件化之间的交互。统计分析使用抽象的可靠性参数，目录级用户—物品数量比仅作为实际实现中的代理变量。为简化记号，在不产生歧义时省略客户端和物品下标。

We examine the bias–variance tradeoff of prior weighting, the geometry of normalized anchoring, and the interaction between routing and conditioning. The statistical analysis uses an abstract reliability parameter; the catalog user-to-item ratio is a practical proxy for it. Client and item indices are omitted where unambiguous.

### 4.1 先验加权的偏差—方差权衡

**命题1：偏差—方差权衡。**

Proposition 1 (Bias–variance tradeoff).

给定目标表示 $z^*$ 和先验 $x$，设估计表示为

Fix a target representation $z^*$ and prior $x$. Let the estimated representation be

$$
\hat z=z^*+\varepsilon,
$$

满足

with

$$
\mathbb E\varepsilon=0,
\qquad
\mathbb E\|\varepsilon\|_2^2=\frac{\sigma^2}{r},
\qquad
r,\sigma^2>0.
$$

记先验偏差为 $b^2=\|x-z^*\|_2^2$。对于未归一化估计

Write $b^2=\|x-z^*\|_2^2$. The unnormalized estimate

$$
\tilde z_\omega=(1-\omega)\hat z+\omega x,
$$

有

satisfies

$$
\begin{aligned}
\mathbb E\|\tilde z_\omega-z^*\|_2^2
&=
(1-\omega)^2\frac{\sigma^2}{r}
+\omega^2b^2,\\
\omega^*
&=
\frac{\sigma^2}{\sigma^2+rb^2}.
\end{aligned}
\tag{8}
$$

其中，$\omega^*$ 在区间 $[0,1]$ 上最小化风险；当 $b>0$ 时，$\omega^*$ 随可靠性 $r$ 增大而减小。

Here $\omega^*$ minimizes the risk over $[0,1]$ and decreases with $r$ when $b>0$.

最优先验权重随估计噪声增大而增大，随先验偏差增大而减小：目标域证据有限时，应保留更多有用的先验信息。AutoAnchor 通过固定的目录级规则近似这一原则，其实际排序效果由第 5.5 节的实验检验。

The optimal weight increases with estimation noise and decreases with prior bias: when target evidence is limited, more of a useful prior should be retained. AutoAnchor approximates this principle with a fixed catalog-level rule, whose ranking effect is evaluated in §5.5.

### 4.2 归一化锚定的几何性质

**命题2：归一化锚定的几何性质。**

Proposition 2 (Normalized anchor geometry).

对于固定单位向量 $x,z$，定义

For fixed unit vectors $x,z$, let

$$
w_\omega=(1-\omega)z+\omega x,
\qquad
y_\omega=\frac{w_\omega}{\|w_\omega\|_2},
\qquad
a=x^\top z.
$$

当 $0\leq\omega<1/2$ 时，

For $0\leq\omega<1/2$,

$$
\begin{aligned}
\|w_\omega\|_2
&\geq 1-2\omega>0,\\
\frac{d(x^\top y_\omega)}{d\omega}
&=
\frac{(1-\omega)(1-a^2)}
{\|w_\omega\|_2^3}
\geq 0.
\end{aligned}
\tag{9}
$$

因此，余弦偏移 $1-x^\top y_\omega$ 不会随锚定权重增加而增大。

Thus the cosine drift $1-x^\top y_\omega$ cannot increase with the anchor weight.

对于固定的适配表示，增大锚定权重会使归一化表示朝先验方向移动。该结论覆盖默认最大权重 $\rho=0.3$。训练结束后的实际偏移在第 5.5 节测量。

For a fixed adapted vector, increasing the anchor weight moves the normalized representation toward the prior. The bound covers the default maximum weight $\rho=0.3$; the drift measured after training is reported in §5.5.

### 4.3 客户端条件化

式（6）中的 $\tanh$ 将每个乘性系数限制在 $[1-\lambda,1+\lambda]$。当 $\lambda=0.2$ 时，该范围为 $[0.8,1.2]$。加性项 $\lambda\beta_c$ 则通过训练学习。两个部分分别根据客户端画像，对路由后的特征进行缩放和平移。

In Eq. (6), $\tanh$ constrains each multiplicative factor to $[1-\lambda,1+\lambda]$, giving $[0.8,1.2]$ at $\lambda=0.2$, while the additive term $\lambda\beta_c$ remains learned. The two terms respectively rescale and shift the routed features according to the client profile.

### 4.4 路由与条件化之间的交互

两个混合系数具有不同作用。当关闭 FiLM，即 $\lambda=0$ 时，式（4）和式（6）可化简为

The two mixture coefficients have distinct effects. When FiLM is disabled ($\lambda=0$), Eqs. (4)–(6) reduce to

$$
z_{c,i}
=
\mathrm{norm}\left(
(1-\alpha q_i')A_h(x_i)
+
\alpha q_i'A_l(x_i)
\right).
\tag{10}
$$

目标域初始化路径的有效系数为 $\alpha q_i'$。物品频率决定该路径的相对分配，$\alpha$ 则限制其最大贡献。即使 $q_i'=1$，表示中仍然保留系数为 $1-\alpha$ 的头部路径。

The effective coefficient of the target-initialized path is $\alpha q_i'$. Item frequency determines its relative allocation, while $\alpha$ limits its maximum contribution. Even when $q_i'=1$, the representation retains a head-path coefficient of $1-\alpha$.

固定路由用于检验是否需要依赖频率进行分配；打乱路由用于检验路由分配是否必须与对应物品的频率匹配。二者均保留两个适配器及其参数量。

Constant routing tests whether frequency-dependent allocation is necessary; shuffled routing tests whether the allocation must match the corresponding item's frequency. Both retain the two adapters and their parameter count.

开启 FiLM 后，客户端画像决定的缩放和平移作用于路由表示，随后再进行归一化与锚定。因此，各模块共同修改同一表示，其排序收益不必具有简单的可加性。我们通过模块移除实验，以及由物品频率和客户端接触情况共同定义的分组实验，评估各组件的作用。

With FiLM enabled, profile-dependent scales and shifts act on the routed vector before normalization and anchoring. The components therefore modify a shared representation, so their effects on ranking need not be additive. We assess their contributions through component removals and through groups defined jointly by item frequency and local exposure.

## 5 实验

### 5.1 实验设置

#### 数据集与评价指标

我们评估 Amazon Music（CDs & Vinyl）、Books 和 Movies（Movies & TV）三个领域之间的全部六个有向迁移任务 [18]。不同领域的物品目录互不重叠，实验不使用用户对齐关系。

We evaluate all six ordered transfer directions among Amazon Music (CDs & Vinyl), Books, and Movies (Movies & TV) [18], with disjoint item catalogs and no user alignment.

域内实验采用 MovieLens-100K、MovieLens-1M [7]、Digital Music、Video Games 和 Luxury Beauty 五个数据集。数据规模见表6。保留至少具有三次交互的用户，将其最后一次交互作为测试集，倒数第二次交互作为验证集，其余交互用于训练。

The in-domain benchmarks are MovieLens-100K, MovieLens-1M [7], Digital Music, Video Games, and Luxury Beauty (Table 6). Each eligible user has at least three events; the last event is test and the preceding event is validation.

使用相同编码器的适配方法共享冻结的 DistilBERT 特征 [22]、用户划分和数据切分。跨域评估使用通过随机种子 777 选取的相同 5,000 名用户。

Common-backbone methods share frozen DistilBERT features [22], user partitions, and splits. Cross-domain evaluation uses the same 5,000-user subset selected with seed 777.

我们同时报告 99 个负样本下的采样排序结果和全目录排序结果。负样本从候选物品中均匀、有放回地采样，并排除用户在训练集、验证集中的已交互物品及测试正样本，因此负样本之间可能重复。适配器方法使用配对的候选采样，FedMF 的用户遍历顺序不同。全目录排序屏蔽训练和验证物品。域内实验在所有符合条件的用户上进行全目录排序。

We report both 99-negative sampled ranking and full-catalog ranking. The sampler draws uniformly with replacement, rejecting observed training/validation items and the held-out positive; duplicate negatives remain possible. Adapter methods use paired candidate draws, while FedMF traverses users in a different order. Full ranking masks training and validation items, and in-domain evaluation ranks the full catalog for all eligible users.

由于每个用户仅保留一个测试正样本，HR 与 Recall 在该设置下相同。采样排序和全目录排序对应不同的候选集合，不应直接混合比较其数值。

With one held-out positive, HR equals recall. Sampled and full ranking measure different candidate tasks, and their values are not directly comparable.

#### 对比方法与训练设置

对比方法包括 FedMF、FedAvg-Adapter、FedProx [11]、FedAdam [20]、TransFR 风格适配 [24]、L2-SP [12]、FedPer [2]、pFedMe 风格个性化 [4]，以及采用可学习门控的 FedMoE。

Common-backbone controls are FedMF, FedAvg-Adapter, FedProx [11], FedAdam [20], TransFR-style adaptation [24], L2-SP [12], FedPer [2], pFedMe-style personalization [4], and learned-gate FedMoE.

目标域为 Music 或 Movies 时，适配器训练使用 12 轮通信，每轮 60 步本地更新；目标域为 Books 时，使用 8 轮通信，每轮 30 步。域内适配器训练使用 20 轮通信，每轮 60 步；FedMF 使用 25 轮通信，每轮 80 步。

Target adapters use 12 rounds of 60 local steps for Music/Movies targets and 8 rounds of 30 steps for Books; in-domain adapters use 20 rounds of 60 steps; FedMF retains 25 rounds of 80 steps.

FedPASA 和 pFedMe 风格方法额外进行 40 步带近端约束的本地更新。最终个性化与共享适配的效果分别报告。固定路由和打乱路由与 FedPASA 保持完全相同的适配器和条件化网络参数量。

FedPASA and pFedMe-style additionally use 40 proximal local steps, reported separately from shared adaptation. Constant and shuffled routing retain exactly the same adapter and conditioner parameters as FedPASA.

我们还纳入 FFMSR [15]、FedDCSR [25] 和 FeDecider [8]。表2列出了相应数据集及排序截断位置。FFMSR 和 FedDCSR 使用序列表示，其目录预处理与配置见附录 D.2。FeDecider 覆盖七个 Amazon 和 Goodreads 领域。另一个解耦适配器变体使用冻结 LLaMA 特征和方向聚合。

We also consider FFMSR [15], FedDCSR [25], and FeDecider [8]; Table 2 lists their datasets and cutoffs. FFMSR and FedDCSR use sequential representations, with catalog preprocessing and configuration described in Appendix D.2. FeDecider covers seven Amazon and Goodreads domains. A separate decoupled-adapter variant uses directional aggregation with frozen LLaMA features.

#### 超参数选择与随机种子

在 Music→Movies 的验证集上，先从 12 组组合中选择 $(\alpha,\kappa)=(0.85,0.5)$，再从 9 组组合中选择 $(\lambda,\rho)=(0.2,0.3)$，具体结果见表16。

Validation on Music→Movies selects $(\alpha,\kappa)=(.85,.5)$ from 12 pairs, then $(\lambda,\rho)=(.2,.3)$ from nine pairs (Table 16).

选择过程采用随机种子 42、123、456，在 8 轮通信、每轮 30 步本地更新下最大化平均验证 HR@20。选定配置在不同目标域上保持不变，敏感性实验围绕该配置每次改变一个参数。

Selection maximizes mean validation HR@20 over seeds 42/123/456 at eight rounds and 30 local steps. The selected configuration is fixed across targets, and sensitivity analysis varies one parameter at a time around it.

主要对比使用五个随机种子 $\{42,123,456,789,2024\}$；客户端和敏感性实验使用前三个种子；FFMSR 使用 42 和 123；FedDCSR 使用 42。表9和表10报告样本标准差。计算总体不确定性时，先在每个种子内部对迁移方向取平均，再进行跨种子的统计分析。

Common-backbone comparisons use five seeds $\{42,123,456,789,2024\}$; client and sensitivity studies use the first three, FFMSR uses 42/123, and FedDCSR uses 42. Tables 9–10 report sample standard deviations. Direction means are formed within each seed before uncertainty calculations.

### 5.2 推荐准确率

在使用相同编码器时，FedPASA 在全部六个迁移方向上均取得最高的采样排序 HR@20 和 NDCG@20，结果见表1。平均 HR@20 和 NDCG@20 分别为 0.4390 和 0.1965，高于 TransFR 对照方法的 0.3462 和 0.1376，以及 FedMoE 的 0.3548 和 0.1420。相应的 HR@20 相对提升分别为 26.8% 和 23.7%。

With the shared encoder, FedPASA achieves the highest sampled HR@20 and NDCG@20 in all six transfer directions (Table 1). Mean HR@20 and NDCG@20 are .4390 and .1965, compared with .3462 and .1376 for TransFR and .3548 and .1420 for FedMoE; the relative HR gains are 26.8% and 23.7%.

未进行最终个性化的 FedPASA base 已达到 0.4360/0.1944，说明主要收益在最终本地适配之前已经形成。仅在目标域训练的 FedAvg-Adapter 达到 0.3488 的 HR@20，略高于 TransFR 对照方法，表明源域初始化本身不足以解释 FedPASA 的提升。

The unpersonalized model already reaches .4360/.1944, placing most of the gain before final local adaptation. Target-only FedAvg-Adapter attains .3488 HR@20, slightly above TransFR, suggesting that source initialization alone is insufficient to explain the improvement.

**表1：跨域采样排序结果，五个随机种子的均值。M、B、V 分别表示 Music、Books 和 Movies。粗体与下划线分别表示最优值和次优值。FedPASA 包含最终个性化。**

Table 1: Cross-domain sampled ranking (five-seed means). M/B/V: Music/Books/Movies. Bold and underline denote the best and second-best values. FedPASA includes final personalization.

**（a）HR@20**

| **方法** | **M$\to$B** | **M$\to$V** | **B$\to$V** | **B$\to$M** | **V$\to$B** | **V$\to$M** | **平均** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FedMF | 0.2005 | 0.2127 | 0.2127 | 0.2212 | 0.2005 | 0.2212 | 0.2115 |
| FedAvg-Adapter | 0.1949 | <u>0.3440</u> | 0.3440 | 0.5074 | 0.1949 | 0.5074 | 0.3488 |
| FedProx | 0.1998 | 0.3381 | 0.3381 | 0.5024 | 0.1998 | 0.5024 | 0.3468 |
| FedAdam | 0.1986 | 0.3398 | 0.3398 | 0.5073 | 0.1986 | 0.5073 | 0.3485 |
| TransFR control | <u>0.2058</u> | 0.3292 | 0.3330 | 0.5046 | 0.2001 | 0.5048 | 0.3462 |
| L2-SP | 0.2032 | 0.3244 | 0.3141 | 0.4996 | 0.2017 | 0.5040 | 0.3412 |
| FedPer | 0.2031 | 0.3410 | 0.3492 | 0.5052 | 0.2040 | 0.5027 | 0.3509 |
| pFedMe-style | 0.2047 | 0.3284 | 0.3368 | 0.5054 | 0.2026 | 0.5064 | 0.3474 |
| FedMoE | 0.2039 | 0.3424 | <u>0.3566</u> | <u>0.5088</u> | <u>0.2064</u> | <u>0.5107</u> | <u>0.3548</u> |
| **FedPASA** | **0.2719** | **0.4980** | **0.4948** | **0.5408** | **0.2824** | **0.5462** | **0.4390** |

**（b）NDCG@20**

| **方法** | **M$\to$B** | **M$\to$V** | **B$\to$V** | **B$\to$M** | **V$\to$B** | **V$\to$M** | **平均** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FedMF | 0.0715 | 0.0792 | 0.0792 | 0.0826 | 0.0715 | 0.0826 | 0.0778 |
| FedAvg-Adapter | 0.0672 | <u>0.1388</u> | 0.1388 | 0.2129 | 0.0672 | 0.2129 | 0.1396 |
| FedProx | 0.0705 | 0.1351 | 0.1351 | 0.2093 | 0.0705 | 0.2093 | 0.1383 |
| FedAdam | 0.0690 | 0.1359 | 0.1359 | 0.2098 | 0.0690 | 0.2098 | 0.1382 |
| TransFR control | <u>0.0723</u> | 0.1283 | 0.1307 | 0.2121 | 0.0705 | 0.2115 | 0.1376 |
| L2-SP | 0.0718 | 0.1244 | 0.1229 | 0.2083 | 0.0710 | 0.2122 | 0.1351 |
| FedPer | 0.0714 | 0.1353 | 0.1397 | <u>0.2150</u> | 0.0721 | 0.2099 | 0.1406 |
| pFedMe-style | 0.0719 | 0.1285 | 0.1332 | 0.2131 | 0.0712 | 0.2117 | 0.1383 |
| FedMoE | 0.0707 | 0.1355 | <u>0.1444</u> | 0.2143 | <u>0.0731</u> | <u>0.2141</u> | <u>0.1420</u> |
| **FedPASA** | **0.1159** | **0.2254** | **0.2284** | **0.2403** | **0.1249** | **0.2439** | **0.1965** |

FedPASA 的优势在全目录排序中仍然存在。其平均 HR@20 为 0.0111，NDCG@20 为 0.0041，而 FedMoE 分别为 0.0060 和 0.0022。全目录排序的候选集合更大，因此绝对得分低于采样排序。

The advantage persists under full-catalog ranking: mean HR@20 is .0111 and NDCG@20 is .0041, versus .0060 and .0022 for FedMoE. The lower absolute scores reflect the larger candidate set.

不同目标域的结果也存在明显差异。M→B 和 V→B 的 HR@20 分别为 0.0017 和 0.0018，而 M→V 和 B→V 分别为 0.0126 和 0.0140。完整方向级结果见表8。

Results vary across targets: HR@20 is .0017 and .0018 for M→B and V→B, compared with .0126 and .0140 for M→V and B→V. Direction-wise results appear in Table 8.

**表2：不同数据集上的推荐性能。H 和 N 分别表示 HR 和 NDCG；$k_1/k_2$ 表示两个排序截断位置。六方向均值覆盖 Music、Books 和 Movies。**

Table 2: Recommendation performance across datasets. H and N denote HR and NDCG; $k_1/k_2$ gives the two ranking cutoffs. Six-transfer means cover Music, Books, and Movies.

| 方法 | 数据集 | $k_1/k_2$ | H@$k_1$ | H@$k_2$ | N@$k_1$ | N@$k_2$ |
| --- | --- | --- | --- | --- | --- | --- |
| FedMF (FCF) | 六方向均值 | 10/20 | 0.0008 | 0.0016 | 0.0004 | 0.0006 |
| FedAvg-Adapter | 六方向均值 | 10/20 | 0.0028 | 0.0055 | 0.0013 | 0.0019 |
| FedProx | 六方向均值 | 10/20 | 0.0030 | 0.0057 | 0.0014 | 0.0021 |
| FedAdam | 六方向均值 | 10/20 | 0.0025 | 0.0048 | 0.0012 | 0.0018 |
| TransFR control | 六方向均值 | 10/20 | 0.0029 | 0.0054 | 0.0014 | 0.0020 |
| L2-SP | 六方向均值 | 10/20 | 0.0028 | 0.0054 | 0.0013 | 0.0019 |
| FedPer | 六方向均值 | 10/20 | 0.0026 | 0.0054 | 0.0012 | 0.0019 |
| pFedMe-style | 六方向均值 | 10/20 | 0.0028 | 0.0053 | 0.0013 | 0.0019 |
| FedMoE | 六方向均值 | 10/20 | 0.0032 | 0.0060 | 0.0015 | 0.0022 |
| FedPASA base | 六方向均值 | 10/20 | 0.0058 | 0.0104 | 0.0027 | 0.0039 |
| **FedPASA** | 六方向均值 | 10/20 | 0.0060 | 0.0111 | 0.0028 | 0.0041 |
| 解耦适配器$^*$ | 六方向均值 | 10/20 | 0.0037 | 0.0071 | 0.0017 | 0.0026 |
| FFMSR | M–B: B | 10/20 | 0.0177 | 0.0273 | 0.0092 | 0.0116 |
| FFMSR | M–V: V | 10/20 | 0.1079 | 0.1354 | 0.0755 | 0.0824 |
| FFMSR | B–V: V | 10/20 | 0.0956 | 0.1231 | 0.0651 | 0.0720 |
| FFMSR | M–B: M | 10/20 | 0.1185 | 0.1585 | 0.0741 | 0.0841 |
| FFMSR | B–V: B | 10/20 | 0.0149 | 0.0214 | 0.0077 | 0.0094 |
| FFMSR | M–V: M | 10/20 | 0.1335 | 0.1701 | 0.0900 | 0.0992 |
| FedDCSR | Music | 10/20 | 0.0940 | 0.1090 | 0.0794 | 0.0832 |
| FedDCSR | Books | 10/20 | 0.0240 | 0.0254 | 0.0214 | 0.0218 |
| FedDCSR | Movies | 10/20 | 0.0808 | 0.0924 | 0.0695 | 0.0724 |
| FeDecider [8] | Beauty | 5/10 | 0.0190 | 0.0247 | 0.0145 | 0.0164 |
| FeDecider | Clothing | 5/10 | 0.0097 | 0.0161 | 0.0061 | 0.0084 |
| FeDecider | Electronics | 5/10 | 0.0130 | 0.0156 | 0.0096 | 0.0106 |
| FeDecider | Phones | 5/10 | 0.0216 | 0.0272 | 0.0154 | 0.0169 |
| FeDecider | Crime | 5/10 | 0.0352 | 0.0478 | 0.0252 | 0.0292 |
| FeDecider | Comics | 5/10 | 0.0862 | 0.1078 | 0.0701 | 0.0771 |
| FeDecider | Children | 5/10 | 0.0502 | 0.0620 | 0.0401 | 0.0439 |

*解耦适配器使用冻结 LLaMA 特征和方向适配器聚合，结果为三个随机种子的均值。M、B、V 分别表示 Music、Books 和 Movies。*

*The decoupled adapter uses frozen LLaMA features with directional adapter aggregation (three seeds). M/B/V: Music/Books/Movies.*

表2中的序列方法和大语言模型相关方法覆盖了其他实验设置。FFMSR 在使用 BERT 表示及其目录预处理时，平均 HR@20/NDCG@20 为 0.1060/0.0598。解耦适配器变体在采样排序下达到 0.3862/0.1599，在全目录排序下达到 0.0071/0.0026。FeDecider 在七个领域上的 HR@10 范围为 0.0156–0.1078 [8]。这些结果反映各自数据集上的性能，不能据此对不同目录上的方法进行直接排序。

The sequential and LLM-based baselines in Table 2 cover additional settings. FFMSR attains mean HR@20/NDCG@20 of .1060/.0598 with BERT representations and its catalog preprocessing; the decoupled-adapter variant reaches .3862/.1599 under sampled ranking and .0071/.0026 under full ranking; FeDecider's HR@10 ranges from .0156 to .1078 across its seven domains [8]. These scores characterize dataset-specific performance rather than a ranking across different catalogs.

域内收益取决于评价指标，结果见表5。FedPASA 在五个数据集中的四个上取得最高 HR@20，平均为 0.0964，高于 FedProx 的 0.0798；但平均 NDCG@20 为 0.0383，低于 FedProx 的 0.0449。

In-domain gains depend on the metric (Table 5). FedPASA has the highest HR@20 on four of five datasets, averaging .0964 versus FedProx's .0798, while its mean NDCG@20 is .0383 versus .0449.

**表5：域内全目录排序 HR@20/NDCG@20，五个随机种子的均值。评估包含所有符合条件的用户。粗体表示对应指标的最优值。**

Table 5: In-domain full-ranking HR@20/NDCG@20 (five-seed means). All eligible users are evaluated. Bold values mark the best result for each metric.

| **方法** | **ML-100K** | **Digital Music** | **Video Games** | **Luxury Beauty** | **ML-1M** | **平均** |
| --- | --- | --- | --- | --- | --- | --- |
| FedMF | 0.1472/0.0616 | 0.0095/0.0041 | 0.0072/0.0029 | 0.0540/0.0184 | 0.0677/0.0255 | 0.0571/0.0225 |
| FedAvg-Adapter | 0.1139/0.0455 | 0.0036/0.0013 | 0.0061/0.0022 | 0.2002/0.1434 | 0.0513/0.0197 | 0.0750/0.0424 |
| FedProx | 0.1239/0.0478 | 0.0046/0.0017 | 0.0076/0.0028 | **0.2106**/**0.1512** | 0.0526/0.0207 | 0.0798/**0.0449** |
| FedAdam | 0.1158/0.0457 | 0.0036/0.0013 | 0.0060/0.0021 | 0.1861/0.1341 | 0.0469/0.0185 | 0.0717/0.0403 |
| FedPer | 0.1141/0.0432 | 0.0042/0.0016 | 0.0069/0.0025 | 0.2033/0.1472 | 0.0509/0.0200 | 0.0759/0.0429 |
| pFedMe-style | 0.1122/0.0431 | 0.0034/0.0012 | 0.0062/0.0022 | 0.1983/0.1414 | 0.0485/0.0186 | 0.0737/0.0413 |
| FedMoE | 0.1177/0.0458 | 0.0043/0.0017 | 0.0067/0.0025 | 0.2089/0.1497 | 0.0501/0.0196 | 0.0775/0.0438 |
| **FedPASA** | **0.1709**/**0.0693** | **0.0124**/**0.0051** | **0.0134**/**0.0048** | 0.1927/0.0763 | **0.0925**/**0.0359** | **0.0964**/0.0383 |

为分析域内收益与跨域收益的差异，表4在全目录排序下比较未进行最终个性化的 FedPASA base 与 FedAvg-Adapter。跨域 HR 从 0.0055 提升至 0.0105，相对提升 91.6%；域内 HR 从 0.0750 提升至 0.0964，相对提升 28.5%。域内绝对提升为 0.0214，大于跨域的 0.0050。因此，跨域相对提升更大，主要因为其基线性能更低。

To examine the difference between cross-domain and in-domain gains, Table 4 compares the unpersonalized model with FedAvg-Adapter under full ranking. Cross-domain HR increases from .0055 to .0105 (+91.6%), whereas in-domain HR increases from .0750 to .0964 (+28.5%). The absolute gain is larger in-domain (.0214 versus .0050), so the larger relative cross-domain gain mainly reflects its lower baseline.

对于分组 $g$，其对总体 HR 提升的贡献定义为

Group $g$ contributes

$$
\frac{n_g}{N}
\left(
H_g^{\mathrm{base}}-H_g^{\mathrm{ref}}
\right).
$$

先将各组贡献求和，再对数据集取平均。频率大于 5 的物品占跨域评估的 98.3%，贡献了 94.4% 的总体提升。域内低频组占评估样本的 14.3%，其负向贡献抵消了高频组 0.0249 提升中的 0.0035。在 Luxury Beauty 上，低频组的贡献为 −0.0159，超过高频组的正向贡献 +0.0108。

to the overall HR gain, and contributions are summed before averaging datasets. Items with frequency above 5 constitute 98.3% of cross-domain evaluations and account for 94.4% of the gain. In-domain low-frequency groups occupy 14.3% of evaluations and offset .0035 of the .0249 higher-frequency gain. On Luxury Beauty, their −.0159 contribution exceeds the +.0108 gain from higher-frequency items.

总体收益差异主要来自评估样本的构成：跨域评估几乎全部由高频物品构成，而域内低频物品的负贡献会抵消部分高频收益。

The aggregate difference thus stems mainly from evaluation composition: cross-domain evaluation consists almost entirely of high-frequency items, whereas in-domain low-frequency items contribute negatively and cancel part of the high-frequency gain.

### 5.3 路由收益是否仅来自模型容量

图2比较不同路由规则及模块移除变体，均以未进行最终个性化的 FedPASA base 为基准，其 HR@20 为 0.4360。

Figure 2 compares routing rules and component removals against the unpersonalized model (.4360 HR@20).

在参数量完全相同的情况下，固定路由和打乱路由分别达到 0.3481 和 0.3381；移除路由后为 0.3434。保持模型容量不变时性能仍明显下降，支持了按照物品频率分配路径的作用。

With identical parameter counts, constant and shuffled routing yield .3481 and .3381; removing routing yields .3434. The decrease with unchanged capacity indicates that assigning paths according to item frequency is itself a source of the gain.

移除 FiLM 后 HR@20 为 0.4245。采用可学习门控的 FedMoE 达到 0.3548，但其参数量较少，因此该比较不属于严格的等容量对照。

Removing FiLM yields .4245. Learned-gate FedMoE reaches .3548 with fewer parameters, so it is not an exact capacity match.

![图2 模块与等容量对照实验](E:/小论文/小论文4-联邦推荐/小论文aaai-www/FedPASA_WWW2027_最终版_LaTeX/figures/ablation_ppt.png)

**图2：相同未个性化评估条件下的模块与容量对照实验。**结果覆盖六个迁移方向和五个随机种子。固定路由与打乱路由保留两个适配器和条件化网络。所有变体均在最终个性化之前评估，两个指标的完整数值见表12。

Figure 2: Component and capacity controls under the same unpersonalized evaluation (six directions, five seeds). Constant and shuffled routing retain both adapters and the conditioner. All variants are evaluated before final personalization; Table 12 reports both metrics.

移除 AutoAnchor 后，采样排序 HR 从 0.4360 变为 0.4346，而全目录排序 HR 从 0.0105 变为 0.0107。逐物品锚定的采样 HR 为 0.4365，见表20。这表明锚定对准确率的影响相对较小，并随评估方式而变化。

Removing AutoAnchor changes sampled HR from .4360 to .4346 but full-ranking HR from .0105 to .0107. Per-item anchoring yields .4365 sampled HR (Table 20), indicating that anchoring has a smaller, evaluation-dependent effect on accuracy.

最终个性化进一步将采样排序和全目录排序 HR 分别提高至 0.4390 和 0.0111。

Final personalization raises sampled and full-ranking HR to .4390 and .0111.

### 5.4 物品频率与客户端接触情况

表3根据训练证据，将测试交互划分为三个群体：

Table 3 groups test events by training evidence:

- **C：低频且客户端已接触。**物品训练频率较低，且出现在所属客户端的训练历史中。

  C: low-frequency/local-seen—low training frequency and present in the assigned client's histories.

- **U：低频且客户端未接触。**物品训练频率较低，且未出现在所属客户端的训练历史中。

  U: low-frequency/local-unseen—low training frequency and absent from the assigned client's histories.

- **W：频率大于 5。**

  W: frequency above 5.

每个随机种子内先合并不同迁移方向，再对种子取平均。三个群体的平均规模分别为 231.2、268.8 和 29,500 个"迁移方向—用户"评估实例。相同目标用户可能出现在多个方向中，这些实例并不相互独立。

We pool directions within each seed, then average seeds; mean group sizes are 231.2, 268.8, and 29,500 direction–user evaluations. Shared target users recur across directions, so these instances are not independent.

**表3：物品频率—客户端接触情况联合分组的全目录排序 HR@20/NDCG@20，五个随机种子的均值。**

Table 3: Joint frequency–exposure groups, full-ranking HR@20/NDCG@20 (five-seed means).

| 变体 | C | U | W |
| --- | --- | --- | --- |
| FedPASA base | 0.0160/0.0070 | 0.0347/0.0151 | 0.0102/0.0037 |
| 移除路由 | 0.0094/0.0039 | 0.0029/0.0009 | 0.0057/0.0021 |
| 移除 FiLM | 0.0326/0.0156 | 0.0365/0.0147 | 0.0099/0.0037 |
| 移除锚定 | 0.0126/0.0059 | 0.0332/0.0147 | 0.0105/0.0038 |

在低频且客户端已接触的群体中，FedPASA base 的 HR@20 为 0.0160，高于移除路由后的 0.0094 和移除锚定后的 0.0126。但移除 FiLM 后，该组 HR/NDCG 从 0.0160/0.0070 上升至 0.0326/0.0156。

In the low-frequency/local-seen group, base HR@20 is .0160, versus .0094 without routing and .0126 without anchoring. Removing FiLM instead raises the group's HR/NDCG from .0160/.0070 to .0326/.0156.

对于低频且客户端未接触的物品，FedPASA base 的 HR 为 0.0347，而移除路由后仅为 0.0029。对于频率大于 5 的物品，FiLM 将 HR 从 0.0099 提升至 0.0102。

For locally unseen low-frequency items, base HR is .0347 versus .0029 without routing. For items with frequency above 5, FiLM improves HR from .0099 to .0102.

结合表15中更细粒度的频率分组结果可以看出，条件化对不同群体的影响并不一致，而路由对两个低频群体均有明显作用。

Together with the finer frequency slices in Table 15, conditioning favors some groups and is even negative on the local-seen low-frequency group, while routing benefits both low-frequency groups.

**表4：全目录排序 HR@20 提升的分组贡献，单位为百分点。结果为 FedPASA base 减去 FedAvg-Adapter；C、U、W 与表3一致。**

Table 4: Full-ranking HR@20 gain decomposition, in percentage points: base minus FedAvg-Adapter. C/U/W follow Table 3.

| 评估任务 | C | U | W | $\Delta$ |
| --- | --- | --- | --- | --- |
| 跨域均值 | +0.006 | +0.022 | +0.471 | +0.499 |
| ML-100K | +0.000 | +0.000 | +5.514 | +5.514 |
| Digital Music | -0.034 | -0.078 | +0.967 | +0.855 |
| Video Games | -0.015 | -0.045 | +0.748 | +0.689 |
| Luxury Beauty | -0.273 | -1.320 | +1.078 | -0.515 |
| ML-1M | -0.003 | -0.003 | +4.159 | +4.152 |
| 域内均值 | -0.065 | -0.289 | +2.493 | +2.139 |

不使用源域初始化时，HR/NDCG 为 0.4362/0.1958；使用源域初始化时为 0.4360/0.1944。两者平均表现接近，说明路由收益并非主要由源域适配器初始化驱动。

Source-free HR/NDCG is .4362/.1958 versus .4360/.1944 with initialization. The near-equal averages indicate that the routing benefit is not driven by source-adapter initialization.

### 5.5 锚定对表示偏移的影响

我们使用相对于冻结语义先验的平均余弦距离衡量表示偏移：

We measure representation drift by the mean cosine distance from the frozen semantic prior:

$$
D=
\frac{1}{|\mathcal C||\mathcal I|}
\sum_{c\in\mathcal C}
\sum_{i\in\mathcal I}
\left(
1-\cos(z^{\mathrm{anchor}}_{c,i},x_i)
\right).
\tag{11}
$$

余弦距离位于 $[0,2]$。如表21所示，加入锚定后，全体物品的平均偏移从 0.9450 增至 0.9841，但低频物品的偏移从 0.9721 降至 0.9645。因此，锚定在完整训练后的效果具有群体差异。

Cosine distance lies in $[0,2]$. Anchoring increases all-item drift from .9450 to .9841 but decreases cold-item drift from .9721 to .9645 (Table 21). Its effect after training is therefore group-dependent.

命题2描述的是固定适配向量上的收缩性质，而端到端训练还会改变该适配向量本身，两者应予以区分。

Proposition 2 describes contraction for a fixed adapted vector; end-to-end optimization also changes that vector.

### 5.6 客户端异质性与计算预算

在 5、20、50 个客户端以及三种非独立同分布划分下，FedPASA base 均优于 TransFR 对照方法，见图3（a、b）。随着客户端数量增加，HR 从 0.4940 上升至 0.6035，但总计算量也同时增加。

The unpersonalized model outperforms the TransFR control at 5, 20, and 50 clients and under all three non-IID partitions (Figure 3a,b). HR rises from .4940 to .6035 as clients increase, alongside total computation.

在基于活跃度的划分中，当集中参数为 0.1 和 0.5 时，HR 分别为 0.4523 和 0.4839，低于默认划分下的 0.4940。

Activity-based partitions at concentration .1 and .5 yield HR of .4523 and .4839, below .4940 under the default partition.

![图3 客户端规模与异质性实验](E:/小论文/小论文4-联邦推荐/小论文aaai-www/FedPASA_WWW2027_最终版_LaTeX/figures/robustness_ppt.png)

**图3：Music→Movies 上的客户端规模和非独立同分布评估，三个随机种子的均值。**图中的 FedPASA 指未进行最终个性化的基础模型。客户端数量与数据划分规则分别变化。（a、b）使用选定配置，其中（a）的客户端数量增加会带来更多本地计算。（c、d）在八轮训练下分别评估近似固定计算量和部分客户端参与。完整数值见表19。

Figure 3: Client-scale and non-IID evaluation on Music→Movies (three seeds). FedPASA denotes the unpersonalized base. Client counts and partition rules are changed independently. Panels (a,b) use the selected configuration; more clients in (a) increase total local work. Panels (c,d) evaluate approximately fixed computation and partial participation over eight rounds. Detailed results appear in Table 19.

图3（c、d）使用 $(\alpha,\kappa,\lambda,\rho)=(0.55,0.7,0.4,0.2)$，训练八轮。在 5、20、50 个客户端下，每轮总更新步数分别固定为 150、160、150，此时 HR 从 0.2803 降至 0.2419，但仍高于 TransFR。

Panels (c,d) use $(\alpha,\kappa,\lambda,\rho)=(.55,.7,.4,.2)$ and eight rounds. Fixing total work to 150/160/150 client steps per round at 5/20/50 clients lowers HR from .2803 to .2419, still above TransFR.

当参与率为 20% 和 50% 时，FedPASA 的 HR 分别为 0.2583 和 0.2692，TransFR 分别为 0.2090 和 0.2039。这表明，在限制总计算量或客户端参与率的条件下，FedPASA 仍保持优势。

At 20% and 50% participation, HR is .2583 and .2692 versus TransFR's .2090 and .2039. FedPASA retains its advantage when total computation or client participation is restricted.

表19还报告更新扰动与编码器替换实验。对于 $p$ 维客户端更新 $\Delta$，加入标准差为

Table 19 also reports update-perturbation and encoder-replacement experiments. Each $p$-dimensional client update $\Delta$ is perturbed with Gaussian noise of standard deviation

$$
s\frac{\|\Delta\|}{\sqrt p}
$$

的高斯噪声。当 $s=0.1,0.2,0.3$ 时，六个方向上的平均 HR@20 保持在 0.4343–0.4351。

For $s=.1,.2,.3$, mean HR@20 remains .4343–.4351 across six directions.

使用冻结 LLaMA 特征时，FedPASA base 在 M→V、B→V 和 M→B 上分别达到 0.5091、0.4869 和 0.2519，均高于对应 TransFR 对照结果。

With frozen LLaMA features, FedPASA base reaches .5091, .4869, and .2519 on M→V, B→V, and M→B, exceeding the corresponding TransFR controls.

通信参数量方面，TransFR 共享 201,344 个参数，FedPASA 共享 553,280 个参数；固定路由和打乱路由与 FedPASA 的通信参数量相同。

TransFR shares 201,344 parameters versus FedPASA's 553,280; constant and shuffled routing match FedPASA's payload.

### 5.7 敏感性与统计分析

在 Music→Movies 上，默认配置的 HR 为 0.4938，见图4和表18。将 $\alpha$ 从 0.55 增至 1.0，HR 从 0.4543 变为 0.4963；将 $\kappa$ 从 0.35 增至 1.0，HR 从 0.4998 变为 0.4249。

On Music→Movies, the default configuration yields HR of .4938 (Figure 4; Table 18). Varying $\alpha$ from .55 to 1 gives .4543–.4963; varying $\kappa$ from .35 to 1 gives .4998–.4249.

改变 $\rho$ 时，HR 的变化范围较小，为 0.4919–0.4952。相比路由和客户端条件化，锚定强度在该区间内的影响较弱。

The $\rho$ sweep varies less (.4919–.4952). Compared with routing and conditioning, anchor strength has a smaller effect over this range.

![图4 超参数敏感性分析](E:/小论文/小论文4-联邦推荐/小论文aaai-www/FedPASA_WWW2027_最终版_LaTeX/figures/sensitivity_ppt.png)

**图4：Music→Movies 上路由与个性化控制参数的敏感性，三个随机种子的基础模型结果。**空心圆表示通过验证集选择的默认配置。每条曲线仅改变一个参数。HR@20 和 NDCG@20 的完整结果见表18。

Figure 4: Sensitivity of the routing and personalization controls on Music→Movies (three seeds, base model). Open circles mark the validation-selected default. Each curve changes one parameter; Table 18 gives both HR@20 and NDCG@20.

在五个配对随机种子上，FedPASA 的采样 HR@20 为 $0.4390\pm0.0058$，FedMoE 为 $0.3548\pm0.0042$。配对平均差为 0.0842，Student-$t$ 的 95% 置信区间为 $[0.0797,0.0888]$。

Across five paired seeds, sampled HR@20 is $.4390\pm.0058$ for FedPASA and $.3548\pm.0042$ for FedMoE. The paired mean difference is .0842, with a Student-$t$ 95% interval [.0797,.0888].

全目录排序的平均差为 0.0051，95% 置信区间为 $[0.0044,0.0058]$。两项比较的精确双侧符号翻转检验均得到 $p=0.0625$。

The full-ranking difference is .0051 [.0044,.0058]. The exact two-sided sign-flip test gives $p=.0625$ for both comparisons.

因此，参数化置信区间不包含零，但基于五个种子的精确检验尚未达到 0.05 显著性水平。各迁移方向的波动情况见表9和表10。

Thus the parametric intervals exclude zero, while five seeds do not establish significance at .05 under the exact test. Tables 9–10 report direction-wise variability.

## 6 结论

本文提出 FedPASA，将基于物品频率的路由、客户端条件化和目录级语义锚定结合起来，用于联邦语义适配。实验表明，在使用相同编码器的条件下，FedPASA 能够提升跨域推荐性能；等模型容量对照进一步表明，频率路由是主要收益来源。客户端条件化和语义锚定的效果具有群体差异。频率代理与目录级门控都是粗粒度信号，学习更细粒度的可靠性估计留待后续工作。

FedPASA combines routing by item frequency, client conditioning, and catalog-level anchoring for federated semantic adaptation. Experiments with a shared encoder show cross-domain gains, and equal-capacity controls identify routing as the main contributor; the effects of conditioning and anchoring are group-dependent. Both the frequency proxy and the catalog-level gate are coarse signals; learning finer-grained reliability estimates is left to future work.

---

## 参考文献

（参考文献本身为英文，不重复对照。新增 MoE-Rec 类文献后编号为 [26]。）

[1] Muhammad Ammad-Ud-Din, Elena Ivannikova, Suleiman A Khan, Quang Oyomno, Qiang Fu, Kuan Ee Tan, and Adrian Flanagan. 2019. Federated Collaborative Filtering for Privacy-Preserving Personalized Recommendation System. *arXiv preprint arXiv:1901.09888*.

[2] Manoj Ghuhan Arivazhagan, Vinay Aggarwal, Aaditya Kumar Singh, and Sunav Choudhary. 2019. Federated Learning with Personalization Layers. *arXiv preprint arXiv:1912.00818*.

[3] Zhihong Chen, Rong Xiao, Chenliang Li, Gang Ye, Hao Sun, and Hongbo Deng. 2020. ESAM: Discriminative Domain Adaptation with Non-Displayed Items to Improve Long-Tail Performance. *Proceedings of SIGIR*.

[4] Canh T. Dinh, Nguyen H. Tran, and Tuan Dung Nguyen. 2020. Personalized Federated Learning with Moreau Envelopes. *Advances in Neural Information Processing Systems*, 33: 21394–21405.

[5] Lei Guo, Ziang Lu, Junliang Yu, Nguyen Quoc Viet Hung, and Hongzhi Yin. 2024. Prompt-Enhanced Federated Content Representation Learning for Cross-Domain Recommendation. *arXiv preprint arXiv:2401.14678*.

[6] Lei Guo, Ting Yang, Hui Liu, Xu Yu, Xiaohui Han, and Xinhua Wang. 2026. FedCRF: A Federated Cross-Domain Recommendation Method with Semantic-Driven Deep Knowledge Fusion. *Information Processing & Management*. arXiv:2604.17681. DOI: 10.1016/j.ipm.2026.104827.

[7] F Maxwell Harper, and Joseph A Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)*, 5(4): 1–19.

[8] Xinrui He, Ting-Wei Li, Tianxin Wei, Xuying Ning, Xinyu He, Wenxuan Bao, Hanghang Tong, and Jingrui He. 2026. FeDecider: An LLM-Based Framework for Federated Cross-Domain Recommendation. *Proceedings of the ACM Web Conference*. DOI: 10.1145/3774904.3792294.

[9] Hoyeop Lee, Jinbae Im, Seongwon Jang, Hyunsouk Cho, and Sehee Chung. 2019. MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation. *Proceedings of KDD*.

[10] Tian Li, Shengyuan Hu, Ahmad Beirami, and Virginia Smith. 2021. Ditto: Fair and Robust Federated Learning Through Personalization. *Proceedings of ICML*: 6357–6368.

[11] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. 2020. Federated Optimization in Heterogeneous Networks. *Proceedings of Machine Learning and Systems*, 2: 429–450.

[12] Xuhong Li, Yves Grandvalet, and Franck Davoine. 2018. Explicit Inductive Bias for Transfer Learning with Convolutional Networks. *Proceedings of ICML*: 2825–2834.

[13] Xingjian Li, Haoyi Xiong, Haozhe Wang, Yifan Rao, Liping Liu, Jie Chen, and Jun Huan. 2019. DELTA: Deep Learning Transfer using Feature Map with Attention for Convolutional Networks. *Proceedings of ICLR*.

[14] Zhiwei Li, Guodong Long, Jing Jiang, Chengqi Zhang, and Qiang Yang. 2026. Federated Vision-Language-Recommendation with Personalized Fusion. *Proceedings of the AAAI Conference on Artificial Intelligence*, 40: 23337–23345.

[15] Ziang Lu, Lei Guo, Xu Yu, Zhiyong Cheng, Xiaohui Han, and Lei Zhu. 2025. Federated Semantic Learning for Privacy-Preserving Cross-Domain Recommendation. *ACM Transactions on Information Systems*. arXiv:2503.23026.

[16] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H. Chi. 2018. Modeling Task Relationships in Multi-Task Learning with Multi-Gate Mixture-of-Experts. *Proceedings of KDD*.

[17] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. 2017. Communication-Efficient Learning of Deep Networks from Decentralized Data. *Proceedings of AISTATS*: 1273–1282.

[18] Jianmo Ni, Jiacheng Li, and Julian McAuley. 2019. Justifying Recommendations Using Distantly-Labeled Reviews and Fine-Grained Aspects. *Proceedings of EMNLP-IJCNLP*: 188–197.

[19] Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. 2018. FiLM: Visual Reasoning with a General Conditioning Layer. *Proceedings of the AAAI Conference on Artificial Intelligence*, 32.

[20] Sashank J Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Konecný, Sanjiv Kumar, and H Brendan McMahan. 2021. Adaptive Federated Optimization. *Proceedings of ICLR*.

[21] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2009. BPR: Bayesian Personalized Ranking from Implicit Feedback. *Proceedings of UAI*: 452–461.

[22] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter. *arXiv preprint arXiv:1910.01108*.

[23] Chunxu Zhang, Guodong Long, Hongkuan Guo, Xiao Fang, Yang Song, Zhaojie Liu, Guorui Zhou, Zijian Zhang, Yang Liu, and Bo Yang. 2024. Federated Adaptation for Foundation Model-Based Recommendations. *Proceedings of IJCAI*: 5453–5461.

[24] Honglei Zhang, Zhiwei Li, Haoxuan Li, Xin Zhou, Jie Zhang, and Yidong Li. 2026. TransFR: Transferable Federated Recommendation with Adapter Tuning on Pre-trained Language Models. *Proceedings of the AAAI Conference on Artificial Intelligence*, 40: 28212–28220.

[25] Huili Zhang, Dongxia Zheng, Xiang Yang, Juan Feng, and Qimei Liao. 2024. FedDCSR: Federated Cross-Domain Sequential Recommendation via Disentangled Representation Learning. *Proceedings of SDM*: 535–543.

## 附录 A 数据集统计

表6统计至少具有三次交互的保留用户。每个用户最后两次交互分别构成验证集和测试集。AutoAnchor 使用保留用户数与物品目录规模的比值。跨域评估使用随机种子 777 采样 5,000 名用户，域内评估使用所有保留用户。

Table 6 counts users retained after the three-interaction filter. Their last two events form the validation and test sets. AutoAnchor uses the retained-user count divided by catalog size. Cross-domain tests sample 5,000 users with seed 777; in-domain tests use all retained users.

**表6：经过三次交互筛选后的数据统计。$N$ 包含训练、验证和测试交互；评估用户数表示测试用户数量。**

Table 6: Retained data after the three-interaction filter. $N$ includes train, validation, and test interactions; "Eval." is the test-user count.

| 数据集 | 用户数 | 物品数 | $N$ | 评估用户数 |
| --- | --- | --- | --- | --- |
| Music | 12,315 | 10,680 | 339,923 | 5,000 |
| Books | 262,698 | 170,389 | 10,848,388 | 5,000 |
| Movies | 28,226 | 19,252 | 1,056,608 | 5,000 |
| ML-100K | 943 | 1,682 | 100,000 | 943 |
| Digital Music | 16,506 | 11,358 | 168,248 | 16,506 |
| Video Games | 55,218 | 17,294 | 497,220 | 55,218 |
| Luxury Beauty | 3,804 | 1,438 | 33,829 | 3,804 |
| ML-1M | 6,040 | 3,706 | 1,000,209 | 6,040 |

排序评估屏蔽训练与验证物品。99 个负样本的采样方式及 FedMF 的非配对用户遍历方式见第 5.1 节。

Ranking masks training and validation items. The 99-negative sampler and FedMF's unpaired traversal follow §5.1.

DistilBERT 使用考虑有效 token 掩码的均值池化，并对所得特征进行归一化，最大文本长度为 96 个 token。编码器替换实验采用 LLaMA-2-7B 特征，通过固定的带符号分块投影映射到 512 维。

DistilBERT uses masked-mean pooling over valid tokens with normalization and a maximum length of 96 tokens. The encoder-replacement check uses LLaMA-2-7B features with a fixed signed block projection to 512 dimensions.

### A.1 符号说明

**表7：FedPASA 的主要符号。**

Table 7: Notation used in FedPASA.

| 符号 | 含义 |
| --- | --- |
| $x_i$, $e_u$ | 单位化冻结物品向量；私有用户向量 (unit frozen item vector; private user vector) |
| $A_h,A_l$ | 源域初始化适配器；目标域初始化适配器 (source-initialized and target-initialized adapters) |
| $F_\ell$ | 第 $\ell$ 个残差块中的瓶颈 MLP (bottleneck MLP in residual block $\ell$) |
| $q_i,q_i'$ | 归一化冷度得分；重塑后的路由得分 (normalized coldness score; reshaped routing score) |
| $t,\kappa$ | 路由阈值与指数 (routing threshold and exponent) |
| $h_i,m_{c,i},z_{c,i}$ | 路由表示、调节表示和归一化残差表示 (routed, modulated, and normalized residual features) |
| $p_c,g_\phi$ | 私有客户端画像；共享 FiLM 条件化网络 (private client profile; shared FiLM conditioner) |
| $\alpha,\lambda$ | 残差混合强度；FiLM 强度 (residual-mixture and FiLM strengths) |
| $\eta_T,\rho_{\mathrm{eff}}$ | 用户—物品数量比；最终锚定权重 (user-to-item ratio; resulting anchor weight) |
| $\rho,\tau_l,\tau_h$ | 最大锚定权重；数量比阈值 (maximum anchor weight; ratio thresholds) |
| $\theta,\mu,\mu_p$ | 共享参数；联邦训练与个性化的近端系数 (shared parameters; federation and personalization penalties) |
| $r,b^2,\sigma^2$ | 命题1中的可靠性、先验偏差平方与噪声尺度 (reliability, squared prior bias, and noise scale in Proposition 1) |

## 附录 B 全目录排序与随机种子波动

**表8：各迁移方向的全目录排序 HR@20 和 NDCG@20，五个随机种子的均值，补充表2的结果。粗体与下划线分别表示最优值和次优值。FedPASA 包含最终个性化。**

Table 8: Direction-wise full-ranking HR@20 and NDCG@20 (five seeds), complementing Table 2. Bold/underline denote the best/second-best values; FedPASA includes final personalization.

**（a）HR@20**

| **方法** | **M$\to$B** | **M$\to$V** | **B$\to$V** | **B$\to$M** | **V$\to$B** | **V$\to$M** | **平均** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FedMF | <u>0.0003</u> | 0.0022 | 0.0022 | 0.0024 | <u>0.0003</u> | 0.0024 | 0.0016 |
| FedAvg-Adapter | 0.0002 | 0.0037 | 0.0037 | 0.0125 | 0.0002 | 0.0125 | 0.0055 |
| FedProx | 0.0001 | <u>0.0041</u> | 0.0041 | 0.0130 | 0.0001 | 0.0130 | 0.0057 |
| FedAdam | 0.0001 | 0.0033 | 0.0033 | 0.0110 | 0.0001 | 0.0110 | 0.0048 |
| TransFR control | 0.0001 | 0.0037 | 0.0030 | 0.0126 | 0.0000 | 0.0132 | 0.0054 |
| L2-SP | 0.0000 | 0.0029 | 0.0028 | 0.0132 | 0.0001 | 0.0132 | 0.0054 |
| FedPer | 0.0001 | 0.0034 | 0.0040 | 0.0127 | 0.0002 | 0.0119 | 0.0054 |
| pFedMe-style | 0.0001 | 0.0029 | 0.0034 | 0.0125 | 0.0000 | 0.0132 | 0.0053 |
| FedMoE | 0.0001 | <u>0.0041</u> | <u>0.0045</u> | <u>0.0134</u> | 0.0002 | <u>0.0137</u> | <u>0.0060</u> |
| **FedPASA** | **0.0017** | **0.0126** | **0.0140** | **0.0177** | **0.0018** | **0.0188** | **0.0111** |

**（b）NDCG@20**

| **方法** | **M$\to$B** | **M$\to$V** | **B$\to$V** | **B$\to$M** | **V$\to$B** | **V$\to$M** | **平均** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FedMF | <u>0.0001</u> | 0.0009 | 0.0009 | 0.0009 | <u>0.0001</u> | 0.0009 | 0.0006 |
| FedAvg-Adapter | 0.0000 | 0.0015 | 0.0015 | 0.0043 | 0.0000 | 0.0043 | 0.0019 |
| FedProx | 0.0000 | <u>0.0015</u> | 0.0015 | 0.0047 | 0.0000 | 0.0047 | 0.0021 |
| FedAdam | 0.0000 | 0.0011 | 0.0011 | 0.0041 | 0.0000 | 0.0041 | 0.0018 |
| TransFR control | 0.0000 | 0.0012 | 0.0011 | 0.0047 | 0.0000 | 0.0050 | 0.0020 |
| L2-SP | 0.0000 | 0.0011 | 0.0011 | 0.0046 | 0.0000 | 0.0049 | 0.0019 |
| FedPer | 0.0000 | 0.0012 | 0.0014 | 0.0045 | 0.0001 | 0.0042 | 0.0019 |
| pFedMe-style | 0.0000 | 0.0009 | 0.0012 | 0.0046 | 0.0000 | 0.0049 | 0.0019 |
| FedMoE | 0.0000 | 0.0015 | <u>0.0016</u> | <u>0.0050</u> | 0.0001 | <u>0.0052</u> | <u>0.0022</u> |
| **FedPASA** | **0.0006** | **0.0046** | **0.0050** | **0.0065** | **0.0009** | **0.0069** | **0.0041** |

**表9：采样排序 HR@20 的均值与样本标准差。平均列的标准差在每个种子内先对六个方向取平均后计算。**

Table 9: Sampled HR@20 mean and sample SD. The SD of the mean column is computed after averaging six directions within each seed.

| 方法 | M$\to$B | M$\to$V | B$\to$V | B$\to$M | V$\to$B | V$\to$M | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TransFR control | 0.2058$\pm$0.0051 | 0.3292$\pm$0.0065 | 0.3330$\pm$0.0210 | 0.5046$\pm$0.0127 | 0.2001$\pm$0.0071 | 0.5048$\pm$0.0074 | 0.3462$\pm$0.0066 |
| FedMoE | 0.2039$\pm$0.0032 | 0.3424$\pm$0.0120 | 0.3566$\pm$0.0077 | 0.5088$\pm$0.0056 | 0.2064$\pm$0.0059 | 0.5107$\pm$0.0085 | 0.3548$\pm$0.0042 |
| FedPASA base | 0.2692$\pm$0.0063 | 0.4950$\pm$0.0055 | 0.4881$\pm$0.0039 | 0.5394$\pm$0.0072 | 0.2796$\pm$0.0053 | 0.5449$\pm$0.0085 | 0.4360$\pm$0.0044 |
| FedPASA | 0.2719$\pm$0.0068 | 0.4980$\pm$0.0085 | 0.4948$\pm$0.0053 | 0.5408$\pm$0.0126 | 0.2824$\pm$0.0052 | 0.5462$\pm$0.0094 | 0.4390$\pm$0.0058 |

**表10：采样排序 NDCG@20 的均值与样本标准差。计算方式同表9。**

Table 10: Sampled NDCG@20 mean and sample SD, computed as in Table 9.

| 方法 | M$\to$B | M$\to$V | B$\to$V | B$\to$M | V$\to$B | V$\to$M | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TransFR control | 0.0723$\pm$0.0032 | 0.1283$\pm$0.0028 | 0.1307$\pm$0.0090 | 0.2121$\pm$0.0044 | 0.0705$\pm$0.0030 | 0.2115$\pm$0.0042 | 0.1376$\pm$0.0027 |
| FedMoE | 0.0707$\pm$0.0010 | 0.1355$\pm$0.0035 | 0.1444$\pm$0.0027 | 0.2143$\pm$0.0031 | 0.0731$\pm$0.0024 | 0.2141$\pm$0.0059 | 0.1420$\pm$0.0019 |
| FedPASA base | 0.1140$\pm$0.0053 | 0.2204$\pm$0.0031 | 0.2268$\pm$0.0042 | 0.2389$\pm$0.0051 | 0.1232$\pm$0.0034 | 0.2432$\pm$0.0046 | 0.1944$\pm$0.0032 |
| FedPASA | 0.1159$\pm$0.0054 | 0.2254$\pm$0.0053 | 0.2284$\pm$0.0032 | 0.2403$\pm$0.0076 | 0.1249$\pm$0.0035 | 0.2439$\pm$0.0069 | 0.1965$\pm$0.0035 |

## 附录 C 补充基线比较与消融实验

### C.1 补充跨域推荐比较

**表11：Amazon Music、Books 和 Movies 上的跨域推荐结果 [24]。每个单元格表示 HR@10/NDCG@10。**

Table 11: Cross-domain recommendation on Amazon Music, Books, and Movies [24]. Each cell is HR@10/NDCG@10.

| 方法 | M$\to$B | M$\to$V | B$\to$V | 平均 |
| --- | --- | --- | --- | --- |
| CMF | .1833/.0955 | .1378/.0764 | .1455/.0662 | .1555/.0794 |
| EMCDR | .2149/.1121 | .1893/.0901 | .1652/.0705 | .1898/.0909 |
| DisenCDR | .2243/.1182 | .1988/.0923 | .1912/.0821 | .2048/.0975 |
| FCF | .1380/.0663 | .1154/.0583 | .1004/.0421 | .1179/.0556 |
| IFedRec | .1981/.0834 | .1810/.0673 | .1703/.0497 | .1831/.0668 |
| FELLRec | .2003/.0995 | .1844/.0812 | .1731/.0831 | .1859/.0879 |
| FedCT | .2034/.1122 | .1765/.0899 | .1585/.0845 | .1795/.0955 |
| FedDCSR | .2217/.1085 | .1881/.0892 | .1719/.0732 | .1939/.0903 |
| **TransFR** | **.2335/.1133** | **.2123/.0934** | **.2089/.0893** | **.2182/.0987** |

### C.2 完整消融结果

**表12：跨域消融结果，六个迁移方向、五个随机种子的均值。前七行对应图2；仅最后一行包含最终个性化。**

Table 12: Cross-domain ablation results (six directions, five seeds). The first seven rows correspond to Figure 2; only the last row includes final personalization.

| 变体 | HR@20 | NDCG@20 |
| --- | --- | --- |
| FedPASA base | 0.4360 | 0.1944 |
| 移除路由 | 0.3434 | 0.1373 |
| 移除 FiLM | 0.4245 | 0.1867 |
| 移除 AutoAnchor | 0.4346 | 0.1934 |
| 固定路由 | 0.3481 | 0.1404 |
| 打乱路由 | 0.3381 | 0.1350 |
| 逐物品锚定 | 0.4365 | 0.1950 |
| 不使用源域初始化 | 0.4362 | 0.1958 |
| 可学习门控 FedMoE | 0.3548 | 0.1420 |
| 最终个性化 | 0.4390 | 0.1965 |

### C.3 FFMSR、PFCR 与 FedCRF

**表13：FFMSR 和 PFCR 的推荐性能。R 表示 Recall，N 表示 NDCG。**

Table 13: FFMSR and PFCR recommendation performance. R: recall; N: NDCG.

**（a）FFMSR [15]**

| 领域 | R10 | R50 | N10 | N50 |
| --- | --- | --- | --- | --- |
| Office | .1312 | .2069 | .0890 | .1054 |
| Arts | .1257 | .2377 | .0711 | .0955 |
| OnlineRetail | .1655 | .4110 | .0763 | .1300 |
| Pantry | .0695 | .1818 | .0326 | .0568 |

**（b）PFCR，完整提示配置 [5]**

(b) PFCR, full prompting [5]

| 领域 | R10 | R50 | N10 | N50 |
| --- | --- | --- | --- | --- |
| Office | .1206 | .1938 | .0782 | .0941 |
| Arts | .1153 | .2199 | .0677 | .0904 |
| OnlineRetail | .1561 | .3982 | .0739 | .1266 |
| Pantry | .0616 | .1591 | .0293 | .0503 |

**表14：FedCRF 在不同领域对上的全目录排序性能 [6]。**

Table 14: FedCRF full-ranking performance across domain pairs [6].

| 领域对 | 领域 | R@10 | R@20 | N@10 | N@20 |
| --- | --- | --- | --- | --- | --- |
| Kitchen–Food | Kitchen | .04465 | .06601 | .02523 | .03112 |
|  | Food | .11078 | .16105 | .06708 | .08193 |
| Care–Beauty | Care | .07115 | .09897 | .04250 | .05001 |
|  | Beauty | .09342 | .13384 | .05398 | .06513 |
| Food–OnlineRetail | Food | .11142 | .16185 | .06713 | .08189 |
|  | OnlineRetail | .22864 | .31488 | .19232 | .23368 |

### C.4 更细粒度的频率分组

**表15：细粒度物品频率分组结果。采用 99 个负样本的采样排序，模型均未进行最终个性化。**

Table 15: Fine-grained frequency slices (99-negative ranking; unpersonalized models).

| 训练频率 | 方法 | HR@20 | NDCG@20 |
| --- | --- | --- | --- |
| 未见 | TransFR | 0.3020 | 0.1070 |
|  | FedMoE | 0.2460 | 0.0927 |
|  | FedPASA base | 0.5360 | 0.3783 |
| 1–5 | TransFR | 0.3711 | 0.1562 |
|  | FedMoE | 0.3915 | 0.1662 |
|  | FedPASA base | 0.3439 | 0.2045 |
| 6–20 | TransFR | 0.3465 | 0.1395 |
|  | FedMoE | 0.3541 | 0.1438 |
|  | FedPASA base | 0.2631 | 0.0990 |
| >20 | TransFR | 0.3455 | 0.1361 |
|  | FedMoE | 0.3544 | 0.1404 |
|  | FedPASA base | 0.4956 | 0.2258 |

未见物品分组包含 20 个非空评估单元，平均每个单元包含 13 名用户；其他分组均包含 30 个评估单元。命中数为零的评估结果也计入统计。

The unseen slice contains 20 nonempty evaluation cells (mean support: 13 users); other slices contain 30. Evaluations with zero hits are included.

## 附录 D 超参数与鲁棒性分析

### D.1 验证集选择与默认配置

**表16：Music→Movies 上仅使用验证集进行超参数选择的结果。指标为平均采样 HR@20，随机种子为 42、123、456，采用八轮通信、每轮 30 步本地更新。**

Table 16: Validation-only selection on Music→Movies (mean sampled HR@20, seeds 42/123/456, eight rounds of 30 local steps).

第一阶段固定 $\lambda=0.4,\rho=0.2$：

Stage 1 fixes $\lambda=.4,\rho=.2$:

| $\alpha\backslash\kappa$ | .5 | .7 | 1.0 |
| --- | --- | --- | --- |
| 0.4 | 0.2807 | 0.2640 | 0.2517 |
| 0.55 | 0.3261 | 0.2871 | 0.2626 |
| 0.7 | 0.3657 | 0.3139 | 0.2721 |
| 0.85 | 0.3851 | 0.3541 | 0.2883 |

第二阶段固定选定的 $\alpha=0.85,\kappa=0.5$：

Stage 2 fixes the selected $\alpha=.85,\kappa=.5$:

| $\lambda\backslash\rho$ | .1 | .2 | .3 |
| --- | --- | --- | --- |
| 0.2 | 0.3917 | 0.3917 | 0.3925 |
| 0.4 | 0.3879 | 0.3851 | 0.3893 |
| 0.6 | 0.3842 | 0.3874 | 0.3817 |

最终配置为 $(\alpha,\kappa,\lambda,\rho)=(0.85,0.5,0.2,0.3)$。

The final configuration is $(\alpha,\kappa,\lambda,\rho)=(.85,.5,.2,.3)$.

**表17：默认实验配置。敏感性实验每次仅改变一个设置。**

Table 17: Default configuration. Sensitivity experiments vary one setting at a time.

| 实验设置 | 取值 |
| --- | --- |
| 冻结编码器 (frozen encoder) | DistilBERT-base-uncased，$d=768$ |
| 适配器 (adapter) | 2 个残差块；瓶颈维度 64；Dropout 0.1 |
| 优化器 (optimizer) | AdamW；学习率 $10^{-3}$；权重衰减 $10^{-4}$ |
| BPR 批次 (BPR batches) | 32 个正样本；每个正样本对应 3 个负样本；得分缩放系数 10 |
| 联邦训练 (federation) | 跨域 $12\times60$ 轮/步；目标域为 Books：$8\times30$；域内 $20\times60$ |
| 客户端数量 (clients) | 跨域 5；域内 10 |
| 混合与 FiLM (mixing / FiLM) | $\alpha=.85$，$t=0$，$\kappa=.5$，$\lambda=.2$ |
| 锚定 (anchor) | $\rho=.3$；$[\tau_l,\tau_h]=[1,1.5]$ |
| 近端约束 (proximal terms) | FedProx $.1$；L2-SP $.01$；FedPASA base $0$ |
| 最终个性化 (final personalization) | 40 步本地更新，$\mu_p=.15$（完整 FedPASA 和 pFedMe 风格方法） |
| FedMF 基线 (FedMF baseline) | 64 维隐因子；25 轮通信；每轮 80 步 |
| 硬件 (hardware) | 单张 RTX 4080 GPU，16 GB 显存 |

**表18：图4对应的 HR@20/NDCG@20，Music→Movies，三个随机种子的均值。"默认"表示验证集选定配置。**

Table 18: HR@20/NDCG@20 for Figure 4 (Music→Movies, three seeds). "Default" marks the validation-selected configuration.

| 参数 | 取值 | HR@20/NDCG@20 |
| --- | --- | --- |
| $\alpha$ | 0.55 | 0.4543/0.1935 |
|  | 0.7 | 0.4782/0.2079 |
|  | 0.85 | 0.4938/0.2194（默认） |
|  | 1.0 | 0.4963/0.2223 |
| $\kappa$ | 0.35 | 0.4998/0.2233 |
|  | 0.5 | 0.4938/0.2194（默认） |
|  | 0.7 | 0.4842/0.2100 |
|  | 1.0 | 0.4249/0.1779 |
| $\lambda$ | 0.0 | 0.4563/0.2010 |
|  | 0.1 | 0.4845/0.2110 |
|  | 0.2 | 0.4938/0.2194（默认） |
|  | 0.4 | 0.4955/0.2208 |
| $\rho$ | 0.0 | 0.4949/0.2190 |
|  | 0.1 | 0.4952/0.2208 |
|  | 0.2 | 0.4943/0.2199 |
|  | 0.3 | 0.4938/0.2194（默认） |
|  | 0.5 | 0.4919/0.2208 |

### D.2 基线配置

FFMSR 使用 BERT 特征，并通过验证集选择模型。M–V 任务的预训练和微调分别进行 5 和 20 个 epoch；涉及 Books 的领域对分别进行 1 和 5 个 epoch。

FFMSR uses BERT features with validation-based model selection. Pretraining/fine-tuning last 5/20 epochs for M–V and 1/5 epochs for pairs involving Books.

FedDCSR 使用 10 轮通信，每轮进行 3 个本地 epoch。序列基准的物品规模分别为 Music 10,634、Books 89,254、Movies 18,643。

FedDCSR uses 10 communication rounds with three local epochs per round. The sequential benchmarks contain 10,634 Music, 89,254 Books, and 18,643 Movies items.

FFMSR 对随机种子 42 和 123 的结果取平均；FedDCSR 使用随机种子 42。各评估领域的完整结果见表2。

FFMSR averages seeds 42/123; FedDCSR uses seed 42. Table 2 lists each evaluated domain.

### D.3 客户端、噪声与编码器实验

**表19：不同条件下的 HR@20/NDCG@20，三个随机种子的基础模型结果。客户端数量、固定计算量和参与率实验对应图3。噪声实验取六个方向的均值，编码器实验使用表中指定方向。**

Table 19: HR@20/NDCG@20 under varying conditions (three seeds, base model). Client, fixed-compute, and participation rows match Figure 3. Noise averages six directions; encoder rows use the indicated direction.

| 实验设置 | TransFR | FedPASA base |
| --- | --- | --- |
| 5 个客户端 | 0.3308/0.1289 | 0.4940/0.2183 |
| 20 个客户端 | 0.4375/0.1841 | 0.5769/0.2788 |
| 50 个客户端 | 0.5015/0.2205 | 0.6035/0.2935 |
| 活跃度划分 .1 | 0.3189/0.1238 | 0.4523/0.2059 |
| 活跃度划分 .5 | 0.3413/0.1357 | 0.4839/0.2167 |
| 流行度划分 .5 | 0.3347/0.1320 | 0.4895/0.2199 |
| 固定计算量：5 个客户端 | 0.2431/0.0876 | 0.2803/0.1089 |
| 固定计算量：20 个客户端 | 0.2304/0.0815 | 0.2460/0.0919 |
| 固定计算量：50 个客户端 | 0.2285/0.0804 | 0.2419/0.0889 |
| 参与率 .2 | 0.2090/0.0754 | 0.2583/0.1034 |
| 参与率 .5 | 0.2039/0.0724 | 0.2692/0.1074 |
| 更新噪声 0.1 | 0.3425/0.1361 | 0.4346/0.1938 |
| 更新噪声 0.2 | 0.3420/0.1359 | 0.4351/0.1944 |
| 更新噪声 0.3 | 0.3443/0.1362 | 0.4343/0.1941 |
| LLaMA M$\to$V | 0.4257/0.1798 | 0.5091/0.2243 |
| LLaMA B$\to$V | 0.4186/0.1807 | 0.4869/0.2203 |
| LLaMA M$\to$B | 0.2001/0.0714 | 0.2519/0.0998 |

固定计算量和参与率对照均采用八轮通信，设置 $\alpha=0.55,\kappa=0.7,\lambda=0.4,\rho=0.2$。在 5、20、50 个客户端下，每轮总更新步数分别为 150、160、150。全参与设置对应五客户端条件。其余实验使用选定的默认配置。

Fixed-compute and participation controls use eight rounds with $\alpha=.55,\kappa=.7,\lambda=.4,\rho=.2$. Total steps per round are 150/160/150 at 5/20/50 clients. Full participation corresponds to the five-client setting. Other rows use the selected configuration.

### D.4 锚定粒度与表示偏移

**表20：不同锚定粒度的比较，五个随机种子的跨域均值。**

Table 20: Anchor granularity (five-seed cross-domain means).

| 锚定方式 | HR@20 | NDCG@20 |
| --- | --- | --- |
| 不使用锚定 | 0.4346 | 0.1934 |
| 目录级门控 | 0.4360 | 0.1944 |
| 逐物品门控 | 0.4365 | 0.1950 |

**表21：适配表示与冻结先验之间的余弦距离。跨域参考模型为 TransFR，域内参考模型为 FedAvg。"—"表示未评估。**

Table 21: Cosine distance from the frozen prior. References are TransFR cross-domain and FedAvg in-domain. "—": not evaluated.

| 评估群体 | 参考模型 | 无锚定 | FedPASA base |
| --- | --- | --- | --- |
| 跨域全部物品 | 0.8916 | 0.9450 | 0.9841 |
| 跨域低频物品 | 0.8983 | 0.9721 | 0.9645 |
| ML-100K 全部物品 | 0.7724 | – | 1.3331 |
| ML-100K 低频物品 | 0.7750 | – | 1.1864 |

---

## 投稿前清单

1. §2.2 的 `[待补引用]`（MoE-Rec 类）补真实文献，参考文献加 [26]。
2. 确认图1 底部面板画出残差跳跃连接，图中符号统一为 $A_h$、$A_l$、$g_\phi$。
3. 本稿引用编号沿用 [1]–[25]，正式 tex 中用 `\citep` 自动编号，不必手工对应。
