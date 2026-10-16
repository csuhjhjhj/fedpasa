# WWW 2027 补充 Baseline：公开实验结果汇总

> 用途：整理目前已经从论文原文/正式出版页检索到的实验数值，并标明哪些数据可以**直接作为原论文结果引用**，哪些不能进入 FedPASA 的统一公平比较主表、必须重新复现。
>
> **重要说明：**下表中的“可直接用”指“可以在论文中作为 *published result under the original protocol* 引用”，**不等于**可以直接和 FedPASA 当前统一协议结果做公平数值比较。  
> 如果最终 WWW 主表写成“same split + full ranking + five seeds”，那么所有方法仍需在统一协议下重跑。

---

# 1. 最接近 FedPASA 的现成结果：Amazon Music / Books / Movies

目前公开论文里，**TransFR（AAAI 2026）**最接近我们的数据：它同样使用 Amazon Music、Books、Movies，并在自己的协议下运行了多个 federated / cross-domain baseline。

## 1.1 TransFR 官方论文：Target-domain HR@10

| Method | Music→Books | Music→Movies | Books→Movies | Average |
|---|---:|---:|---:|---:|
| CMF | 0.1833 | 0.1378 | 0.1455 | 0.1555 |
| EMCDR | 0.2149 | 0.1893 | 0.1652 | 0.1898 |
| DisenCDR | 0.2243 | 0.1988 | 0.1912 | 0.2048 |
| FCF | 0.1380 | 0.1154 | 0.1004 | 0.1179 |
| IFedRec | 0.1981 | 0.1810 | 0.1703 | 0.1831 |
| FELLRec | 0.2003 | 0.1844 | 0.1731 | 0.1859 |
| FedCT | 0.2034 | 0.1765 | 0.1585 | 0.1795 |
| FedDCSR | 0.2217 | 0.1881 | 0.1719 | 0.1939 |
| **TransFR** | **0.2335** | **0.2123** | **0.2089** | **0.2182** |

**来源：** TransFR, AAAI 2026, Table 1。  
官方 PDF：<https://ojs.aaai.org/index.php/AAAI/article/download/40048/44009>  
DOI：<https://doi.org/10.1609/aaai.v40i33.40048>

---

## 1.2 TransFR 官方论文：Target-domain NDCG@10

| Method | Music→Books | Music→Movies | Books→Movies | Average |
|---|---:|---:|---:|---:|
| CMF | 0.0955 | 0.0764 | 0.0662 | 0.0794 |
| EMCDR | 0.1121 | 0.0901 | 0.0705 | 0.0909 |
| DisenCDR | 0.1182 | 0.0923 | 0.0821 | 0.0975 |
| FCF | 0.0663 | 0.0583 | 0.0421 | 0.0556 |
| IFedRec | 0.0834 | 0.0673 | 0.0497 | 0.0668 |
| FELLRec | 0.0995 | 0.0812 | 0.0831 | 0.0879 |
| FedCT | 0.1122 | 0.0899 | 0.0845 | 0.0955 |
| FedDCSR | 0.1085 | 0.0892 | 0.0732 | 0.0903 |
| **TransFR** | **0.1133** | **0.0934** | **0.0893** | **0.0987** |

**来源：** TransFR, AAAI 2026, Table 2。  
官方 PDF：<https://ojs.aaai.org/index.php/AAAI/article/download/40048/44009>

### 这些值怎么用？

- **可以直接引用为 TransFR 原论文结果 / sanity check。**
- **不能直接当成我们新统一 full-ranking 五随机种子主表中的最终 baseline 数值。**
- 原因：
  1. TransFR 只报告 M→B、M→V、B→V 三个方向；
  2. 只有 HR@10 / NDCG@10；
  3. 论文没有报告 HR@20 / NDCG@20；
  4. 没有 5-seed mean±std；
  5. 原论文没有明确声明其该表为 full-catalog ranking；
  6. TransFR 的数据过滤/预处理与当前 FedPASA 不完全一致。

---

# 2. FeDecider（WWW 2026）

**论文：** FeDecider: An LLM-Based Framework for Federated Cross-Domain Recommendation  
**会议：** The ACM Web Conference 2026  
**DOI：** <https://doi.org/10.1145/3774904.3792294>  
**arXiv：** <https://arxiv.org/abs/2602.16034>

原论文数据：
- Goodreads：Crime / Comics / Children
- Amazon：Beauty / Clothing
- Amazon：Electronics / Phones
- 指标：H@5、H@10、NDCG@5、NDCG@10
- 没有 H@20 / NDCG@20
- 没有 Music / Books / Movies 结果

---

## 2.1 FeDecider：Amazon Beauty / Clothing

| Method | Beauty H@5 | Beauty H@10 | Beauty N@5 | Beauty N@10 | Clothing H@5 | Clothing H@10 | Clothing N@5 | Clothing N@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Local train | 0.0172 | 0.0220 | 0.0124 | 0.0139 | 0.0080 | 0.0107 | 0.0057 | 0.0062 |
| FedAvg | 0.0168 | 0.0220 | 0.0130 | 0.0146 | 0.0068 | 0.0100 | 0.0054 | 0.0064 |
| PFedAvg | 0.0173 | 0.0210 | 0.0132 | 0.0144 | 0.0078 | 0.0105 | 0.0062 | 0.0071 |
| FedProx | 0.0160 | 0.0200 | 0.0121 | 0.0134 | 0.0075 | 0.0100 | 0.0060 | 0.0068 |
| Ditto | 0.0158 | 0.0210 | 0.0122 | 0.0140 | 0.0088 | 0.0113 | 0.0067 | 0.0075 |
| FFA-LoRA | 0.0140 | 0.0163 | 0.0109 | 0.0116 | 0.0073 | 0.0120 | 0.0052 | 0.0067 |
| RoLoRA | 0.0165 | 0.0218 | 0.0124 | 0.0141 | 0.0083 | 0.0103 | 0.0065 | 0.0072 |
| FellRec | 0.0173 | 0.0210 | 0.0128 | 0.0140 | 0.0085 | 0.0105 | 0.0068 | 0.0075 |
| FDLoRA | 0.0163 | 0.0205 | 0.0123 | 0.0143 | 0.0090 | 0.0115 | 0.0069 | 0.0077 |
| **FeDecider** | **0.0190** | **0.0247** | **0.0145** | **0.0164** | **0.0097** | **0.0161** | **0.0061** | **0.0084** |

**来源：** FeDecider Table 2。  
可核对页面：<https://www.researchgate.net/publication/400930348_FeDecider_An_LLM-Based_Framework_for_Federated_Cross-Domain_Recommendation>

---

## 2.2 FeDecider：Amazon Electronics / Phones

| Method | Electronics H@5 | H@10 | N@5 | N@10 | Phones H@5 | H@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 0.0114 | 0.0130 | 0.0092 | 0.0097 | 0.0164 | 0.0194 | 0.0128 | 0.0137 |
| PFedAvg | 0.0122 | 0.0150 | 0.0095 | 0.0105 | 0.0186 | 0.0248 | 0.0139 | 0.0159 |
| FedProx | 0.0116 | 0.0136 | 0.0088 | 0.0095 | 0.0204 | 0.0262 | 0.0146 | 0.0164 |
| Ditto | 0.0124 | 0.0150 | 0.0095 | 0.0103 | 0.0205 | 0.0258 | 0.0148 | 0.0166 |
| **FeDecider** | **0.0130** | **0.0156** | **0.0096** | **0.0106** | **0.0216** | **0.0272** | **0.0154** | **0.0169** |

**来源：** FeDecider Appendix Table 7。  
arXiv：<https://arxiv.org/abs/2602.16034>

---

## 2.3 FeDecider：Goodreads Crime / Comics / Children

| Method | Crime H@5 | H@10 | N@5 | N@10 | Comics H@5 | H@10 | N@5 | N@10 | Children H@5 | H@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Local train | 0.0339 | 0.0426 | 0.0232 | 0.0260 | 0.0820 | 0.1071 | 0.0625 | 0.0706 | 0.0494 | 0.0624 | 0.0376 | 0.0418 |
| FedAvg | 0.0198 | 0.0252 | 0.0154 | 0.0171 | 0.0598 | 0.0790 | 0.0458 | 0.0520 | 0.0354 | 0.0456 | 0.0283 | 0.0315 |
| PFedAvg | 0.0328 | 0.0420 | 0.0223 | 0.0236 | 0.0804 | 0.1006 | 0.0620 | 0.0684 | 0.0486 | 0.0614 | 0.0367 | 0.0409 |
| FedProx | 0.0340 | 0.0412 | 0.0233 | 0.0257 | 0.0838 | 0.1056 | 0.0661 | 0.0738 | 0.0500 | 0.0620 | 0.0381 | 0.0420 |
| Ditto | 0.0358 | 0.0452 | 0.0241 | 0.0271 | 0.0836 | 0.1036 | 0.0651 | 0.0716 | 0.0482 | 0.0592 | 0.0368 | 0.0403 |
| FFA-LoRA | 0.0304 | 0.0370 | 0.0200 | 0.0222 | 0.0676 | 0.0888 | 0.0517 | 0.0586 | 0.0374 | 0.0468 | 0.0297 | 0.0328 |
| RoLoRA | 0.0322 | 0.0408 | 0.0217 | 0.0245 | 0.0820 | 0.1032 | 0.0635 | 0.0703 | 0.0494 | 0.0640 | 0.0378 | 0.0425 |
| FellRec | 0.0316 | 0.0406 | 0.0211 | 0.0240 | 0.0840 | 0.1056 | 0.0637 | 0.0706 | 0.0474 | 0.0618 | 0.0359 | 0.0406 |
| **FeDecider** | **0.0352** | **0.0478** | **0.0252** | **0.0292** | **0.0862** | **0.1078** | **0.0701** | **0.0771** | **0.0502** | **0.0620** | **0.0401** | **0.0439** |

**主表是否能直接使用？**  
**不能。** FeDecider 必须重新适配到 Music / Books / Movies，并按我们的统一 full-ranking 协议重跑。

---

# 3. FFMSR（TOIS 2025）

**论文：** Federated Semantic Learning for Privacy-preserving Cross-domain Recommendation  
**arXiv：** <https://arxiv.org/abs/2503.23026>  
**代码：** <https://github.com/Sapphire-star/FFMSR>

原论文明确使用 **full sorting**：validation/test 时对所有 items 排序。

原始数据：
- Office–Arts
- OnlineRetail–Pantry

指标：
- Recall@10 / Recall@50
- NDCG@10 / NDCG@50

## 3.1 FFMSR 原论文主结果

| Domain | Recall@10 | Recall@50 | NDCG@10 | NDCG@50 |
|---|---:|---:|---:|---:|
| Office | **0.1312** | **0.2069** | **0.0890** | **0.1054** |
| Arts | **0.1257** | **0.2377** | **0.0711** | **0.0955** |
| OnlineRetail | **0.1655** | **0.4110** | **0.0763** | **0.1300** |
| Pantry | **0.0695** | **0.1818** | **0.0326** | **0.0568** |

**来源：** FFMSR Table 3。  
论文正文明确写：validation/testing 使用 full sorting。

**主表是否能直接使用？**  
**不能。** 数据集不同，且缺 @20；但其官方代码和 full-sorting evaluator 很适合直接改成 M/B/V。

---

# 4. FedCRF（IPM 2026）

**论文：** FedCRF: A Federated Cross-domain Recommendation Method with Semantic-driven Deep Knowledge Fusion  
**正式期刊：** Information Processing & Management, 2026  
**DOI：** <https://doi.org/10.1016/j.ipm.2026.104827>  
**arXiv：** <https://arxiv.org/abs/2604.17681>

FedCRF 明确使用：
- 80/10/10 split
- 所有用户未交互 items 作为候选
- **full ranking**
- Recall@10 / Recall@20 / NDCG@10 / NDCG@20

原数据：
- Kitchen–Food
- Care–Beauty
- Food–OnlineRetail

---

## 4.1 FedCRF：Kitchen–Food / Care–Beauty

| Domain | Method | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|---|---|---:|---:|---:|---:|
| Kitchen | FFMSR | 0.02636 | 0.04373 | 0.01463 | 0.01993 |
| Kitchen | FedDCSR | 0.02572 | 0.04281 | 0.01421 | 0.01936 |
| Kitchen | **FedCRF** | **0.04465** | **0.06601** | **0.02523** | **0.03112** |
| Food | FFMSR | 0.06939 | 0.12793 | 0.04358 | 0.06264 |
| Food | FedDCSR | 0.06741 | 0.12486 | 0.04263 | 0.06112 |
| Food | **FedCRF** | **0.11078** | **0.16105** | **0.06708** | **0.08193** |
| Care | FFMSR | 0.04763 | 0.08411 | 0.02875 | 0.03953 |
| Care | FedDCSR | 0.04602 | 0.08157 | 0.02802 | 0.03891 |
| Care | **FedCRF** | **0.07115** | **0.09897** | **0.04250** | **0.05001** |
| Beauty | FFMSR | 0.06557 | 0.10502 | 0.03960 | 0.05146 |
| Beauty | FedDCSR | 0.06321 | 0.10188 | 0.03842 | 0.05037 |
| Beauty | **FedCRF** | **0.09342** | **0.13384** | **0.05398** | **0.06513** |

> 注：FedCRF 在其统一 protocol 下重新实现了 FFMSR 与 FedDCSR，这一点对我们很重要：**在自己的统一数据/评测协议下重跑 published baselines 是合理且已有先例的。**

**来源：** FedCRF main comparison / ablation tables。  
arXiv：<https://arxiv.org/abs/2604.17681>

---

## 4.2 FedCRF：Food–OnlineRetail full-ranking

| Method | Food R@10 | Food R@20 | Food N@10 | Food N@20 | OnlineRetail R@10 | R@20 | N@10 | N@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NGCF | 0.01749 | 0.02783 | 0.00951 | 0.01227 | 0.16147 | 0.22961 | 0.13282 | 0.16482 |
| LightGCN | 0.08191 | 0.12058 | 0.04992 | 0.06159 | 0.21805 | 0.30070 | 0.18208 | 0.21982 |
| XSimGCL | 0.03032 | 0.04556 | 0.01674 | 0.02081 | 0.22377 | 0.30931 | 0.18785 | 0.22951 |
| GWCDR | 0.08872 | 0.12635 | 0.04791 | 0.06102 | 0.21884 | 0.30615 | 0.17963 | 0.21767 |
| SRTrans | 0.08111 | 0.12591 | 0.04868 | 0.06173 | 0.19818 | 0.28043 | 0.16253 | 0.20075 |
| MITrans | 0.06840 | 0.10284 | 0.04174 | 0.05219 | 0.19499 | 0.27551 | 0.16212 | 0.19896 |
| FFMSR | 0.08436 | 0.12318 | 0.04712 | 0.06344 | 0.20142 | 0.28491 | 0.16573 | 0.20384 |
| FedDCSR | 0.08194 | 0.11976 | 0.04837 | 0.06102 | 0.19685 | 0.27866 | 0.16104 | 0.19821 |
| FedCRF-Fed | 0.10184 | 0.14898 | 0.06157 | 0.07554 | 0.22108 | 0.30757 | 0.18312 | 0.22291 |
| FedCRF-CL | 0.10727 | 0.15666 | 0.06454 | 0.07894 | 0.22322 | 0.31066 | 0.18500 | 0.22416 |
| FedCRF-FGSAT | 0.10682 | 0.15854 | 0.06359 | 0.07851 | 0.22401 | 0.30943 | 0.18330 | 0.22242 |
| **FedCRF** | **0.11142** | **0.16185** | **0.06713** | **0.08189** | **0.22864** | **0.31488** | **0.19232** | **0.23368** |

**来源：** FedCRF Table 4。

**主表是否能直接使用？**  
**不能。** 需要换成 M/B/V 并与 FedPASA 同 split 重跑。

---

# 5. FedDCSR（SDM 2024）

**论文：** FedDCSR: Federated Cross-domain Sequential Recommendation via Disentangled Representation Learning  
**arXiv：** <https://arxiv.org/abs/2309.08420>  
**官方代码：** <https://github.com/orion-orion/FedDCSR>

原论文评测：
- leave-one-out
- 每个 positive 搭配 **999 个 negative items**
- 因此不是 full ranking
- 指标：MRR、HR@10、NDCG@10

## 5.1 Movie–Book–Game

原论文表格单位是百分比；下面已转换为 0–1。

| Domain | MRR | HR@10 | NDCG@10 |
|---|---:|---:|---:|
| Movie | **0.1611** | **0.2832** | **0.1809** |
| Book | **0.1038** | **0.1868** | **0.1156** |
| Game | **0.0765** | **0.1060** | **0.0783** |

**来源：** FedDCSR 原论文 Table（Movie–Book–Game scenario）。  
原文明确说明每个 validation/test sample 使用 999 negative items。

### 更适合我们参考的是 TransFR 对 FedDCSR 的重跑结果

| Method | M→B H@10 | N@10 | M→V H@10 | N@10 | B→V H@10 | N@10 |
|---|---:|---:|---:|---:|---:|---:|
| FedDCSR（TransFR protocol） | **0.2217** | **0.1085** | **0.1881** | **0.0892** | **0.1719** | **0.0732** |

这组数值来自 TransFR 官方论文，不是 FedDCSR 原论文。

**主表是否能直接使用？**  
**不能。** 最终需要统一 full-ranking + 六方向 + 五种子重跑。

---

# 6. PFCR（WWW 2024，可选新增 baseline）

**论文：** Prompt-enhanced Federated Content Representation Learning for Cross-domain Recommendation  
**arXiv：** <https://arxiv.org/abs/2401.14678>

原数据：
- Office–Arts
- OnlineRetail–Pantry

指标：
- Recall@10 / Recall@50
- NDCG@10 / NDCG@50

PFCR(f) = full prompting，是论文后续默认所称 PFCR。

## 6.1 PFCR Full Prompting

| Domain | Recall@10 | Recall@50 | NDCG@10 | NDCG@50 |
|---|---:|---:|---:|---:|
| Office | **0.1206** | **0.1938** | **0.0782** | **0.0941** |
| Arts | **0.1153** | **0.2199** | **0.0677** | **0.0904** |
| OnlineRetail | **0.1561** | **0.3982** | **0.0739** | **0.1266** |
| Pantry | **0.0616** | **0.1591** | **0.0293** | **0.0503** |

**来源：** PFCR Table 1。  
论文原文 `*` 表示相对当时 best baseline 的 t-test \(p<0.05\)。

**注意：**原论文显著性符号不能拿来表示相对 FedPASA 显著。

---

# 7. 当前 FedPASA 稿件已有实验结果

以下来自当前 `FedPASA_WWW_12pages_updated.pdf`。

## 7.1 当前 sampled-99 六方向结果

| Direction | FedPASA HR@20 | FedPASA NDCG@20 |
|---|---:|---:|
| Music→Books | 0.2356 | 0.0911 |
| Music→Movies | 0.2898 | 0.1141 |
| Books→Movies | 0.3757 | 0.1633 |
| Books→Music | 0.3875 | 0.1563 |
| Movies→Books | 0.2450 | 0.0977 |
| Movies→Music | 0.4017 | 0.1627 |
| **Mean** | **0.3225** | **0.1309** |

这些是当前论文 **sampled-99** 主实验，不是 full ranking。

## 7.2 当前 full-ranking cross-check

当前 Appendix 只给出了六方向平均 HR@20：

| Method | Sampled-99 HR@20 | Full-ranking HR@20 |
|---|---:|---:|
| FedMF | 0.2115 | 0.0016 |
| FedAvg-Adapter | 0.2629 | 0.0028 |
| FedProx | 0.2616 | 0.0029 |
| FedAdam | 0.2562 | 0.0025 |
| TransFR-style | 0.2652 | 0.0025 |
| L2-SP | 0.2593 | 0.0027 |
| FedPer | 0.2705 | 0.0029 |
| pFedMe-style | 0.2749 | 0.0026 |
| FedMoE | 0.2731 | 0.0029 |
| **FedPASA base** | **0.3053** | **0.0048** |

### 当前仍缺

- 六方向分别的 full-ranking HR@10
- 六方向分别的 full-ranking HR@20
- full-ranking NDCG@10
- full-ranking NDCG@20
- 五种子方向级 mean±std
- 对新 baseline 的统一显著性检验

因此 FedPASA 本身也需要补一次完整统一 full-ranking evaluation。

---

# 8. 哪些结果可以“直接拿来写”，哪些必须复现？

| Method / 结果 | 有公开数值 | 与 M/B/V 相同 | 六方向 | Full ranking 明确 | 有 @20 | 5 seeds ± std | 主表处理 |
|---|---:|---:|---:|---:|---:|---:|---|
| TransFR 官方 HR/NDCG@10 | ✓ | ✓ | ✗，仅 3 方向 | 未明确 | ✗ | ✗ | **可作 published reference；主表重跑** |
| TransFR 论文中的 FedDCSR | ✓ | ✓ | ✗，仅 3 方向 | 未明确 | ✗ | ✗ | **可作 sanity check；主表重跑** |
| FeDecider | ✓ | ✗ | ✗ | 与我们协议不同 | ✗ | ✗ | **重跑** |
| FFMSR | ✓ | ✗ | ✗ | ✓ | 原文 @10/@50 | ✗ | **重跑** |
| FedCRF | ✓ | ✗ | ✗ | ✓ | ✓ | 未给 | **重跑** |
| FedDCSR 原论文 | ✓ | 部分（Movie/Book） | ✗ | ✗，999 negatives | ✗ | ✗ | **重跑** |
| PFCR | ✓ | ✗ | ✗ | 原协议不同 | 原文 @10/@50 | ✗ | **可选重跑** |
| FedPASA 当前 full-ranking | ✓ | ✓ | 只有平均值 | ✓ | 只有 H@20 | 5 seeds 但未完整报告 | **补齐** |

---

# 9. 建议最终 WWW 主表统一成这一张

> **所有格子均由我们在同一 split 下重跑，不直接混用原论文数字。**

| Method | M→B | B→M | M→V | V→M | B→V | V→B | Mean |
|---|---|---|---|---|---|---|---|
| TransFR | H10 / H20 / N10 / N20 | … | … | … | … | … | … |
| FFMSR | H10 / H20 / N10 / N20 | … | … | … | … | … | … |
| FedCRF | H10 / H20 / N10 / N20 | … | … | … | … | … | … |
| FedDCSR | H10 / H20 / N10 / N20 | … | … | … | … | … | … |
| FeDecider | H10 / H20 / N10 / N20 | … | … | … | … | … | … |
| PFCR（可选） | H10 / H20 / N10 / N20 | … | … | … | … | … | … |
| **FedPASA** | **H10 / H20 / N10 / N20** | … | … | … | … | … | … |

每个结果建议记录：

```text
mean ± std
```

固定随机种子：

```text
42, 123, 456, 789, 2024
```

显著性：

```text
*  : FedPASA significantly outperforms the baseline, p < 0.05
** : FedPASA significantly outperforms the baseline, p < 0.01
```

建议所有方法都使用相同 user split、candidate set 与 seed，使 paired significance test 有明确对应关系。

---

# 10. 最推荐的复现优先级

1. **TransFR**
   - 最接近 FedPASA
   - 同 M/B/V
   - 先补反向 3 方向 + @20 + full ranking + 5 seeds

2. **FFMSR**
   - 有官方代码
   - 已有 full sorting
   - 换数据和指标相对容易

3. **FedDCSR**
   - 官方代码可用
   - 需要把 999-negative evaluator 改成 full-ranking evaluator

4. **FedCRF**
   - 评测定义最接近我们要的新主表
   - M/B/V 上需要重新实现/适配

5. **FeDecider**
   - WWW 2026，非常重要
   - LLM + LoRA 复现成本最高
   - 公开代码完整度目前较低，放最后

6. **PFCR**
   - WWW 2024
   - 若篇幅/算力允许可加入，能增强 WWW relevance

---

# 11. 来源索引

## TransFR
- 正式页面：<https://ojs.aaai.org/index.php/AAAI/article/view/40048>
- PDF：<https://ojs.aaai.org/index.php/AAAI/article/download/40048/44009>
- DOI：<https://doi.org/10.1609/aaai.v40i33.40048>

## FeDecider
- ACM DOI：<https://doi.org/10.1145/3774904.3792294>
- arXiv：<https://arxiv.org/abs/2602.16034>

## FFMSR
- arXiv：<https://arxiv.org/abs/2503.23026>
- GitHub：<https://github.com/Sapphire-star/FFMSR>

## FedCRF
- ScienceDirect / DOI：<https://doi.org/10.1016/j.ipm.2026.104827>
- arXiv：<https://arxiv.org/abs/2604.17681>

## FedDCSR
- arXiv：<https://arxiv.org/abs/2309.08420>
- GitHub：<https://github.com/orion-orion/FedDCSR>

## PFCR
- arXiv：<https://arxiv.org/abs/2401.14678>

---

# 12. 最重要的使用原则

论文中建议把结果明确分成两类：

### Published results under original protocols
用于 Related Work、复现 sanity check、说明已有方法原始性能。

### Reproduced under our unified protocol
用于 WWW 主实验公平比较。

**不要把两种结果混进同一张“公平比较”主表中。**

最终真正具有说服力的主实验应当是：

> **same M/B/V data + same split + same test users + full ranking + same 5 seeds + HR@10/20 + NDCG@10/20 + mean±std + paired significance**

这样 reviewer 很难再从 protocol fairness 上攻击。
