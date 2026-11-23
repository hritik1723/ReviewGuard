# ReviewGuard: Comprehensive Model Evaluation Summary

This document provides a detailed comparative analysis of all classical machine learning models, regularized regressions, and ensemble architectures implemented within **ReviewGuard**. It examines empirical diagnostic trade-offs, learning dynamics, feature selection results, and inherent dataset constraints.

---

## 1. Classical Model Benchmarks

All models were evaluated on the balanced Kaggle Fake Reviews Benchmark partition ($N = 10,000$ stratified samples, 80/20 train/test split, random seed = 42) using TF-IDF n-grams (1, 2) coupled with statistical linguistic features (character length, word count, exclamation frequency, capitalization ratio, and VADER sentiment polarity).

| Algorithm | Model Paradigm | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV (Mean $\pm$ Std) | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM** | Max-Margin Hyperplane | **86.8%** | **87.2%** | **86.4%** | **86.8%** | **0.932** | **86.5% $\pm$ 0.8%** | ~4.2 ms |
| **Logistic Regression (L1)** | Sparse Regularization (LASSO) | 86.1% | 86.5% | 85.8% | 86.1% | 0.925 | 85.9% $\pm$ 0.9% | ~3.8 ms |
| **Multinomial Naive Bayes** | Generative Probabilistic | 85.3% | 86.0% | 84.5% | 85.2% | 0.914 | 85.0% $\pm$ 1.1% | ~2.1 ms |
| **CART Decision Tree** | Non-Linear Recursive Partitioning | 79.4% | 80.1% | 78.5% | 79.3% | 0.792 | 78.8% $\pm$ 1.4% | ~3.1 ms |
| **KNN (k=5, Cosine)** | Instance-Based Memory Learner | 71.2% | 73.0% | 67.5% | 70.1% | 0.761 | 69.8% $\pm$ 1.8% | ~38.5 ms |

---

## 2. Ensemble Architectures & Generalization Lift

Ensemble learning methods were deployed to test variance and bias reduction over single hypotheses:

| Ensemble Method | Architecture | Aggregation Strategy | Test F1-Score | F1 Delta vs Best Single Model | ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Random Forest** | Bagging (100 Trees) | Bootstrap Sampling + Random Subspaces | 85.4% | -1.4% | 0.928 |
| **AdaBoost** | Boosting (Sequential) | Exponential Loss Sample Reweighting | 84.8% | -2.0% | 0.918 |
| **Gradient Boosting (GBDT)**| Boosting (Stage-Wise) | Negative Binomial Deviance Optimization | 86.2% | -0.6% | 0.930 |
| **Soft Voting Consensus** | Multi-Model Consensus | Probability Mixture ($w_{SVM}=0.35, w_{LR}=0.35, w_{RF}=0.30$) | **87.5%** | **+0.7% Lift** | **0.938** |

### Key Architectural Findings:
1. **Linear SVM Superiority in High-Dimensional Sparse Spaces:** Because TF-IDF yields high-dimensional, sparse sparse feature matrices ($D \approx 2,000$), Linear SVM and L1-regularized Logistic Regression significantly outperform tree-based and distance-based methods. Hyperplane separation avoids the curse of dimensionality that impacts KNN.
2. **Soft Voting Synergy:** Combining Linear SVM, L1 Logistic Regression, and Random Forest achieves an empirical peak F1 of **87.5%** (AUC: 0.938), outperforming any single model alone by mitigating individual classification edge cases.
3. **Decision Tree Depth Saturation:** CART Decision Trees suffer from high variance when splitting on sparse lexical terms; pruning at depth 16 stabilized cross-validation variance ($\pm 1.4\%$) but capped overall F1 at 79.4%.

---

## 3. Unsupervised Reviewer Clustering & Pattern Mining

Using the behavioral metadata dataset (`sample_reviews_dataset.csv`), reviewer profiles were aggregated along 7 behavioral dimensions:
- Average star rating, rating variance ($\sigma$), reviews per day, verified buyer ratio, average review character length, exclamation marks per review, and user account age.

### K-Means & Hierarchical Findings:
- **Optimal Clusters ($k = 4$):** Evaluated via the Elbow Method (inflection point at $k=4$, WCSS drops from 1240 to 480) and Silhouette peak of **0.38**.
- **Cluster 0 (Organic Typical Buyers):** 50.8% of users, average rating 4.1★, 0.08 reviews/day, 92% verified purchase ratio, average review length 180 characters. Tagged as *Normal / Organic*.
- **Cluster 1 (Bursty Review-Bombing / Bot Farm):** 10.6% of users, average rating 4.85★, 2.69 reviews/day, 12% verified purchase ratio, short repetitive reviews with high exclamation rates ($> 2.5$). Tagged as *Suspicious Ring / Bot Farm*.
- **Minimum Spanning Tree (MST) Cut:** Centroid Euclidean distances connected via MST revealed a natural cut threshold at $d = 5.974$, severing Cluster 1 from Cluster 0 without requiring manual density tuning.

### Association Rules (FP-Growth):
Discretized transaction itemsets revealed high-confidence behavioral markers for deceptive reviews:
- `[unverified_buyer, new_account, rating_5] => [label_fake]` (Support: 9.2%, Confidence: 88.5%, Lift: **3.44x**)
- `[high_exclamations, short_review, rating_5] => [label_fake]` (Support: 6.8%, Confidence: 82.1%, Lift: **3.19x**)
- `[verified_buyer, mature_account] => [label_genuine]` (Support: 52.4%, Confidence: 94.2%, Lift: **1.27x**)

---

## 4. Sequence Modeling via Hidden Markov Models (HMM)

To capture temporal reviewer dynamics, a 2-State Ergotic HMM was trained over user review timelines:
- **Hidden State 0:** Normal / Organic posting state
- **Hidden State 1:** Spammer / Sybil burst state
- **Emissions:** Discrete tuples combining star ratings ($1\star..5\star$) and inter-review intervals ($\Delta t \le 3$ days vs $\Delta t > 3$ days).

### Empirical Transition Matrix ($A$):
$$A = \begin{bmatrix} 0.880 & 0.120 \\ 0.250 & 0.750 \end{bmatrix}$$

- **State Persistence:** Normal reviewers exhibit an 88% probability of remaining in an organic state. Spammers exhibit a 75% probability of continued burst spamming.
- **Viterbi Sequence Decoding:** Decoded user trajectories demonstrate that accounts switching from State 0 to State 1 coincide with burst posting spikes ($\ge 3$ reviews in 48 hours) coupled with extreme rating bias. Spammer risk probability ($P(\text{Spam} \mid O)$) reliably flags accounts with sequence log-likelihoods below -10.0.

---

## 5. Dataset Constraints & Critical Limitations

1. **Synthetic Nature of Sample Behavioral Dataset:**
   - The primary Kaggle benchmark (`fake reviews dataset.csv`) contains ground-truth text and labels but **no reviewer IDs or timestamps**.
   - Consequently, reviewer clustering, association rule mining, and HMM sequence modeling rely on `sample_reviews_dataset.csv`. While structurally faithful to e-commerce logs, this dataset is **synthetic** and exhibits cleaner cluster separation than real-world adversarial environments.
2. **Generative Artifacts in Kaggle Benchmark:**
   - The Kaggle Fake Reviews dataset was generated using early conditional language models. In this dataset, fake reviews (`CG`) tend to be grammatically correct, formal, and longer, whereas human reviews (`OR`) are frequently shorter and contain colloquial typos.
   - Real-world human click-farm reviews often exhibit inverted patterns (e.g., promotional hype, broken English, excessive capitalization). ReviewGuard addresses this domain discrepancy by fusing text model consensus with rule-based heuristics and behavioral signals.
3. **Absence of Real-Time Graph Topology:**
   - Reviewer syndicates in production e-commerce ecosystems frequently operate across bipartite reviewer-product bipartite graphs. ReviewGuard's classical MST clustering operates on reviewer behavioral vectors; incorporating full bipartite graph mining (e.g. Co-Review Graph Spectral Bisection) would further enhance collusion ring detection.
