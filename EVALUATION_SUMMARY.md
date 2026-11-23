# ReviewGuard: Comprehensive Model Evaluation Summary

This document reports the empirical results of the classical machine learning models persisted in `reviewguard/models/`. All numbers below are taken from `summary.json`, `ensemble_metrics.json`, `clustering_results.json`, `association_rules.json`, and `hmm_model.json` so that the write-up stays consistent with the shipped artifacts and the `README.md` results table.

---

## 1. Classical Model Benchmarks

All single models were trained on a stratified 15,000-review sample of the balanced 40,432-review Kaggle Fake Reviews Benchmark (12,000 train / 3,000 test, random seed 42). Features consist of 2,000 TF-IDF unigrams + bigrams combined with 9 statistical linguistic attributes (2,009 features total).

| Algorithm | Model Paradigm | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV (Mean ± Std) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (Calibrated)** | Max-Margin Hyperplane | **87.83%** | **88.47%** | **86.99%** | **87.72%** | **0.9501** | **87.66% ± 0.14%** |
| **Logistic Regression (L1 LASSO)** | Sparse Regularization | 86.83% | 88.23% | 84.99% | 86.58% | 0.9428 | 86.07% ± 0.45% |
| **Multinomial Naive Bayes** | Generative Probabilistic | 80.50% | 79.03% | 82.99% | 80.96% | 0.8929 | 80.42% ± 0.75% |
| **Decision Tree (CART)** | Non-Linear Recursive Partitioning | 70.80% | 74.35% | 63.44% | 68.47% | 0.7595 | 69.51% ± 2.02% |
| **K-Nearest Neighbors** | Instance-Based Memory Learner | 63.77% | 60.96% | 76.45% | 67.83% | 0.7233 | 67.76% ± 1.32% |

---

## 2. Ensemble Architectures & Generalization Lift

Ensemble architectures were evaluated on the same 3,000-review test partition. `ensemble_metrics.json` reports Random Forest as the best-performing ensemble, but **no ensemble beat the calibrated Linear SVM** (F1 87.72%).

| Ensemble Method | Architecture | Aggregation Strategy | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | F1 Delta vs Linear SVM |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | Bagging (100 Trees) | Bootstrap Sampling + Random Subspaces | 81.92% | 83.54% | 79.50% | 81.47% | 0.9169 | −6.25 pp |
| **Soft Voting Consensus** | Model Averaging | RF 0.40 / GBDT 0.35 / AdaBoost 0.25 | 81.33% | 82.98% | 78.83% | 80.85% | 0.9120 | −6.87 pp |
| **Gradient Boosting (GBDT)** | Stage-Wise Boosting | Additive Loss Optimization | 80.92% | 82.04% | 79.17% | 80.58% | 0.9048 | −7.14 pp |
| **AdaBoost** | Sequential Boosting | Exponential Loss Sample Reweighting | 73.25% | 73.21% | 73.33% | 73.27% | 0.7995 | −14.45 pp |

### Key Architectural Findings:
1. **Linear SVM Superiority in High-Dimensional Sparse Spaces:** TF-IDF produces an extremely sparse ~2,000-dimensional feature matrix. A regularized max-margin hyperplane captures diffuse lexical patterns more effectively than axis-aligned tree splits or distance-based voting.
2. **Tree Ensembles Underperform on Sparse Text:** Orthogonal axis splits in Random Forest, GBDT, and AdaBoost cannot model distributed token combinations as efficiently, capping F1 at 81.47% for the best ensemble.
3. **Voting Dilution:** Soft Voting averages heterogeneous probabilities (RF, GBDT, AdaBoost). It reaches 80.85% F1 but remains well below the linear baselines, so the SVM remains the default production model in `summary.json` and in the Live Review Checker.

---

## 3. Unsupervised Reviewer Clustering & Pattern Mining

The behavioral modules operate on the synthetic `sample_reviews_dataset.csv` (1,000 reviews, 236 distinct reviewer profiles), because the Kaggle benchmark contains no reviewer IDs or timestamps. Each profile is described by 7 behavioral dimensions: average star rating, rating variance, reviews per day, verified-buyer ratio, average review length, exclamation average, and account age.

### K-Means & Hierarchical Findings (cached `clustering_results.json`):
- **Cluster count:** the cached result uses **k = 3** across 236 reviewers. Elbow inertia falls from 805.16 (k=2) to 625.01 (k=3) to 534.86 (k=4); the highest silhouette score is 0.6271 at k=2 (k=3: 0.2701), so the optimal k depends on which criterion is prioritized.
- **Cluster 0 (Organic Typical Buyers):** 50.8% of users, average rating 4.12★, 0.267 reviews/day, 82% verified, 140.2-character reviews, 0% fake ratio → *Normal / Organic Behavior*.
- **Cluster 1 (Bursty Bot Farm):** 10.6% of users, average rating 4.82★, 2.693 reviews/day, 19% verified, 95.2-character reviews, 100% fake ratio → *Suspicious Ring / Bot Farm* (high severity).
- **Cluster 2 (Low-Activity Normal Users):** 38.6% of users, average rating 3.04★, 0.071 reviews/day, 86% verified, 140.4-character reviews, 0% fake ratio → *Normal / Organic Behavior*.
- **2D PCA** explains 56.02% and 16.16% of variance. The **MST** connects centroids with a 5.974-weight edge between clusters 0 and 1 (flagged as an anomaly cut) and a 1.866-weight edge between clusters 0 and 2.

### Association Rules (FP-Growth, cached `association_rules.json`):
Mined at minimum support 0.05 and minimum confidence 0.30, yielding 50 rules. Representative high-lift rules targeting fake labels:

| Antecedents | Consequents | Support | Confidence | Lift |
| :--- | :--- | :---: | :---: | :---: |
| `[short_review]` | `[label_fake, unverified_buyer, burst_posting]` | 8.16% | 82.72% | 4.53× |
| `[rating_5, short_review]` | `[label_fake, unverified_buyer, burst_posting]` | 6.70% | 82.09% | 4.49× |
| `[new_account]` | `[label_fake, unverified_buyer, burst_posting]` | 11.69% | 81.36% | 4.45× |

---

## 4. Sequence Modeling via Hidden Markov Models (HMM)

A 2-state HMM was fitted over the 236 reviewer timelines in the synthetic behavioral dataset (`hmm_model.json`):

- **Hidden State 0:** Normal Reviewer
- **Hidden State 1:** Spammer / Burst
- **Emissions:** 4 discrete buckets combining star rating and inter-review interval.
- **Start probabilities:** `[0.80, 0.20]`

### Empirical Transition Matrix (A):
$$A = \begin{bmatrix} 0.880 & 0.120 \\ 0.250 & 0.750 \end{bmatrix}$$

- **State Persistence:** Normal reviewers have an 88% probability of remaining in the organic state; spammers have a 75% probability of continuing a burst. The API exposes per-user state sequences and anomaly probability through `GET /api/hmm/{user_id}`.
- **Emission Contrast:** Normal-state emissions are dominated by rating 1–3 / long-gap buckets (0.627, 0.360), while the burst state is dominated by the extreme-rating / short-gap bucket (0.756).

---

## 5. Dataset Constraints & Critical Limitations

1. **Synthetic Nature of Sample Behavioral Dataset:**
   - The primary Kaggle benchmark (`fake reviews dataset.csv`) contains ground-truth text and labels but **no reviewer IDs or timestamps**.
   - Consequently, reviewer clustering, association rule mining, and HMM sequence modeling rely on `sample_reviews_dataset.csv`. While structurally faithful to e-commerce logs, this dataset is **synthetic** and exhibits cleaner cluster separation than real-world adversarial environments.
2. **Generative Artifacts in Kaggle Benchmark:**
   - The Kaggle Fake Reviews dataset was generated using early conditional language models. In this dataset, fake reviews (`CG`) tend to be grammatically correct, formal, and longer, whereas human reviews (`OR`) are frequently shorter and contain colloquial typos.
   - Real-world human click-farm reviews often exhibit inverted patterns (e.g., promotional hype, broken English, excessive capitalization). ReviewGuard addresses this domain discrepancy by fusing text model consensus with rule-based heuristics and behavioral signals.
3. **Absence of Real-Time Graph Topology:**
   - Reviewer syndicates in production e-commerce ecosystems frequently operate across bipartite reviewer-product graphs. ReviewGuard's classical MST clustering operates on reviewer behavioral vectors; incorporating full bipartite graph mining (e.g., Co-Review Graph Spectral Bisection) would further enhance collusion ring detection.
