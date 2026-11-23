# ReviewGuard: Classical ML Fake Product Review Detection System

> An end-to-end machine learning system for detecting deceptive product reviews, profiling reviewer behavioral clusters, and modeling temporal review sequences using classical algorithms.

---

## About the Project

ReviewGuard was developed as an academic machine learning project for **B.Tech Computer Science & Engineering (Artificial Intelligence & Machine Learning)** at **Maulana Abul Kalam Azad University of Technology (MAKAUT), West Bengal**.

The project operates under a strict **classical machine learning curriculum requirement**:
- **Constraint**: Pure classical statistical and machine learning algorithms only.
- **Excluded**: Deep learning architectures, recurrent networks, transformers, BERT, and large language models.
- **Objective**: Demonstrate that rigorous text preprocessing, linguistic feature engineering, hypothesis-driven feature selection, linear margin classifiers, tree ensembles, unsupervised clustering, association rules, and Markov models can effectively identify deceptive patterns in consumer reviews.

### Attribution & Acknowledgments
- **Core ML Formulation & Validation**: Algorithm selection, feature design, experimental benchmarks, and diagnostic evaluations were authored by the project team.
- **UI & Integration Assistance**: Google Stitch aided the visual interface layout, and an AI coding assistant (Google Antigravity) supported full-stack integration and responsive wiring.

---

## Key Features

ReviewGuard provides a unified single-page web dashboard divided into 10 interactive modules:

1. **System Overview**: High-level platform telemetry, model benchmark radar charts, overall classification metrics, and live detection counters.
2. **Dataset Explorer**: In-depth inspection of review corpora, category distributions, star rating histograms, class balance visualizers, and paginated record search.
3. **Preprocessing & Features**: Visual 6-stage NLP cleaning pipeline, Chi-Square ($\chi^2$) and Mutual Information feature scoring charts, and a real-time class rebalancing simulator (SMOTE, oversampling, undersampling).
4. **Model Training Workbench**: Interactive training console supporting model selection, cross-validation fold configuration, train/test split adjustment, and live background training progress tracking.
5. **Model Evaluation Diagnostics**: Comprehensive diagnostic comparison displaying Accuracy, Precision, Recall, F1-Score, ROC-AUC curves, and an interactive 2x2 Confusion Matrix with sensitivity and specificity calculations.
6. **Ensemble Learning Studio**: Side-by-side analysis of Bagging (Random Forest), Boosting (AdaBoost and Gradient Boosting), and Soft Voting consensus, alongside Gini feature importance rankings.
7. **Reviewer Clustering**: Unsupervised reviewer profiling using K-Means with WCSS Elbow analysis, 2D PCA projection scatter, Agglomerative Hierarchical clustering dendrograms (Ward, Complete, Average, Single linkage), and Minimum Spanning Tree (MST) anomaly graph identification.
8. **Pattern Mining**: Frequent itemset discovery using Apriori and FP-Growth algorithms, surfacing behavioral rules with Support, Confidence, and Lift indicators.
9. **Sequence Analysis (HMM)**: A 2-State Hidden Markov Model (Organic vs. Burst Spammer) featuring empirical transition matrices, emission likelihoods, and Viterbi dynamic programming decoding across reviewer timelines.
10. **Live Review Checker**: Real-time review evaluation workbench providing calibrated probability scores, individual model predictions, linear feature attributions, and token-level linguistic highlights.

---

## University Syllabus Mapping

The implementation maps directly to the classical Machine Learning syllabus:

| Module | Curriculum Topic | Implemented Technique | Source Path / Dashboard Page |
| :--- | :--- | :--- | :--- |
| **Module 1** | Data Ingestion & Hygiene | CSV normalization ($CG \to 1$, $OR \to 0$), deduplication, and missing value handling | `reviewguard/src/load_data.py`<br>*(Dataset Explorer)* |
| **Module 1** | Classical Text Preprocessing | Lowercasing, regex cleaning, NLTK stopword filtering, and WordNet lemmatization | `reviewguard/src/preprocess.py`<br>*(Preprocessing & Features)* |
| **Module 2** | Feature Extraction | Sublinear TF-IDF (unigrams + bigrams), scalar text statistics, and VADER sentiment scores | `reviewguard/src/features.py`<br>*(Preprocessing & Features)* |
| **Module 2** | Feature Selection & Importance | Chi-Square test ($\chi^2$) and Mutual Information / Information Gain scoring | `reviewguard/src/feature_selection.py`<br>*(Preprocessing & Features)* |
| **Module 2** | Class Imbalance Handling | SMOTE (k-NN synthetic synthesis), Random Oversampling, Undersampling, and Class Weighting | `reviewguard/src/imbalance.py`<br>*(Preprocessing & Features)* |
| **Module 3** | Linear & Regularized Models | L1-regularized Logistic Regression (LASSO via liblinear) | `reviewguard/src/train.py`<br>*(Model Training & Evaluation)* |
| **Module 3** | Generative Classification | Multinomial Naive Bayes with Laplace smoothing | `reviewguard/src/train.py`<br>*(Model Training & Evaluation)* |
| **Module 4** | Margin-Based Classification | Linear Support Vector Machine (LinearSVC) with Platt probability calibration | `reviewguard/src/train.py`<br>*(Model Training & Live Checker)* |
| **Module 4** | Instance-Based Learning | K-Nearest Neighbors (KNN, $k=5$) with cosine distance weighting | `reviewguard/src/train.py`<br>*(Model Training & Evaluation)* |
| **Module 4** | Non-Linear Decision Trees | CART Decision Tree with Gini impurity splitting | `reviewguard/src/train.py`<br>*(Model Training & Evaluation)* |
| **Module 5** | Model Evaluation Diagnostics | Stratified 5-Fold CV, Precision, Recall, F1, ROC-AUC, and Confusion Matrix analysis | `reviewguard/src/evaluate.py`<br>*(Model Evaluation)* |
| **Module 5** | Ensemble: Bagging | Random Forest Classifier (100 decorrelated trees with random feature subspacing) | `reviewguard/src/ensemble.py`<br>*(Ensemble Learning)* |
| **Module 5** | Ensemble: Boosting | AdaBoost (exponential loss) and Gradient Boosted Decision Trees (GBDT deviance) | `reviewguard/src/ensemble.py`<br>*(Ensemble Learning)* |
| **Module 5** | Ensemble: Voting | Soft Voting consensus averaging calibrated class probabilities | `reviewguard/src/ensemble.py`<br>*(Ensemble Learning & Live Checker)* |
| **Module 6** | Unsupervised: Centroid | K-Means clustering with Within-Cluster Sum of Squares (WCSS / Elbow) and 2D PCA | `reviewguard/src/clustering.py`<br>*(Reviewer Clustering)* |
| **Module 6** | Unsupervised: Hierarchical | Agglomerative Clustering supporting Ward, Complete, Average, and Single linkages | `reviewguard/src/clustering.py`<br>*(Reviewer Clustering)* |
| **Module 6** | Graph-Based Clustering | Minimum Spanning Tree (MST) distance thresholding to isolate outlier clusters | `reviewguard/src/clustering.py`<br>*(Reviewer Clustering)* |
| **Module 7** | Association Rule Mining | Apriori and FP-Growth rule generation evaluated by Support, Confidence, and Lift | `reviewguard/src/association.py`<br>*(Pattern Mining)* |
| **Module 7** | Sequential Probabilistic Models| 2-State Hidden Markov Model with Forward, Backward, and Viterbi decoding | `reviewguard/src/hmm_model.py`<br>*(Sequence Analysis)* |

*(Note: Deep neural networks and transformers were intentionally excluded in accordance with syllabus constraints.)*

---

## Tech Stack

| Layer | Component / Library | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Runtime** | Python | 3.10+ (tested on 3.15) | Core application programming language |
| **Machine Learning** | `scikit-learn` | $\ge$ 1.4.0 (1.9.1) | Classifiers, feature extractors, calibration, metrics |
| **Scientific Computing**| `numpy`, `scipy` | $\ge$ 1.26.0, $\ge$ 1.12.0 | Sparse matrix operations, distance calculations, HMM |
| **Data Manipulation** | `pandas` | $\ge$ 2.2.0 | Tabular dataset management and aggregation |
| **NLP & Lexicons** | `nltk` | $\ge$ 3.8.1 | Tokenization, stopwords, WordNet, VADER sentiment |
| **Imbalance Handling** | `imbalanced-learn`| $\ge$ 0.12.0 | SMOTE and random resampling techniques |
| **Pattern Mining** | `mlxtend` | $\ge$ 0.23.0 | Apriori and FP-Growth transactional mining |
| **Web API Backend** | `fastapi`, `uvicorn` | $\ge$ 0.110.0, $\ge$ 0.28.0 | High-performance asynchronous REST API server |
| **Form Handling** | `python-multipart` | $\ge$ 0.0.9 | Dataset upload parsing |
| **Model Serialization**| `joblib` | $\ge$ 1.3.0 | Disk caching of fitted models and pipelines |
| **Testing** | `pytest` | $\ge$ 8.0.0 | Unit and integration test suite (32 tests) |
| **Frontend UI** | HTML5, Tailwind CSS, Vanilla JS | Standalone SPA | Single-page UI with zero build-step requirement |
| **Charts & Graphs** | `Chart.js` | 4.4.1 (CDN) | Interactive radar, ROC, PCA, bar, and histogram charts |

---

## Project Structure

```text
Review Guard/
├── requirements.txt            # Python dependencies
├── run.bat                     # Windows launch script
├── run.sh                      # Linux/macOS launch script
├── .gitignore                  # Git ignore rules for environments, caches, and raw data
├── README.md                   # Project documentation
├── docs/
│   └── screenshots/            # Clean UI captures of all 10 dashboard pages
├── reviewguard/
│   ├── app/                    # FastAPI application layer
│   │   ├── main.py             # Server entry point, middleware, static mounting
│   │   ├── schemas.py          # Pydantic request/response validation schemas
│   │   └── routes/             # REST API modular routers
│   │       ├── overview.py     # System telemetry and leaderboard metrics
│   │       ├── datasets.py     # Dataset listing, paginated explorer, CSV uploads
│   │       ├── features.py     # Chi-Square, Mutual Information, SMOTE simulator
│   │       ├── train.py        # Asynchronous model training trigger & status polling
│   │       ├── evaluation.py   # Diagnostic metrics, confusion matrix, ROC curves
│   │       ├── ensemble.py     # Bagging, Boosting, and Soft Voting benchmarks
│   │       ├── clustering.py   # K-Means, Hierarchical, PCA, and MST graph endpoints
│   │       ├── association.py  # Apriori and FP-Growth rule mining
│   │       ├── hmm.py          # HMM transition matrix, user sequences, Viterbi
│   │       └── predict.py      # Live review inference and prediction audit history
│   ├── src/                    # Classical Machine Learning intelligence core
│   │   ├── load_data.py        # Dataset discovery, schema validation, normalization
│   │   ├── preprocess.py       # Regex sanitization, stopwords, lemmatization
│   │   ├── features.py         # Sublinear TF-IDF and linguistic feature extraction
│   │   ├── feature_selection.py# Chi-square and Mutual Information algorithms
│   │   ├── imbalance.py        # SMOTE and class rebalancing handlers
│   │   ├── train.py            # Model fitting, calibration, and parameter logging
│   │   ├── evaluate.py         # 5-fold cross-validation and classification metrics
│   │   ├── ensemble.py         # Random Forest, AdaBoost, GBDT, Soft Voting
│   │   ├── clustering.py       # Centroid, Agglomerative, PCA, MST graph clustering
│   │   ├── association.py      # Discretized itemset preparation and rule mining
│   │   ├── hmm_model.py        # First-principles NumPy HMM (Forward/Backward/Viterbi)
│   │   └── predict.py          # Real-time multi-model predictor and explainability
│   ├── frontend/               # Embedded single-page application
│   │   ├── index.html          # Semantic HTML structure with responsive sidebar
│   │   └── app.js              # State management, router, API client, Chart.js views
│   ├── models/                 # Pre-trained joblib weights and benchmark JSON records
│   ├── data/
│   │   └── raw/
│   │       └── sample_reviews_dataset.csv # 1,000-sample synthetic behavioral dataset
│   └── tests/                  # Automated pytest test suites
│       ├── test_loader.py      # Dataset loading and schema validation tests
│       ├── test_preprocess.py  # Text cleaning and lemmatization tests
│       ├── test_features.py    # TF-IDF and linguistic feature pipeline tests
│       ├── test_models.py      # Classical model training and evaluation tests
│       ├── test_predict.py     # Inference predictor and explainability tests
│       ├── test_clustering_association_hmm.py # Unsupervised and Markov model tests
│       └── test_api.py         # FastAPI route and integration tests
```

---

## Getting Started

### 1. Prerequisites
- **Python**: Version 3.10 or higher (Python 3.10 to 3.15 supported).
- **Git**: Installed and available in your terminal path.

### 2. Environment Setup

Clone the repository and establish an isolated virtual environment:

```bash
# Clone the repository
git clone <repository-url>
cd "Review Guard"

# Create a virtual environment
python -m venv .venv
```

Activate the virtual environment:

- **Windows (PowerShell)**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt)**:
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
- **macOS / Linux**:
  ```bash
  source .venv/bin/activate
  ```

Install dependencies:
```bash
pip install -r requirements.txt
```

Download the required NLTK corpora:
```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

### 3. Dataset Configuration

ReviewGuard works with two datasets:

#### Primary Benchmark: Kaggle Fake Reviews Dataset
- **Volume**: 40,432 balanced reviews (20,216 computer-generated fakes `CG`, 20,216 original human reviews `OR`).
- **Columns**: `category`, `rating`, `label`, `text_`.
- **Download**: Obtain `fake reviews dataset.csv` from [Kaggle Fake Reviews Dataset](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset).
- **Placement**: Place the file at either:
  - `reviewguard/data/raw/fake reviews dataset.csv` (recommended)
  - Or directly in the repository root: `fake reviews dataset.csv`
- The system automatically detects and loads the dataset on startup.

#### Demonstration Dataset: Synthetic Behavioral Reviews
- **Volume**: 1,000 synthesized e-commerce transactions (`sample_reviews_dataset.csv`).
- **Columns**: `review_id`, `user_id`, `product_id`, `category`, `rating`, `review_date`, `verified_purchase`, `account_age_days`, `text_`, `label`.
- **Purpose**: Evaluates user-level features, rating variance, reviewer clustering, association rules, and HMM sequence modeling.
- **Notice**: Included directly in the repository for immediate evaluation. It is explicitly synthetic and intended for architectural demonstration.

---

### 4. Training the Models

Pre-trained weights and metrics on a balanced 15,000-sample partition are bundled in `reviewguard/models/`. 

To retrain from scratch:
1. **Via the Web Interface**: Navigate to the **Model Training** page (`#model-training`), select your desired models, adjust sample size, and click **Train Selected Models**.
2. **Via Command Line**:
   ```bash
   python -c "from reviewguard.src.train import train_pipeline; train_pipeline(dataset_name='kaggle', max_samples=15000)"
   python -c "from reviewguard.src.ensemble import train_ensemble_pipeline; train_ensemble_pipeline(dataset_name='kaggle', max_samples=15000)"
   ```

---

### 5. Running the Application

Start the web server using Uvicorn:

```bash
python -m uvicorn reviewguard.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or execute the provided launcher scripts:
- **Windows**: `run.bat`
- **macOS / Linux**: `chmod +x run.sh && ./run.sh`

Once running:
- **Application Dashboard**: Open [http://localhost:8000](http://localhost:8000)
- **Interactive OpenAPI Documentation**: Open [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative API Reference (ReDoc)**: Open [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

### 6. Running Automated Tests

Execute the automated test suite covering loaders, feature engineering, models, and API endpoints:

```bash
python -m pytest reviewguard/tests -v
```

All 32 test cases should pass successfully.

---

## REST API Reference

All backend routes are mounted under the `/api` prefix:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the single-page application (`index.html`) |
| `GET` | `/api/health` | Service health status check |
| `GET` | `/api/overview` | High-level telemetry, primary KPIs, and radar benchmark values |
| `GET` | `/api/datasets` | Metadata and statistics for available datasets |
| `GET` | `/api/dataset/explore` | Paginated review explorer with text search and category filters |
| `POST` | `/api/dataset/upload` | Upload and validate a custom CSV review dataset |
| `GET` | `/api/features/importance` | Chi-Square and Mutual Information rankings for lexical features |
| `POST` | `/api/imbalance` | Test class rebalancing techniques (SMOTE, Random Oversample/Undersample) |
| `POST` | `/api/train` | Trigger asynchronous classical model training job |
| `GET` | `/api/train/status` | Poll status, message, and completion percentage of training jobs |
| `GET` | `/api/evaluation` | Model diagnostics, confusion matrix parameters, and ROC curve data |
| `GET` | `/api/ensemble` | Benchmarks for Bagging, Boosting, and Soft Voting ensembles |
| `GET` | `/api/clustering` | K-Means WCSS curve, 2D PCA points, Dendrogram links, and MST graph |
| `GET` | `/api/association` | Discretized association rule generation with Support, Confidence, Lift |
| `GET` | `/api/hmm/summary` | 2-State HMM transition matrix, emission matrix, and reviewer risk distribution |
| `GET` | `/api/hmm/{user_id}` | Individual reviewer chronological history with Viterbi state sequence |
| `POST` | `/api/predict` | Real-time review classification, probability calibration, and explainability |
| `GET` | `/api/history` | Audit log of recent live review predictions |

---

## Experimental Benchmark Results

Evaluated on the Kaggle Fake Reviews Benchmark ($N = 15,000$ balanced partition, 80/20 stratified train/test split, random seed 42, 5-fold cross-validation) using unigram/bigram sublinear TF-IDF and linguistic statistical features:

| Model Category | Algorithm | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear / Margin** | **Linear SVM (Calibrated)** | **87.8%** | **88.5%** | **87.0%** | **87.7%** | **0.950** |
| **Linear / Regularized**| **Logistic Regression (L1 LASSO)**| 86.8% | 88.2% | 85.0% | 86.6% | 0.943 |
| **Generative** | **Multinomial Naive Bayes** | 80.5% | 79.0% | 83.0% | 81.0% | 0.893 |
| **Ensemble (Bagging)** | **Random Forest (100 Trees)** | 81.9% | 83.5% | 79.5% | 81.5% | 0.917 |
| **Ensemble (Voting)** | **Soft Voting Consensus** | 81.3% | 83.0% | 78.8% | 80.8% | 0.912 |
| **Ensemble (Boosting)**| **Gradient Boosting (GBDT)** | 80.9% | 82.0% | 79.2% | 80.6% | 0.905 |
| **Ensemble (Boosting)**| **AdaBoost** | 73.3% | 73.2% | 73.3% | 73.3% | 0.800 |
| **Decision Tree** | **CART Decision Tree (depth=16)** | 70.8% | 74.4% | 63.4% | 68.5% | 0.760 |
| **Instance-Based** | **K-Nearest Neighbors ($k=5$)** | 63.8% | 61.0% | 76.5% | 67.8% | 0.723 |

### Honest Diagnostic Discussion
1. **Linear SVM and L1 Regularization Dominate**: In high-dimensional sparse lexical feature spaces ($D \approx 2,009$), maximum-margin hyperplanes (Linear SVM) and sparse feature shrinkage (LASSO) consistently outperform tree and distance-based classifiers. The maximum-margin boundary resists overfitting in sparse feature distributions.
2. **Ensembles Did Not Surpass the Linear SVM**: While Random Forest (81.5% F1) and Gradient Boosting (80.6% F1) significantly improve upon single decision trees (68.5% F1), tree ensembles did not outperform the calibrated Linear SVM (87.7% F1). Tree splits on sparse term matrices divide individual orthogonal features rather than computing linear feature combinations.
3. **The Curse of Dimensionality on KNN**: K-Nearest Neighbors showed the lowest accuracy (63.8%), as Euclidean and Cosine distances degrade in high-dimensional sparse TF-IDF spaces.

---

## Limitations

- **Computer-Generated Benchmark Artifacts**: The Kaggle dataset fakes (`CG`) were generated using early conditional language models. They tend to feature formal grammar and longer sentences, whereas genuine human reviews (`OR`) contain conversational colloquialisms. Real-world human click-farm reviews frequently exhibit the reverse (broken grammar, promotional hype, excessive exclamation marks).
- **Absence of User Metadata in the Primary Benchmark**: The Kaggle dataset provides text and star ratings only, with no reviewer identifiers or timestamps.
- **Synthetic Behavioral Demonstrations**: Unsupervised reviewer clustering, association rules, and HMM sequence modeling rely on `sample_reviews_dataset.csv`. While structurally reflective of real platform logs, it is synthetically generated and exhibits cleaner cluster separation than noisy production environments.
- **Linguistic Scope**: Text preprocessing and sentiment lexicons (VADER) currently support English text only.
- **Partitioning Constraints**: Training was conducted on a 15,000-sample partition to ensure interactive model fitting on standard laptop hardware.

---

## Future Work

- **Real-World Metadata Datasets**: Integrate production review corpora with authentic reviewer histories and timestamps (such as the Yelp Open Dataset).
- **Stylometric & Syntactic Features**: Incorporate Part-of-Speech (POS) distribution ratios, syntactic parse tree depth, and readability indexes (Flesch-Kincaid).
- **Expectation-Maximization (EM) & GMM**: Implement Gaussian Mixture Models with EM for probabilistic soft clustering of reviewer behaviors.
- **Multilingual Support**: Extend tokenization and sentiment lexicons to regional languages, specifically Hindi and Bengali.
- **Browser Extension**: Build a client-side browser plugin to run live inferences on e-commerce product pages.

---

## Authors & Contact

- **Author**: [Your Name]
- **Institution**: [Your College / University Name]
- **Degree**: B.Tech in Computer Science & Engineering (Artificial Intelligence & Machine Learning)
- **University**: Maulana Abul Kalam Azad University of Technology (MAKAUT), West Bengal
- **LinkedIn**: [Your LinkedIn Profile URL]
- **GitHub**: [Your GitHub Profile URL]

*If you found this project helpful for classical machine learning research or academic coursework, please consider starring the repository.*
