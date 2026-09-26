# MOOC Review Sentiment Analysis

Distributed Sentiment Classification with PySpark

A sentiment analysis pipeline that classifies MOOC (Massive Open Online Course) reviews as positive or negative, built with Apache Spark for scalable data processing and machine learning.

The project trains and compares two classifiers on TF-IDF features extracted from real course reviews, then investigates *why* the models succeed or fail through detailed error analysis.

---

## Overview

Online course platforms collect large volumes of student reviews, but manually reading through them to gauge sentiment doesn't scale. This project builds an automated sentiment classifier that:

- Processes review data using **PySpark** for distributed, scalable computation
- Converts review text into numerical features using **TF-IDF**
- Trains and compares **Logistic Regression** and **Naive Bayes** classifiers
- Investigates model errors through mismatch analysis and feature importance
- Visualizes model behavior, confidence, and failure patterns

---

## Dataset

The dataset consists of MOOC course reviews, each paired with a 1–5 star rating.

**Labeling approach:** ratings are used only as labels, not as model features. Ratings 4–5 are mapped to **positive (1)**, and ratings 1–3 to **negative (0)**. This is a form of *weak supervision* — using an existing signal (the rating) as a proxy for the sentiment label, which is standard practice when explicit sentiment annotations aren't available.

> The raw dataset is not included in this repository. See [`data/README.md`](data/README.md) for the dataset source and instructions for obtaining it.

---

## Methodology

### Phase 1 — Foundation Model (Baseline)

- Load reviews into a Spark DataFrame
- Clean text: remove URLs, emails, punctuation, extra whitespace; lowercase everything
- Tokenize and extract features with **HashingTF + IDF** (TF-IDF)
- Create binary sentiment labels from the original 1–5 rating
- Split into train/test sets (80/20, stratified by label)
- Train a baseline **Logistic Regression** model

### Phase 2 — Model Comparison

- Train a **Naive Bayes** classifier on the same features
- Evaluate both models on Accuracy, AUC, and F1-score
- Compare results in a summary table to identify the stronger baseline

### Phase 3 — Deep Analysis & Insights

- **Sentiment vs. rating mismatch analysis** — find cases where the model's prediction disagrees with the original rating (e.g. a 5-star review predicted as negative), revealing where sentiment and rating diverge
- **Feature importance** — extract the words with the strongest positive/negative influence on predictions
- **Error analysis** — break down false positives vs. false negatives to understand *what kind* of reviews the model struggles with (e.g. short reviews, mixed sentiment, sarcasm)
- **Class imbalance handling** — reviews are heavily skewed toward positive; the pipeline addresses this using **class weighting** (rather than resampling) so Logistic Regression is trained with a weight column that penalizes misclassifying the minority (negative) class more heavily

---

## Key Findings

- The dataset is large (100k+ reviews) but **highly imbalanced**, skewed toward positive sentiment, and contains multilingual and variable-length text
- **Naive Bayes generalizes better than unweighted Logistic Regression** on this imbalanced data — unweighted Logistic Regression tends to predict the majority class almost exclusively
- Adding **class weights** to Logistic Regression significantly improves its ability to recognize the minority (negative) class
- The models struggle most with **mixed-sentiment reviews** (e.g. "Good but confusing"), where positive and negative signals coexist in the same text

---

## Why Spark?

| Task | Tool Used | Why |
|---|---|---|
| Load & clean large CSV | Spark | Distributed processing handles 100k+ rows efficiently |
| Text preprocessing & TF-IDF | Spark ML | Built-in, scalable feature pipeline |
| Model training (LR, NB) | Spark ML | Optimized, distributed training algorithms |
| Post-hoc analysis & visualization | Python (Pandas/Matplotlib/Seaborn) | Predictions are a small dataset by this point — pandas and matplotlib are simpler and more flexible for detailed analysis and plotting |

---

## Project Structure

```text
mooc-sentiment-analysis/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md
│
└── notebooks/
    └── mooc_sentiment_analysis.ipynb
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/mooc-sentiment-analysis.git
cd mooc-sentiment-analysis

python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

**Note:** PySpark requires a Java runtime (JDK 8, 11, or 17). If Java isn't installed:

```bash
# Ubuntu/Debian
sudo apt-get install openjdk-11-jdk-headless

# macOS (Homebrew)
brew install openjdk@11
```

---

## Running the Project

The pipeline is provided as a Jupyter/Colab notebook:

```text
notebooks/mooc_sentiment_analysis.ipynb
```

The notebook was originally built for **Google Colab**. It includes a file-upload cell for `reviews.csv`; when run outside Colab, place `reviews.csv` in the working directory (see [`data/README.md`](data/README.md)) and skip the upload cell.

---

## Limitations

- Labels are derived from star ratings rather than explicit sentiment annotations (weak supervision), so some noise is expected — a review can carry mixed sentiment that a single star rating doesn't fully capture
- The dataset is imbalanced toward positive reviews; while class weighting mitigates this, results should be interpreted with that skew in mind
- TF-IDF with `HashingTF` does not preserve an exact vocabulary mapping, which limits fully precise word-level interpretability
- The dataset includes multilingual reviews; the current cleaning pipeline is oriented toward English text
- No external validation set was used; all evaluation is on a held-out split from the same source data

---

## References

Dataset: MOOC course reviews (see [`data/README.md`](data/README.md) for source).
