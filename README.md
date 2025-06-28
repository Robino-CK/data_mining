# Opinion Spam & Bug Classification using Machine Learning

This repository contains two independent classification projects focused on real-world applications of machine learning techniques in text classification and software defect prediction. Both projects explore model evaluation, feature importance, statistical significance, and overfitting prevention techniques in supervised learning settings.

---

## 🔍 Projects Overview

### 1. **Detection of Opinion Spam**
This project aims to detect fake hotel reviews using various supervised learning algorithms. We compare generative and discriminative models (e.g., Naive Bayes vs. Logistic Regression), as well as linear and non-linear classifiers (e.g., Decision Trees, Random Forests). A Bag-of-Words (BOW) approach is used for text vectorization, with additional preprocessing techniques like TF-IDF and feature selection.

**Highlights:**
- Text preprocessing (lowercasing, punctuation removal, BOW, bigrams)
- Feature engineering (TF-IDF, feature selection)
- Classifiers: Naive Bayes, Logistic Regression, Decision Tree, Random Forest
- Evaluation metrics: Accuracy, Precision, Recall, F1 Score
- Statistical significance testing using McNemar’s Test
- Feature importance analysis (Odds ratios, coefficients, Gini reduction)

### 2. **Bug Prediction in Software Packages**
This project uses tree-based classifiers to predict post-release bugs in software packages based on structural code metrics. We train a single classification tree, a bagged tree ensemble, and a random forest, evaluating the impact of overfitting constraints and bagging.

**Highlights:**
- Structured numeric dataset (Eclipse releases 2.0 & 3.0)
- Tree growth control via `minleaf`, `nmin`, `nfeat` parameters
- Comparison of raw and ensemble methods (tree vs. bagging vs. random forest)
- Confusion matrices and model metrics (Accuracy, Precision, Recall)
- Model significance assessment with McNemar’s Test
- Insights into interpretability and tree structure

---

## 📊 Topics Covered

### 🧠 Machine Learning Models
- Classification Trees
- Bagging
- Random Forests
- Logistic Regression (LASSO)
- Multinomial Naive Bayes

### 📚 Text & Feature Processing
- Bag-of-Words (BOW)
- Bigrams
- TF-IDF
- Feature Selection (percentile-based)

### 📈 Model Evaluation
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrices
- Cross-validation (5-fold)
- Hyperparameter tuning (Grid Search)

### 📉 Statistical Testing
- McNemar Test for paired classifier comparison
- Chi-square significance analysis
- P-value interpretation

### 📌 Feature Analysis
- Feature importance (Gini, odds ratios, logistic coefficients)
- Tree structure visualization
- Overfitting prevention and generalization

### 👥 Authors

- Robin Kollmann (099435)
- Patrick Junghenn (1140761)
- Ellora Keemink (6529771)
