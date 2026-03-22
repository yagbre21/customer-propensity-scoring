# Customer Propensity Scoring Pipeline

**CSCI E-96: Data Mining for Business - Fall 2021**
Harvard University, Division of Continuing Education
Instructor: Ted Kwartler

**Author:** Yves Agbre

## Abstract

This project builds a customer propensity scoring system for a retail bank (National City Bank case study). Given a portfolio of existing customers with known marketing outcomes and a pool of prospective customers, the system predicts which prospects are most likely to accept a product offer. Four classification models are trained and compared on the same dataset, and the best-performing model scores prospective customers to produce a ranked target list.

## Problem

The bank wants to identify which prospective customers to target with a marketing campaign. Targeting everyone is expensive. Targeting randomly wastes budget. The goal is to build a model that ranks prospects by their likelihood of accepting an offer, so the bank can focus outreach on the highest-probability targets.

This is a classic **eligibility-scoring-ranking** pipeline - the same architecture used in credit pre-qualification, ad targeting, content recommendation, and messaging decisioning systems.

## Approach

### Data Integration

Four data sources joined on household ID to create a unified customer profile:

| Source | Features |
|--------|----------|
| Customer Marketing Results | Contact history, past outcomes, communication channel |
| Household Vehicle Data | Car model, car year |
| Household Credit Data | Car loan, insurance status |
| Household Axiom Demographics | Job, education, marital status, age, affluence |

### Feature Engineering

- **Vtreat** for automated categorical variable treatment, missing value handling, and high-cardinality feature encoding
- 10 input features selected based on domain relevance
- 80/20 train/test split (seed: 1234)

### Models Compared

| Model | Technique | Key Configuration |
|-------|-----------|-------------------|
| **Random Forest** | Ensemble of 50 decision trees | mtry = 1, bagging |
| **Decision Tree** | Single pruned tree (RPART) | Complexity parameter grid search (0.01-0.1) |
| **Logistic Regression** | GLM with backward stepwise selection | Automatic multicollinearity reduction |
| **KNN** | K-Nearest Neighbors | Center/scale preprocessing, 47 k-values searched |

Each model evaluated with:
- Confusion matrix (train and test)
- Accuracy comparison across train/test splits
- Variable importance analysis (Random Forest)
- Hyperparameter visualization (Decision Tree CP, KNN k-value)

## Results

The winning model was applied to score prospective customers, producing a ranked list of the top 100 targets with their acceptance probabilities and demographic profiles. Output includes visualizations segmenting the top prospects by job, education, marital status, and communication channel to inform campaign strategy.

## Project Structure

```
customer_propensity_scoring.R     # Full pipeline: data loading, 4 models, scoring, visualization
data/                             # Training and prospect datasets (not included)
output/                           # Scored prospect rankings
```

## Tech Stack

- R (caret, randomForest, rpart, glm, class)
- vtreat (categorical variable treatment)
- ggplot2 (visualization)
- MLmetrics / pROC (evaluation)

## Why This Matters

This pipeline is structurally identical to production ML decisioning systems:

| This Project | Production Equivalent |
|---|---|
| Customer accepts offer? | User clicks ad? Opens email? Watches content? |
| Multi-source data join | Feature store aggregation |
| 4-model comparison | Model selection / champion-challenger testing |
| Backward stepwise selection | Feature importance / dimensionality reduction |
| Top 100 prospect ranking | Eligibility-scoring-ranking pipeline |
| Demographic segmentation | Audience targeting / cohort analysis |

The architecture - join heterogeneous data, engineer features, compare models, score and rank - is the foundation of ad targeting (Amazon), messaging decisioning (HBO Max), credit pre-qualification (Macy's), and content recommendation (Netflix).

## Course Reference

- **CSCI E-96**: Data Mining for Business, Harvard Extension School
- **Instructor**: [Ted Kwartler](https://github.com/kwartler) (Harvard DataMining Business Student repo)

## Author

**Yves Agbre** - Harvard ALM, Management | [LinkedIn](https://www.linkedin.com/in/yagbre)
