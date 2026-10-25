# Bank Marketing: Predicting Term-Deposit Subscriptions

Which customers should a bank's call centre contact first? This project builds a leakage-free classification model on 41,000+ telephone-campaign records and translates it into a call-prioritisation strategy.

## Key results

| Metric | Value |
|---|---|
| Best model | Gradient Boosting (class-weighted) |
| ROC AUC (test set) | **0.81** |
| ROC AUC (5-fold cross-validation) | 0.80 ± 0.01 |
| Recall of subscribers | 65% |
| Subscribers reached by calling the top 20% of customers | **66% (3.3x lift over random calling)** |

**Main drivers of subscription:** the economic climate (number of employees, consumer price index, Euribor rate), the contact channel (mobile beats landline), the month of contact and the outcome of the previous campaign.

## What makes this analysis reliable

- **No data leakage:** call `duration` is only known *after* a call ends, so it is excluded. Including it would inflate ROC AUC to 0.95; the notebook shows this comparison explicitly.
- **Split before preprocessing:** scaling and encoding live inside a scikit-learn `Pipeline`, fitted on training data only.
- **Imbalance handled properly:** the classes are about 8 : 1, so models use class weighting and are judged on ROC AUC and recall, not accuracy.
- **Data quality checks:** duplicates removed, `"unknown"` categories kept as information, and the `pdays = 999` code converted into a "previously contacted" flag.

## Methods

Python · pandas · scikit-learn · SciPy · seaborn

Data quality checks → exploratory analysis → Welch t-tests and ANOVA F-tests → Logistic Regression, Decision Tree, Random Forest and Gradient Boosting → ROC curves, confusion matrix, cross-validation → cumulative gains (lift) chart → permutation feature importance. An optional SMOTE-ENN comparison runs if `imbalanced-learn` is installed.

## Repository contents

| File | Description |
|---|---|
| `Bank_Marketing_Subscription_Prediction.ipynb` | Full analysis with outputs and charts |
| `data/bank-additional-full.csv` | UCI Bank Marketing dataset |

## How to run

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn jupyter
jupyter notebook Bank_Marketing_Subscription_Prediction.ipynb
```

## Data source

Moro, S., Cortez, P. & Rita, P. (2014). *Bank Marketing* dataset, UCI Machine Learning Repository (CC BY 4.0).

## Author

**Teja Manjusree Koduru** · M.Sc. Big Data & Business Analytics, FOM Essen · [LinkedIn](https://www.linkedin.com/in/teja-manjusree-25856b138)
