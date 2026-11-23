# Customer Churn Prediction using Machine Learning

> **Internship Project | Provided by UNP Education**
>
> This project was completed as part of a Data Science / Machine Learning internship. The goal was to apply the end-to-end ML workflow (data cleaning, exploratory analysis, model building, evaluation and comparison) to a real-world telecom problem: identifying customers who are likely to leave, so the business can act before they do.

Predicts which telecom customers are likely to churn (stop using the service) using 9 machine learning classifiers and an Artificial Neural Network (ANN), compared across five train–test splits.

## Dataset
- **Source:** `Customer Churn.xlsx`, 3,150 rows × 14 columns
- **After pre-processing:** 2,850 rows × 13 columns (12 input features + `Churn` target)
- **Features:** Call Failure, Complains, Subscription Length, Charge Amount, Seconds of Use, Frequency of Use, Frequency of SMS, Distinct Called Numbers, Tariff Plan, Status, Age, Customer Value

## Workflow
1. **Pre-processing:** removed 300 duplicates, dropped `Age Group`, recoded `Tariff Plan` and `Status` to binary
2. **EDA:** bar, histogram, box, line and pie plots, plus a correlation heatmap
3. **ML models:** Logistic Regression, KNN, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, SVC, Naive Bayes
4. **Pipeline (per split):** stratified split → StandardScaler → SMOTE on training data → fit → evaluate
5. **Splits:** 60:40, 65:35, 70:30, 75:25, 80:20
6. **ANN:** 14 configurations tested (Adam and SGD); best is 64→32→16→1 with Adam (lr 0.0005), 150 epochs, batch size 16

## Results
| Model | Split | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|---|
| Random Forest (best ML) | 65:35 | 96.29% | 0.8825 | 0.9752 |
| ANN (best overall) | 80:20 | 97.02% | 0.9081 | 0.9916 |

Tree ensembles (Random Forest, XGBoost) performed best among the ML models across all splits.

## Applications
Customer retention campaigns, high-risk customer identification, personalized offers, and proactive support.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, imbalanced-learn (SMOTE), XGBoost, TensorFlow/Keras, Matplotlib, Seaborn
