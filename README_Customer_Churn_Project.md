# Customer Churn Prediction & Retention Analysis

**Internship:** Veda Technology - AI & ML Internship  
**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib, Google Colab  
**Dataset:** IBM Telco Customer Churn sample dataset

## Project Overview
This project builds a supervised machine-learning workflow to estimate whether a telecom customer may leave the service (churn), explore customer patterns, and suggest retention actions.

## Objectives
- Clean and prepare customer data.
- Perform exploratory data analysis (EDA).
- Engineer and encode features.
- Train and compare at least three classification models.
- Evaluate using confusion matrix, classification report, and ROC-AUC.
- Identify influential features.
- Provide a churn prediction function and retention recommendations.

## Dataset
The notebook downloads the IBM Telco Customer Churn sample CSV from IBM's public GitHub repository:

`https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`

The commonly distributed file contains 7,043 customer records and 21 columns, including the `Churn` target. Internet access is required when running the notebook.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Google Colab / Jupyter Notebook

## Workflow
1. Load and inspect the IBM Telco Customer Churn dataset.
2. Check missing values and duplicates; convert `TotalCharges` to numeric.
3. Explore churn by target distribution, contract, tenure, monthly charges, and internet service.
4. Create `ServiceCount` as an engineered feature.
5. Separate predictors (`X`) and target (`y`).
6. Split into stratified training and test sets.
7. Impute and scale numeric data; impute and one-hot encode categorical data.
8. Train Logistic Regression, Decision Tree, and Random Forest models.
9. Compare Accuracy, Precision, Recall, F1-score, and ROC-AUC.
10. Plot ROC curves and a confusion matrix.
11. Calculate permutation feature importance.
12. Predict churn risk for a customer record and save the best pipeline.

## Models Compared
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

The notebook selects the model with the highest ROC-AUC on the held-out test set. Actual metric values are produced when the notebook runs; they are not hard-coded here.

## Evaluation Metrics
- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Among customers predicted to churn, the proportion who actually churned.
- **Recall:** Among actual churners, the proportion identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Ability to rank churners above non-churners across thresholds.
- **Confusion matrix:** Counts correct and incorrect predictions by class.

Because churn classes may be imbalanced, accuracy should not be used alone.

## Retention Recommendations
The notebook suggests reviewing high-risk customers, improving onboarding, considering suitable contract options, checking billing clarity, and offering relevant technical support. These are hypotheses to validate; model associations do not prove that an action causes customers to stay.

## How to Run in Google Colab
1. Download `Customer_Churn_Prediction_Retention_Analysis.ipynb`.
2. Open https://colab.research.google.com/.
3. Choose **File → Upload notebook** and select the downloaded notebook.
4. Connect to a runtime.
5. Run all cells from top to bottom.
6. Keep internet access enabled for dataset download.
7. The final section saves `customer_churn_model.joblib`.

If Colab shows a backend assignment/connection error, it is a runtime or network issue rather than a Python exception. Try again later or use local Jupyter.

## Project Structure
```text
Customer-Churn-Prediction-Retention-Analysis/
├── Customer_Churn_Prediction_Retention_Analysis.ipynb
└── README.md
```

## Learning Outcomes
- Practised data cleaning and EDA on tabular business data.
- Understood categorical encoding and numeric scaling.
- Built Scikit-learn preprocessing/model pipelines.
- Compared classification algorithms using multiple metrics.
- Interpreted confusion matrices, ROC-AUC, and permutation importance.
- Created a reusable churn prediction function.

## Limitations
- Educational sample data, not live company data.
- Random stratified split, not a time-based forecast.
- Model performance must be revalidated on new data.
- Retention recommendations are not causal claims.
- Validate privacy, fairness, thresholds, and business impact before real deployment.

## Interview Questions
**What is customer churn?** When a customer stops using a company's service.

**What is classification?** A supervised learning task that predicts a category.

**Why compare multiple models?** Different algorithms learn patterns differently; comparing metrics helps identify a suitable model.

**Why is recall important?** It measures how many actual churners the model identifies.

**What is ROC-AUC?** A metric measuring how well a model distinguishes the positive and negative classes across thresholds.

**What is feature importance?** An estimate of how much input variables contribute to model performance.
