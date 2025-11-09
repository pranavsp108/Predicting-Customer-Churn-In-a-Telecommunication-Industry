# Predicting Customer Churn in the Telecommunications Industry

This project is a comprehensive analysis of customer churn within a telecommunications company. The goal is to identify the key factors that lead to customers leaving the service and to build a machine learning model that can accurately predict churn. By understanding why customers leave, the business can proactively implement retention strategies to reduce churn, which is often far less expensive than acquiring new customers.

**Business Problem:** Identify customers who are at high risk of "churning" (leaving the company).
**Model Type:** Supervised Learning, Binary Classification.

## 2. The Dataset

The dataset used for this analysis contains customer-level information from a fictional telecommunications provider. Each row represents a unique customer, and the columns include:

* **Demographic Info:** gender, SeniorCitizen, Partner, Dependents
* **Account Information:** tenure (months), Contract, PaymentMethod, PaperlessBilling, MonthlyCharges, TotalCharges
* **Services Subscribed:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
* **Target Variable:** Churn (Yes/No) - This is the variable we aim to predict.

**Source:** Kaggle IBM Watson Telecom Churn Dataset

## 3. Project Workflow

My process for this project followed a standard data science methodology:

1.  **Data Cleaning & Preprocessing:**
    * Handled missing values (11 missing values in TotalCharges were filled with the mean).
    * Converted TotalCharges from an object/string to a numeric type.
    * Standardized column names and categorical feature values for consistency.

2.  **Exploratory Data Analysis (EDA):**
    * Analyzed the distribution of the target variable Churn and confirmed a class imbalance (approx. 73% 'No' vs. 27% 'Yes').
    * Visualized the relationship between churn and key features like Contract, tenure, InternetService, and MonthlyCharges.
    * Used count plots, histograms, and box plots to understand customer profiles.

3.  **Feature Engineering & Selection:**
    * Converted categorical variables (like Contract, Payment Method) into numerical format using One-Hot Encoding.
    * Scaled numerical features (like tenure, MonthlyCharges, TotalCharges) using StandardScaler to prepare them for modeling.

4.  **Model Building & Training:**
    * Split the data into training (80%) and testing (20%) sets.
    * Trained and compared several classification models, including Logistic Regression and Random Forest.
    * Performed hyperparameter tuning using GridSearchCV on the Random ForestClassifier to find the optimal settings for the best-performing model.

5.  **Model Evaluation:**
    * Evaluated models based on Accuracy, Precision, Recall, and F1-Score.
    * Given the imbalanced dataset, I focused heavily on the ROC-AUC (Receiver Operating Characteristic - Area Under Curve) score to measure the model's ability to distinguish between churning and non-churning customers.

## 4. Key Findings & Results

### EDA Insights:
* **Contract Type:** Customers on a Month-to-month contract are significantly more likely to churn than those on One-year or Two-year contracts.
* **Tenure:** New customers (low tenure) are far more likely to churn. The churn rate drops off dramatically as customer tenure increases.
* **Internet Service:** Customers with Fiber optic internet churn more often than those with DSL, possibly indicating issues with price, speed, or reliability for that service.

### Model Performance:
The final, tuned Random Forest Classifier was the best-performing model. It achieved the following results on the unseen test data:

* **Accuracy:** 81%
* **Precision (for 'Yes' class):** 0.66 (Correctly identifies 66% of predicted churners)
* **Recall (for 'Yes' class):** 0.54 (Successfully finds 54% of all actual churners)
* **ROC-AUC Score:** 0.85 (Excellent score for distinguishing between classes)

The model's feature importance analysis showed that the top 3 predictors of churn are:
1.  TotalCharges
2.  MonthlyCharges
3.  tenure

This suggests that a customer's financial commitment and history with the company are the most critical factors in their decision to stay or leave.

## 5. Technologies Used

* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn, Plotly
* **Machine Learning:** Scikit-learn (sklearn)
* **Environment:** Jupyter Notebook

## 6. How to Run This Project

To replicate this analysis, you can follow these steps:

1.  Clone this repository:
    ```bash
    git clone [https://github.com/pranavsp108/Predicting-Customer-Churn-In-a-Telecommunication-Industry.git](https://github.com/pranavsp108/Predicting-Customer-Churn-In-a-Telecommunication-Industry.git)
    ```
2.  Navigate to the project directory:
    ```bash
    cd Predicting-Customer-Churn-In-a-Telecommunication-Industry
    ```
3.  Install the required libraries (it's good practice to create and include a requirements.txt file):
    ```bash
    pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyter
    ```
4.  Launch the Jupyter Notebook:
    ```bash
    jupyter notebook Churn_Prediction_Project.ipynb
    ```
