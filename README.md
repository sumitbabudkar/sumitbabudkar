#ABC Motors, a successful Japanese automobile manufacturer, is evaluating entry into the Indian market.
This project uses data science and predictive modeling to estimate market potential and assess whether the company can achieve a target of 10,000 car sales per year in India.

# Business Objective

Analyze Japanese customer behavior to understand purchase drivers

Build an interpretable classification model to predict car purchases

Apply the model to Indian customer data to estimate potential buyers

Support a GO / NO-GO market entry decision using data

# Datasets Used

Japanese Dataset: Historical customer data with purchase labels

Indian Dataset: Customer data used for purchase probability prediction

Key variables include:

Age

Gender

Annual Income

Days since last vehicle maintenance

# Methodology

Data Cleaning & Feature Engineering

Standardized columns

Created maintenance segments based on days since last service

Exploratory Data Analysis (EDA)

Identified income and maintenance cycle as key purchase drivers

Model Building

Logistic Regression (chosen for interpretability)

Model Validation

Train–test split

Feature scaling

Threshold optimization (selected threshold = 0.4)

Market Estimation

Applied the validated model to Indian data

Estimated realistic conversion rates

# Key Results

ROC-AUC: ~0.75 (validated model)

High recall, ensuring minimal loss of potential buyers

Model predicts a sufficient pool of high-probability buyers in India

Annual sales target of 10,000 cars is achievable

# Tableau Dashboard

An interactive Tableau dashboard was built to:

Visualize customer behavior (Japan)

Display model predictions (India)

Support executive-level decision making

# Tools & Technologies


Python (Pandas, NumPy, Scikit-learn)

Tableau

Excel

Logistic Regression
