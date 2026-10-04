Fraud \& Anomaly Detection System



A machine learning-based transaction monitoring system that identifies potentially fraudulent transactions using anomaly detection with Isolation Forest.



Overview



This project demonstrates how machine learning can be applied to financial transaction monitoring.



The system generates a synthetic transaction dataset, analyzes transaction behavior, detects anomalous transactions using an Isolation Forest model, assigns a relative risk score, evaluates model performance, and exports detected transactions for further investigation.



«Note: The dataset used in this project is synthetic and created for demonstration and portfolio purposes. The results should not be interpreted as real-world fraud detection performance.»



Problem Statement



Financial systems process large numbers of transactions, making manual identification of suspicious activity difficult.



This project explores an automated approach to flag transactions that differ significantly from normal transaction patterns.



The system considers factors such as:



\- Transaction amount

\- Transaction time

\- Distance from the user's home

\- Transaction frequency



Objectives



\- Generate realistic synthetic transaction data

\- Apply anomaly detection using machine learning

\- Identify potentially fraudulent transactions

\- Generate a relative transaction risk score from 0–100

\- Evaluate model performance using standard classification metrics

\- Visualize transaction and fraud patterns

\- Export detected transactions for further analysis



Technologies Used



\- Python

\- Pandas — data manipulation and analysis

\- NumPy — numerical computation

\- Scikit-learn — machine learning

\- Matplotlib — data visualization

\- Seaborn — statistical visualization

\- Isolation Forest — anomaly detection



Dataset



The project uses a synthetic dataset containing:



\- 5,000 total transactions

\- 150 labelled fraudulent transactions

\- 4,850 normal transactions



Features



Feature| Description

Transaction\_ID| Unique transaction identifier

Amount| Transaction amount

Hour| Hour of the day when the transaction occurred

Distance\_From\_Home| Transaction distance from the user's home

Transaction\_Frequency| Number of transactions associated with the activity

Is\_Fraud| Synthetic ground-truth fraud label



The synthetic fraud cases represent several unusual behavioral patterns, including unusually high transaction amounts, unusual transaction locations, and high transaction frequency.



Methodology



1\. Data Generation



A synthetic dataset of 5,000 transactions is generated using Python, NumPy, and Pandas.



A controlled number of transactions are modified to represent different types of potentially fraudulent behavior.



2\. Feature Selection



The Isolation Forest model uses four transaction-level features:



Amount

Hour

Distance\_From\_Home

Transaction\_Frequency



3\. Train-Test Split



The dataset is divided into:



\- 80% training data — 4,000 transactions

\- 20% testing data — 1,000 transactions



A stratified split is used so that the fraud proportion is maintained across the training and testing sets.



4\. Anomaly Detection



The project uses Isolation Forest, an unsupervised anomaly detection algorithm.



Model configuration:



\- Number of trees: 200

\- Expected anomaly rate: 3%

\- Random state: 42



The model assigns each transaction an anomaly score. Transactions identified as anomalies are flagged as potentially fraudulent.



5\. Risk Scoring



The anomaly scores are converted into a relative 0–100 risk score.



Higher scores indicate that a transaction is more anomalous relative to the transactions in the dataset.



«The risk score is a relative anomaly ranking, not a probability of fraud.»



6\. Evaluation



The model is evaluated on the held-out test set using:



\- Accuracy

\- Precision

\- Recall

\- F1 Score

\- Confusion Matrix



Results



Metric| Result

Accuracy| 97.40%

Precision| 60.00%

Recall| 40.00%

F1 Score| 48.00%



Confusion Matrix



| Predicted Normal| Predicted Fraud

Actual Normal| 4789| 61

Actual Fraud| 71| 79



The model identified 79 of the 150 fraudulent transactions in the evaluated test set.



Because fraud detection is an imbalanced classification problem, accuracy alone is not sufficient for judging model quality. Precision, recall, and F1 score provide additional insight into detection performance.



Visualizations



Visualizations



The system generates visualizations to help analyze transaction behavior and model performance.



\### Normal vs Fraudulent Transactions



Shows the distribution of normal and fraudulent transactions in the synthetic dataset.



!\[Normal vs Fraudulent Transactions](fraud\_vs\_normal.png)



\### Transaction Amount Distribution



Compares transaction amount patterns between normal and fraudulent transactions.



!\[Transaction Amount Distribution](transaction\_amount\_distribution.png)



\### Fraudulent Transactions by Hour



Shows the distribution of fraudulent transactions across different hours of the day.



!\[Fraudulent Transactions by Hour](fraud\_by\_hour.png)



\### Confusion Matrix



Shows the model's correct and incorrect predictions on the evaluated dataset.



!\[Confusion Matrix](confusion\_matrix.png)



\### Risk Score Distribution



Shows the distribution of relative anomaly risk scores across transactions.



!\[Risk Score Distribution](risk\_score\_distribution.png)



Project Structure



Fraud-Anomaly-Detection-System/

│

├── fraud\_detection.py

├── generate\_data.py

│

├── fraud\_transactions.csv

├── detected\_fraud\_transactions.csv

│

├── confusion\_matrix.png

├── fraud\_vs\_normal.png

├── transaction\_amount\_distribution.png

├── fraud\_by\_hour.png

└── risk\_score\_distribution.png



How to Run



1\. Generate the dataset



python generate\_data.py



2\. Run the fraud detection system



python fraud\_detection.py



The system will train the Isolation Forest model, generate predictions and risk scores, display evaluation metrics, create visualizations, and export detected transactions.



Example System Output



FRAUD \& ANOMALY DETECTION SYSTEM

==================================================

Machine Learning-Based Transaction Monitoring

==================================================



Training transactions: 4000

Testing transactions: 1000



Isolation Forest Configuration:

Number of trees: 200

Expected anomaly rate: 3%



Accuracy:  97.40%

Precision: 60.00%

Recall:    40.00%

F1 Score:  48.00%



Limitations



This project is intended as a machine learning portfolio demonstration and has several limitations:



\- The dataset is synthetic.

\- Fraud patterns are generated using predefined rules.

\- The feature set is relatively small.

\- The model is an anomaly detection prototype rather than a production fraud prevention system.

\- The risk score is not a calibrated fraud probability.

\- Real-world fraud detection would require larger datasets, richer behavioral features, threshold optimization, and continuous model monitoring.



Future Improvements



Possible improvements include:



\- Feature engineering using transaction history

\- Time-series behavioral analysis

\- Threshold optimization

\- Comparison with supervised machine learning models

\- ROC-AUC and Precision-Recall analysis

\- Model explainability

\- Real-world fraud datasets

\- Automated model monitoring

\- Interactive dashboard using Streamlit

\- REST API for real-time transaction scoring



Learning Outcomes



Through this project, I practiced:



\- Data generation and preprocessing

\- Feature selection

\- Anomaly detection

\- Machine learning model training

\- Train-test evaluation

\- Classification metrics

\- Confusion matrix analysis

\- Risk scoring

\- Data visualization

\- CSV data processing

\- Building a complete Python-based ML workflow



Author



Vineet Thakran



BCA — Maharshi Dayanand University



This project was developed as part of my machine learning and data analytics portfolio.

