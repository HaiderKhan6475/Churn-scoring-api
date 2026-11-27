# Customer Churn Prediction API

An end-to-end Machine Learning project that predicts customer churn using a Support Vector Machine (SVM) classifier. The model is deployed as a REST API using FastAPI and containerized with Docker.

## Project Overview

Customer churn prediction helps businesses identify customers who may stop using their services.

This project uses customer activity, spending, contract tenure, support tickets, and payment delays to predict whether a customer is likely to churn.

**Note:** This project uses a synthetic dataset created for educational and demonstration purposes. Predictions should not be treated as validated real-world business outcomes.

## Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- Support Vector Machine (SVM)
- GridSearchCV
- FastAPI
- Pydantic
- Joblib
- Docker

## Dataset

The project uses a synthetic dataset containing 2,000 customer records.

Features:

| Feature | Description |
|---|---|
| monthly_spend_usd | Monthly customer spending |
| support_tickets_30d | Support tickets in the last 30 days |
| weekly_active_hours | Weekly active hours |
| contract_tenure_months | Contract duration in months |
| payment_delays_count | Number of payment delays |

Target:
- `0` = Retained
- `1` = Churned

## Machine Learning Workflow

1. Dataset generation and exploration
2. Exploratory Data Analysis (EDA)
3. Train-test split
4. Feature scaling using StandardScaler
5. SVM model training
6. Hyperparameter tuning using GridSearchCV
7. Model evaluation
8. Model serialization using Joblib

## Model Performance

The tuned SVM model achieved:

- Test Accuracy: **63.25%**
- Cross-validation Accuracy: **62.69%**
- Kernel: Linear
- C: 0.1
- Gamma: scale

These results are based on synthetic data and do not establish real-world predictive performance.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Health check |
| POST | `/predict` | Predict customer churn |

## Run Locally

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd churn-scoring-api
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Open Swagger documentation:

```text
http://localhost:8000/docs
```

## Run with Docker

Build the Docker image:

```bash
docker build -t churn-scoring-api .
```

Run the container:

```bash
docker run -d -p 8001:8000 --name churn-container churn-scoring-api
```

Open API documentation:

```text
http://localhost:8001/docs
```

## Sample Prediction Request

```json
{
  "monthly_spend_usd": 1500.50,
  "support_tickets_30d": 8,
  "weekly_active_hours": 5.5,
  "contract_tenure_months": 6,
  "payment_delays_count": 4
}
```

## Sample Response

```json
{
  "status": "success",
  "customer_prediction": {
    "prediction": 1,
    "result": "Churned"
  },
  "message": "Customer is predicted to be churned."
}
```

## Project Structure

```text
churn-scoring-api/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── data/
│   └── customer_churn.csv
├── models/
│   └── churn_model.joblib
├── scripts/
│   ├── __init__.py
│   ├── eda.py
│   ├── train_model.py
│   └── grid_search.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Limitations

- Dataset is synthetic, not collected from real customers.
- Model accuracy is limited and requires further experimentation.
- Predictions are for demonstration and educational purposes only.
- Real-world deployment would require representative data, validation, monitoring, and additional testing.

## Author

Developed as an end-to-end Machine Learning and API deployment portfolio project.