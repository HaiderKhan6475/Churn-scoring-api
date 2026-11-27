
from fastapi import FastAPI,HTTPException
import joblib
from pathlib import Path
import pandas as pd
from app.schemas import CustomerData
# Model Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"

# Load Trained Model
model = joblib.load(MODEL_PATH)

print("Churn Prediction Model Loaded Successfully!")
# Create FastAPI Application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="An SVM-based API for predicting customer churn.",
    version="1.0.0"
)

# Home Route
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is Running!"
    }

# Health Check Route
@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }





@app.post("/predict")
def predict_churn(customer: CustomerData):

    try:
        customer_data = pd.DataFrame([{
            "monthly_spend_usd": customer.monthly_spend_usd,
            "support_tickets_30d": customer.support_tickets_30d,
            "weekly_active_hours": customer.weekly_active_hours,
            "contract_tenure_months": customer.contract_tenure_months,
            "payment_delays_count": customer.payment_delays_count
        }])

        prediction = model.predict(customer_data)[0]


        result = "Churned" if prediction == 1 else "Retained"
        logger.info("Customer churn prediction completed successfully.")

        return {
            "status": "success",
            "customer_prediction": {
                "prediction": int(prediction),
                "result": result
            },
            "message": f"Customer is predicted to be {result.lower()}."
        }

    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
