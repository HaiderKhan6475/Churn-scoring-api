
from pydantic import BaseModel, Field


class CustomerData(BaseModel):

    monthly_spend_usd: float = Field(
        ge=50,
        le=5000,
        description="Monthly customer spending in USD"
    )

    support_tickets_30d: int = Field(
        ge=0,
        le=15,
        description="Support tickets in the last 30 days"
    )

    weekly_active_hours: float = Field(
        ge=0.5,
        le=40,
        description="Weekly active hours"
    )

    contract_tenure_months: int = Field(
        ge=1,
        le=48,
        description="Customer contract tenure in months"
    )

    payment_delays_count: int = Field(
        ge=0,
        le=6,
        description="Number of payment delays"
    )
