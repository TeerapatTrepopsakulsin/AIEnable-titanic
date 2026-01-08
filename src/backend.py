import fastapi
import uvicorn
from pydantic import BaseModel, Field
import joblib
import pandas as pd


# model
model = joblib.load('../data/titanic_model.pkl')

# Request schema
class Passenger(BaseModel):
    Pclass: int = Field(ge=1, le=3)
    Sex: str
    Age: float | None = None
    SibSp: int
    Parch: int
    HasPrefix: int
    TicketNumber: int
    TicketIsLine: int
    TicketLength: int
    Fare: float | None = None
    Embarked: str | None = None

app = fastapi.FastAPI(title="Titanic Survival API", version="1.1")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(passenger: Passenger):
    cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
     'HasPrefix', 'TicketNumber', 'TicketIsLine', 'TicketLength']
    X = pd.DataFrame([[getattr(passenger, c) for c in cols]], columns=cols)
    prob = model.predict_proba(X)[0, 1]
    pred = "Survived" if prob >= 0.5 else "Not Survived"
    return {
        "prediction": pred,
        "prob_survive": round(float(prob), 4)
    }

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)