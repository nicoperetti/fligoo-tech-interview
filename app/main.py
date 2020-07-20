import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# define model for post request.
class ModelParams(BaseModel):
    pid: str
    default: int


# load model
model = pickle.load(open("models/model.pkl", 'rb'))


def get_prediction(pid, default):
    """Get predictions

    :param pid: product_id
    :type pid: str
    :param default: default
    :type default: int
    :return: predictions
    :rtype: dict
    """
    x = [[pid, default]]
    dfp = pd.DataFrame(x, columns=["product_id", "default"])
    y = model.predict(dfp)[0]
    return {'churn': int(y), "score": float(y)}


# API instance
app = FastAPI()


@app.get("/")
def health_check():
    """Helth Check

    :return: helth check status
    :rtype: dict
    """
    return {"Check": "Ok!"}


@app.get("/predict")
def predict(params: ModelParams):
    """Predict endpoint

    :param params: model features
    :type params: ModelParams
    :return: predictions
    :rtype: dict
    """
    pid, default = params.pid, params.default

    pred = get_prediction(pid, default)

    return {"preds": pred, "feats": [pid, default]}
