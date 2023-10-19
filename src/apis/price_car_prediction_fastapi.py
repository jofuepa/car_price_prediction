# FastAPI service to infer the price of a car
# activate Pyenv: $ source venv/bin/activate
# run locally: $ uvicorn price_car_prediction_fastapi:app --reload
# test terminal: $ 
# test browser: http://localhost:8000/docs 
# kill TCP connections on port 8080: sudo lsof -t -i tcp:8000 | xargs kill -9
#request body example:
"""
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
"city_mpg": 18,
"driven_wheels": "all_wheel_drive",
"engine_cylinders": 6.0,
"engine_fuel_type": "regular_unleaded",
"engine_hp": 268.0,
"highway_mpg": 25,
"make": "toyota",
"market_category": "crossover,performance",
"model": "venza",
"number_of_doors": 4.0,
"popularity": 2031,
"transmission_type": "automatic",
"vehicle_size": "midsize",
"vehicle_style": "wagon",
"year": 2013
}'

{
"city_mpg": 18,
"driven_wheels": "all_wheel_drive",
"engine_cylinders": 6.0,
"engine_fuel_type": "regular_unleaded",
"engine_hp": 268.0,
"highway_mpg": 25,
"make": "toyota",
"market_category": "crossover,performance",
"model": "venza",
"number_of_doors": 4.0,
"popularity": 2031,
"transmission_type": "automatic",
"vehicle_size": "midsize",
"vehicle_style": "wagon",
"year": 2013
}

"""
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle


class Car(BaseModel):
    city_mpg: int
    driven_wheels: str
    engine_cylinders: float
    engine_fuel_type: str
    engine_hp: float
    highway_mpg: int
    make: str
    market_category: str
    model: str
    number_of_doors: float
    popularity: int
    transmission_type: str
    vehicle_size: str
    vehicle_style: str
    year: int

app = FastAPI(title="Price Car Prediction API", description="API for price car prediction ml model")
with open('../../models/car-model.bin', 'rb') as f_in:
    dv, model, = pickle.load(f_in)

#define a POST endpoint at the /predict path
@app.post("/predict")
async def prediction(car:Car):
    data = car.dict()
    X = dv.transform(data)
    y_pred = model.predict(X)
    #return a JSON respons
    return {"Price car prediction = $":"{:0.3f}".format(np.expm1(y_pred[0]))}
   