import pickle
import numpy as np

def predict_sklearn_single(car, dv, model):
    X = dv.transform([car])
    y_pred = model.predict(X)
    return y_pred

with open('../../models/car-model.bin', 'rb') as f_in:
    dv, model, = pickle.load(f_in)


car_instance = {
    'city_mpg': 18,
    'driven_wheels': 'all_wheel_drive',
    'engine_cylinders': 6.0,
    'engine_fuel_type': 'regular_unleaded',
    'engine_hp': 268.0,
    'highway_mpg': 25,
    'make': 'toyota',
    'market_category': 'crossover,performance',
    'model': 'venza',
    'number_of_doors': 4.0,
    'popularity': 2031,
    'transmission_type': 'automatic',
    'vehicle_size': 'midsize',
    'vehicle_style': 'wagon',
    'year': 2013

}

prediction = predict_sklearn_single(car_instance, dv, model)

print("MSRP value is: {:0.3f}".format(np.expm1(prediction[0])))