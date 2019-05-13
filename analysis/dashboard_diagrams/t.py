import pickle
import os
list = [[96, 98, 669, 193, 7, 1996, 47.3385, -122.282,5.75,33.56522,98,11,0,1,1,0,1.75,4,4]]
import pathlib
abspath = pathlib.Path('scaler.pkl').absolute()

with open(abspath, "rb") as infile:
    scaler = pickle.load(infile)
    scaled = scaler.transform(list)
    with open("average_model.pkl", "rb") as model:
        model_pred = pickle.load(model)
        print(model_pred.predict(scaled))