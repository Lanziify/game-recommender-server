import os
import joblib

current_dir = os.path.dirname(__file__)

dt_model = os.path.join(current_dir, "model.pkl")

model = joblib.load(dt_model)
