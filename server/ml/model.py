import os
import joblib

current_dir = os.path.dirname(__file__)

dt_model = os.path.join(current_dir, "model_v2.pkl")

model = joblib.load(dt_model)