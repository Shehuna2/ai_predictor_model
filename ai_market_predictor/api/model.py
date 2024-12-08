import joblib

def load_model():
    # Load the trained model
    return joblib.load("../models/market_predictor.pkl")

def predict(model, data):
    # Preprocess and predict
    features = [data["price"], data["volume"]]
    return model.predict([features])[0]
