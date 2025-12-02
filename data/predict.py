import joblib

def predict(data):
    knn = joblib.load("knn_model.sav")
    return knn.predict(data)