import unittest
import joblib
import numpy as np


class ModelTest(unittest.TestCase):
    def test_model_and_scaler_loading(self):
        model = joblib.load('./gender_model.pkl')
        scaler = joblib.load('./scaler.pkl')
        feature_names = joblib.load('./feature_names.pkl')

        self.assertIsNotNone(model, "Failed to load the logistic regression model.")
        self.assertIsNotNone(scaler, "Failed to load the scaler.")
        self.assertIsInstance(feature_names, list, "Failed to load feature names list.")

    def test_model_prediction(self):
        test_features = np.random.rand(1, len(joblib.load('./feature_names.pkl')))
        scaler = joblib.load('./scaler.pkl')
        model = joblib.load('./gender_model.pkl')

        scaled_features = scaler.transform(test_features)
        prediction = model.predict(scaled_features)
        self.assertIsNotNone(prediction, "Model failed to make a prediction.")


if __name__ == '__main__':
    unittest.main()
