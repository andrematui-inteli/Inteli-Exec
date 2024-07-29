import numpy as np
import streamlit as st
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from category_encoders import WOEEncoder
from beta_calibration import BetaCalibration


class ModelTypes:
    LOG_REG = 'Regressão Logística'
    LGBM = 'LightGBM'
    ANN = 'Rede Neural'
    # KNN = 'kNN'
    XGB = 'XGBoost'
    

class AutoWOEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.woe_encoder = None
        self.categorical_features = None
    
    def fit(self, X, y=None):
        # Identify categorical columns
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.woe_encoder = WOEEncoder(cols=self.categorical_features)
        self.woe_encoder.fit(X, y)
        return self
    
    def transform(self, X):
        return self.woe_encoder.transform(X)
    

class BetaCalibratedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator or LogisticRegression()
        self.beta_calibrator = BetaCalibration()
        
    def fit(self, X, y):
        
        self.base_estimator.fit(X, y)
        probs = self.base_estimator.predict_proba(X)[:, 1] 
        self.beta_calibrator.fit(probs, y) # uses train set predictions to calibrate -- not ideal
        return self
    
    def predict(self, X):
        probs = self.base_estimator.predict_proba(X)[:, 1]
        calibrated_probs = self.beta_calibrator.predict(probs)
        return calibrated_probs >= 0.5
    
    def predict_proba(self, X):
        probs = self.base_estimator.predict_proba(X)[:, 1]
        calibrated_probs = self.beta_calibrator.predict(probs)
        return np.vstack((1-calibrated_probs, calibrated_probs)).T
    
