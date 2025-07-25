import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class F1FeatureEngineer:
    """Handles all features for f1 predcitions"""

    def __init__(self):
        self.label_encoders = {}
    
    def create_features(self, results_df, races_df, drivers_df, constructors_df,
                        circuits_df, qualifying_df=None, weather_df=None):
        """create comprehensive features for race winner prediction"""
        print("creating features...")
        df = self._merge_data(results_df, races_df, drivers_df, constructors_df, circuits_df)
        if qualifying_df is not None:
            df = self._add_qualifying_data(df, qualifying_df)
        if weather_df is not None:
            df = self._add_weather_data(df, weather_df)
