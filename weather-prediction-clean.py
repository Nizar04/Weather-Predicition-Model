import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class WeatherPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self):
        df = pd.read_csv(self.data_path)
        return self.wrangle(df)
    
    def wrangle(self, df):
        df = df.copy()
        
        uncertainty_cols = [col for col in df.columns if 'Uncertainty' in col]
        df = df.drop(columns=uncertainty_cols)
        
        temp_cols = [col for col in df.columns if 'Temperature' in col]
        for col in temp_cols:
            df[col] = df[col].apply(lambda x: (x * 1.8) + 32 if pd.notnull(x) else x)
        
        df["dt"] = pd.to_datetime(df["dt"])
        df["Month"] = df["dt"].dt.month
        df["Year"] = df["dt"].dt.year
        
        df = df[df.Year >= 1850]
        df = df.set_index(["Year", "Month"])
        df = df.dropna()
        
        return df
    
    def visualize_correlations(self, df):
        plt.figure(figsize=(10, 8))
        corrMatrix = df.corr()
        sns.heatmap(corrMatrix, annot=True, cmap='coolwarm')
        plt.title('Temperature Correlations')
        plt.tight_layout()
        plt.show()
        
    def prepare_features(self, df):
        target = "LandAndOceanAverageTemperature"
        features = ["LandAverageTemperature", "LandMaxTemperature", "LandMinTemperature"]
        
        X = df[features]
        y = df[target]
        
        return train_test_split(X, y, test_size=0.25, random_state=42)
    
    def train_model(self, X_train, y_train):
        self.model = make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=100,
                max_depth=50,
                random_state=77,
                n_jobs=-1
            )
        )
        self.model.fit(X_train, y_train)
    
    def evaluate_model(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        errors = abs(predictions - y_test)
        mape = 100 * (errors / y_test)
        accuracy = 100 - np.mean(mape)
        
        return {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'Accuracy': accuracy
        }
    
    def plot_feature_importance(self, X):
        rf_model = self.model.named_steps['randomforestregressor']
        importance = rf_model.feature_importances_
        
        plt.figure(figsize=(10, 6))
        plt.bar(X.columns, importance)
        plt.title('Feature Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    def predict(self, features):
        if not self.model:
            raise ValueError("Train the model first")
        return self.model.predict(features)

def main():
    predictor = WeatherPredictor('GlobalTemperatures.csv')
    df = predictor.load_and_prepare_data()
    print("Data Shape:", df.shape)
    
    predictor.visualize_correlations(df)
    
    X_train, X_test, y_train, y_test = predictor.prepare_features(df)
    predictor.train_model(X_train, y_train)
    
    metrics = predictor.evaluate_model(X_test, y_test)
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    
    predictor.plot_feature_importance(X_train)

if __name__ == "__main__":
    main()
