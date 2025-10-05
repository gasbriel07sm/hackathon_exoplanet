# model.py
import joblib
import pandas as pd
import numpy as np

class ExoplanetPredictor:
    def __init__(self, artifacts_path='artifacts/'):
        """Carrega todos os artefatos do modelo (IA) do disco."""
        self.model = None
        try:
            # Garanta que está carregando o seu melhor modelo XGBoost
            self.model = joblib.load(f'{artifacts_path}exoplanet_xgboost_best_model.pkl')
            self.imputer = joblib.load(f'{artifacts_path}super_imputer.pkl')
            self.scaler = joblib.load(f'{artifacts_path}super_scaler.pkl')
            self.label_encoder = joblib.load(f'{artifacts_path}super_label_encoder.pkl')
            self.X_columns = joblib.load(f'{artifacts_path}X_columns.pkl')
            print("Modelo e transformadores carregados com sucesso!")
        except FileNotFoundError as e:
            print(f"Erro Crítico ao carregar artefatos: {e}")

    def predict(self, input_data):
        """Faz uma previsão em novos dados, garantindo o formato correto."""
        if not all([self.model, self.imputer, self.scaler, self.label_encoder, self.X_columns]):
            return {"error": "Modelo não carregado corretamente."}

        try:
            # Cria um DataFrame com todas as colunas esperadas, preenchido com NaN
            input_df = pd.DataFrame(columns=self.X_columns)
            # Concatena com os dados de entrada, alinhando as colunas
            input_df = pd.concat([input_df, pd.DataFrame(input_data)], ignore_index=True)
            # Garante que as colunas estejam na ordem correta e preenche o resto com NaN
            input_df = input_df[self.X_columns]

            # Aplica o pipeline de transformação
            data_imputed = self.imputer.transform(input_df)
            data_scaled = self.scaler.transform(data_imputed)

            # Faz a previsão e obtém as probabilidades
            prediction_encoded = self.model.predict(data_scaled)
            prediction_proba = self.model.predict_proba(data_scaled)
            prediction_label = self.label_encoder.inverse_transform(prediction_encoded)[0]
            
            confidence_scores = {self.label_encoder.classes_[i]: prob for i, prob in enumerate(prediction_proba[0])}

            return {"prediction": prediction_label, "confidence": confidence_scores, "error": None}
        except Exception as e:
            # Return a detailed error if prediction fails
            return {"prediction": None, "confidence": None, "error": f"Erro durante a previsão: {e}"}