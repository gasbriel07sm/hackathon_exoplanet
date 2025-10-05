# model.py

import joblib
import pandas as pd
import numpy as np

class ExoplanetModel:
    def __init__(self, artifacts_path='artifacts/'):
        """
        Carrega o modelo treinado, os transformadores e a lista de colunas.
        """
        print("Carregando artefatos do modelo...")
        try:
            # Carrega todos os 5 artefatos salvos
            self.model = joblib.load(f'{artifacts_path}exoplanet_super_model.pkl')
            self.imputer = joblib.load(f'{artifacts_path}super_imputer.pkl')
            self.scaler = joblib.load(f'{artifacts_path}super_scaler.pkl')
            self.label_encoder = joblib.load(f'{artifacts_path}super_label_encoder.pkl')
            self.X_columns = joblib.load(f'{artifacts_path}X_columns.pkl')
            
            print("Modelo e transformadores carregados com sucesso!")
        except FileNotFoundError as e:
            print(f"Erro Crítico: Não foi possível encontrar um dos arquivos .pkl em '{artifacts_path}'.")
            print(f"Detalhe do erro: {e}")
            self.model = None

    def predict(self, input_data):
        """
        Faz uma previsão em novos dados de entrada.

        Args:
            input_data (pd.DataFrame): Um DataFrame com os novos dados.

        Returns:
            dict: Um dicionário com a previsão e as probabilidades de confiança.
        """
        if self.model is None:
            return {"error": "Modelo não foi carregado. Verifique os logs."}

        try:
            # Garante que o input_data tenha exatamente as mesmas colunas na mesma ordem
            # que o modelo foi treinado.
            input_df = pd.DataFrame(input_data, columns=self.X_columns)

            # Aplica a MESMA sequência de transformações dos dados de teste
            data_imputed = self.imputer.transform(input_df)
            data_scaled = self.scaler.transform(data_imputed)

            # Faz a previsão numérica e obtém as probabilidades
            prediction_encoded = self.model.predict(data_scaled)
            prediction_proba = self.model.predict_proba(data_scaled)

            # Converte a previsão numérica de volta para o rótulo de texto
            prediction_label = self.label_encoder.inverse_transform(prediction_encoded)[0]
            
            # Cria um dicionário de confiança para todas as classes
            confidence_scores = {self.label_encoder.classes_[i]: prob for i, prob in enumerate(prediction_proba[0])}

            return {
                "prediction": prediction_label,
                "confidence": confidence_scores,
                "error": None
            }

        except Exception as e:
            return {
                "prediction": None,
                "confidence": None,
                "error": f"Erro durante a previsão: {e}"
            }