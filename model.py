# model.py

import joblib
import os
import pandas as pd
import numpy as np

class ExoplanetModel:
    """
    Uma classe para carregar os artefactos de IA e prever a classificação 
    de um candidato a exoplaneta.
    """
    def __init__(self, artifacts_path='artifacts/'):
        """
        Carrega o modelo e todos os transformadores necessários do disco.

        Args:
            artifacts_path (str): O caminho para a pasta que contém os artefactos salvos.
        """
        self.artifacts_path = artifacts_path
        self.model = None
        self.imputer = None
        self.scaler = None
        self.label_encoder = None
        self.columns = None
        self._load_artifacts()

    def _load_artifacts(self):
        """
        Função auxiliar para carregar todos os componentes do modelo.
        """
        try:
            # Adaptado para carregar o modelo XGBoost final do seu notebook
            self.model = joblib.load(os.path.join(self.artifacts_path, 'exoplanet_xgboost_best_model.pkl'))
            self.imputer = joblib.load(os.path.join(self.artifacts_path, 'super_imputer.pkl'))
            self.scaler = joblib.load(os.path.join(self.artifacts_path, 'super_scaler.pkl'))
            self.label_encoder = joblib.load(os.path.join(self.artifacts_path, 'super_label_encoder.pkl'))
            self.columns = joblib.load(os.path.join(self.artifacts_path, 'X_columns.pkl'))
            print("Artefactos do modelo carregados com sucesso.")
        except FileNotFoundError as e:
            print(f"Erro: Não foi possível encontrar um artefacto do modelo - {e}. "
                  f"Verifique se a pasta '{self.artifacts_path}' está correta e contém todos os ficheiros .pkl.")
            self.model = None # Garante que o modelo é None se algo falhar
        except Exception as e:
            print(f"Ocorreu um erro inesperado ao carregar os artefactos: {e}")
            self.model = None

    def predict(self, input_data):
        """
        Realiza uma previsão em novos dados, preenchendo colunas ausentes automaticamente.

        Args:
            input_data (pd.DataFrame): Um DataFrame com os dados a serem previstos.

        Returns:
            dict: Um dicionário com a previsão, confiança, erros e avisos.
        """
        if not self.model:
            return {'prediction': None, 'confidence': None, 'error': "Modelo não carregado.", 'warning': None}

        try:
            user_df = pd.DataFrame(input_data)
            template_df = pd.DataFrame(columns=self.columns, index=user_df.index)
            common_cols = [col for col in user_df.columns if col in self.columns]
            missing_cols = sorted(list(set(self.columns) - set(common_cols)))
            template_df[common_cols] = user_df[common_cols]

            warning_message = None
            if missing_cols:
                warning_message = (
                    f"{len(missing_cols)} colunas não foram encontradas no seu ficheiro "
                    f"e foram preenchidas com valores padrão pela IA. "
                    f"Isto pode afetar a precisão da previsão."
                )

            data_to_process = template_df
            data_imputed = self.imputer.transform(data_to_process)
            data_scaled = self.scaler.transform(data_imputed)
            prediction_encoded = self.model.predict(data_scaled)
            prediction_proba = self.model.predict_proba(data_scaled)
            prediction_label = self.label_encoder.inverse_transform(prediction_encoded)[0]
            
            confidence_scores = {self.label_encoder.inverse_transform([i])[0]: prob for i, prob in enumerate(prediction_proba[0])}

            return {
                'prediction': prediction_label,
                'confidence': confidence_scores,
                'error': None,
                'warning': warning_message
            }
        
        except Exception as e:
            return {
                'prediction': None, 
                'confidence': None, 
                'error': f"Ocorreu um erro inesperado durante a previsão: {e}",
                'warning': None
            }

