# app.py

import pandas as pd
from model import ExoplanetModel

# Função para carregar o modelo (usa cache para não recarregar a cada interação)
def load_model():
    predictor = ExoplanetModel(artifacts_path='artifacts/')
    return predictor

# Carrega o modelo
predictor = load_model()

def _predict(input_df):
    result = predictor.predict(input_df)
    if result["error"]:
         raise Exception("Error whilst predicting")
    return result

def predict_from_file(file):
    input_df = None
    try:
            input_df = pd.read_csv(file)
            
    except Exception as e:
        raise e
    return _predict(input_df)


def predict_random_example():
    # Carrega o dataset de teste para pegar um exemplo
    # (Em um app real, você teria um arquivo de exemplos separado)
    test_data_path = 'data/kepler.csv' # Usando Kepler como fonte de exemplos
    example_df = pd.read_csv(test_data_path, comment='#').head(100) # Pega as 100 primeiras linhas
    
    # Seleciona uma linha aleatória como nosso exemplo
    # Garante que tem as colunas certas, mesmo que não usemos todas
    sample = example_df.sample(1) 
    return _predict(sample)

print(predict_from_file("test.csv"))