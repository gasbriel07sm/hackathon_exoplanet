# app.py

import streamlit as st
import pandas as pd
from model import ExoplanetModel
from json import load
# Configurações da página
lang = "us"
TEXTS_BR = load(open("text_BR.json",encoding="UTF-8"))
TEXTS_US = load(open("text_US.json",encoding="UTF-8"))
texts = TEXTS_BR if lang == "br" else TEXTS_US
st.set_page_config(page_title=texts["0"], layout="wide")

# Função para carregar o modelo (usa cache para não recarregar a cada interação)
@st.cache_resource
def load_model():
    predictor = ExoplanetModel(artifacts_path='artifacts/')
    return predictor

# Carrega o modelo
predictor = load_model()

# --- Interface do Usuário ---

st.title(texts["1"])
st.write(texts["2"])

# Área de Upload de Arquivo
uploaded_file = st.file_uploader(
    texts["3"],
    type="csv"
)

# Botão de Exemplo
if st.button(texts["4"]):
    # Carrega o dataset de teste para pegar um exemplo
    # (Em um app real, você teria um arquivo de exemplos separado)
    test_data_path = 'data/kepler.csv' # Usando Kepler como fonte de exemplos
    example_df = pd.read_csv(test_data_path, comment='#').head(100) # Pega as 100 primeiras linhas
    
    # Seleciona uma linha aleatória como nosso exemplo
    # Garante que tem as colunas certas, mesmo que não usemos todas
    sample = example_df.sample(1) 
    st.session_state.sample = sample # Salva o exemplo no estado da sessão
    uploaded_file = None # Limpa o uploader para usar o exemplo


if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.write(texts["5"])
        st.dataframe(input_df.head())
    except Exception as e:
        if lang == "br":
            st.error(f"Erro ao ler o arquivo CSV: {e}")
        else:
            st.error(f"Error when reading CSV file: {e}")
        input_df = None
else: # Usa o exemplo
    input_df = st.session_state.sample
    st.write(texts["6"])
    st.dataframe(input_df)

if 'sample' in st.session_state:
    if input_df is not None:
        # Botão para iniciar a classificação
        if st.button(texts["7"]):
            with st.spinner(texts["8"]):
                # Faz a previsão
                result = predictor.predict(input_df)
                
                if result['error']:
                    st.error(result['error'])
                else:
                    prediction = result['prediction']
                    confidence = result['confidence']
                    
                    st.success(texts["9"])
                    
                    # Exibe o resultado de forma destacada
                    st.subheader(texts["10"])
                    
                    if lang == "br":
                        if prediction == "Confirmed":
                            st.markdown(f"## 🪐 <span style='color:green;'>Confirmado</span>", unsafe_allow_html=True)
                        elif prediction == "Candidate":
                            st.markdown(f"## 🔭 <span style='color:blue;'>Candidato Promissor</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"## ☄️ <span style='color:red;'>Falso Positivo</span>", unsafe_allow_html=True)
                    else:
                        if prediction == "Confirmed":
                            st.markdown(f"## 🪐 <span style='color:green;'>Confirmed</span>", unsafe_allow_html=True)
                        elif prediction == "Candidate":
                            st.markdown(f"## 🔭 <span style='color:blue;'>Promissing Candidate</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"## ☄️ <span style='color:red;'>False Positive</span>", unsafe_allow_html=True)
                    # Mostra as probabilidades de confiança
                    st.subheader(texts["14"])
                    
                    # Cria colunas para as métricas
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(texts["11"], f"{confidence.get('Candidate', 0):.1%}")
                    with col2:
                        st.metric(texts["12"], f"{confidence.get('Confirmed', 0):.1%}")
                    with col3:
                        st.metric(texts["13"], f"{confidence.get('False Positive', 0):.1%}")

# Limpa o exemplo se um novo arquivo for carregado
if uploaded_file is not None and 'sample' in st.session_state:
    del st.session_state.sample