# app.py

import streamlit as st
import pandas as pd
from model import ExoplanetModel

# Configurações da página
st.set_page_config(page_title="Detector de Exoplanetas", layout="wide")

# Função para carregar o modelo (usa cache para não recarregar a cada interação)
@st.cache_resource
def load_model():
    predictor = ExoplanetModel(artifacts_path='artifacts/')
    return predictor

# Carrega o modelo
predictor = load_model()

# --- Interface do Usuário ---

st.title("🛰️ Detector de Exoplanetas com Inteligência Artificial")
st.write("""
    Esta aplicação utiliza um modelo de Machine Learning treinado com dados das missões Kepler, K2 e TESS 
    da NASA para classificar objetos de interesse. Faça o upload de um arquivo CSV com os dados de um 
    novo candidato para obter uma previsão.
""")

# Área de Upload de Arquivo
uploaded_file = st.file_uploader(
    "Escolha um arquivo CSV com os dados do candidato a exoplaneta:",
    type="csv"
)

# Botão de Exemplo
if st.button("Usar um Exemplo Aleatório"):
    # Carrega o dataset de teste para pegar um exemplo
    # (Em um app real, você teria um arquivo de exemplos separado)
    test_data_path = 'data/kepler.csv' # Usando Kepler como fonte de exemplos
    example_df = pd.read_csv(test_data_path, comment='#').head(100) # Pega as 100 primeiras linhas
    
    # Seleciona uma linha aleatória como nosso exemplo
    # Garante que tem as colunas certas, mesmo que não usemos todas
    sample = example_df.sample(1) 
    st.session_state.sample = sample # Salva o exemplo no estado da sessão
    uploaded_file = None # Limpa o uploader para usar o exemplo

if uploaded_file is not None or 'sample' in st.session_state:
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.write("Amostra dos dados enviados:")
            st.dataframe(input_df.head())
        except Exception as e:
            st.error(f"Erro ao ler o arquivo CSV: {e}")
            input_df = None
    else: # Usa o exemplo
        input_df = st.session_state.sample
        st.write("Usando um exemplo aleatório do dataset Kepler:")
        st.dataframe(input_df)

    if input_df is not None:
        # Botão para iniciar a classificação
        if st.button("Classificar Objeto"):
            with st.spinner("Analisando os dados..."):
                # Faz a previsão
                result = predictor.predict(input_df)
                
                if result['error']:
                    st.error(result['error'])
                else:
                    prediction = result['prediction']
                    confidence = result['confidence']
                    
                    st.success("Análise Concluída!")
                    
                    # Exibe o resultado de forma destacada
                    st.subheader("Resultado da Classificação:")
                    
                    if prediction == "Confirmed":
                        st.markdown(f"## 🪐 <span style='color:green;'>Confirmado</span>", unsafe_allow_html=True)
                    elif prediction == "Candidate":
                        st.markdown(f"## 🔭 <span style='color:blue;'>Candidato Promissor</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"## ☄️ <span style='color:red;'>Falso Positivo</span>", unsafe_allow_html=True)
                    
                    # Mostra as probabilidades de confiança
                    st.subheader("Nível de Confiança:")
                    
                    # Cria colunas para as métricas
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Candidato", f"{confidence.get('Candidate', 0):.1%}")
                    with col2:
                        st.metric("Confirmado", f"{confidence.get('Confirmed', 0):.1%}")
                    with col3:
                        st.metric("Falso Positivo", f"{confidence.get('False Positive', 0):.1%}")

# Limpa o exemplo se um novo arquivo for carregado
if uploaded_file is not None and 'sample' in st.session_state:
    del st.session_state.sample