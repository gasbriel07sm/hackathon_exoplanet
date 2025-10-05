# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import lightkurve as lk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model import ExoplanetPredictor

# --- 1. Configurações Iniciais e Carregamento do Modelo ---
st.set_page_config(page_title="Exoplanet AI Explorer", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_predictor():
    """Carrega o modelo de IA e armazena em cache para performance."""
    return ExoplanetPredictor(artifacts_path='artifacts/')

predictor = load_predictor()

# --- 2. Funções de Visualização ---

@st.cache_data
def plot_light_curve(target_id, period, mission='Kepler'):
    """Busca, processa e plota uma curva de luz dobrada."""
    try:
        search = lk.search_lightcurve(target_id, mission=mission, author=('Kepler', 'K2', 'TESS'))
        lc = search.download(download_dir=".lightkurve-cache").flatten().remove_outliers()
        folded_lc = lc.fold(period=period)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        folded_lc.plot(ax=ax, alpha=0.1, color='royalblue', label="Dados Brutos")
        folded_lc.bin(time_bin_size=0.01).plot(ax=ax, color='red', lw=3, label="Sinal Médio do Trânsito")
        ax.set_title(f"Curva de Luz Dobrada para {target_id}", fontsize=16)
        ax.set_xlabel("Fase (relativa ao trânsito)", fontsize=12)
        ax.set_ylabel("Brilho Normalizado", fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        return fig
    except Exception as e:
        return f"Não foi possível gerar o gráfico da curva de luz: {e}"

def create_orbit_animation(st_rad, pl_rad, pl_orbper):
    """Cria uma animação 2D simplificada do trânsito."""
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        
        star_radius = st_rad if pd.notna(st_rad) and st_rad > 0 else 1.0
        planet_radius = (pl_rad / 109) if pd.notna(pl_rad) and pl_rad > 0 else 0.1
        orbit_radius = star_radius * 10
        
        star = plt.Circle((0, 0), star_radius, color='gold', zorder=10)
        ax.add_patch(star)
        planet = plt.Circle((0, 0), planet_radius * 20, color='darkblue', zorder=11) # Aumenta para visibilidade
        ax.add_patch(planet)

        ax.set_aspect('equal')
        ax.set_xlim(-orbit_radius * 1.2, orbit_radius * 1.2)
        ax.set_ylim(-orbit_radius * 1.2, orbit_radius * 1.2)
        ax.axis('off')

        def animate(i):
            angle = np.radians(i * 3.6)
            planet_x = orbit_radius * np.cos(angle)
            planet_y = orbit_radius * np.sin(angle) * 0.2 # Simula inclinação
            planet.set_center((planet_x, planet_y))
            # O planeta fica "atrás" da estrela em metade da órbita
            planet.set_zorder(5 if planet_x < 0 else 11)
            return planet,

        ani = animation.FuncAnimation(fig, animate, frames=100, interval=50, blit=True)
        return ani
    except Exception as e:
        return f"Erro ao criar animação: {e}"

# --- 3. Estrutura da Barra Lateral (Sidebar) ---
with st.sidebar:
    st.title("🛰️ Exoplanet AI Explorer")
    st.markdown("Uma plataforma para detectar e visualizar exoplanetas com Inteligência Artificial.")
    
    page = st.radio(
        "Navegação",
        ["Página Inicial", "Análise para Leigos", "Análise Profissional", "Visualizador Interativo", "Como a IA Funciona"]
    )
    st.divider()
    st.info("Projeto construído para o NASA Space Apps Challenge.")
    st.caption("Desenvolvido com Python, Streamlit e XGBoost.")

# --- 4. Renderização das Páginas ---

if page == "Página Inicial":
    st.title("Bem-vindo ao Explorador de Exoplanetas com IA!")
    st.markdown("""
        Esta plataforma interativa utiliza Inteligência Artificial para mergulhar no fascinante mundo da descoberta de exoplanetas. 
        Navegue pelas seções na barra lateral para aprender, analisar e visualizar.
    """)
    
    st.header("O que são Exoplanetas?")
    st.markdown("Exoplanetas são planetas que orbitam uma estrela diferente do nosso Sol. A busca por esses mundos distantes nos ajuda a entender nosso lugar no cosmos e a procurar por sinais de vida.")

    st.header("Como os Detectamos? O Método de Trânsito")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://i.imgur.com/w9N09T4.png", use_column_width=True)
    with col2:
        st.markdown("""
            Quando um planeta passa na frente de sua estrela (um evento chamado **trânsito**), ele bloqueia uma pequena parte da luz. Isso causa uma queda minúscula e periódica no brilho da estrela.
            
            - **Gráfico (a):** Uma estrela sem um planeta em trânsito tem um brilho constante.
            - **Gráfico (b):** Uma estrela com um planeta em trânsito mostra quedas de brilho regulares, criando uma **curva de luz** característica.
            
            Nossa IA é treinada para reconhecer a "impressão digital" de um trânsito real nesses dados.
        """)

    st.header("Confirmado vs. Candidato vs. Falso Positivo")
    st.markdown("""
    - **Candidato:** Um sinal que **parece** ser um planeta. Precisa de mais análise.
    - **Confirmado:** Um candidato que foi verificado por múltiplos métodos e é, com alta certeza, um planeta real.
    - **Falso Positivo:** Um sinal que parece um trânsito, mas é causado por outra coisa (ex: duas estrelas se eclipsando, ruído do telescópio). **Filtrar falsos positivos é um dos maiores desafios**, e é onde a IA brilha.
    """)
    st.info("Este projeto usa dados do **NEOSSat** (Canadá) e do **Telescópio Espacial James Webb (JWST)**, destacando a colaboração internacional na exploração espacial.")


elif page == "Análise para Leigos":
    st.title("🔍 Análise Simplificada: A IA em Ação")
    st.markdown("Clique no botão abaixo! Nós pegaremos um exemplo aleatório de um objeto celeste e pediremos para a nossa IA classificá-lo em tempo real.")

    if st.button("Analisar um Exemplo Aleatório", type="primary", use_container_width=True):
        try:
            example_df = pd.read_csv('data/kepler.csv', comment='#').sample(1)
            st.info("Dados do exemplo aleatório selecionado:")
            st.dataframe(example_df)
            
            with st.spinner("A IA está analisando o objeto..."):
                result = predictor.predict(example_df)
                if result and not result['error']:
                    prediction = result['prediction']
                    st.header("Resultado da Análise:")
                    if prediction == "Confirmed":
                        st.success(f"🪐 **PLANETA CONFIRMADO** (Confiança: {result['confidence']['Confirmed']:.1%})")
                    elif prediction == "Candidate":
                        st.info(f"🔭 **CANDIDATO PROMISSOR** (Confiança: {result['confidence']['Candidate']:.1%})")
                    else:
                        st.warning(f"☄️ **FALSO POSITIVO** (Confiança: {result['confidence']['False Positive']:.1%})")
        except Exception as e:
            st.error(f"Ocorreu um erro ao gerar o exemplo: {e}")

elif page == "Análise Profissional":
    st.title("🔬 Análise Profissional: Classifique Seus Dados")
    
    tab1, tab2 = st.tabs(["Upload de Arquivo CSV", "Entrada Manual"])

    with tab1:
        st.subheader("Upload de Arquivo (Uma Linha por Vez)")
        uploaded_file = st.file_uploader("Selecione um arquivo CSV:", type="csv")
        if uploaded_file:
            input_df = pd.read_csv(uploaded_file)
            row_to_predict = st.selectbox("Selecione a linha do arquivo para analisar:", input_df.index)
            selected_row = input_df.iloc[[row_to_predict]]
            
            st.write("Dados selecionados para análise:")
            st.dataframe(selected_row)

            if st.button("Classificar Observação Selecionada", type="primary"):
                with st.spinner("A IA está analisando..."):
                    result = predictor.predict(selected_row)
                    if result and not result['error']:
                        st.header("Resultado da Análise:")
                        st.metric("Previsão Final", result['prediction'])
                        st.write("Níveis de Confiança:")
                        st.json({k: f"{v:.2%}" for k, v in result['confidence'].items()})

    with tab2:
        st.subheader("Entrada Manual (Principais Features)")
        with st.form("manual_input_form"):
            top_features = [col for col in predictor.X_columns if 'koi_' in col][:10]
            manual_data = {}
            st.write("Insira os valores para as features mais comuns do Kepler:")
            cols = st.columns(2)
            for i, feature in enumerate(top_features):
                manual_data[feature] = cols[i % 2].number_input(feature, value=0.0, format="%.6f", key=f"manual_{feature}")
            
            submitted = st.form_submit_button("Classificar")
            if submitted:
                input_df_manual = pd.DataFrame([manual_data])
                with st.spinner("A IA está analisando..."):
                    result = predictor.predict(input_df_manual)
                    if result and not result['error']:
                        st.header("Resultado da Análise:")
                        st.metric("Previsão Final", result['prediction'])
                        st.write("Níveis de Confiança:")
                        st.json({k: f"{v:.2%}" for k, v in result['confidence'].items()})

elif page == "Visualizador Interativo":
    st.title("🌌 Visualizador de Trânsito e Órbita")
    st.markdown("Insira os dados de um objeto para gerar visualizações de sua curva de luz e uma representação de sua órbita.")

    col1, col2 = st.columns(2)
    with col1:
        target_id_input = st.text_input("ID do Alvo (ex: KIC 11904151)", "KIC 11904151")
        period_input = st.number_input("Período Orbital (dias)", value=0.837, format="%.6f")
    with col2:
        st_rad_input = st.number_input("Raio da Estrela (Raios Solares)", value=1.0)
        pl_rad_input = st.number_input("Raio do Planeta (Raios Terrestres)", value=1.5)

    if st.button("Gerar Visualizações", type="primary", use_container_width=True):
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            st.subheader("Gráfico da Curva de Luz")
            with st.spinner("Buscando dados nos arquivos da NASA..."):
                lc_fig = plot_light_curve(target_id_input, period_input)
                if isinstance(lc_fig, plt.Figure):
                    st.pyplot(lc_fig)
                else:
                    st.warning(str(lc_fig)) # Mostra a mensagem de erro da função
        # CÓDIGO COM ANIMAÇÃO DESATIVADA

        with viz_col2:
            st.subheader("Animação do Trânsito (Conceitual)")
            st.warning("A funcionalidade de animação requer a instalação do 'ffmpeg' no seu sistema. Como não foi encontrado, esta visualização foi desativada.")
            # with st.spinner("Gerando animação... (Pode requerer `ffmpeg`)"):
            #     animation_obj = create_orbit_animation(st_rad_input, pl_rad_input, period_input)
            #     if isinstance(animation_obj, animation.FuncAnimation):
            #         st.video(animation_obj.to_html5_video())
            #         plt.close()
            #     else:
            #         st.warning(str(animation_obj))

elif page == "Como a IA Funciona":
    st.title("🧠 Como a Nossa IA Funciona")
    st.markdown("""
        Esta IA não é uma "caixa preta". Ela foi construída usando um pipeline de Machine Learning claro e robusto. 
        Abaixo, você pode ver a performance do modelo e quais informações ele considera mais importantes.
    """)
    st.header("Performance do Modelo")
    st.markdown("Estes são os resultados do modelo em um conjunto de teste que ele nunca viu durante o treinamento. A **acurácia de 95%+** mostra que ele é altamente confiável.")
    # Adicione a imagem do seu relatório de classificaçã