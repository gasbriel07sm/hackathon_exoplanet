# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import ExoplanetModel
import os

# --- CONFIGURAÇÕES DA PÁGINA E ESTILO ---
st.set_page_config(page_title="Exoplanet Detector AI", layout="wide", initial_sidebar_state="expanded")
sns.set_theme(style="whitegrid", palette="viridis")

# --- CSS CUSTOMIZADO ---
st.markdown("""
<style>
/* Seletor de Idioma */
.language-selector {
    position: relative;
    display: inline-block;
    width: 100%;
    margin-bottom: 10px;
}
.language-button {
    background-color: #222;
    color: white;
    padding: 10px 15px;
    font-size: 16px;
    border: 1px solid #444;
    border-radius: 0.5rem;
    cursor: default;
    width: 100%;
    text-align: left;
}
.language-dropdown {
    display: none;
    position: absolute;
    background-color: #f1f1f1;
    min-width: 100%;
    box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
    z-index: 1;
    border-radius: 0.5rem;
}
.language-dropdown a {
    color: black;
    padding: 12px 16px;
    text-decoration: none;
    display: block;
    font-size: 16px;
}
.language-dropdown a:hover {background-color: #ddd;}
.language-selector:hover .language-dropdown {display: block;}

/* Cartões de Navegação Animados */
div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] div[data-testid="stButton"] button {
    width: 100%;
    height: 100%;
    padding: 2rem;
    border-radius: 0.5rem;
    transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
}
div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] div[data-testid="stButton"] button:hover {
    transform: scale(1.03);
    box-shadow: 0 8px 25px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# --- DICIONÁRIO DE TRADUÇÕES (i18n) ---
translations = {
    'en': {
        # ... UI Metadata
        "sidebar_title": "🛰️ Exoplanet Detector",
        "sidebar_subtitle": "An AI to classify objects of interest based on NASA data.",
        "lang_current": "🇬🇧 English",
        "lang_switch_to": "🇧🇷 Português",
        "lang_switch_code": "pt",
        "back_to_home": "⬅️ Back to Home",

        # ... Home Page
        "home_title": "Welcome to the Exoplanet Detector AI",
        "home_subtitle": "Choose a tool below to begin your exploration.",
        "home_card1_title": "🤖 AI Classifier",
        "home_card1_text": "Use our trained AI to classify new exoplanet candidates from a file or by manually entering data.",
        "home_card1_button": "Access Classifier",
        "home_card2_title": "📊 Data Analysis",
        "home_card2_text": "Visually explore the Kepler dataset with interactive charts and discover the patterns our AI learned from.",
        "home_card2_button": "Explore Data",
        "home_card3_title": "📖 Reference Guide",
        "home_card3_text": "Learn about exoplanets, how our AI works step-by-step, its performance, and details about our project.",
        "home_card3_button": "Read the Guide",
        
        # ... Classifier Texts
        "classifier_title": "Exoplanet Classification Panel",
        "classifier_tab_file": "Classify by File",
        "classifier_tab_manual": "Classify Manually",
        "file_uploader_label": "Choose a CSV file with candidate data:",
        "example_button": "Use a Random Example",
        "classify_file_button": "Classify Object from File",
        "manual_header": "Insert data manually (key features)",
        "classify_manual_button": "Classify with Manual Data",
        "form_koi_score": "KOI Score",
        "form_koi_period": "Orbital Period [days]",
        "form_koi_prad": "Planetary Radius [Earth radii]",
        "form_koi_duration": "Transit Duration [hours]",
        "form_koi_depth": "Transit Depth [ppm]",
        "form_koi_teq": "Equilibrium Temp. [K]",
        "spinner_text": "Analyzing data with the AI...",
        "analysis_finished": "✅ Analysis Complete!",
        "result_header": "Classification Result:",
        "confidence_header": "Confidence Level:",
        "class_candidate": "Candidate",
        "class_confirmed": "Confirmed",
        "class_false_positive": "False Positive",
        "warning_ia": "⚠️ **AI Warning:** {}",
        "error_prediction": "❌ Prediction Error: {}",
        "example_success": "Random example loaded!",
        "sample_data_header": "Sample of uploaded data:",
        "example_data_header": "Using a random example:",
        "file_read_error": "Error reading CSV file: {}",

        # ... Data Analysis Texts
        "analysis_title": "Exploratory Data Analysis",
        "analysis_subtitle": "Visualizing the Kepler dataset to uncover patterns and understand the foundation upon which our AI was trained.",
        "analysis_chart1_title": "Exoplanet Dispositions in the Dataset",
        "analysis_chart1_desc": """
        This bar chart shows the distribution of the three main classes in our dataset. We can observe a significant class imbalance: **False Positives** are the most common, while **Confirmed** planets are the rarest. This is why techniques like SMOTE were used during training to help the AI learn effectively.
        """,
        "analysis_chart2_title": "Orbital Period vs. Planetary Radius",
        "analysis_chart2_desc": """
        This scatter plot reveals the relationship between a planet's "year" and its size. The log scale helps visualize the wide range of values. We can see a dense cluster of planets with short orbital periods (less than 100 days), which are easier to detect because they transit their star more frequently.
        """,
        "analysis_chart3_title": "Distribution of Stellar Equilibrium Temperatures",
        "analysis_chart3_desc": """
        This histogram shows that the Kepler mission was particularly effective at observing Sun-like stars (peaking around 5500-6000 K), which is ideal for the search for potentially habitable exoplanets.
        """,
        "analysis_chart4_title": "What Does the AI Consider Most Important?",
        "analysis_chart4_desc": """
        This chart displays the **feature importances** from our AI. It shows which data columns the AI relies on most to make a decision. Unsurprisingly, `koi_score` and the various `koi_fpflag` (False Positive Flags) are highly important, confirming that the AI is focusing on scientifically relevant variables.
        """,
        "analysis_chart1_xlabel": "Disposition Class",
        "analysis_chart1_ylabel": "Number of Samples",
        "analysis_chart2_xlabel": "Orbital Period [log scale, days]",
        "analysis_chart2_ylabel": "Planetary Radius [log scale, Earth radii]",
        "analysis_chart3_xlabel": "Equilibrium Temperature [K]",
        "analysis_chart3_ylabel": "Frequency",
        "analysis_chart4_xlabel": "Importance Score",
        "analysis_chart4_ylabel": "Feature Name",

        # ... Reference Guide Texts
        "ref_title": "Reference Guide: Understanding Exoplanets and Our AI",
        "ref_about_title": "About This Project",
        "ref_about_text": """
        This application was developed in response to the **[A World Away: Hunting for Exoplanets with AI](https://www.spaceappschallenge.org/2025/challenges/a-world-away-hunting-for-exoplanets-with-ai/?tab=details)** challenge from the NASA International Space Apps Challenge 2025. Our goal was to create a tool that not only fulfills the challenge's objective but is also powerful for analysis and accessible for educational purposes.

        **Why use our AI?**
        Manually analyzing hundreds of thousands of light curves is impossible. Our AI automates and accelerates this process. It acts as an incredibly effective, intelligent filter that:
        - **Saves Time:** Classifies a new candidate in a fraction of a second, allowing scientists to efficiently sift through massive datasets.
        - **Increases Efficiency:** By accurately identifying False Positives, it prevents astronomers from wasting valuable telescope time on signals that are not real planets.
        - **Accelerates Discovery:** By highlighting the most promising candidates, it accelerates the pace of discovery in the search for new worlds.
        
        **Project Links:**
        - **[View the Source Code on GitHub](https://github.com/gasbriel07sm/hackathon_exoplanet)**
        """,
        "ref_pipeline_title": "How Does Our AI Work? The Pipeline Step-by-Step",
        "ref_pipeline_text": """
        Our AI uses a **Machine Learning Pipeline** to ensure every new piece of data is analyzed consistently.
        
        1.  **During Training:** We loaded and unified NASA data, split it, and preprocessed it by filling missing values (imputation), converting text to numbers (encoding), balancing rare classes (SMOTE), and normalizing feature scales (scaling). We then trained an XGBoost model.
        2.  **During Prediction:** When you classify data, the app applies the exact same saved preprocessing steps before feeding it to the trained model to get a result. This rigor ensures every prediction is reliable.
        """,
        "ref_performance_title": "Our AI's Performance",
        "ref_performance_text": """
        To ensure our model is reliable, it was rigorously evaluated on a separate test set. The model achieved:
        - **Overall Accuracy: 98.46%**
        This high accuracy means the model is extremely effective at correctly classifying signals, which is crucial for efficiently filtering through vast amounts of data to find genuine exoplanet signals.
        """,
        "ref_what_are_exoplanets_title": "What are Exoplanets?",
        "ref_what_are_exoplanets_text": """
        An exoplanet is any planet beyond our solar system. They come in a wide variety of sizes and orbits, from gas giants to rocky worlds like Earth.
        """,
        "ref_how_ai_helps_title": "What Our AI Does",
        "ref_how_ai_helps_text": """
        Our AI uses a powerful **XGBoost** model to recognize the subtle patterns in transit data, determine which features are most important, and classify candidates in a fraction of a second.
        """,
        "ref_dispositions_title": "Understanding the Classifications",
        "ref_dispositions_text": """
        - **Candidate:** A promising signal that looks like a planet and is worthy of follow-up observations.
        - **False Positive:** A signal that mimics a planet but is caused by something else (e.g., other stars, instrument noise).
        - **Confirmed:** A candidate that has been verified as a true exoplanet through further observations.
        """,
    },
    'pt': {
        # ... Metadados da UI
        "sidebar_title": "🛰️ Detector de Exoplanetas",
        "sidebar_subtitle": "Uma IA para classificar objetos de interesse com base em dados da NASA.",
        "lang_current": "🇧🇷 Português",
        "lang_switch_to": "🇬🇧 Inglês",
        "lang_switch_code": "en",
        "back_to_home": "⬅️ Voltar ao Início",

        # ... Página Principal
        "home_title": "Bem-vindo ao Detector de Exoplanetas com IA",
        "home_subtitle": "Escolha uma ferramenta abaixo para começar a sua exploração.",
        "home_card1_title": "🤖 Classificador de IA",
        "home_card1_text": "Use a nossa IA treinada para classificar novos candidatos a exoplanetas a partir de um arquivo ou inserindo dados manualmente.",
        "home_card1_button": "Aceder ao Classificador",
        "home_card2_title": "📊 Análise de Dados",
        "home_card2_text": "Explore visualmente o dataset Kepler com gráficos interativos e descubra os padrões que a nossa IA aprendeu.",
        "home_card2_button": "Explorar os Dados",
        "home_card3_title": "📖 Guia de Referência",
        "home_card3_text": "Aprenda sobre exoplanetas, como a nossa IA funciona passo a passo, a sua performance e detalhes sobre o nosso projeto.",
        "home_card3_button": "Ler o Guia",

        # ... Textos do Classificador
        "classifier_title": "Painel de Classificação de Exoplanetas",
        "classifier_tab_file": "Classificar por Arquivo",
        "classifier_tab_manual": "Classificar Manualmente",
        "file_uploader_label": "Escolha um arquivo CSV com os dados do candidato:",
        "example_button": "Usar um Exemplo Aleatório",
        "classify_file_button": "Classificar Objeto do Arquivo",
        "manual_header": "Inserir dados manualmente (principais características)",
        "classify_manual_button": "Classificar com Dados Manuais",
        "form_koi_score": "Score KOI",
        "form_koi_period": "Período Orbital [dias]",
        "form_koi_prad": "Raio Planetário [raios terrestres]",
        "form_koi_duration": "Duração do Trânsito [horas]",
        "form_koi_depth": "Profundidade do Trânsito [ppm]",
        "form_koi_teq": "Temp. de Equilíbrio [K]",
        "spinner_text": "Analisando os dados com a IA...",
        "analysis_finished": "✅ Análise Concluída!",
        "result_header": "Resultado da Classificação:",
        "confidence_header": "Nível de Confiança:",
        "class_candidate": "Candidato",
        "class_confirmed": "Confirmado",
        "class_false_positive": "Falso Positivo",
        "warning_ia": "⚠️ **Aviso da IA:** {}",
        "error_prediction": "❌ Erro na previsão: {}",
        "example_success": "Exemplo aleatório carregado!",
        "sample_data_header": "Amostra dos dados enviados:",
        "example_data_header": "Usando um exemplo aleatório:",
        "file_read_error": "Erro ao ler o arquivo CSV: {}",
        
        # ... Textos de Análise de Dados
        "analysis_title": "Análise Exploratória de Dados",
        "analysis_subtitle": "Visualizando o dataset Kepler para descobrir padrões e entender a base sobre a qual a nossa IA foi treinada.",
        "analysis_chart1_title": "Disposições de Exoplanetas no Dataset",
        "analysis_chart1_desc": """
        Este gráfico de barras mostra a distribuição das três classes principais no nosso dataset. Podemos observar um desequilíbrio de classes significativo: **Falsos Positivos** são a classe mais comum, enquanto planetas **Confirmados** são os mais raros. É por isso que técnicas como o SMOTE foram usadas durante o treino para ajudar a IA a aprender eficazmente.
        """,
        "analysis_chart2_title": "Período Orbital vs. Raio Planetário",
        "analysis_chart2_desc": """
        Este gráfico de dispersão revela a relação entre o "ano" de um planeta e o seu tamanho. A escala logarítmica ajuda a visualizar a vasta gama de valores. Vemos um denso aglomerado de planetas com períodos orbitais curtos (menos de 100 dias), que são mais fáceis de detetar porque transitam pela sua estrela com mais frequência.
        """,
        "analysis_chart3_title": "Distribuição da Temperatura de Equilíbrio das Estrelas",
        "analysis_chart3_desc": """
        Este histograma mostra que a missão Kepler foi particularmente eficaz na observação de estrelas do tipo solar (com pico por volta de 5500-6000 K), o que é ideal para a busca por exoplanetas potencialmente habitáveis.
        """,
        "analysis_chart4_title": "O Que a IA Considera Mais Importante?",
        "analysis_chart4_desc": """
        Este gráfico exibe a **importância das características** da nossa IA. Ele mostra em quais colunas de dados a IA mais se baseia para tomar uma decisão. Não surpreendentemente, o `koi_score` e as várias `koi_fpflag` (Bandeiras de Falso Positivo) são altamente importantes, o que confirma que a IA está a focar-se em variáveis cientificamente relevantes.
        """,
        "analysis_chart1_xlabel": "Classe de Disposição",
        "analysis_chart1_ylabel": "Número de Amostras",
        "analysis_chart2_xlabel": "Período Orbital [escala log, dias]",
        "analysis_chart2_ylabel": "Raio Planetário [escala log, raios terrestres]",
        "analysis_chart3_xlabel": "Temperatura de Equilíbrio [K]",
        "analysis_chart3_ylabel": "Frequência",
        "analysis_chart4_xlabel": "Pontuação de Importância",
        "analysis_chart4_ylabel": "Nome da Característica",

        # ... Textos do Guia de Referência
        "ref_title": "Guia de Referência: Entendendo os Exoplanetas e a Nossa IA",
        "ref_about_title": "Sobre Este Projeto",
        "ref_about_text": """
        Esta aplicação foi desenvolvida em resposta ao desafio **[A World Away: Hunting for Exoplanets with AI](https://www.spaceappschallenge.org/2025/challenges/a-world-away-hunting-for-exoplanets-with-ai/?tab=details)** do NASA International Space Apps Challenge 2025. O nosso objetivo foi criar uma ferramenta que não apenas cumprisse o objetivo do desafio, mas que também fosse poderosa para análise e acessível para fins educativos.

        **Por que usar a nossa IA?**
        Analisar manualmente centenas de milhares de curvas de luz é impossível. A nossa IA automatiza e acelera este processo. Ela atua como um filtro inteligente e incrivelmente eficaz que:
        - **Poupa Tempo:** Classifica um novo candidato numa fração de segundo, permitindo que os cientistas analisem eficientemente conjuntos de dados massivos.
        - **Aumenta a Eficiência:** Ao identificar com precisão os Falsos Positivos, evita que os astrónomos desperdicem tempo valioso de telescópio em sinais que não são planetas reais.
        - **Acelera a Descoberta:** Ao destacar os candidatos mais promissores, acelera o ritmo da descoberta na busca por novos mundos.
        
        **Links do Projeto:**
        - **[Ver o Código Fonte no GitHub](https://github.com/gasbriel07sm/hackathon_exoplanet)**
        """,
        "ref_pipeline_title": "Como Funciona a Nossa IA? O Pipeline Passo a Passo",
        "ref_pipeline_text": """
        A nossa IA utiliza um **Pipeline de Machine Learning** para garantir que cada novo dado seja analisado de forma consistente.
        
        1.  **Durante o Treino:** Carregamos e unificamos dados da NASA, os dividimos, e os pré-processamos preenchendo valores em falta (imputação), convertendo texto para números (codificação), balanceando classes raras (SMOTE) e normalizando as escalas das características (scaling). Em seguida, treinamos um modelo XGBoost.
        2.  **Durante a Previsão:** Quando você classifica um dado, a aplicação usa os mesmos componentes de pré-processamento salvos para tratar os seus dados antes de os entregar ao modelo treinado para obter um resultado. Este rigor garante que cada previsão seja confiável.
        """,
        "ref_performance_title": "Performance da Nossa IA",
        "ref_performance_text": """
        Para garantir que o nosso modelo é confiável, ele foi rigorosamente avaliado num conjunto de dados de teste que nunca tinha visto durante o treino. O modelo alcançou:
        - **Precisão Geral: 98.46%**
        Esta alta pontuação de precisão significa que o modelo é extremamente eficaz na classificação correta de sinais, o que é crucial para filtrar eficientemente grandes quantidades de dados para encontrar sinais genuínos de exoplanetas.
        """,
        "ref_what_are_exoplanets_title": "O que são Exoplanetas?",
        "ref_what_are_exoplanets_text": """
        Um exoplaneta é qualquer planeta para além do nosso sistema solar. Eles existem numa grande variedade de tamanhos e órbitas, desde gigantes gasosos a mundos rochosos como a Terra.
        """,
        "ref_how_ai_helps_title": "O Que a Nossa IA Faz",
        "ref_how_ai_helps_text": """
        Analisar manualmente centenas de milhares de curvas de luz é impossível. A nossa IA automatiza este processo usando um poderoso modelo **XGBoost** para reconhecer os padrões subtis nos dados de trânsito, determinar quais características são mais importantes e classificar candidatos numa fração de segundo.
        """,
        "ref_dispositions_title": "Entendendo as Classificações",
        "ref_dispositions_text": """
        - **Candidato:** Um sinal promissor que se parece com um planeta e merece mais observações.
        - **Falso Positivo:** Um sinal que imita um planeta, mas é causado por outra coisa (ex: outras estrelas, ruído do instrumento).
        - **Confirmado:** Um candidato que foi verificado como um verdadeiro exoplaneta através de observações adicionais.
        """,
    }
}
# Adicionar chaves em falta para evitar KeyErrors
for lang in translations:
    for key, value in translations['en'].items():
        if key not in translations[lang]:
            translations[lang][key] = value
    for key, value in translations['pt'].items():
        if key not in translations[lang]:
            translations[lang][key] = value


# --- FUNÇÕES AUXILIARES ---
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        predictor = ExoplanetModel(artifacts_path='artifacts/')
        return predictor if predictor.model else None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_analysis_data():
    """Carrega os dados do Kepler para análise e exemplos."""
    file_path = 'data/kepler.csv'
    try:
        return pd.read_csv(file_path, comment='#')
    except FileNotFoundError:
        st.error(f"File '{file_path}' not found.")
        return None

def display_classification_result(result, texts):
    # CORREÇÃO: Remover a exibição do aviso
    # if result.get('warning'): st.warning(texts['warning_ia'].format(result['warning']))
    if result.get('error'): st.error(texts['error_prediction'].format(result['error']))
    else:
        st.success(texts.get('analysis_finished', "Analysis Complete!"))
        st.subheader(texts.get('result_header', "Classification Result:"))
        prediction = result.get('prediction')
        confidence = result.get('confidence', {})
        class_map = {
            "Confirmed": f"## 🪐 <span style='color:green;'>{texts.get('class_confirmed', 'Confirmed')}</span>",
            "Candidate": f"## 🔭 <span style='color:blue;'>{texts.get('class_candidate', 'Candidate')}</span>",
            "False Positive": f"## ☄️ <span style='color:red;'>{texts.get('class_false_positive', 'False Positive')}</span>"
        }
        st.markdown(class_map.get(prediction, f"## {prediction}"), unsafe_allow_html=True)
        st.subheader(texts.get('confidence_header', "Confidence Level:"))
        c1, c2, c3 = st.columns(3)
        c1.metric(texts.get('class_candidate', 'Candidate'), f"{confidence.get('Candidate', 0):.1%}")
        c2.metric(texts.get('class_confirmed', 'Confirmed'), f"{confidence.get('Confirmed', 0):.1%}")
        c3.metric(texts.get('class_false_positive', 'False Positive'), f"{confidence.get('False Positive', 0):.1%}")

# --- PÁGINAS DA APLICAÇÃO ---
def render_home_page(texts):
    st.title(texts['home_title'])
    st.markdown(texts['home_subtitle'])
    st.markdown("---")

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.header(texts['home_card1_title'])
        st.write(texts['home_card1_text'])
        if st.button(texts['home_card1_button'], key='nav_classifier'):
            st.session_state.page = 'classifier'
            st.rerun()

    with col2:
        st.header(texts['home_card2_title'])
        st.write(texts['home_card2_text'])
        if st.button(texts['home_card2_button'], key='nav_analysis'):
            st.session_state.page = 'analysis'
            st.rerun()

    with col3:
        st.header(texts['home_card3_title'])
        st.write(texts['home_card3_text'])
        if st.button(texts['home_card3_button'], key='nav_reference'):
            st.session_state.page = 'reference'
            st.rerun()

def render_classifier_page(predictor, texts):
    if st.button(texts['back_to_home']):
        st.session_state.page = 'home'
        st.rerun()
    st.title(texts['classifier_title'])
    tab1, tab2 = st.tabs([texts['classifier_tab_file'], texts['classifier_tab_manual']])
    with tab1:
        uploaded_file = st.file_uploader(texts['file_uploader_label'], type="csv", key="file_uploader")
        if st.button(texts['example_button']):
            df = load_analysis_data()
            if df is not None:
                st.session_state.sample = df.sample(1)
                st.success(texts.get('example_success', 'Success!'))
        input_df = None
        if uploaded_file:
            st.session_state.sample = None
            try:
                input_df = pd.read_csv(uploaded_file)
                st.write(texts.get('sample_data_header', 'Data sample:'))
                st.dataframe(input_df.head())
            except Exception as e:
                st.error(texts.get('file_read_error', 'Error reading file').format(e))
        elif 'sample' in st.session_state and st.session_state.sample is not None:
            input_df = st.session_state.sample
            st.write(texts.get('example_data_header', 'Example data:'))
            st.dataframe(input_df)
        if input_df is not None and st.button(texts['classify_file_button']):
            with st.spinner(texts.get('spinner_text', 'Analyzing...')):
                display_classification_result(predictor.predict(input_df.iloc[[0]]), texts)
    with tab2:
        submitted = False
        with st.form(key="manual_input_form"):
            st.header(texts['manual_header'])
            c1, c2, c3 = st.columns(3)
            with c1:
                koi_score = st.number_input(texts['form_koi_score'], value=0.9, format="%.4f")
                koi_period = st.number_input(texts['form_koi_period'], value=5.0, format="%.4f")
            with c2:
                koi_prad = st.number_input(texts['form_koi_prad'], value=1.5, format="%.2f")
                koi_duration = st.number_input(texts['form_koi_duration'], value=3.0, format="%.4f")
            with c3:
                koi_depth = st.number_input(texts['form_koi_depth'], value=100.0, format="%.2f")
                koi_teq = st.number_input(texts['form_koi_teq'], value=300, format="%d")
            submitted = st.form_submit_button(texts['classify_manual_button'])

        if submitted:
            manual_data = pd.DataFrame([{'koi_score': koi_score, 'koi_period': koi_period, 'koi_prad': koi_prad, 'koi_duration': koi_duration, 'koi_depth': koi_depth, 'koi_teq': koi_teq}])
            with st.spinner(texts.get('spinner_text', 'Analyzing...')):
                display_classification_result(predictor.predict(manual_data), texts)

def render_analysis_page(predictor, texts):
    if st.button(texts['back_to_home']):
        st.session_state.page = 'home'
        st.rerun()
    st.title(texts['analysis_title'])
    st.markdown(texts['analysis_subtitle'])
    
    df = load_analysis_data()
    
    if df is not None:
        st.subheader(texts['analysis_chart1_title'])
        st.markdown(texts['analysis_chart1_desc'])
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x='koi_disposition', ax=ax1, order=df['koi_disposition'].value_counts().index)
        ax1.set_xlabel(texts.get('analysis_chart1_xlabel'))
        ax1.set_ylabel(texts.get('analysis_chart1_ylabel'))
        st.pyplot(fig1)
        
        st.subheader(texts['analysis_chart2_title'])
        st.markdown(texts['analysis_chart2_desc'])
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='koi_period', y='koi_prad', hue='koi_disposition', alpha=0.5, ax=ax2)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel(texts.get('analysis_chart2_xlabel'))
        ax2.set_ylabel(texts.get('analysis_chart2_ylabel'))
        st.pyplot(fig2)

        st.subheader(texts['analysis_chart3_title'])
        st.markdown(texts['analysis_chart3_desc'])
        fig3, ax3 = plt.subplots()
        sns.histplot(df['koi_steff'].dropna(), kde=True, ax=ax3, bins=50)
        ax3.set_xlabel(texts.get('analysis_chart3_xlabel'))
        ax3.set_ylabel(texts.get('analysis_chart3_ylabel'))
        st.pyplot(fig3)
        
        st.subheader(texts['analysis_chart4_title'])
        st.markdown(texts['analysis_chart4_desc'])
        try:
            importances = predictor.model.feature_importances_
            if hasattr(predictor.imputer, 'get_feature_names_out'):
                 feature_names = predictor.imputer.get_feature_names_out()
            else:
                 feature_names = predictor.columns

            if len(importances) == len(feature_names):
                importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                importance_df = importance_df.sort_values('importance', ascending=False).head(20)
                
                fig4, ax4 = plt.subplots(figsize=(10, 8))
                sns.barplot(x='importance', y='feature', data=importance_df, ax=ax4)
                ax4.set_xlabel(texts.get('analysis_chart4_xlabel'))
                ax4.set_ylabel(texts.get('analysis_chart4_ylabel'))
                st.pyplot(fig4)
            else:
                 st.warning("Could not display feature importance chart due to a length mismatch.")

        except Exception as e:
            st.warning(f"Could not display feature importance chart. Error: {e}")

    else:
        st.error("Kepler dataset not found. Cannot display analysis.")

def render_reference_page(texts):
    if st.button(texts['back_to_home']):
        st.session_state.page = 'home'
        st.rerun()
    st.title(texts.get('ref_title', "Reference"))
    
    st.header(texts.get('ref_about_title'))
    st.markdown(texts.get('ref_about_text'))

    st.header(texts.get('ref_pipeline_title'))
    st.markdown(texts.get('ref_pipeline_text'))
    
    st.header(texts.get('ref_performance_title'))
    st.markdown(texts.get('ref_performance_text'))

    st.header(texts.get('ref_what_are_exoplanets_title'))
    st.markdown(texts.get('ref_what_are_exoplanets_text'))

    st.header(texts.get('ref_how_ai_helps_title'))
    st.markdown(texts.get('ref_how_ai_helps_text'))
    
    st.header(texts.get('ref_dispositions_title'))
    st.markdown(texts.get('ref_dispositions_text'))
    
# --- LÓGICA PRINCIPAL DA APLICAÇÃO ---
if 'lang' not in st.session_state: 
    if 'lang' in st.query_params:
        st.session_state.lang = st.query_params.get('lang')
    else:
        st.session_state.lang = 'en'

if 'page' not in st.session_state:
    st.session_state.page = 'home'


if 'lang' in st.query_params:
    lang_code = st.query_params.get('lang')
    if lang_code in ['en', 'pt'] and lang_code != st.session_state.lang:
        st.session_state.lang = lang_code
        st.rerun()

texts = translations[st.session_state.lang]

st.sidebar.markdown(f"""
<div class="language-selector">
  <button class="language-button">{texts['lang_current']}</button>
  <div class="language-dropdown">
    <a href="?lang={texts['lang_switch_code']}" target="_self">{texts['lang_switch_to']}</a>
  </div>
</div>
""", unsafe_allow_html=True)

predictor = load_model()

st.sidebar.title(texts['sidebar_title'])
st.sidebar.write(texts['sidebar_subtitle'])


if predictor:
    if st.session_state.page == 'home':
        render_home_page(texts)
    elif st.session_state.page == 'classifier':
        render_classifier_page(predictor, texts)
    elif st.session_state.page == 'analysis':
        render_analysis_page(predictor, texts)
    elif st.session_state.page == 'reference':
        render_reference_page(texts)
else:
    st.error("CRITICAL: AI model could not be loaded. The application cannot continue.")