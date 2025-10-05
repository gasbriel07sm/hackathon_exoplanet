# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import ExoplanetModel

# --- CONFIGURAÇÕES DA PÁGINA E ESTILO ---
st.set_page_config(page_title="Exoplanet Detector AI", layout="wide", initial_sidebar_state="expanded")
sns.set_theme(style="whitegrid", palette="viridis")

# --- CSS CUSTOMIZADO PARA O SELETOR DE IDIOMA COM HOVER ---
st.markdown("""
<style>
.language-selector {
    position: relative;
    display: inline-block;
    width: 100%;
    margin-bottom: 10px; /* Adiciona espaço abaixo */
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
</style>
""", unsafe_allow_html=True)

# --- DICIONÁRIO DE TRADUÇÕES (i18n) - VERSÃO EXPANDIDA ---
translations = {
    'en': {
        # ... Metadados da UI
        "sidebar_title": "🛰️ Exoplanet Detector",
        "sidebar_subtitle": "An AI to classify objects of interest based on NASA data.",
        "sidebar_select_mode": "Choose your tool:",
        "mode_classifier": "AI Classifier",
        "mode_analysis": "Data Analysis",
        "mode_reference": "Reference Guide",
        "lang_current": "🇬🇧 English",
        "lang_switch_to": "🇧🇷 Português",
        "lang_switch_code": "pt",

        # ... Textos do Classificador
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


        # ... Textos de Análise de Dados
        "analysis_title": "Exploratory Data Analysis",
        "analysis_subtitle": "Visualizing the Kepler dataset to uncover patterns and understand the foundation upon which our AI was trained.",
        "analysis_chart1_title": "Exoplanet Dispositions in the Dataset",
        "analysis_chart1_desc": """
        This bar chart shows the distribution of the three main classes in our dataset. We can observe a significant class imbalance:
        - **False Positives** are the most common class, which is expected. The vast majority of signals detected by telescopes are not actual planets.
        - **Candidates** represent a smaller, but still significant, portion of promising signals.
        - **Confirmed** planets are the rarest class, as they require extensive verification.
        This imbalance is a key reason why techniques like SMOTE were used during the AI's training to ensure it learns to identify the rare classes effectively.
        """,
        "analysis_chart2_title": "Orbital Period vs. Planetary Radius",
        "analysis_chart2_desc": """
        This scatter plot reveals the relationship between a planet's "year" (orbital period) and its size (radius). Both axes are on a logarithmic scale to better visualize the wide range of values.
        - We can see a dense cluster of planets with short orbital periods (less than 100 days), which are easier to detect because they transit their star more frequently.
        - There is no simple linear relationship, indicating that planets of all sizes can be found at various distances from their stars. The confirmed planets, represented by one of the colors in the legend, are scattered across the plot, highlighting the diversity of discovered worlds.
        """,
        "analysis_chart3_title": "Distribution of Stellar Equilibrium Temperatures",
        "analysis_chart3_desc": """
        This histogram displays the frequency of different stellar effective temperatures in the dataset. This temperature is a proxy for the type of star being observed.
        - The distribution peaks around 5500-6000 Kelvin, which is very similar to our Sun (~5,778 K). This indicates that the Kepler mission was particularly effective at observing Sun-like stars (G-type main-sequence stars).
        - This focus is partly by design, as Sun-like stars are of high interest in the search for potentially habitable exoplanets.
        """,
        "analysis_chart4_title": "What Does the AI Consider Most Important?",
        "analysis_chart4_desc": """
        This chart displays the **feature importances** as determined by our trained XGBoost model. It essentially shows which data columns the AI relies on most to make its classification decision.
        - Unsurprisingly, `koi_score` (a pre-computed score of how likely a signal is to be a planet) is the most influential feature.
        - The various `koi_fpflag` (False Positive Flags) are also highly important. These are flags assigned by the Kepler pipeline that indicate if a signal resembles known types of false positives.
        - By visualizing this, we gain trust in our model, as it confirms that the AI is focusing on the most scientifically relevant variables to distinguish real planets from impostors.
        """,
        "analysis_chart1_xlabel": "Disposition Class",
        "analysis_chart1_ylabel": "Number of Samples",
        "analysis_chart2_xlabel": "Orbital Period [log scale, days]",
        "analysis_chart2_ylabel": "Planetary Radius [log scale, Earth radii]",
        "analysis_chart3_xlabel": "Equilibrium Temperature [K]",
        "analysis_chart3_ylabel": "Frequency",
        "analysis_chart4_xlabel": "Importance Score",
        "analysis_chart4_ylabel": "Feature Name",

        # ... Textos do Guia de Referência
        "ref_title": "Reference Guide: Understanding Exoplanets and Our AI",
        "ref_pipeline_title": "How Does Our AI Work? The Pipeline Step-by-Step",
        "ref_pipeline_text": """
        To ensure every new piece of data is analyzed consistently and accurately, our AI uses a **Machine Learning Pipeline**. This is a sequence of automated steps that process data from its raw form to a final prediction. Here's how it works:
        
        **During Training (what we did once):**
        1.  **Data Ingestion & Unification:** We loaded data from multiple NASA missions (Kepler, K2, TESS) and merged them into a single, massive dataset.
        2.  **Preparation & Splitting:** We separated the data into features (the inputs, e.g., `koi_period`) and the target (the output we want to predict, `disposition`). This data was then split into a training set (to teach the AI) and a test set (to evaluate it).
        3.  **Preprocessing:** The training data went through a rigorous "treatment" process:
            - **Imputation:** Filled in any missing data points using the median value of each column.
            - **Label Encoding:** Converted text labels ('Candidate', 'Confirmed') into numbers (0, 1, 2) that the model can understand.
            - **SMOTE (Balancing):** Since 'Confirmed' planets are rare, this technique created intelligent synthetic examples of the rare classes to prevent the AI from becoming biased towards 'False Positives'.
            - **Scaling:** Normalized all numeric features to a common scale, ensuring no single feature could unfairly dominate the learning process.
        4.  **Model Training:** Our core model, an **XGBoost Classifier**, was trained on this fully preprocessed data.
        5.  **Artifact Serialization:** We saved not just the trained model, but *every single component* of the preprocessing pipeline (the imputer, the scaler, the label encoder, and the column list). These are the 'artifacts'.

        **During Prediction (what happens every time you click 'Classify'):**
        1.  **Load Artifacts:** The application loads all the saved components of the pipeline.
        2.  **Apply Identical Pipeline:** Your new data is passed through the exact same sequence of steps, using the already-fitted components:
            - The **fitted imputer** fills missing values based on the original data's medians.
            - The **fitted scaler** normalizes your data based on the original data's scale.
            - The **trained XGBoost model** receives the fully processed data and makes a prediction.
            - The **fitted label encoder** translates the model's numeric output back into a human-readable label ('Candidate', 'Confirmed', or 'False Positive').
            
        This strict, repeatable process ensures that every prediction is made with the same rigor and logic that was used to train and validate the AI.
        """,
        "ref_performance_title": "Our AI's Performance",
        "ref_performance_text": """
        To ensure our model is reliable, it was rigorously evaluated on a separate test set of data that it had never seen during training. Based on the final evaluation in our training notebook, the model achieved:
        - **Overall Accuracy: 98.46%**
        This high accuracy score means the model is extremely effective at correctly classifying signals into 'Confirmed', 'Candidate', and 'False Positive' categories. This level of performance is crucial for efficiently filtering through vast amounts of data to find genuine exoplanet signals.
        """,
        "ref_what_are_exoplanets_title": "What are Exoplanets?",
        "ref_what_are_exoplanets_text": """
        An exoplanet is any planet beyond our solar system. They come in a wide variety of sizes and orbits. Some are gigantic gas-giants hugging their parent star, others are icy, and some are rocky, like Earth. As of 2024, more than 5,000 exoplanets have been found. Key types include:
        - **Gas Giants:** Large planets composed mostly of helium and/or hydrogen, like Jupiter and Saturn.
        - **Super-Earths:** A class of planets with a mass higher than Earth's, but substantially below those of the Solar System's ice giants, Uranus and Neptune.
        - **Neptune-like:** Planets similar in size to Neptune or Uranus, with hydrogen/helium-dominated atmospheres.
        - **Terrestrial:** Earth-sized or smaller planets, composed of rock, silicate, water or carbon. The search for habitable worlds focuses on this category.
        """,
        "ref_how_ai_helps_title": "How Does Our AI Help?",
        "ref_how_ai_helps_text": """
        Manually analyzing hundreds of thousands of light curves is impossible. Our AI automates and accelerates this process. Specifically, it is a **XGBoost (Extreme Gradient Boosting) model**, a powerful machine learning algorithm.
        
        1.  **Pattern Recognition:** The XGBoost model is an ensemble of "decision trees". It learns by sequentially adding new trees that correct the errors of the previous ones. This allows it to learn highly complex and non-linear patterns in the transit data that a human eye would miss.
        2.  **Feature Importance:** It can determine which of the hundreds of features (like transit depth, stellar radius, etc.) are most predictive for classifying a signal. We can see in the 'Data Analysis' tab that features like `koi_score` are highly influential.
        3.  **Speed and Efficiency:** Once trained, the model can classify a new candidate in a fraction of a second. This allows scientists to efficiently sift through massive datasets from missions like TESS and prioritize the most promising signals for follow-up verification with other telescopes. It acts as an incredibly effective, intelligent filter.
        """,
        "ref_dispositions_title": "Understanding the Classifications",
        "ref_dispositions_text": """
        - **Candidate:** A signal that has passed initial automated tests and exhibits planet-like characteristics. It's a promising lead, but not yet a confirmed planet. The AI's job is to assess the strength of this candidacy.
        - **False Positive:** A signal that mimics a planetary transit but is caused by something else. Common causes include eclipsing binary star systems (two stars orbiting each other), starspots, or noise from the spacecraft's instruments. A robust AI is crucial for weeding these out with high accuracy.
        - **Confirmed:** A candidate that has been verified as a true exoplanet through rigorous follow-up observations, often using different detection methods (like the radial velocity method) to confirm its mass and planetary nature.
        """,
    },
    'pt': {
        # ... Metadados da UI
        "sidebar_title": "🛰️ Detector de Exoplanetas",
        "sidebar_subtitle": "Uma IA para classificar objetos de interesse com base em dados da NASA.",
        "sidebar_select_mode": "Escolha a sua ferramenta:",
        "mode_classifier": "Classificador IA",
        "mode_analysis": "Análise de Dados",
        "mode_reference": "Guia de Referência",
        "lang_current": "🇧🇷 Português",
        "lang_switch_to": "🇬🇧 English",
        "lang_switch_code": "en",

        # ... Textos do Classificador
        "classifier_title": "Painel de Classificação de Exoplanetas",
        "classifier_tab_file": "Classificar por Ficheiro",
        "classifier_tab_manual": "Classificar Manualmente",
        "file_uploader_label": "Escolha um ficheiro CSV com os dados do candidato:",
        "example_button": "Usar um Exemplo Aleatório",
        "classify_file_button": "Classificar Objeto do Ficheiro",
        "manual_header": "Inserir dados manualmente (principais características)",
        "classify_manual_button": "Classificar com Dados Manuais",
        "form_koi_score": "Score KOI",
        "form_koi_period": "Período Orbital [dias]",
        "form_koi_prad": "Raio Planetário [raios terrestres]",
        "form_koi_duration": "Duração do Trânsito [horas]",
        "form_koi_depth": "Profundidade do Trânsito [ppm]",
        "form_koi_teq": "Temp. de Equilíbrio [K]",
        "spinner_text": "A analisar os dados com a IA...",
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
        "example_data_header": "A usar um exemplo aleatório:",
        "file_read_error": "Erro ao ler o ficheiro CSV: {}",
        
        # ... Textos de Análise de Dados
        "analysis_title": "Análise Exploratória de Dados",
        "analysis_subtitle": "A visualizar o dataset Kepler para descobrir padrões e entender a base sobre a qual a nossa IA foi treinada.",
        "analysis_chart1_title": "Disposições de Exoplanetas no Dataset",
        "analysis_chart1_desc": """
        Este gráfico de barras mostra a distribuição das três classes principais no nosso dataset. Podemos observar um desequilíbrio de classes significativo:
        - **Falsos Positivos** são a classe mais comum, o que é esperado. A grande maioria dos sinais detetados pelos telescópios não são planetas reais.
        - **Candidatos** representam uma porção menor, mas ainda significativa, de sinais promissores.
        - **Planetas Confirmados** são a classe mais rara, pois exigem uma verificação extensiva.
        Este desequilíbrio é uma razão fundamental pela qual técnicas como o SMOTE foram usadas durante o treino da IA para garantir que ela aprende a identificar eficazmente as classes raras.
        """,
        "analysis_chart2_title": "Período Orbital vs. Raio Planetário",
        "analysis_chart2_desc": """
        Este gráfico de dispersão revela a relação entre o "ano" de um planeta (período orbital) e o seu tamanho (raio). Ambos os eixos estão em escala logarítmica para melhor visualizar a vasta gama de valores.
        - Podemos ver um denso aglomerado de planetas com períodos orbitais curtos (menos de 100 dias), que são mais fáceis de detetar porque transitam pela sua estrela com mais frequência.
        - Não existe uma relação linear simples, indicando que planetas de todos os tamanhos podem ser encontrados a várias distâncias das suas estrelas. Os planetas confirmados, representados por uma das cores na legenda, estão espalhados pelo gráfico, destacando a diversidade dos mundos descobertos.
        """,
        "analysis_chart3_title": "Distribuição das Temperaturas de Equilíbrio Estelar",
        "analysis_chart3_desc": """
        Este histograma exibe a frequência de diferentes temperaturas efetivas estelares no dataset. Esta temperatura é um indicador do tipo de estrela que está a ser observada.
        - A distribuição atinge o pico por volta de 5500-6000 Kelvin, o que é muito semelhante ao nosso Sol (~5,778 K). Isto indica que a missão Kepler foi particularmente eficaz na observação de estrelas do tipo solar (estrelas da sequência principal do tipo G).
        - Este foco é parcialmente intencional, uma vez que as estrelas do tipo solar são de grande interesse na busca por exoplanetas potencialmente habitáveis.
        """,
        "analysis_chart4_title": "O Que a IA Considera Mais Importante?",
        "analysis_chart4_desc": """
        Este gráfico exibe a **importância das características** (feature importances) conforme determinado pelo nosso modelo XGBoost treinado. Essencialmente, mostra em quais colunas de dados a IA mais se baseia para tomar a sua decisão de classificação.
        - Não surpreendentemente, o `koi_score` (uma pontuação pré-calculada da probabilidade de um sinal ser um planeta) é a característica mais influente.
        - As várias `koi_fpflag` (Bandeiras de Falso Positivo) também são altamente importantes. Estas são bandeiras atribuídas pelo pipeline do Kepler que indicam se um sinal se assemelha a tipos conhecidos de falsos positivos.
        - Ao visualizar isto, ganhamos confiança no nosso modelo, pois confirma que a IA está a focar-se nas variáveis cientificamente mais relevantes para distinguir planetas reais de impostores.
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
        "ref_title": "Guia de Referência: A Entender os Exoplanetas e a Nossa IA",
        "ref_pipeline_title": "Como Funciona a Nossa IA? O Pipeline Passo a Passo",
        "ref_pipeline_text": """
        Para garantir que cada novo dado seja analisado de forma consistente e precisa, a nossa IA utiliza um **Pipeline de Machine Learning**. Esta é uma sequência de passos automatizados que processam os dados desde a sua forma bruta até uma previsão final. Eis como funciona:
        
        **Durante o Treino (o que fizemos uma vez):**
        1.  **Ingestão e Unificação de Dados:** Carregámos dados de múltiplas missões da NASA (Kepler, K2, TESS) e fundimo-los num único e massivo conjunto de dados.
        2.  **Preparação e Divisão:** Separamos os dados em características (as entradas, ex: `koi_period`) e o alvo (a saída que queremos prever, `disposition`). Estes dados foram então divididos num conjunto de treino (para ensinar a IA) e num conjunto de teste (para a avaliar).
        3.  **Pré-processamento:** Os dados de treino passaram por um rigoroso processo de "tratamento":
            - **Imputação:** Preencheu quaisquer pontos de dados em falta usando o valor mediano de cada coluna.
            - **Codificação de Rótulos (Label Encoding):** Converteu rótulos de texto ('Candidato', 'Confirmado') em números (0, 1, 2) que o modelo consegue entender.
            - **SMOTE (Balanceamento):** Como os planetas 'Confirmados' são raros, esta técnica criou exemplos sintéticos inteligentes das classes raras para evitar que a IA ficasse viciada em 'Falsos Positivos'.
            - **Normalização (Scaling):** Normalizou todas as características numéricas para uma escala comum, garantindo que nenhuma característica pudesse dominar injustamente o processo de aprendizagem.
        4.  **Treino do Modelo:** O nosso modelo principal, um **Classificador XGBoost**, foi treinado com estes dados totalmente pré-processados.
        5.  **Serialização de Artefactos:** Guardámos não só o modelo treinado, mas *todos os componentes* do pipeline de pré-processamento (o imputer, o scaler, o codificador de rótulos e a lista de colunas). Estes são os 'artefactos'.

        **Durante a Previsão (o que acontece sempre que clica em 'Classificar'):**
        1.  **Carregar Artefactos:** A aplicação carrega todos os componentes guardados do pipeline.
        2.  **Aplicar Pipeline Idêntico:** Os seus novos dados passam pela mesma sequência exata de passos, usando os componentes já ajustados:
            - O **imputer ajustado** preenche os valores em falta com base nas medianas dos dados originais.
            - O **scaler ajustado** normaliza os seus dados com base na escala dos dados originais.
            - O **modelo XGBoost treinado** recebe os dados totalmente processados e faz uma previsão.
            - O **codificador de rótulos ajustado** traduz a saída numérica do modelo de volta para um rótulo legível por humanos ('Candidato', 'Confirmado' ou 'Falso Positivo').
            
        Este processo rigoroso e repetível garante que cada previsão é feita com o mesmo rigor e lógica que foram usados para treinar e validar a IA.
        """,
        "ref_performance_title": "Performance da Nossa IA",
        "ref_performance_text": """
        Para garantir que o nosso modelo é fiável, ele foi rigorosamente avaliado num conjunto de dados de teste separado que nunca tinha visto durante o treino. Com base na avaliação final no nosso notebook de treino, o modelo alcançou:
        - **Precisão Geral: 98.46%**
        Esta alta pontuação de precisão significa que o modelo é extremamente eficaz na classificação correta de sinais nas categorias 'Confirmado', 'Candidato' e 'Falso Positivo'. Este nível de performance é crucial para filtrar eficientemente grandes quantidades de dados para encontrar sinais genuínos de exoplanetas.
        """,
        "ref_what_are_exoplanets_title": "O que são Exoplanetas?",
        "ref_what_are_exoplanets_text": """
        Um exoplaneta é qualquer planeta para além do nosso sistema solar. Eles existem numa grande variedade de tamanhos e órbitas. Alguns são gigantes gasosos gigantescos muito próximos da sua estrela-mãe, outros são gelados, e alguns são rochosos, como a Terra. Em 2024, mais de 5,000 exoplanetas foram encontrados. Os tipos principais incluem:
        - **Gigantes Gasosos:** Grandes planetas compostos maioritariamente por hélio e/ou hidrogénio, como Júpiter e Saturno.
        - **Super-Terras:** Uma classe de planetas com uma massa superior à da Terra, mas substancialmente abaixo da dos gigantes de gelo do Sistema Solar, Urano e Neptuno.
        - **Tipo Neptuno:** Planetas de tamanho semelhante a Neptuno ou Urano, com atmosferas dominadas por hidrogénio/hélio.
        - **Terrestres:** Planetas do tamanho da Terra ou menores, compostos por rocha, silicato, água ou carbono. A busca por mundos habitáveis foca-se nesta categoria.
        """,
        "ref_how_ai_helps_title": "Como a Nossa IA Ajuda?",
        "ref_how_ai_helps_text": """
        Analisar manualmente centenas de milhares de curvas de luz é impossível. A nossa IA automatiza e acelera este processo. Especificamente, é um modelo **XGBoost (Extreme Gradient Boosting)**, um poderoso algoritmo de machine learning.
        
        1.  **Reconhecimento de Padrões:** O modelo XGBoost é um conjunto de "árvores de decisão". Ele aprende adicionando sequencialmente novas árvores que corrigem os erros das anteriores. Isto permite-lhe aprender padrões altamente complexos e não lineares nos dados de trânsito que um olho humano não detetaria.
        2.  **Importância das Características:** Ele pode determinar qual das centenas de características (como profundidade do trânsito, raio estelar, etc.) é mais preditiva para classificar um sinal. Podemos ver na aba 'Análise de Dados' que características como o `koi_score` são altamente influentes.
        3.  **Velocidade e Eficiência:** Uma vez treinado, o modelo pode classificar um novo candidato numa fração de segundo. Isto permite que os cientistas analisem eficientemente conjuntos de dados massivos de missões como a TESS e priorizem os sinais mais promissores para verificação de acompanhamento com outros telescópios. Atua como um filtro inteligente e incrivelmente eficaz.
        """,
        "ref_dispositions_title": "A Entender as Classificações",
        "ref_dispositions_text": """
        - **Candidato:** Um sinal que passou nos testes automatizados iniciais e exibe características semelhantes às de um planeta. É uma pista promissora, mas ainda não é um planeta confirmado. O trabalho da IA é avaliar a força desta candidatura.
        - **Falso Positivo:** Um sinal que imita um trânsito planetário, mas é causado por outra coisa. Causas comuns incluem sistemas de estrelas binárias eclipsantes (duas estrelas a orbitar-se mutuamente), manchas estelares ou ruído dos instrumentos da nave espacial. Uma IA robusta é crucial para eliminar estes com alta precisão.
        - **Confirmado:** Um candidato que foi verificado como um verdadeiro exoplaneta através de observações de acompanhamento rigorosas, muitas vezes usando métodos de deteção diferentes (como o método de velocidade radial) para confirmar a sua massa e natureza planetária.
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
    if result.get('warning'): st.warning(texts['warning_ia'].format(result['warning']))
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
def render_classifier_page(predictor, texts):
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
            # CORREÇÃO: Usar a fonte mais robusta para os nomes das características
            if hasattr(predictor.imputer, 'get_feature_names_out'):
                 feature_names = predictor.imputer.get_feature_names_out()
            else: # Fallback para versões mais antigas
                 feature_names = predictor.columns

            # Verificação de segurança para garantir que os comprimentos correspondem
            if len(importances) == len(feature_names):
                importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                importance_df = importance_df.sort_values('importance', ascending=False).head(20)
                
                fig4, ax4 = plt.subplots(figsize=(10, 8))
                sns.barplot(x='importance', y='feature', data=importance_df, ax=ax4)
                ax4.set_xlabel(texts.get('analysis_chart4_xlabel'))
                ax4.set_ylabel(texts.get('analysis_chart4_ylabel'))
                st.pyplot(fig4)
            else:
                 st.warning("Could not display feature importance chart due to a length mismatch between features and importances.")

        except Exception as e:
            st.warning(f"Could not display feature importance chart. Error: {e}")

    else:
        st.error("Kepler dataset not found. Cannot display analysis.")

def render_reference_page(texts):
    st.title(texts.get('ref_title', "Reference"))

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
        st.session_state.lang = st.query_params['lang']
    else:
        st.session_state.lang = 'en'


# CORREÇÃO: Lógica de Botão para Troca de Idioma
def set_language(lang_code):
    st.session_state.lang = lang_code

st.sidebar.markdown(f"""
<div class="language-selector">
  <button class="language-button">{translations[st.session_state.lang]['lang_current']}</button>
  <div class="language-dropdown">
     <!-- Esta parte é agora apenas visual, a lógica está nos botões abaixo -->
  </div>
</div>
""", unsafe_allow_html=True)

if st.sidebar.button(translations[st.session_state.lang]['lang_switch_to']):
    set_language(translations[st.session_state.lang]['lang_switch_code'])
    st.rerun()


texts = translations[st.session_state.lang]

predictor = load_model()

st.sidebar.title(texts['sidebar_title'])
st.sidebar.write(texts['sidebar_subtitle'])
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox(
    texts['sidebar_select_mode'],
    [texts['mode_classifier'], texts['mode_analysis'], texts['mode_reference']]
)

if predictor:
    if app_mode == texts['mode_classifier']: render_classifier_page(predictor, texts)
    elif app_mode == texts['mode_analysis']: render_analysis_page(predictor, texts) # Passar o predictor
    elif app_mode == texts['mode_reference']: render_reference_page(texts)
else:
    st.error("CRITICAL: AI model could not be loaded. The application cannot continue.")