# Este arquivo implementa a página de treinamento de modelos da aplicação
# Permite ao usuário carregar dados, selecionar features, treinar diferentes 
# tipos de modelos de machine learning e visualizar métricas de performance

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import os
import sys
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import messages as msn
import metrics as met
from models import ModelTypes, AutoWOEEncoder, BetaCalibratedClassifier
from config.config import set_page_config
set_page_config()

def get_features_and_target(data):
    """
    Função para extrair features e variável target dos dados carregados
    
    Args:
        data: Arquivo Excel carregado pelo usuário
        
    Returns:
        X: DataFrame com as features selecionadas
        y: Serie com a variável target
    """
    X, y = None, None
    if data is None:
        # Se nenhum dado for carregado, retorna None
        pass
    else:
        # Configura o layout em duas colunas
        col1, col2 = st.columns(2)
        #st.write("Carregando dados...")
        df = pd.read_excel(data)
        col1.write("Visualização da sua base de dados (30 linhas)")
        col1.dataframe(df.head(30))
        cols = df.columns.tolist()

        # Assume por padrão que todas as colunas exceto a última são features
        # e a última coluna é o target
        feats_list, target_list = cols[:-1], cols[-1]

        # Permite ao usuário confirmar ou modificar as features selecionadas
        features = col2.multiselect(label="Verifique que essas são as variáveis de treino",
                                    options=cols, default=feats_list)
        # Permite ao usuário confirmar ou modificar a variável target
        target = col2.multiselect(
            label="Verifique que esse é seu alvo de treino",
            options=cols,
            default=target_list,
            max_selections=1)

        # Verifica se as seleções são válidas:
        # - Features e target não se sobrepõem
        # - Exatamente um target foi selecionado
        exclusive = set(features).intersection(target) == set()
        has_target = len(target) == 1
        if exclusive & has_target:
            col2.write("✅ Sua escolha de covariáveis e alvo estão boas.\n\n Pode ir em frente com o treino.")
            X, y = df[features], df[target[0]]
        else:
            col2.write("❌ Garanta que você escolheu apenas um alvo e que ele não faça parte das variáveis de treino!")
    return X, y


def reset_model_state():
    """
    Função para limpar o estado do modelo quando uma nova seleção é feita
    Remove variáveis do session_state relacionadas ao modelo treinado
    """
    # Remove as variáveis de estado que armazenam informações do modelo atual
    for key in ['is_fit', 'model', 'features']:
        if key in st.session_state:
            del st.session_state[key]

def initialize_model_history():
    """
    Inicializa o histórico de modelos no session_state se não existir
    """
    # Cria uma lista vazia para armazenar o histórico de modelos se ainda não existir
    if 'model_history' not in st.session_state:
        st.session_state['model_history'] = []

def add_model_to_history(model_name, metrics_dict, features_used, model_wrapper):
    """
    Adiciona um modelo treinado ao histórico
    
    Args:
        model_name: Nome do modelo (ex: 'Regressão Logística')
        metrics_dict: Dicionário com as métricas do modelo
        features_used: Lista de features utilizadas no treino
        model_wrapper: O objeto model_wrapper completo para uso futuro
    """
    # Garante que o histórico esteja inicializado
    initialize_model_history()
    
    # Criando registro do modelo com timestamp atual e métricas
    model_record = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': model_name,
        'roc_auc_train': metrics_dict['roc_auc_train'],
        'roc_auc_test': metrics_dict['roc_auc_test'],
        'ks_train': metrics_dict['ks_train'],
        'ks_test': metrics_dict['ks_test'],
        'features_count': len(features_used),
        'features_used': list(features_used),
        'model_wrapper': model_wrapper  # Salvando o model_wrapper completo
    }
    
    # Adicionando ao histórico
    st.session_state['model_history'].append(model_record)

def clear_model_history():
    """
    Limpa todo o histórico de modelos
    """
    # Reseta o histórico para uma lista vazia
    st.session_state['model_history'] = []

def display_metrics_explanation():
    """
    Exibe explicações sobre as métricas ROC AUC e KS Score
    """
    # Informações didáticas sobre as métricas utilizadas
    st.markdown("""
    **ROC AUC (Area Under the Receiver Operating Characteristic Curve):** É uma métrica que mede a capacidade de um modelo de classificação de distinguir entre duas classes (por exemplo, "bom pagador" e "mau pagador"). O valor da ROC AUC representa a probabilidade de um caso positivo aleatório ter uma pontuação de risco maior do que um caso negativo aleatório. Os valores variam entre 0.5 (desempenho equivalente ao acaso) e 1 (desempenho perfeito).

    **KS Score (Kolmogorov-Smirnov):** É outra métrica para avaliar o poder de separação de um modelo de classificação. Embora esteja sendo gradualmente substituída, é uma métrica tradicionalmente muito utilizada na indústria de crédito no Brasil.

    **A diferenciação entre (treino) e (teste) refere-se aos conjuntos de dados utilizados:**

    - **ROC AUC (treino) e KS Score (treino):** Medem o desempenho do modelo nos mesmos dados que foram usados para criá-lo e treiná-lo.

    - **ROC AUC (teste) e KS Score (teste):** Avaliam o desempenho do modelo em dados que não foram utilizados durante o treinamento, oferecendo uma perspectiva sobre a capacidade de generalização do modelo.
    """)

def display_model_history():
    """
    Exibe o histórico de modelos treinados em formato de tabela
    """
    # Garante que o histórico está inicializado
    initialize_model_history()
    
    # Verifica se há modelos no histórico
    if len(st.session_state['model_history']) == 0:
        st.info("Nenhum modelo foi treinado ainda. Treine um modelo para ver o histórico.")
        return
    
    # Convertendo histórico para DataFrame para melhor visualização
    df_history = pd.DataFrame(st.session_state['model_history'])
    
    # Reorganizando colunas para melhor visualização
    df_display = df_history[['timestamp', 'model_name', 'roc_auc_train', 'roc_auc_test', 
                           'ks_train', 'ks_test', 'features_count']].copy()
    
    # Renomeando colunas para melhor apresentação
    df_display.columns = ['Data/Hora', 'Modelo', 'ROC AUC (Treino)', 'ROC AUC (Teste)', 
                         'KS (Treino)', 'KS (Teste)', 'Qtd Features']
    
    # Formatando valores numéricos para melhor legibilidade
    for col in ['ROC AUC (Treino)', 'ROC AUC (Teste)']:
        df_display[col] = df_display[col].round(4)
    
    for col in ['KS (Treino)', 'KS (Teste)']:
        df_display[col] = df_display[col].round(2)
    
    # Exibe a tabela de histórico
    st.write("### 📊 Histórico de Modelos Treinados")
    st.dataframe(df_display, use_container_width=True)
    
    # Adicionar explicação das métricas logo após a tabela do histórico
    display_metrics_explanation()
    
    # Gráfico comparativo de modelos (apenas se houver mais de um modelo)
    if len(df_history) > 1:
        st.write("### 📈 Comparação de Performance")
        
        # Cria um gráfico interativo com Plotly
        fig = go.Figure()
        
        # Adiciona linha para ROC AUC Teste
        fig.add_trace(go.Scatter(
            x=list(range(len(df_history))),
            y=df_history['roc_auc_test'],
            mode='lines+markers',
            name='ROC AUC (Teste)',
            text=df_history['model_name'],
            hovertemplate='%{text}<br>ROC AUC: %{y:.4f}<extra></extra>'
        ))
        
        # Adiciona linha para KS Teste (eixo Y secundário)
        fig.add_trace(go.Scatter(
            x=list(range(len(df_history))),
            y=df_history['ks_test'],
            mode='lines+markers',
            name='KS Score (Teste)',
            yaxis='y2',  # Usar segundo eixo Y
            text=df_history['model_name'],
            hovertemplate='%{text}<br>KS: %{y:.2f}<extra></extra>'
        ))
        
        # Configuração do layout com dois eixos Y
        fig.update_layout(
            template="plotly_dark",
            title="Evolução da Performance dos Modelos",
            xaxis_title='Ordem de Treinamento',
            yaxis=dict(title='ROC AUC', side='left'),
            yaxis2=dict(title='KS Score', side='right', overlaying='y'),
            legend=dict(
                orientation="h",
                y=-0.2,
                x=0.5,
                xanchor="center",
                yanchor="top"
            )
        )
        
        # Exibe o gráfico de comparação
        st.plotly_chart(fig, use_container_width=True)

def get_model_display_name(model_type):
    """
    Converte nomes internos dos modelos para nomes de exibição na interface
    """
    if model_type == ModelTypes.KNN:
        return "Regressão Linear"
    return model_type

if __name__ == '__main__':
    # Inicializar histórico de modelos
    initialize_model_history()

    # Exibe a mensagem de boas-vindas/instruções
    st.write(msn.treino)
    
    # Inicializa variáveis para armazenar os dados de treino e validação
    X_train, X_val = None, None
    
    # Interface para upload do arquivo com os dados
    df = st.file_uploader("Envie seu arquivo para treino", type=['xlsx'], accept_multiple_files=False)
    
    # Extrai features e target dos dados carregados
    X, y = get_features_and_target(df)

    if X is not None:
        # Marca no session_state que há dados disponíveis
        st.session_state['has_data'] = True
        if X is not None and y is not None:
            # Remove linhas onde y é NaN para evitar problemas no treinamento
            mask = ~y.isna()
            X = X[mask]
            y = y[mask]
            if y.isna().sum() > 0:
                st.warning(f"Foram removidas {y.isna().sum()} linhas com alvo (y) nulo.")
            
            # Checagem se ainda há dados suficientes após remover valores nulos
            if len(X) == 0 or len(y) == 0:
                st.error("Não há dados suficientes para treinar o modelo após remover valores nulos. Verifique seu arquivo de entrada.")
                X_train = X_val = y_train = y_val = None
            else:
                # Divide os dados em treino e validação (70% / 30%)
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=123)

    # Seção de Histórico - antes da seleção do modelo
    st.write("---")
    # Layout com três colunas para os botões de histórico
    col_hist1, col_hist2, col_hist3 = st.columns([2, 1, 1])
    
    # Botão para mostrar/ocultar o histórico
    with col_hist1:
        if st.button("📈 Ver Histórico de Modelos"):
            st.session_state['show_history'] = not st.session_state.get('show_history', False)
    
    # Botão para atualizar o histórico
    with col_hist2:
        if st.button("🔄 Atualizar Histórico", type="primary"):
            st.rerun()
    
    # Botão para limpar o histórico
    with col_hist3:
        if st.button("🗑️ Limpar Histórico", type="secondary"):
            clear_model_history()
            st.success("Histórico limpo com sucesso!")
            st.rerun()
    
    # Exibe o histórico de modelos se o botão foi acionado
    if st.session_state.get('show_history', False):
        display_model_history()
    
    st.write("---")

    # Seleção do modelo a ser treinado com nomes mais amigáveis
    model_types = [val for key, val in ModelTypes.__dict__.items() if not key.startswith('__')]
    display_names = [get_model_display_name(model_type) for model_type in model_types]
    
    model_choice = st.selectbox(
        label="Selecione o modelo para treinar",
        options=["Selecionar..."] + display_names,
        index=0,
        key="model_choice",
        on_change=reset_model_state  # Reseta o estado quando o usuário muda de modelo
    )

    # Configuração dos modelos disponíveis
    # Usando o padrão match/case do Python 3.10+
    match model_choice:
        case ModelTypes.LOG_REG:
            # Regressão Logística com regularização L1 (Lasso)
            base_model = LogisticRegression(class_weight='balanced', penalty='l1', C=0.01, solver='liblinear')

        case ModelTypes.LGBM:
            # LightGBM - algoritmo de gradient boosting eficiente
            from lightgbm import LGBMClassifier
            base_model = LGBMClassifier(n_estimators=300, learning_rate=0.007, reg_alpha=0.5, reg_lambda=0.5,  random_state=123)

        case "Regressão Linear":
            # Regressão Linear calibrada para fornecer probabilidades
            from sklearn.linear_model import RidgeClassifier
            from sklearn.calibration import CalibratedClassifierCV
            
            # Usamos CalibratedClassifierCV para adicionar funcionalidade predict_proba
            # ao RidgeClassifier que naturalmente não a possui
            ridge_model = RidgeClassifier(alpha=1.0, class_weight='balanced', random_state=123)
            base_model = CalibratedClassifierCV(ridge_model, method='sigmoid', cv=3)

        case ModelTypes.XGB:
            # XGBoost - implementação popular de gradient boosting
            from xgboost import XGBClassifier
            base_model = XGBClassifier(learning_rate=0.08, n_estimators=125, max_depth=6, colsample_bytree=0.9,gamma=0.5,
                                  min_child_weight=1,subsample=0.8)

        case ModelTypes.ANN:
            # Rede Neural Artificial - implementação do scikit-learn
            from sklearn.neural_network import MLPClassifier
            base_model = MLPClassifier((4, 8, 4), random_state=123)
        case _:
            # Caso nenhum modelo seja selecionado
            base_model = None

    # Se um modelo foi selecionado e há dados disponíveis
    if base_model is not None and st.session_state.get('has_data'):
        if X_train is None:
            print("Selecione seus dados antes de treinar o modelo")
        else:
            # Tratando valores nulos antes do processamento
            X_train = X_train.fillna(0)
            X_val = X_val.fillna(0)
            
            # Criando pipeline compatível com todos os modelos
            if model_choice in [ModelTypes.XGB, ModelTypes.ANN]:
                # Para XGBoost e ANN, usamos uma abordagem em duas etapas
                # para evitar problemas de compatibilidade com o pipeline do sklearn
                encoder = AutoWOEEncoder()
                scaler = StandardScaler().set_output(transform="pandas")
                
                # Pipeline para pré-processamento apenas
                preprocess_pipeline = Pipeline([
                    ('auto_woe_encoder', encoder),
                    ('scaler', scaler)
                ])
                
                # Aplicar pré-processamento aos dados
                X_train_processed = preprocess_pipeline.fit_transform(X_train, y_train)
                X_val_processed = preprocess_pipeline.transform(X_val)
                
                # Wrapper que contém tanto o pipeline de pré-processamento quanto o modelo
                model_wrapper = {
                    'type': 'two_step',
                    'preprocess': preprocess_pipeline,
                    'model': base_model
                }
            else:
                # Para outros modelos (mais compatíveis), usamos o pipeline padrão
                # que combina pré-processamento e modelo em um único objeto
                model_wrapper = {
                    'type': 'pipeline',
                    'pipeline': Pipeline([
                        ('auto_woe_encoder', AutoWOEEncoder()),
                        ('scaler', StandardScaler().set_output(transform="pandas")),
                        ('model', base_model)
                    ])
                }

            # Botão para iniciar o treinamento do modelo
            fit = st.button("Treinar modelo (pode levar alguns minutos)")
            if fit:
                try:
                    # Treinar o modelo conforme o tipo de wrapper
                    if model_wrapper['type'] == 'two_step':
                        # Treinar o modelo separadamente após pré-processamento
                        model_wrapper['model'].fit(X_train_processed, y_train)
                        is_fit = True
                    else:
                        # Treinar o pipeline completo
                        model_wrapper['pipeline'].fit(X_train, y_train)
                        is_fit = True
                    
                    st.write("Modelo treinado!")
                    
                    # Salvar no session_state para uso posterior
                    st.session_state['is_fit'] = True
                    st.session_state['model_wrapper'] = model_wrapper
                    st.session_state['features'] = X_train.columns
                    
                    # Calcular métricas imediatamente após o treinamento
                    if model_wrapper['type'] == 'two_step':
                        # Para modelos de duas etapas, precisamos aplicar o pré-processamento
                        y_probs = model_wrapper['model'].predict_proba(X_val_processed)[:, 1]
                        y_probs_treino = model_wrapper['model'].predict_proba(X_train_processed)[:, 1]
                    else:
                        # Para pipeline padrão, podemos passar os dados diretamente
                        y_probs = model_wrapper['pipeline'].predict_proba(X_val)[:, 1]
                        y_probs_treino = model_wrapper['pipeline'].predict_proba(X_train)[:, 1]

                    # Calcula métricas de performance
                    roc_auc_teste = met.roc_auc(y_val, y_probs)
                    ks_teste = met.ks_score(y_val, y_probs)
                    roc_auc_train = met.roc_auc(y_train, y_probs_treino)
                    ks_train = met.ks_score(y_train, y_probs_treino)

                    # Adicionar ao histórico de modelos
                    metrics_dict = {
                        'roc_auc_train': roc_auc_train,
                        'roc_auc_test': roc_auc_teste,
                        'ks_train': ks_train,
                        'ks_test': ks_teste
                    }
                    add_model_to_history(model_choice, metrics_dict, X_train.columns, model_wrapper)
                    
                    # Confirmação visual para o usuário
                    st.success(f"✅ Modelo {model_choice} adicionado ao histórico!")
                    st.info("💡 Clique em 'Atualizar Histórico' para ver as mudanças se o histórico estiver aberto.")
                    
                    # Adicionar explicações específicas para cada modelo
                    if model_choice == ModelTypes.LOG_REG:
                        st.markdown("""
                        ### Regressão Logística
                        **Vantagens**: É um modelo muito rápido, que consome poucos recursos computacionais e é extremamente fácil de interpretar. Os coeficientes de cada variável mostram de forma clara e direta como elas influenciam a previsão, o que é excelente para explicar os resultados e gerar insights de negócio.

                        **Desvantagens**: Sua principal fraqueza é a incapacidade de capturar relações complexas e não-lineares nos dados. O modelo assume uma fronteira de decisão linear, o que limita seu poder preditivo em problemas mais complexos, onde modelos mais modernos geralmente apresentam performance superior.
                        """)
                    elif model_choice == ModelTypes.LGBM:
                        st.markdown("""
                        ### LightGBM
                        **Vantagens**: Sua maior vantagem é a velocidade de treinamento e o baixo uso de memória. Ele é significativamente mais rápido que seus concorrentes (como o XGBoost) em grandes volumes de dados, permitindo iterações e experimentos muito mais ágeis. Mantém um altíssimo poder preditivo.

                        **Desvantagens**: Em datasets pequenos (com poucos milhares de linhas), ele é mais propenso a overfitting (se ajustar demais aos dados de treino e generalizar mal). A grande quantidade de hiperparâmetros, embora flexível, pode tornar sua otimização um processo complexo.
                        """)
                    elif model_choice == "Regressão Linear":
                        st.markdown("""
                        ### Regressão Linear
                        **Vantagens**: É o modelo mais simples e intuitivo para prever valores numéricos contínuos. É muito rápido para treinar e seus resultados são totalmente interpretáveis, permitindo entender exatamente quanto cada variável contribui para a previsão final.

                        **Desvantagens**: Sua maior limitação é assumir que a relação entre as variáveis é puramente linear. Ele não consegue modelar curvas ou interações complexas, além de ser muito sensível a outliers (valores extremos), o que o torna inadequado para a maioria dos problemas do mundo real que buscam máxima precisão.
                        """)
                    elif model_choice == ModelTypes.XGB:
                        st.markdown("""
                        ### XGBoost (Extreme Gradient Boosting)
                        **Vantagens**: É famoso por seu altíssimo poder preditivo e robustez. Frequentemente, é o modelo que apresenta os melhores resultados em competições de Machine Learning com dados estruturados (tabelas). Possui mecanismos internos de regularização que ajudam a controlar o overfitting.

                        **Desvantagens**: Seu principal ponto fraco é o alto custo computacional. Ele tende a ser mais lento para treinar e consumir mais memória do que alternativas como o LightGBM. A sintonia fina de seus diversos hiperparâmetros também pode ser um processo demorado e complexo.
                        """)
                    elif model_choice == ModelTypes.ANN:
                        st.markdown("""
                        ### Rede Neural
                        **Vantagens**: Tem uma capacidade incomparável de aprender padrões muito complexos e não-lineares, sendo o modelo de escolha para dados não estruturados como imagens, áudio e texto. Quando bem treinada e com dados suficientes, pode atingir o maior poder preditivo entre todos os modelos.

                        **Desvantagens**: São modelos "caixa-preta" (black box), ou seja, é extremamente difícil entender o porquê de suas decisões. Exigem um volume massivo de dados, alto custo computacional (tempo e hardware potentes) e sua arquitetura e ajuste de hiperparâmetros são notoriamente complexos.
                        """)
                except Exception as e:
                    # Tratamento de erros durante o treinamento
                    st.error(f"Erro ao treinar o modelo: {str(e)}")
                    st.exception(e)

    # Seção de diagnóstico - exibida apenas se um modelo foi treinado
    if st.session_state.get('is_fit') is not None and st.session_state.get('has_data'):
        model_wrapper = st.session_state['model_wrapper']
        st.write("---")
        
        # Título do diagnóstico
        st.write("# Diagnóstico do modelo")
        
        if X_val is None or X_train is None:
            st.write("Dê upload de base de treino novamente")
        else:
            try:
                # Fazer predições de acordo com o tipo de modelo
                if model_wrapper['type'] == 'two_step':
                    # Para modelos de duas etapas, aplicamos o pré-processamento primeiro
                    X_val_processed = model_wrapper['preprocess'].transform(X_val)
                    X_train_processed = model_wrapper['preprocess'].transform(X_train)
                    
                    y_probs = model_wrapper['model'].predict_proba(X_val_processed)[:, 1]
                    y_probs_treino = model_wrapper['model'].predict_proba(X_train_processed)[:, 1]
                else:
                    # Para pipeline padrão, podemos passar os dados diretamente
                    y_probs = model_wrapper['pipeline'].predict_proba(X_val)[:, 1]
                    y_probs_treino = model_wrapper['pipeline'].predict_proba(X_train)[:, 1]
                
                # Calcular métricas de performance novamente
                roc_auc_teste = met.roc_auc(y_val, y_probs)
                ks_teste = met.ks_score(y_val, y_probs)
                roc_auc_train = met.roc_auc(y_train, y_probs_treino)
                ks_train = met.ks_score(y_train, y_probs_treino)
                
                # Layout para métricas e explicações lado a lado
                metrics_col, explanation_col = st.columns([1, 1])
                
                # Coluna esquerda: Exibição das métricas
                with metrics_col:
                    st.subheader("Métricas de Performance")
                    col1, col2 = st.columns(2)
                    
                    # Exibir métricas formatadas com a função metric()
                    col1.metric(label="ROC AUC (teste)", value=f"{round(roc_auc_teste, 4)}")
                    col1.metric(label="KS Score (teste)", value=f"{round(ks_teste, 2)}")

                    col2.metric(label="ROC AUC (treino)", value=f"{round(roc_auc_train, 4)}")
                    col2.metric(label="KS Score (treino)", value=f"{round(ks_train, 2)}")
                
                # Coluna direita: Explicação das métricas
                with explanation_col:
                    st.subheader("📚 O que significam estas métricas?")
                    display_metrics_explanation()

                # Seção de visualizações gráficas
                st.write("### Visualizações")
                
                # Gráfico 1: Curva ROC
                # Calcula pontos para a curva ROC
                fpr, tpr = met.roc_curve(y_val, y_probs)
                fpr_train, tpr_train = met.roc_curve(y_train, y_probs_treino)

                # Layout com gráfico e explicação lado a lado
                roc_graph_col, roc_explanation_col = st.columns([3, 2])
                
                # Coluna esquerda: Gráfico da Curva ROC
                with roc_graph_col:
                    fig = go.Figure()
                    # Adiciona curva ROC para dados de teste
                    fig.add_trace(go.Scatter(x=fpr.round(3), y=tpr.round(3),
                                                mode='lines',
                                                name='ROC (Teste)',
                                                hovertemplate='FPR: %{x}, TPR: %{y} <extra></extra>'))
                    # Adiciona curva ROC para dados de treino
                    fig.add_trace(go.Scatter(x=fpr_train.round(3), y=tpr_train.round(3),
                                                mode='lines',
                                                name='ROC (Train)',
                                                hovertemplate='FPR: %{x}, TPR: %{y} <extra></extra>'))
                    # Adiciona linha de referência (baseline aleatório)
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                                mode='lines',
                                                line=dict(dash='dot'),
                                                name='Baseline aleatório'))

                    # Configuração do layout do gráfico
                    fig.update_layout(template="plotly_dark",
                                        title="Curva ROC",
                                        xaxis_title='Taxa de falsos positivos (FPR)',
                                        yaxis_title='Taxa de positivos verdadeiros (TPR)',
                                        legend=dict(
                                            orientation="h",
                                            y=-0.2,
                                            x=0.5,
                                            xanchor="center",
                                            yanchor="top"
                                        ))

                    # Exibe o gráfico da curva ROC
                    st.plotly_chart(fig, theme=None, use_container_width=True)
                
                # Coluna direita: Explicação da Curva ROC
                with roc_explanation_col:
                    st.subheader("📊 Entendendo a Curva ROC")
                    st.markdown("""
                    **Curva ROC** avalia a performance de um modelo preditivo, mostrando o quão bem ele consegue separar dois grupos (por exemplo, bons e maus pagadores). O objetivo é ter uma curva que se aproxime o máximo possível do canto superior esquerdo.

                    **Linhas azul e vermelha (ROC)**: Mostram o desempenho do modelo. O objetivo é que elas se afastem da linha pontilhada e cheguem o mais perto possível do canto superior esquerdo.

                    **Linha azul (Teste)**: É a mais importante, pois mostra o desempenho do modelo com dados novos, como aconteceria na vida real.

                    **Linha pontilhada (Baseline)**: Representa um palpite ou um chute. O modelo precisa ser melhor do que isso para ser útil.
                    """)

                # Gráfico 2: Curvas de Erro
                st.subheader("Curvas de Erro")
                
                # Layout com gráfico e explicação lado a lado
                error_graph_col, error_explanation_col = st.columns([3, 2])
                
                # Coluna esquerda: Gráfico de Curvas de Erro
                with error_graph_col:
                    # Calcula taxas de erro para diferentes thresholds
                    fpr, fnr, thresh = met.false_positive_negative_rates(y_val, y_probs)

                    fig = go.Figure()
                    # Adiciona curva de falsos positivos
                    fig.add_trace(go.Scatter(x=thresh.round(3), y=fpr.round(3),
                                                mode='lines',
                                                name='Taxa de falsos positivos (FPR)',
                                                hovertemplate='Thresh: %{x}, FPR: %{y}<extra></extra>'))
                    # Adiciona curva de falsos negativos
                    fig.add_trace(go.Scatter(x=thresh.round(3), y=fnr.round(3),
                                                mode='lines',
                                                name='Taxa de falsos negativos (FNR)',
                                                hovertemplate='Thresh: %{x}, FNR: %{y}<extra></extra>'))

                    # Configuração do layout do gráfico
                    fig.update_xaxes(range=[0.0, 1.0])
                    fig.update_layout(template="plotly_dark",
                                        title="Curvas de falsos positivos / negativos (somente conjunto de teste)",
                                        xaxis_title='Limiar de cutoff (threshold)',
                                        legend=dict(
                                            orientation="h",
                                            y=-0.2,
                                            x=0.5,
                                            xanchor="center",
                                            yanchor="top"
                                        ))

                    # Exibe o gráfico de curvas de erro
                    st.plotly_chart(fig, theme=None, use_container_width=True)
                
                # Coluna direita: Explicação das Curvas de Erro
                with error_explanation_col:
                    st.subheader("📉 Entendendo as Curvas de Erro")
                    st.markdown("""
                    Este gráfico ajuda a decidir qual "nota de corte" (ou threshold) usar para tomar uma decisão com o modelo. A "nota de corte" é a regra que define, por exemplo, a partir de qual pontuação de risco um cliente terá o crédito negado. O gráfico mostra como a escolha dessa regra afeta dois tipos de erro.

                    **Linha azul (Falsos positivos)**: É o erro de "alarme falso" (negar crédito a um bom cliente).

                    **Linha vermelha (Falsos negativos)**: É o erro de "deixar passar" (aprovar crédito para um mau cliente).

                    A escolha da melhor "nota de corte" (o ponto no eixo horizontal) depende de qual desses dois erros é pior para o seu negócio, pois não é possível zerar ambos ao mesmo tempo.
                    """)

                # Seção de interpretabilidade - disponível apenas para regressão logística
                if model_choice == ModelTypes.LOG_REG:
                    st.subheader("Interpretabilidade do modelo")
                    
                    # Layout com gráfico e explicação lado a lado
                    coef_graph_col, coef_explanation_col = st.columns([3, 2])
                    
                    # Coluna esquerda: Gráfico de coeficientes
                    with coef_graph_col:
                        # Obter o modelo correto dependendo do tipo de wrapper
                        if model_wrapper['type'] == 'two_step':
                            model_obj = model_wrapper['model']
                        else:
                            model_obj = model_wrapper['pipeline'].named_steps['model']
                        
                        # Verificar se o modelo tem coeficientes disponíveis
                        if hasattr(model_obj, 'coef_'):
                            # Extrai os coeficientes e cria um DataFrame para visualização
                            coefs = model_obj.coef_
                            aux = pd.DataFrame({'var': X_train.columns, 'coef': coefs[0]}).sort_values('coef', ascending=True).reset_index(drop=True)
                            
                            # Cria um gráfico de barras horizontais com os coeficientes
                            fig, ax = plt.subplots(figsize=(5,6))
                            ax.barh(aux.index, aux['coef'])
                            ax.set_yticks(aux.index, aux['var'])
                            ax.set_title('Coeficientes da regressão logística')
                            ax.grid()
                            ax.set_facecolor('#0C1017')
                            st.pyplot(fig)
                        else:
                            # Mensagem se os coeficientes não estiverem disponíveis
                            st.warning("Não foi possível acessar os coeficientes do modelo.")
                    
                    # Coluna direita: Explicação dos coeficientes
                    with coef_explanation_col:
                        st.subheader("📊 Entendendo os Coeficientes")
                        st.markdown("""
                        Este gráfico mostra a **importância e o impacto** de cada variável no modelo de regressão logística:

                        **Valores positivos** (barras para a direita): Indicam que a variável **aumenta** a chance de um resultado positivo (ex: inadimplência).

                        **Valores negativos** (barras para a esquerda): Indicam que a variável **diminui** a chance de um resultado positivo.

                        **Tamanho da barra**: Quanto maior o valor absoluto (comprimento da barra), maior o impacto da variável na decisão do modelo.

                        Isso permite identificar quais características mais contribuem para o risco e quais são "protetivas" na sua análise.
                        """)
                        
            except Exception as e:
                # Tratamento de erros na seção de diagnóstico
                st.error(f"Erro ao gerar diagnóstico: {str(e)}")
                st.exception(e)
