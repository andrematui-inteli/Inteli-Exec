import streamlit as st
import pandas as pd
import io
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import set_page_config

set_page_config()

# Reimplementar as funções necessárias aqui em vez de importar
def initialize_model_history():
    """
    Inicializa o histórico de modelos no session_state se não existir
    """
    if 'model_history' not in st.session_state:
        st.session_state['model_history'] = []

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

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.close()
    processed_data = output.getvalue()
    return processed_data

# Garantir que o histórico de modelos esteja inicializado
initialize_model_history()

# Inicializar variáveis de controle no session_state
if 'showing_prediction_section' not in st.session_state:
    st.session_state['showing_prediction_section'] = True

if 'model_switched' not in st.session_state:
    st.session_state['model_switched'] = False

if 'just_updated_page' not in st.session_state:
    st.session_state['just_updated_page'] = False

if st.session_state.get('is_fit') is None:
    st.write("Você precisa treinar um modelo antes")
    st.write("### Importante! \n As features devem ser *exatamente* as mesmas usadas para treino")
else:
    if st.session_state['is_fit']:
        st.write("# Aplicar score na base de teste")
        
        # Seção de Histórico - antes do upload de arquivo
        st.write("## Histórico e seleção de modelos")
        st.write("Você pode selecionar um modelo diferente do histórico para usar nas predições:")
        
        # Layout com três colunas para os botões de histórico
        col_hist1, col_hist2, col_hist3 = st.columns([2, 1, 1])
        
        # Botão para mostrar/ocultar o histórico
        with col_hist1:
            if st.button("📈 Ver Histórico de Modelos"):
                st.session_state['show_history_test'] = not st.session_state.get('show_history_test', False)
        
        # Botão para atualizar a página
        with col_hist2:
            if st.button("🔄 Atualizar Página", type="primary"):
                st.session_state['just_updated_page'] = True
                st.rerun()
        
        # Coluna para seleção do modelo (vazia por enquanto)
        with col_hist3:
            pass
        
        # Exibe o histórico de modelos se o botão foi acionado
        if st.session_state.get('show_history_test', False):
            display_model_history()
        
        # Selecionar modelo do histórico para usar
        if 'model_history' in st.session_state and len(st.session_state['model_history']) > 0:
            # Preparar opções para o selectbox
            model_options = []
            for i, model_record in enumerate(st.session_state['model_history']):
                timestamp = model_record['timestamp']
                model_name = model_record['model_name']
                roc_auc = round(model_record['roc_auc_test'], 4)
                option_text = f"[{i+1}] {model_name} ({timestamp}) - ROC AUC: {roc_auc}"
                model_options.append(option_text)
            
            # Adicionar opção para o modelo atual
            model_options.insert(0, "Modelo Atual")
            
            # Se já trocamos o modelo, voltar para "Modelo Atual"
            default_index = 0
            if st.session_state.get('model_switched', False):
                default_index = 0  # Forçar "Modelo Atual" se acabamos de trocar o modelo
            
            # Seletor de modelo
            selected_model = st.selectbox(
                "Selecione o modelo para fazer predições:",
                options=model_options,
                index=default_index  # Use o índice que determinamos acima
            )
            
            # Limpar a flag após usá-la
            if st.session_state.get('model_switched', False):
                st.session_state['model_switched'] = False
            
            # Ao selecionar um novo modelo, oculte a seção de predição
            if selected_model == "Modelo Atual":
                st.session_state['showing_prediction_section'] = True
            else:
                # Se selecionou um modelo do histórico, ocultamos a seção de predição
                st.session_state['showing_prediction_section'] = False
            
            # Mostrar informações sobre o modelo selecionado (seja atual ou histórico)
            if selected_model == "Modelo Atual":
                # Mostrar informações sobre o modelo atual
                st.info("""
                **Modelo selecionado:** Modelo atual (último treinado)
                """)
                
                # Determinar o tipo do modelo atual
                model_wrapper = st.session_state['model_wrapper']
                if model_wrapper['type'] == 'two_step':
                    if hasattr(model_wrapper['model'], '_final_estimator'):
                        model_name = model_wrapper['model']._final_estimator.__class__.__name__
                    else:
                        model_name = model_wrapper['model'].__class__.__name__
                else:
                    if hasattr(model_wrapper['pipeline'], 'named_steps') and 'model' in model_wrapper['pipeline'].named_steps:
                        model_name = model_wrapper['pipeline'].named_steps['model'].__class__.__name__
                    else:
                        model_name = "Desconhecido"
                
                # Mostrar o tipo do modelo atual
                st.write("## Sobre o modelo atual")
                
                # Mapear o nome técnico do modelo para um nome amigável
                display_name = "Desconhecido"
                if "LogisticRegression" in model_name:
                    display_name = "Regressão Logística"
                elif "LGBMClassifier" in model_name:
                    display_name = "LightGBM"
                elif "Ridge" in model_name or "CalibratedClassifierCV" in model_name:
                    display_name = "Regressão Linear"
                elif "XGBClassifier" in model_name:
                    display_name = "XGBoost"
                elif "MLP" in model_name:
                    display_name = "Rede Neural"
                
                st.write(f"### Tipo de modelo: {display_name}")
                
                # Mostrar explicação com base no tipo de modelo identificado
                if "LogisticRegression" in model_name:
                    st.markdown("""
                    **Vantagens**: É um modelo muito rápido, que consome poucos recursos computacionais e é extremamente fácil de interpretar. Os coeficientes de cada variável mostram de forma clara e direta como elas influenciam a previsão, o que é excelente para explicar os resultados e gerar insights de negócio.

                    **Desvantagens**: Sua principal fraqueza é a incapacidade de capturar relações complexas e não-lineares nos dados. O modelo assume uma fronteira de decisão linear, o que limita seu poder preditivo em problemas mais complexos, onde modelos mais modernos geralmente apresentam performance superior.
                    """)
                elif "LGBMClassifier" in model_name:
                    st.markdown("""
                    ### LightGBM
                    **Vantagens**: Sua maior vantagem é a velocidade de treinamento e o baixo uso de memória. Ele é significativamente mais rápido que seus concorrentes (como o XGBoost) em grandes volumes de dados, permitindo iterações e experimentos muito mais ágeis. Mantém um altíssimo poder preditivo.

                    **Desvantagens**: Em datasets pequenos (com poucos milhares de linhas), ele é mais propenso a overfitting (se ajustar demais aos dados de treino e generalizar mal). A grande quantidade de hiperparâmetros, embora flexível, pode tornar sua otimização um processo complexo.
                    """)
                elif "Ridge" in model_name or "CalibratedClassifierCV" in model_name:
                    st.markdown("""
                    ### Regressão Linear
                    **Vantagens**: É o modelo mais simples e intuitivo para prever valores numéricos contínuos. É muito rápido para treinar e seus resultados são totalmente interpretáveis, permitindo entender exatamente quanto cada variável contribui para a previsão final.

                    **Desvantagens**: Sua maior limitação é assumir que a relação entre as variáveis é puramente linear. Ele não consegue modelar curvas ou interações complexas, além de ser muito sensível a outliers (valores extremos), o que o torna inadequado para a maioria dos problemas do mundo real que buscam máxima precisão.
                    """)
                elif "XGBClassifier" in model_name:
                    st.markdown("""
                    ### XGBoost (Extreme Gradient Boosting)
                    **Vantagens**: É famoso por seu altíssimo poder preditivo e robustez. Frequentemente, é o modelo que apresenta os melhores resultados em competições de Machine Learning com dados estruturados (tabelas). Possui mecanismos internos de regularização que ajudam a controlar o overfitting.

                    **Desvantagens**: Seu principal ponto fraco é o alto custo computacional. Ele tende a ser mais lento para treinar e consumir mais memória do que alternativas como o LightGBM. A sintonia fina de seus diversos hiperparâmetros também pode ser um processo demorado e complexo.
                    """)
                elif "MLP" in model_name:
                    st.markdown("""
                    ### Rede Neural
                    **Vantagens**: Tem uma capacidade incomparável de aprender padrões muito complexos e não-lineares, sendo o modelo de escolha para dados não estruturados como imagens, áudio e texto. Quando bem treinada e com dados suficientes, pode atingir o maior poder preditivo entre todos os modelos.

                    **Desvantagens**: São modelos "caixa-preta" (black box), ou seja, é extremamente difícil entender o porquê de suas decisões. Exigem um volume massivo de dados, alto custo computacional (tempo e hardware potentes) e sua arquitetura e ajuste de hiperparâmetros são notoriamente complexos.
                    """)
            elif len(model_options) > 1:
                # O código existente para modelos selecionados do histórico
                # Extrair índice do modelo selecionado (formato "[X] Nome...")
                model_idx = int(selected_model.split(']')[0].replace('[', '')) - 1
                
                # Carregar o modelo do histórico
                selected_model_record = st.session_state['model_history'][model_idx]
                
                # Exibir informações do modelo selecionado
                st.info(f"""
                **Modelo selecionado:** {selected_model_record['model_name']}
                **Treinado em:** {selected_model_record['timestamp']}
                **ROC AUC (teste):** {selected_model_record['roc_auc_test']}
                **KS Score (teste):** {selected_model_record['ks_test']}
                """)
                
                # Exibir alerta se as features não corresponderem
                current_features = set(st.session_state['features'])
                historic_features = set(selected_model_record['features_used'])
                if current_features != historic_features:
                    st.warning("""
                    ⚠️ **Atenção**: As features deste modelo são diferentes do modelo atual.
                    Certifique-se de que seu arquivo de teste contenha todas as features necessárias.
                    """)
                
                # Adicionar explicações sobre o modelo selecionado
                st.write("## Sobre o modelo selecionado")
                
                # Determinar o tipo de modelo com base no nome
                model_name = selected_model_record['model_name']
                
                # Mostrar explicação com base no nome do modelo
                if "Regressão Logística" in model_name:
                    st.markdown("""
                    ### Regressão Logística
                    **Vantagens**: É um modelo muito rápido, que consome poucos recursos computacionais e é extremamente fácil de interpretar. Os coeficientes de cada variável mostram de forma clara e direta como elas influenciam a previsão, o que é excelente para explicar os resultados e gerar insights de negócio.

                    **Desvantagens**: Sua principal fraqueza é a incapacidade de capturar relações complexas e não-lineares nos dados. O modelo assume uma fronteira de decisão linear, o que limita seu poder preditivo em problemas mais complexos, onde modelos mais modernos geralmente apresentam performance superior.
                    """)
                elif "LightGBM" in model_name:
                    st.markdown("""
                    ### LightGBM
                    **Vantagens**: Sua maior vantagem é a velocidade de treinamento e o baixo uso de memória. Ele é significativamente mais rápido que seus concorrentes (como o XGBoost) em grandes volumes de dados, permitindo iterações e experimentos muito mais ágeis. Mantém um altíssimo poder preditivo.

                    **Desvantagens**: Em datasets pequenos (com poucos milhares de linhas), ele é mais propenso a overfitting (se ajustar demais aos dados de treino e generalizar mal). A grande quantidade de hiperparâmetros, embora flexível, pode tornar sua otimização um processo complexo.
                    """)
                elif "Regressão Linear" in model_name:
                    st.markdown("""
                    ### Regressão Linear
                    **Vantagens**: É o modelo mais simples e intuitivo para prever valores numéricos contínuos. É muito rápido para treinar e seus resultados são totalmente interpretáveis, permitindo entender exatamente quanto cada variável contribui para a previsão final.

                    **Desvantagens**: Sua maior limitação é assumir que a relação entre as variáveis é puramente linear. Ele não consegue modelar curvas ou interações complexas, além de ser muito sensível a outliers (valores extremos), o que o torna inadequado para a maioria dos problemas do mundo real que buscam máxima precisão.
                    """)
                elif "XGBoost" in model_name:
                    st.markdown("""
                    ### XGBoost (Extreme Gradient Boosting)
                    **Vantagens**: É famoso por seu altíssimo poder preditivo e robustez. Frequentemente, é o modelo que apresenta os melhores resultados em competições de Machine Learning com dados estruturados (tabelas). Possui mecanismos internos de regularização que ajudam a controlar o overfitting.

                    **Desvantagens**: Seu principal ponto fraco é o alto custo computacional. Ele tende a ser mais lento para treinar e consumir mais memória do que alternativas como o LightGBM. A sintonia fina de seus diversos hiperparâmetros também pode ser um processo demorado e complexo.
                    """)
                elif "Rede Neural" in model_name:
                    st.markdown("""
                    ### Rede Neural
                    **Vantagens**: Tem uma capacidade incomparável de aprender padrões muito complexos e não-lineares, sendo o modelo de escolha para dados não estruturados como imagens, áudio e texto. Quando bem treinada e com dados suficientes, pode atingir o maior poder preditivo entre todos os modelos.

                    **Desvantagens**: São modelos "caixa-preta" (black box), ou seja, é extremamente difícil entender o porquê de suas decisões. Exigem um volume massivo de dados, alto custo computacional (tempo e hardware potentes) e sua arquitetura e ajuste de hiperparâmetros são notoriamente complexos.
                    """)

                # Botão para confirmar a troca de modelo
                if st.button("Usar este modelo para predições", type="primary"):
                    if 'model_wrapper' in selected_model_record:
                        # Salvar o modelo selecionado no session state
                        st.session_state['model_wrapper'] = selected_model_record['model_wrapper']
                        st.session_state['features'] = selected_model_record['features_used']
                        
                        # Esta é a linha mais importante - marcar que o modelo foi trocado 
                        st.session_state['model_switched'] = True
                        
                        # IMPORTANTE: mostrar a seção de predição novamente
                        st.session_state['showing_prediction_section'] = True
                        
                        # Garantir que nossas mensagens de depuração mostrem as informações corretas
                        st.session_state['selected_model_name'] = selected_model_record['model_name']
                        
                        # Mensagem mais informativa sobre o próximo passo
                        st.success(f"""
                        ✅ Modelo {selected_model_record['model_name']} selecionado com sucesso!""")
                    else:
                        st.info("Este modelo foi treinado antes da implementação do armazenamento completo. Por favor, treine este modelo novamente para usá-lo nas predições.")
        
        st.write("---")
        
        # Verificar se acabou de atualizar a página após trocar o modelo
        if st.session_state.get('just_updated_page', False):
            # Resetar a flag para não exibir sempre
            st.session_state['just_updated_page'] = False
            
            # Exibir instruções claras
            if st.session_state.get('selected_model_name'):
                st.info(f"""
                👇 Agora você pode fazer predições com o modelo: **{st.session_state.get('selected_model_name')}**
                
                Use a área de upload abaixo para carregar seu arquivo de dados.
                """)
        
        # Somente exibir a seção de upload e predições se não estamos selecionando modelo
        # ou se já confirmamos a seleção
        if st.session_state['showing_prediction_section']:
            # Interface para upload do arquivo com os dados de teste
            uploaded_file = st.file_uploader("", type=['xlsx'], accept_multiple_files=False, key="unique_uploader")

            # Salvar o arquivo carregado no session_state para preservá-lo após recarregamento
            if uploaded_file is not None:
                # Processar o arquivo e salvar no session_state
                df_val = pd.read_excel(uploaded_file)
                st.session_state['test_file'] = df_val
                st.session_state['has_test_file'] = True

            # Processar predições se temos um arquivo carregado (do upload atual ou anterior)
            if st.session_state.get('has_test_file', False):
                model_wrapper = st.session_state['model_wrapper']
                df_val = st.session_state['test_file']

                # Verificar se temos todas as features necessárias
                required_features = st.session_state['features']

                # Verificar se as colunas necessárias estão presentes
                missing_features = [f for f in required_features if f not in df_val.columns]
                if missing_features:
                    st.error(f"⚠️ Faltam as seguintes colunas no arquivo: {', '.join(missing_features)}")
                else:
                    # Selecionar apenas as features necessárias na ordem correta
                    X_val = df_val[required_features]

                    # Remove linhas com NaN nas features
                    X_val = X_val.fillna(0)  # Garantir que não haja NaNs

                    df_pred = X_val.copy()

                    # Fazer predições de acordo com o tipo de modelo
                    if model_wrapper['type'] == 'two_step':
                        # Aplicar pré-processamento e então o modelo
                        X_val_processed = model_wrapper['preprocess'].transform(X_val)
                        df_pred['Prediction'] = model_wrapper['model'].predict_proba(X_val_processed)[:,1]
                    else:
                        # Usar o pipeline completo
                        df_pred['Prediction'] = model_wrapper['pipeline'].predict_proba(X_val)[:,1]

                    # Salvar predições no session_state
                    st.session_state['predictions'] = df_pred

                    # Mostrar as predições
                    st.write("### Predições geradas")
                    st.dataframe(df_pred.head(30))

                    # Opção para download
                    file_name = st.text_input("Insira o nome que deseja fornecer ao arquivo", "base_scorada")

                    # Botão para download do arquivo com predições
                    excel_data = to_excel(df_pred)
                    st.download_button(
                        label="📥 Download Excel File",
                        data=excel_data,
                        file_name=f"{file_name}.xlsx",
                        mime="application/vnd.ms-excel"
                    )

                    # Limpar a flag de mudança de modelo
                    if st.session_state.get('model_changed', False):
                        st.session_state['model_changed'] = False
        else:
            # Apenas um espaço ou mensagem indicando o que fazer
            st.info("👆 Clique em 'Usar este modelo para predições' para confirmar a escolha ou selecione 'Modelo Atual' para usar o modelo carregado atualmente.")

