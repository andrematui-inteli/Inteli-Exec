# access this file from cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/ in the Mac terminal
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
import shap
import messages as msn
import metrics as met
from models import ModelTypes, AutoWOEEncoder, BetaCalibratedClassifier
from config.config import set_page_config
set_page_config()

def get_features_and_target(data):
    X, y = None, None
    if data is None:
        pass
    else:
        col1, col2 = st.columns(2)
        st.write("Carregando dados...")
        df = pd.read_excel(data)
        col1.write("Visualização da sua base de dados (30 linhas)")
        col1.dataframe(df.head(30))
        cols = df.columns.tolist()

        feats_list, target_list = cols[:-1], cols[-1]

        features = col2.multiselect(label="Verifique que essas são as variáveis de treino",
                                    options=cols, default=feats_list)
        target = col2.multiselect(
            label="Verifique que esse é seu alvo de treino",
            options=cols,
            default=target_list,
            max_selections=1)

        exclusive = set(features).intersection(target) == set()
        has_target = len(target) == 1
        if exclusive & has_target:
            col2.write("✅ Sua escolha de covariáveis e alvo estão boas.\n\n Pode ir em frente com o treino.")
            X, y = df[features], df[target[0]]

        else:
            col2.write("❌ Garanta que você escolheu apenas um alvo e que ele não faça parte das variáveis de treino!")
    return X, y


if __name__ == '__main__':

    # if 'train_bytes' in st.session_state:
    #     del st.session_state['train_bytes']
    #     del st.session_state['test_bytes']
    #     st.experimental_rerun()

    st.write(msn.treino)
    X_train, X_val = None, None
    df = st.file_uploader("Envie seu arquivo para treino", type=['xlsx'], accept_multiple_files=False)
    X, y = get_features_and_target(df)

    if X is not None:
        st.session_state['has_data'] = True
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=123)

    model_choice = st.selectbox(label="Selecione o modelo para treinar",
                            options=["Selecionar..."] + [val for key, val in ModelTypes.__dict__.items() if not key.startswith('__')],
                            index=0)

    match model_choice:
        case ModelTypes.LOG_REG:
            base_model = LogisticRegression(class_weight='balanced', penalty='l1', C=0.01, solver='liblinear')

        case ModelTypes.LGBM:
            from lightgbm import LGBMClassifier
            base_model = LGBMClassifier(n_estimators=300, learning_rate=0.007, reg_alpha=0.5, reg_lambda=0.5,  random_state=123)
            # model = LGBMClassifier(n_estimators=125, learning_rate=0.08, colsample_bytree=0.9, min_child_weight=1, subsample=0.8)

        # case ModelTypes.KNN:
        #     from sklearn.neighbors import KNeighborsClassifier
        #     base_model = KNeighborsClassifier(n_neighbors=300, weights='uniform', n_jobs=3)

        # case ModelTypes.XGB:
        #     from xgboost import XGBClassifier
        #     # https://analytics-nuts.github.io/Comparative-Study-of-Classification-Techniques-on-Credit-Defaults/
        #     base_model = XGBClassifier(learning_rate=0.08, n_estimators=125, max_depth=6, colsample_bytree=0.9,gamma=0.5,
        #                           min_child_weight=1,subsample=0.8)

        case ModelTypes.ANN:
            from sklearn.neural_network import MLPClassifier
            base_model = MLPClassifier((4, 8, 4), random_state=123)
        case _:
            base_model = None

    if base_model is not None and st.session_state.get('has_data'):

        if X_train is None:
            print("Selecione seus dados antes de treinar o modelo")
        else:
            model = Pipeline([
                ('auto_woe_encoder', AutoWOEEncoder()),
                ('scaler', StandardScaler().set_output(transform="pandas")),
                ('model', base_model)
                # ('beta_calibrated_classifier', BetaCalibratedClassifier(base_estimator=base_model))
            ])

            fit = st.button("Treinar modelo (pode levar alguns minutos)")
            if fit:
                model.fit(X_train, y_train)
                is_fit = True
                st.write("Modelo treinado!")
                st.session_state['is_fit'] = True
                st.session_state['model'] = model
                st.session_state['features'] = X_train.columns

    if st.session_state.get('is_fit') is not None and st.session_state.get('has_data'):
        model = st.session_state['model']
        st.write("---")
        st.write("# Diagnóstico do modelo")
        if X_val is None or X_train is None:
            st.write("Dê upload de base de treino novamente")
        else:
            y_probs = model.predict_proba(X_val)[:, 1]
            y_probs_treino = model.predict_proba(X_train)[:, 1]

            roc_auc_teste = met.roc_auc(y_val, y_probs)
            ks_teste = met.ks_score(y_val, y_probs)
            roc_auc_train = met.roc_auc(y_train, y_probs_treino)
            ks_train = met.ks_score(y_train, y_probs_treino)

            col1, col2 = st.columns(2)

            col1.metric(label="ROC AUC (teste)", value=f"{round(roc_auc_teste, 4)}")
            col1.metric(label="KS Score (teste)", value=f"{round(ks_teste, 2)}")

            col2.metric(label="ROC AUC (treino)", value=f"{round(roc_auc_train, 4)}")
            col2.metric(label="KS Score (treino)", value=f"{round(ks_train, 2)}")


            # show ROC curve
            fpr, tpr = met.roc_curve(y_val, y_probs)
            fpr_train, tpr_train = met.roc_curve(y_train, y_probs_treino)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr.round(3), y=tpr.round(3),
                                        mode='lines',
                                        name='ROC (Teste)',
                                        hovertemplate='FPR: %{x}, TPR: %{y} <extra></extra>'))
            fig.add_trace(go.Scatter(x=fpr_train.round(3), y=tpr_train.round(3),
                                        mode='lines',
                                        name='ROC (Train)',
                                        hovertemplate='FPR: %{x}, TPR: %{y} <extra></extra>'))

            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                        mode='lines',
                                        line=dict(dash='dot'),
                                        name='Baseline aleatório'))

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

            col1.plotly_chart(fig, theme=None)

            # show precision and recalls
            fpr, fnr, thresh = met.false_positive_negative_rates(y_val, y_probs)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=thresh.round(3), y=fpr.round(3),
                                        mode='lines',
                                        name='Taxa de falsos positivos (FPR)',
                                        hovertemplate='Thresh: %{x}, FPR: %{y}<extra></extra>'))
            fig.add_trace(go.Scatter(x=thresh.round(3), y=fnr.round(3),
                                        mode='lines',
                                        name='Taxa de falsos negativos (FNR)',
                                        hovertemplate='Thresh: %{x}, FNR: %{y}<extra></extra>'))

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

            col2.plotly_chart(fig, theme=None)

            # st.write("# Predições no dataset inteiro")
            # df_pred = X.copy()
            # df_pred[' '] = ' '
            # df_pred['Ground truth'] = y
            # df_pred['Prediction'] = model.predict_proba(X)[:,1]
            # st.dataframe(df_pred)

            # Analise de interpretabilidade


            if model_choice == ModelTypes.LOG_REG:
                col1.write("Interpretabilidade do modelo (somente disponível para reg. log.)")
                # coefs = model['beta_calibrated_classifier'].base_estimator.coef_
                coefs = model['model'].coef_
                aux = pd.DataFrame({'var': X_train.columns, 'coef': coefs[0]}).sort_values('coef', ascending=True).reset_index(drop=True)

                fig, ax = plt.subplots(figsize=(5,6))
                ax.barh(aux.index, aux['coef'])
                ax.set_yticks(aux.index, aux['var'])
                ax.set_title('Coeficientes da regressão logística')
                ax.grid()
                ax.set_facecolor('#0C1017')
                col1.pyplot(fig)

            # OLD: does interpretability for more models, but is heavy
            # if model_choice in (ModelTypes.LGBM, ModelTypes.XGB):
            #     clf = model['beta_calibrated_classifier'].base_estimator
            #     mini_pipeline = make_pipeline(model['auto_woe_encoder'], model['scaler'])
            #     X_test_ = mini_pipeline.transform(X_val)


            #     explainer = shap.TreeExplainer(clf)
            #     shap_values = explainer.shap_values(X_test_)
            #     fig = plt.figure()

            #     if model_choice == ModelTypes.LGBM:
            #         shap.summary_plot(shap_values[0], X_test_, feature_names=X_train.columns, max_display=15, show=False)

            #     if model_choice == ModelTypes.XGB:
            #         shap.summary_plot(shap_values, X_test_, feature_names=X_train.columns, max_display=15, show=False)

            #     plt.gcf().set_size_inches(8,6)
            #     col1.pyplot(fig)
