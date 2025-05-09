import streamlit as st
import pandas as pd
import io
import boto3

from config.config import set_page_config
set_page_config()

st.markdown(
                """
                <style>
                [data-testid="stElementToolbar"] {
                    display: none;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

def to_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.close()
        processed_data = output.getvalue()
        return processed_data

def upload_file_to_s3(df, company_name, file_name):
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, engine='openpyxl')
    excel_buffer.seek(0)
    s3_client = boto3.client('s3')
    s3_client.upload_fileobj(
        excel_buffer,
        Bucket='inteli-exec-dados',
        Key=f'{company_name}/{file_name}.xlsx'
    )
    st.write("O upload do arquivo foi realizado com sucesso!")

if  st.session_state.get('is_fit') is None:
    st.write("Você precisa treinar um modelo antes")
    st.write("### Importante! \n As features devem ser *exatamente* as mesmas usadas para treino")
else:
    if st.session_state['is_fit']:
        st.write("# Aplicar score na base de teste")
        val = st.file_uploader("Envie seu arquivo para teste", type=['xlsx'], accept_multiple_files=False)
        if val is not None:
            model = st.session_state['model']
            df_val = pd.read_excel(val)
            X_val = df_val[st.session_state['features']]
            df_pred = X_val.copy()
            df_pred['Prediction'] = model.predict_proba(X_val)[:,1]
            st.dataframe(df_pred.head(30))
            company_name = st.selectbox("Qual o nome da sua empresa?", ["Stone", "Ovo", "Inteli", "Renova", "Pine", "KPMG", "Base Comunica", "Estapar", "AG Antecipa", "SINPETRO MS", "Outra"], index=None)
            file_name = st.text_input("Insira o nome que deseja fornecer ao arquivo", "")
            if file_name != "" and company_name != None:
                upload_button = st.button("Realizar upload")
                if upload_button:
                    upload_file_to_s3(df_pred, company_name, file_name)

