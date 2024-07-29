import os
import io
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import messages as msn
import models
import metrics as met

from PIL import Image

from config.config import set_page_config
set_page_config()

plt.style.use("dark_background")
mpl.rcParams['figure.dpi'] = 210
font = {'family': 'Tahoma', 'size': 14}
mpl.rc('font', **font)

if __name__ == '__main__':
    # banner = Image.open(os.path.join('assets', 'inteli_logo_exec.png'))
    # st.image(banner)

    st.write(
"""Use os botões abaixo para baixar os dados. 
             
Ambos os arquivos tem as mesmas variáveis, mas somente o de treino contém as marcações de `default` realizadas.

Seu objetivo será de estudar as variáveis no treino, construir variáveis novas se achar necessário, e realizar o treinamento. 
Com isso, poderá obter os scores (probabilidade de inadimplência) para a base de teste.
""")
    # Carregar o arquivo zip
    zip_path = "zip/Train_test_files_excel.zip"

    # Abrir o arquivo zip em modo binário
    with open(zip_path, "rb") as f:
        zip_data = f.read()

    # Botão de download para o arquivo zip
    st.download_button(
        label="Baixar dados",
        data=zip_data,
        file_name='arquivos.zip',
        mime='application/zip'
    )

    # import webbrowser
    # link = "https://www.dropbox.com/scl/fi/uh9k4iwz3snq7yibnl12v/Train_test_files.zip?rlkey=4d9hfqpq6squwrwb156uhlzn1&dl=0"
    # link = "https://www.dropbox.com/scl/fi/i1ys54vmfsqgfhje3vaf6/Train_test_files_excel.zip?rlkey=h7qx8n09c5hbbih605nw98dpo&st=dczvbwjz&dl=0"
    # st.link_button("Baixar dados", link)

    # if st.button('Baixar arquivos'):
    #     webbrowser.open_new_tab(link)

    # @st.cache_data
    # def load_dfs():
    #     train = pd.read_csv("data/processed/InteliBank_Inadimplencia_de_credito__Treino.csv")
    #     test = pd.read_csv("data/processed/InteliBank_Inadimplencia_de_credito__Avaliacao.csv")
    #     return train, test
    
    # train, test = load_dfs()

    # def to_excel(df):
    #     output = io.BytesIO()
    #     with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    #         df.to_excel(writer, index=False, sheet_name='Sheet1')
    #         writer.close()
    #     processed_data = output.getvalue()
    #     return processed_data
    
    # # Verifica se o DataFrame já foi processado e o botão de download já foi criado
    # if 'download_ready' not in st.session_state or not st.session_state.download_ready:
    #     with st.spinner('Aguarde... Preparando o arquivo para download.'):
    #         # Chamando a função to_excel para obter o Excel em formato de bytes
    #         train_bytes = to_excel(train)
    #         test_bytes = to_excel(test)

    #         st.session_state['train_bytes'] = train_bytes
    #         st.session_state['test_bytes'] = test_bytes
    #         st.session_state['download_ready'] = True

    # # Sempre mostrar o botão de download após o processamento
    # if 'download_ready' in st.session_state and st.session_state.download_ready:
    #     st.download_button(
    #         label="Download Train set",
    #         data=st.session_state.train_bytes,
    #         file_name="base_treino.xlsx",
    #         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    #     )

    #     st.download_button(
    #         label="Download Test set",
    #         data=st.session_state.test_bytes,
    #         file_name="base_test.xlsx",
    #         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    #     )

