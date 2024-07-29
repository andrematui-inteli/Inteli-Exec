import streamlit as st
from PIL import Image
import os

from config.config import set_page_config
set_page_config()

st.image(Image.open(os.path.join('assets', 'inteli_exec.png')))

st.write("# Inteli Exec - Treinamento de modelos de machine learning")

#st.sidebar.success("Selecione .")

st.markdown(
    """
    Esta página contém conteúdos específicos para o Módulo 3:
    
    * Dados para download
    * Interface para treino de modelos de machine learning

    Use o menu à esquerda para acessar as diferentes páginas.
"""
)