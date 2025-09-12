import streamlit as st
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

def set_page_config():
    icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets', 'inteli_logo.png'))
    st.set_page_config(
        page_title="Inteli Exec",
        page_icon=Image.open(icon_path),
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get help": "mailto:jordan.piva@inteli.edu.br",
            "About": """Página construída para curso de dados do Inteli (2023)"""
        }
    )
    plt.style.use("dark_background")
    mpl.rcParams['figure.dpi'] = 210
    font = {'family': 'Tahoma', 'size': 14}
    mpl.rc('font', **font)
