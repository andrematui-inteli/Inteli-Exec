"""
Build train and test datasets for students to work on
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_raw_credit_data():

    NTD_BRL = 0.072054

    df = pd.read_csv('data/raw/UCI_Credit_Card.csv', index_col='ID')
    df = df.rename({'default.payment.next.month': 'default'}, axis=1)
    df = df.rename(dict(
        ID='id',
        LIMIT_BAL= 'limite_credito', 
        SEX='sexo',
        EDUCATION='educacao',
        MARRIAGE='estado_civil',
        AGE='idade',
        PAY_0='status_pagamento_mes_09',
        PAY_2='status_pagamento_mes_08',
        PAY_3='status_pagamento_mes_07',
        PAY_4='status_pagamento_mes_06',
        PAY_5='status_pagamento_mes_05',
        PAY_6='status_pagamento_mes_04',
        BILL_AMT1='fatura_mes_09',
        BILL_AMT2='fatura_mes_08',
        BILL_AMT3='fatura_mes_07',
        BILL_AMT4='fatura_mes_06',
        BILL_AMT5='fatura_mes_05',
        BILL_AMT6='fatura_mes_04',
        PAY_AMT1='pago_mes_09',
        PAY_AMT2='pago_mes_08',
        PAY_AMT3='pago_mes_07',
        PAY_AMT4='pago_mes_06',
        PAY_AMT5='pago_mes_05',
        PAY_AMT6='pago_mes_04',
    ),axis=1)

    df['sexo'] = df['sexo'].map({1: 'masculino', 2:'feminino'})
    df['educacao'] = df['educacao'].map({0: 'outros/desconhecido', 1:'pos_graduacao', 2:'graduacao', 3:'ensino_medio', 4: 'outros/desconhecido', 5: 'outros/desconhecido', 6: 'outros/desconhecido'})
    df['estado_civil'] = df['estado_civil'].map({1: 'casado', 2: 'solteiro', 3: 'outros/desconhecido', 0: 'outros/desconhecido'})

    for col in df.columns:
        if ('fatura_mes' in col) or ('pago_mes' in col) or (col == 'limite_credito'):
            df[col] = (df[col] * NTD_BRL).round(2)
    
    assert df.isna().sum().sum() == 0

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=123, stratify=df['default'])

    df_test_copy = df_test.copy()
    df_test_copy['default'] = np.nan
    df_test_copy['score'] = np.nan

    print("Saving...")
    df_train.to_csv('data/processed/InteliBank_Inadimplencia_de_credito__Treino.csv')
    df_test_copy.to_csv('data/processed/InteliBank_Inadimplencia_de_credito__Avaliacao.csv')
    df_test.to_csv('data/processed/InteliBank_Inadimplencia_de_credito__Gabarito.csv')
    print("Saved to data/processed")
    return df_train, df_test_copy

if __name__ == "__main__":
    preprocess_raw_credit_data()