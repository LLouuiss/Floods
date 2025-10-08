import pandas as pd
import scipy as sp

file_path = 'DGH_5962_Q_DayMean.xlsx'

def load_data():
    df = pd.read_excel(file_path, skiprows=9)
    df.rename(columns={"#Timestamp": "Date", "Value": "Q"}, inplace=True)
    return df

def lognormal_moments(data):
    m_Q = data['Q'].mean()
    sigma2_Q = data['Q'].var()
    m_lnQ = sp.log(m_Q) - 0.5 * sigma2_Q
    sigma2_lnQ = sp.ln(sigma2_Q / m_Q**2 + 1)
    return m_lnQ, sigma2_lnQ

def lognormal_max_likelihood(data):
    m_lnQ = sp.log(data['Q']).mean()
    sigma2_lnQ = sp.log(data['Q']).var()  #BIAISE
    return m_lnQ, sigma2_lnQ

