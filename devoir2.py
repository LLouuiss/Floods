import pandas as pd
import scipy.stats as stat
import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.copy_on_write = True


file_path = 'DGH_5962_Q_DayMean.xlsx'

plots = True


#######################
####  Definitions  ####
#######################

def lognormal_moments_params(data):
    sigma2_lnQ = np.log(stat.variation(data["X_i"])**2 + 1)
    m_lnY = np.log(data["X_i"].mean()) - 0.5 * sigma2_lnQ
    return (m_lnY, sigma2_lnQ)

def lognormal_max_likelihood_params(data):
    m_lnY = np.log(data["X_i"]).mean()
    sigma2_lnQ = np.log(data["X_i"]).var(ddof=1)  #non biaisé
    return (m_lnY, sigma2_lnQ)
    
def gumbel_params(data):
    alpha = np.sqrt(6) * data["X_i"].std(ddof=1) / np.pi
    u = data["X_i"].mean() - 0.5772 / alpha
    return (u, alpha)

def plotter(typeyear, endyear, method):
    datas = {"Calendar 2020": calen_20, "Calendar 2025": calen_25, "Hydro 2020": hydro_20}
    tickers = {"Lognormal: Moments": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999],
               "Lognormal: MaxLikelihood": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999],
               "Gumbel": [0.001, 0.01, 0.1, 0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.998, 0.999]
               }
    mappers = {"Lognormal: Moments": lambda t: stat.norm.ppf(t),
               "Lognormal: MaxLikelihood": lambda t: stat.norm.ppf(t),
               "Gumbel": lambda t: -np.log(-np.log(t))
               }
    data = datas[typeyear + " " + endyear]
    ticks_label = tickers[method]
    mapper = mappers[method]
    ticks = mapper(ticks_label)

    plt.plot(data["X_i"], mapper(data["Weibull"]), marker='o', markersize=5, linestyle='', label='Empirical CDF (Weibull estimator)')
    plt.plot(data["X_i"], mapper(data[method]), label='Lognormal CDF', color='red')
    plt.yticks(ticks, ticks_label)
    plt.xlabel('Q [m³/s]')
    plt.ylabel('CDF')
    if method != "Gumbel": plt.xscale('log')
    plt.legend()
    plt.grid()
    title = method + " - 1979-" + endyear + " - " + typeyear
    plt.title(title)
    plt.show()


##############
###  Main  ###
##############

df = pd.read_excel(file_path, skiprows=9, parse_dates=[0])
df.rename(columns={"#Timestamp": "Date", "Value": "X_i"}, inplace=True)
df["CalendarYear"] = df["Date"].dt.year
df["HydroYear"] = df["Date"].apply(lambda d: d.year + 1 if d.month >= 9 else d.year)
df.drop(columns=["Date"], inplace=True)

# Extracting annual maxima
calen_25 = df.groupby("CalendarYear")["X_i"].max().reset_index()
hydro_25 = df.groupby("HydroYear")["X_i"].max().reset_index()
calen_20 = calen_25[calen_25["CalendarYear"] <= 2020]
hydro_20 = hydro_25[hydro_25["HydroYear"] <= 2020]

# Dropping year columns
calen_25.drop(columns=["CalendarYear"], inplace=True)
calen_20.drop(columns=["CalendarYear"], inplace=True)
hydro_20.drop(columns=["HydroYear"], inplace=True)

# Sorting by increasing discharge
for df in [calen_25, calen_20, hydro_20]:
    df.sort_values(by="X_i", inplace=True)
    df.reset_index(drop = True, inplace=True)

# Computing parameters
params_calen_20 = {"n": len(calen_20), 
                   "ln_m": lognormal_moments_params(calen_20),
                   "ln_ml": lognormal_max_likelihood_params(calen_20),
                   "gumbel": gumbel_params(calen_20)}
params_calen_25 = {"n": len(calen_25),
                   "ln_m": lognormal_moments_params(calen_25),
                   "ln_ml": lognormal_max_likelihood_params(calen_25),
                   "gumbel": gumbel_params(calen_25)}
params_hydro_20 = {"n": len(hydro_20),
                   "ln_m": lognormal_moments_params(hydro_20),
                   "ln_ml": lognormal_max_likelihood_params(hydro_20),
                   "gumbel": gumbel_params(hydro_20)}

# Computing CDFs & estimators
for df, params in [(calen_20, params_calen_20), (calen_25, params_calen_25), (hydro_20, params_hydro_20)]:
    df["Lognormal: Moments"] = stat.lognorm.cdf(df["X_i"], s=np.sqrt(params["ln_m"][1]), scale=np.exp(params["ln_m"][0]))
    df["Lognormal: MaxLikelihood"] = stat.lognorm.cdf(df["X_i"], s=np.sqrt(params["ln_ml"][1]), scale=np.exp(params["ln_ml"][0]))
    df["Gumbel"] = stat.gumbel_r.cdf(df["X_i"], loc=params["gumbel"][0], scale=params["gumbel"][1])
    df["Weibull"] = (df.index + 0.5) / params["n"]
    df["i_over_n"] = (df.index + 1) / params["n"]

# Plotting
if plots:
    plotter("Hydro", "2020", "Lognormal: Moments")
    plotter("Hydro", "2020", "Lognormal: MaxLikelihood")
    plotter("Hydro", "2020", "Gumbel")

    plotter("Calendar", "2020", "Lognormal: Moments")
    plotter("Calendar", "2020", "Lognormal: MaxLikelihood")
    plotter("Calendar", "2020", "Gumbel")

    plotter("Calendar", "2025", "Lognormal: Moments")
    plotter("Calendar", "2025", "Lognormal: MaxLikelihood")
    plotter("Calendar", "2025", "Gumbel")