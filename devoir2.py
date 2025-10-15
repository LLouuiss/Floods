import pandas as pd
import scipy.stats as stat
import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.copy_on_write = True


file_path = 'DGH_5962_Q_DayMean.xlsx'

verbose = True
export_file = True
display_plots = False
save_plots = True


##############  TODO  ################
#                                    #
# - Kolmogorov-Smirnov test          #
# - Chi-squared 4 class test         #
# - Verbose tests                    #
# - Improving beauty of export file  #
# - Improving beauty of plots        #
#                                    #
######################################


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
    datas = {"Calendar 2020": calen_20, "Calendar 2025": calen_25, "Hydrological 2020": hydro_20}
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

    plt.figure(figsize=(8,6))
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
    if save_plots:
        plt.savefig("devoir2_plots/" + title.replace(" ", "_").replace(":", "") + ".pdf", dpi=300)
    if display_plots:
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
hydro_25.drop(columns=["HydroYear"], inplace=True)

# Sorting by increasing discharge
for df in [calen_25, calen_20, hydro_20, hydro_25]:
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
params_hydro_25 = {"n": len(hydro_25),
                   "ln_m": lognormal_moments_params(hydro_25),
                   "ln_ml": lognormal_max_likelihood_params(hydro_25),
                   "gumbel": gumbel_params(hydro_25)}

# Computing CDFs & estimators
for df, params in [(calen_20, params_calen_20), (calen_25, params_calen_25), (hydro_20, params_hydro_20), (hydro_25, params_hydro_25)]:
    df["Lognormal: Moments"] = stat.lognorm.cdf(df["X_i"], s=np.sqrt(params["ln_m"][1]), scale=np.exp(params["ln_m"][0]))
    df["Lognormal: MaxLikelihood"] = stat.lognorm.cdf(df["X_i"], s=np.sqrt(params["ln_ml"][1]), scale=np.exp(params["ln_ml"][0]))
    df["Gumbel"] = stat.gumbel_r.cdf(df["X_i"], loc=params["gumbel"][0], scale=params["gumbel"][1])
    df["Weibull"] = (df.index + 0.5) / params["n"]
    df["i_over_n"] = (df.index + 1) / params["n"]

# Kolmogorov-Smirnov test
if verbose:
    print("---------------------------------------------")
    print("---------- Kolmogorov-Smirnov test ----------")
    print("---------------------------------------------")
# Distance between empirical and theoretical CDF
for df in [calen_20, hydro_20]:
    df["D_ln_m"] = np.abs(df["i_over_n"] - df["Lognormal: Moments"])
    df["D_ln_ml"] = np.abs(df["i_over_n"] - df["Lognormal: MaxLikelihood"])
    df["D_gumbel"] = np.abs(df["i_over_n"] - df["Gumbel"])

# Find maximun distance for each distribution and dataset
for df, name, params in [(calen_20, "Calendar 2020",params_calen_20), (hydro_20, "Hydrological 2020",params_hydro_20)]:
    D_ln_m_max = df["D_ln_m"].max()
    D_ln_ml_max = df["D_ln_ml"].max()
    D_gumbel_max = df["D_gumbel"].max()
    C_10 = 1.22/np.sqrt(params["n"])  # alpha = 0.1
    C_5= 1.36/np.sqrt(params["n"])   # alpha = 0.05
    C_1 = 1.63/np.sqrt(params["n"])   # alpha = 0.01
    if verbose:
        print(f"Kolmogorov-Smirnov test - {name}:")
        for dist, D_max in [("Lognormal: Moments", D_ln_m_max), ("Lognormal: MaxLikelihood", D_ln_ml_max), ("Gumbel", D_gumbel_max)]:
            c= None
            if D_max < C_10:
                result = "Accept H0 at alpha = 0.1"
                c = C_10
                print(f"  {dist}: D_max = {D_max:.4f} < {c}, {result}")
            #elif D_max < C_5:
            #    result = "Accept H0 at alpha = 0.05"
            #    c = C_5
            #elif D_max < C_1:
            #    result = "Accept H0 at alpha = 0.01"
            #    c = C_1
            else:
                result = "Reject H0 at alpha = 0.1"
                c= C_10
                print(f"  {dist}: D_max = {D_max:.4f} < {c}, {result}")
            print()


# Chi-squared 4 and 5 class test
if verbose:
    print("--------------------------------------")
    print("---------- Chi-squared test ----------")
    print("--------------------------------------")
for df, name, params in [(calen_20, "Calendar 2020",params_calen_20), (hydro_20, "Hydrological 2020",params_hydro_20)]:
    for dist in ["Lognormal: Moments", "Lognormal: MaxLikelihood", "Gumbel"]:
        for k in [4, 5]:
            observed, bins = np.histogram(df["X_i"], bins=k)
            expected_probs = [stat.lognorm.cdf(bins[i+1], s=np.sqrt(params["ln_m"][1]), scale=np.exp(params["ln_m"][0])) - stat.lognorm.cdf(bins[i], s=np.sqrt(params["ln_m"][1]), scale=np.exp(params["ln_m"][0])) for i in range(k)] if dist == "Lognormal: Moments" else \
                            [stat.lognorm.cdf(bins[i+1], s=np.sqrt(params["ln_ml"][1]), scale=np.exp(params["ln_ml"][0])) - stat.lognorm.cdf(bins[i], s=np.sqrt(params["ln_ml"][1]), scale=np.exp(params["ln_ml"][0])) for i in range(k)] if dist == "Lognormal: MaxLikelihood" else \
                            [stat.gumbel_r.cdf(bins[i+1], loc=params["gumbel"][0], scale=params["gumbel"][1]) - stat.gumbel_r.cdf(bins[i], loc=params["gumbel"][0], scale=params["gumbel"][1]) for i in range(k)]
            expected = np.array(expected_probs) * params["n"]
            chi2_stat = ((observed - expected)**2 / expected).sum()
            dof = k - 1 - 2  # number of classes - 1 - number of estimated parameters
            critical_value_5 = stat.chi2.ppf(0.95, dof)
            critical_value_1 = stat.chi2.ppf(0.99, dof)
            critical_value_10 = stat.chi2.ppf(0.90, dof)
            if verbose:
                c = None
                print(f"Chi-squared test {k} classes - {name} - {dist}:")
                if chi2_stat < critical_value_5:
                    result = "Accept H0 at alpha = 0.05"
                    c = critical_value_5
                    print(f"  Chi2 stat = {chi2_stat:.4f} < {c}, dof = {dof}, {result}")
                #elif chi2_stat < critical_value_1:
                #    result = "Accept H0 at alpha = 0.01"
                #    c = critical_value_1
                #elif chi2_stat < critical_value_10:
                #    result = "Accept H0 at alpha = 0.1"
                #    c = critical_value_10
                else:
                    result = "Reject H0 at alpha = 0.05"
                    c = critical_value_5
                    print(f"  Chi2 stat = {chi2_stat:.4f} < {c}, dof = {dof}, {result}")
                #print(f"  Chi2 stat = {chi2_stat:.4f} < {c}, dof = {dof}, {result}")
                print()
        


# Return periods
Q_2021 = calen_25["X_i"].max()

Q_calen_20 = pd.DataFrame(columns=["Tr", "Q_ln_m", "Q_ln_ml", "Q_gumbel"])
Q_calen_25 = pd.DataFrame(columns=["Tr", "Q_ln_m", "Q_ln_ml", "Q_gumbel"])
Q_hydro_20 = pd.DataFrame(columns=["Tr", "Q_ln_m", "Q_ln_ml", "Q_gumbel"])

for df, params, name in [(Q_calen_20, params_calen_20, "Calendar 2020"), (Q_calen_25, params_calen_25, "Calendar 2025"), (Q_hydro_20, params_hydro_20, "Hydrological 2020")]:
    for Tr in [10, 100, 1000, 10000]:
        p = 1 - 1/Tr
        Q_ln_m = stat.lognorm.ppf(p, s=np.sqrt(params["ln_m"][1]), scale=np.exp(params["ln_m"][0]))
        Q_ln_ml = stat.lognorm.ppf(p, s=np.sqrt(params["ln_ml"][1]), scale=np.exp(params["ln_ml"][0]))
        Q_gumbel = stat.gumbel_r.ppf(p, loc=params["gumbel"][0], scale=params["gumbel"][1])
        df.loc[len(df)] = [Tr, Q_ln_m, Q_ln_ml, Q_gumbel]
    Tr_2021_ln_m = 1 / (1 - stat.lognorm.cdf(Q_2021, s=np.sqrt(params["ln_m"][1]), scale=np.exp(params["ln_m"][0])))
    Tr_2021_ln_ml = 1 / (1 - stat.lognorm.cdf(Q_2021, s=np.sqrt(params["ln_ml"][1]), scale=np.exp(params["ln_ml"][0])))
    Tr_2021_gumbel = 1 / (1 - stat.gumbel_r.cdf(Q_2021, loc=params["gumbel"][0], scale=params["gumbel"][1]))
    if (name == "Calendar 2020" or name == "Calendar 2025"):
        df.loc[len(df)] = ["T_r for Q_2021", Tr_2021_ln_m, Tr_2021_ln_ml, Tr_2021_gumbel]


# Exporting results to a file
if export_file:
    with pd.ExcelWriter('devoir2_results.xlsx') as writer:
        params_df = pd.DataFrame({
            "Dataset": ["Calendar 2020", "Calendar 2025", "Hydrological 2020"],
            "n": [params_calen_20["n"], params_calen_25["n"], params_hydro_20["n"]],
            "ln_m_sigma": [np.sqrt(params_calen_20["ln_m"][1]), np.sqrt(params_calen_25["ln_m"][1]), np.sqrt(params_hydro_20["ln_m"][1])],
            "ln_m_m": [params_calen_20["ln_m"][0], params_calen_25["ln_m"][0], params_hydro_20["ln_m"][0]],
            "ln_ml_sigma": [np.sqrt(params_calen_20["ln_ml"][1]), np.sqrt(params_calen_25["ln_ml"][1]), np.sqrt(params_hydro_20["ln_ml"][1])],
            "ln_ml_m": [params_calen_20["ln_ml"][0], params_calen_25["ln_ml"][0], params_hydro_20["ln_ml"][0]],
            "gumbel_alpha": [params_calen_20["gumbel"][1], params_calen_25["gumbel"][1], params_hydro_20["gumbel"][1]],
            "gumbel_u": [params_calen_20["gumbel"][0], params_calen_25["gumbel"][0], params_hydro_20["gumbel"][0]],
        })
        params_df.to_excel(writer, sheet_name='Parameters', index=False)
        calen_20.to_excel(writer, sheet_name='Calendar 2020', index=False)
        calen_25.to_excel(writer, sheet_name='Calendar 2025', index=False)
        hydro_20.to_excel(writer, sheet_name='Hydrological 2020', index=False)
        Q_calen_20.to_excel(writer, sheet_name='Return periods Calen 20', index=False)
        Q_calen_25.to_excel(writer, sheet_name='Return periods Calen 25', index=False)
        Q_hydro_20.to_excel(writer, sheet_name='Return periods Hydro 20', index=False)

# Plotting
if display_plots or save_plots:
    plotter("Hydrological", "2020", "Lognormal: Moments")
    plotter("Hydrological", "2020", "Lognormal: MaxLikelihood")
    plotter("Hydrological", "2020", "Gumbel")

    plotter("Calendar", "2020", "Lognormal: Moments")
    plotter("Calendar", "2020", "Lognormal: MaxLikelihood")
    plotter("Calendar", "2020", "Gumbel")

    plotter("Calendar", "2025", "Lognormal: Moments")
    plotter("Calendar", "2025", "Lognormal: MaxLikelihood")
    plotter("Calendar", "2025", "Gumbel")
    
# Verbose output
if verbose:
    print("--------------------------------")
    print("---------- Parameters ----------")
    print("--------------------------------")
    print("Calendar 2020:")
    print(f"n: {params_calen_20['n']}")
    print(f"ln_m: sigma = {np.sqrt(params_calen_20['ln_m'][1]):.4f}, m = {params_calen_20['ln_m'][0]:.4f}")
    print(f"ln_ml: sigma = {np.sqrt(params_calen_20['ln_ml'][1]):.4f}, m = {params_calen_20['ln_ml'][0]:.4f}")
    print(f"gumbel: alpha = {params_calen_20['gumbel'][1]:.4f}, u = {params_calen_20['gumbel'][0]:.4f}")
    print()
    print("Calendar 2025:")
    print(f"n: {params_calen_25['n']}")
    print(f"ln_m: sigma = {np.sqrt(params_calen_25['ln_m'][1]):.4f}, m = {params_calen_25['ln_m'][0]:.4f}")
    print(f"ln_ml: sigma = {np.sqrt(params_calen_25['ln_ml'][1]):.4f}, m = {params_calen_25['ln_ml'][0]:.4f}")
    print(f"gumbel: alpha = {params_calen_25['gumbel'][1]:.4f}, u = {params_calen_25['gumbel'][0]:.4f}")
    print()
    print("Hydrological 2020:")
    print(f"n: {params_hydro_20['n']}")
    print(f"ln_m: sigma = {np.sqrt(params_hydro_20['ln_m'][1]):.4f}, m = {params_hydro_20['ln_m'][0]:.4f}")
    print(f"ln_ml: sigma = {np.sqrt(params_hydro_20['ln_ml'][1]):.4f}, m = {params_hydro_20['ln_ml'][0]:.4f}")
    print(f"gumbel: alpha = {params_hydro_20['gumbel'][1]:.4f}, u = {params_hydro_20['gumbel'][0]:.4f}")
    print()
    print("Max discharge 2021 (Calendar 2025):", Q_2021)
    print()
    print("--------------------------------")
    print("------- Return periods ---------")
    print("--------------------------------")
    print("Calendar 2020:")
    print(Q_calen_20)
    print()
    print("Calendar 2025:")
    print(Q_calen_25)
    print()
    print("Hydro 2020:")
    print(Q_hydro_20)
    print()
    