import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor

# ---------------- CONFIG ----------------
CSV_PATH = "C:/Users/Suporte/Documents/GitHub/regressao_energia/sarima/consumo.csv"
OUT_DIR = "C:/Users/Suporte/Documents/GitHub/regressao_energia/sarima/sarimax_plots"
TRAIN_UP_TO = 2024        # treinar com dados até este ano (inclusive)
FORECAST_END = 2030
USE_LOG = True
N_LAGS_RF = 3
RF_N_EST = 500
# grid pequeno para SARIMAX
P_RANGE = range(0, 3)
D_RANGE = range(0, 3)
Q_RANGE = range(0, 3)
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)


def prepare_exog(n_periods, offset=0):
    """Retorna DataFrame com uma coluna 't' (0..n_periods-1) iniciando em `offset`."""
    return pd.DataFrame({"t": np.arange(offset, offset + int(n_periods))})


def choose_best_sarimax(y_train, exog_train):
    """Busca (p,d,q) com menor AIC no grid definido por P_RANGE/D_RANGE/Q_RANGE.
    Retorna dicionário {'res': result, 'order': (p,d,q), 'aic': aic} ou None se falhar.
    """
    best = None
    for p in P_RANGE:
        for d in D_RANGE:
            for q in Q_RANGE:
                try:
                    model = SARIMAX(y_train, exog=exog_train, order=(p, d, q),
                                    enforce_stationarity=False, enforce_invertibility=False)
                    res = model.fit(disp=False)
                    if best is None or res.aic < best['aic']:
                        best = {'res': res, 'order': (p, d, q), 'aic': res.aic}
                except Exception:
                    # ignora combinação inválida
                    continue
    return best


def rf_recursive_forecast(train_years, train_y, n_lags, horizon, n_estimators=RF_N_EST,
                          TREND_WEIGHT=0.25, TREND_YEARS=5):
    """
    Previsão recursiva usando RandomForest com blend por uma tendência linear.
    Retorna numpy.array de tamanho `horizon` com previsões (np.nan se não for possível prever).
    """
    horizon = int(horizon)
    if horizon <= 0:
        return np.array([])

    base_year = int(train_years[-1])
    history = list(train_y)

    # construir lags
    df_lag = pd.DataFrame({'Ano': train_years, 'y': train_y})
    for i in range(1, n_lags + 1):
        df_lag[f'lag_{i}'] = df_lag['y'].shift(i)
    df_lag = df_lag.dropna().reset_index(drop=True)

    # dados insuficientes para treinar
    if len(df_lag) < 4:
        return np.array([np.nan] * horizon)

    df_lag['Ano_norm'] = df_lag['Ano'].astype(float) - base_year
    feature_cols = ['Ano_norm'] + [f'lag_{i}' for i in range(1, n_lags + 1)]
    X = df_lag[feature_cols].values
    y_target = df_lag['y'].values

    rf = RandomForestRegressor(n_estimators=int(n_estimators), random_state=42)
    rf.fit(X, y_target)

    # tendência linear pelos últimos TREND_YEARS (ou menos, se não houver)
    trend_years = min(len(train_years), max(2, int(TREND_YEARS)))
    yrs_for_trend = np.array(train_years[-trend_years:], dtype=float)
    vals_for_trend = np.array(train_y[-trend_years:], dtype=float)
    coef = np.polyfit(yrs_for_trend, vals_for_trend, 1)
    trend_poly = np.poly1d(coef)

    preds = []
    last_year = int(train_years[-1])
    for step in range(horizon):
        future_year = last_year + step + 1
        if len(history) < n_lags:
            preds.append(np.nan)
            continue
        last_lags = [history[-i] for i in range(1, n_lags + 1)]
        ano_norm_future = float(future_year - base_year)
        x_input = np.array([ano_norm_future] + last_lags).reshape(1, -1)
        rf_pred = float(rf.predict(x_input)[0])
        trend_pred = float(trend_poly(future_year))
        final_pred = (1.0 - TREND_WEIGHT) * rf_pred + TREND_WEIGHT * trend_pred
        preds.append(final_pred)
        history.append(final_pred)

    return np.array(preds)


def fit_and_forecast_region(df_region):
    """
    Ajusta SARIMAX usando dados até TRAIN_UP_TO e retorna dicionário com fitted e forecast.
    """
    df_region = df_region.sort_values('Ano').reset_index(drop=True)
    if df_region['Ano'].max() < TRAIN_UP_TO:
        return None

    train_df = df_region[df_region['Ano'] <= TRAIN_UP_TO].copy()
    years_train = train_df['Ano'].values
    y_train = train_df['Soma de Consumo'].values.astype(float)

    if len(y_train) < 6:
        return None

    y_train_t = np.log(y_train) if USE_LOG else y_train.copy()

    exog_train = prepare_exog(len(y_train_t), offset=0)

    best = choose_best_sarimax(y_train_t, exog_train)
    if best is None:
        return None
    res = best['res']
    order = best['order']

    last_train_year = int(years_train[-1])
    horizon = int(max(0, FORECAST_END - last_train_year))
    exog_future = prepare_exog(horizon, offset=len(exog_train))

    if horizon > 0:
        pred_t = res.predict(start=len(y_train_t), end=len(y_train_t) + horizon - 1, exog=exog_future)
        pred = np.exp(pred_t) if USE_LOG else pred_t
    else:
        pred = np.array([])

    fitted_t = res.predict(start=0, end=len(y_train_t) - 1, exog=exog_train)
    fitted = np.exp(fitted_t) if USE_LOG else fitted_t

    return {
        'years_train': years_train,
        'y_train': y_train,
        'fitted': fitted,
        'order': order,
        'forecast_years': np.arange(last_train_year + 1, last_train_year + 1 + horizon),
        'forecast': np.array(pred)
    }


# ---------------- Main ----------------

df = pd.read_csv(CSV_PATH)
summary = []

for region in df['Regiao'].unique():
    subset = df[df['Regiao'] == region].sort_values('Ano')
    res = fit_and_forecast_region(subset)
    if res is None:
        print(f"[SKIP] região {region}: dados insuficientes até {TRAIN_UP_TO} ou erro no ajuste.")
        continue

    rf_for = rf_recursive_forecast(
        res['years_train'],
        res['y_train'],
        N_LAGS_RF,
        FORECAST_END - res['years_train'][-1],
        RF_N_EST
    )

    plt.figure(figsize=(10, 5))
    plt.plot(subset['Ano'].values, subset['Soma de Consumo'].values, marker='o', label='Real (histórico)')

    mask_after_2020 = res['years_train'] > 2020
    if mask_after_2020.any():
        plt.plot(res['years_train'][mask_after_2020], res['fitted'][mask_after_2020], linestyle='-', label='SARIMAX (fitted)')

    if len(res['forecast']) > 0:
        plt.plot(res['forecast_years'], res['forecast'], marker='s', linestyle='-.', label='SARIMAX (forecast)')
    if len(rf_for) > 0:
        plt.plot(res['forecast_years'], rf_for, marker='x', linestyle='--', label='RF (forecast)')

    plt.title(f"{region} — treino até {res['years_train'][-1]} — SARIMA order {res['order']}")
    plt.xlabel('Ano')
    plt.ylabel('Consumo (MWh)')
    plt.grid(True)
    plt.legend()
    out_file = os.path.join(OUT_DIR, f"{region.replace(' ', '_')}_forecast.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()

    summary.append({'Regiao': region, 'SARIMAX_order': res['order'], 'Train_up_to': int(res['years_train'][-1])})

print("Pronto. Plots salvos em:", OUT_DIR)
