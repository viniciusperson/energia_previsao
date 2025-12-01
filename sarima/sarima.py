import streamlit as st
import pandas as pd
import numpy as np
import math
import io
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Previsão de Consumo — RF vs SARIMA", layout="wide")

# ---------- utilitários ----------
def rmse(a, b):
    return math.sqrt(mean_squared_error(a, b))

@st.cache_data(show_spinner=False)
def load_dataframe_from_file(uploaded_file):
    """Lê um arquivo enviado via upload e tenta detectar separador automaticamente."""
    if uploaded_file is None:
        return None
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';')
        except Exception:
            uploaded_file.seek(0)
            text = uploaded_file.read().decode('utf-8', errors='ignore')
            df = pd.read_csv(io.StringIO(text), sep=None, engine='python')
    df = df.rename(columns=lambda x: x.strip())
    # garantir tipos
    df['Ano'] = df['Ano'].astype(int)
    df['Soma de Consumo'] = df['Soma de Consumo'].astype(float)
    return df

def rf_recursive_forecast(train_years, train_y, n_lags, horizon,
                          n_estimators=500, trend_weight=0.25, trend_years=5):
    import numpy as _np
    history = list(train_y)
    df_lag = pd.DataFrame({'Ano': train_years, 'y': train_y})
    for i in range(1, n_lags+1):
        df_lag[f'lag_{i}'] = df_lag['y'].shift(i)
    df_lag = df_lag.dropna().reset_index(drop=True)
    if len(df_lag) < max(4, n_lags + 1):
        return _np.array([_np.nan] * int(horizon))
    base_year = int(train_years[-1])
    df_lag['Ano_norm'] = df_lag['Ano'] - base_year
    feature_cols = ['Ano_norm'] + [f'lag_{i}' for i in range(1, n_lags+1)]
    X = df_lag[feature_cols].values
    y_t = df_lag['y'].values
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42).fit(X, y_t)
    trend_years = min(len(train_years), max(2, trend_years))
    yrs = np.array(train_years[-trend_years:], dtype=float)
    vals = np.array(train_y[-trend_years:], dtype=float)
    coef = np.polyfit(yrs, vals, 1)
    poly = np.poly1d(coef)
    preds = []
    last_year = int(train_years[-1])
    for step in range(int(horizon)):
        future_year = last_year + step + 1
        if len(history) < n_lags:
            preds.append(_np.nan)
            continue
        last_lags = [history[-i] for i in range(1, n_lags+1)]
        ano_norm = float(future_year - base_year)
        x_input = np.array([ano_norm] + last_lags).reshape(1, -1)
        rf_pred = rf.predict(x_input)[0]
        trend_pred = float(poly(future_year))
        final_pred = (1.0 - trend_weight) * rf_pred + trend_weight * trend_pred
        preds.append(final_pred)
        history.append(final_pred)
    return np.array(preds)

# ---------- interface ----------
st.title("Previsão de Consumo de Energia — Random Forest vs SARIMA")
st.markdown("Carregue seu arquivo CSV com as colunas: Regiao, Ano, Soma de Consumo. Ajuste parâmetros e execute a comparação.")

with st.sidebar:
    st.header("Parâmetros")
    uploaded_file = st.file_uploader("Envie o arquivo CSV", type=["csv", "txt"], accept_multiple_files=False)
    test_years = st.number_input("Últimos N anos para teste", value=3, min_value=1, max_value=10)
    n_lags = st.number_input("Número de lags (RF)", value=3, min_value=1, max_value=10)
    rf_n_estimators = st.number_input("RF n_estimators", value=500, min_value=10, max_value=1000000)
    trend_weight = st.slider("Peso da tendência no RF (0..1)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
    trend_years = st.number_input("Anos para ajustar tendência (RF)", value=5, min_value=2, max_value=20)
    p_max = st.number_input("ARIMA p max", value=3, min_value=0, max_value=6)
    d_max = st.number_input("ARIMA d max", value=2, min_value=0, max_value=3)
    q_max = st.number_input("ARIMA q max", value=3, min_value=0, max_value=6)
    forecast_end = st.number_input("Ano final do forecast (extrapolação)", value=2030, min_value=2025, max_value=2050)
    run_button = st.button("Executar comparação")

if uploaded_file is None:
    st.info("Faça upload do arquivo CSV para iniciar. O CSV deve conter as colunas: Regiao, Ano, Soma de Consumo.")
    st.stop()

df = load_dataframe_from_file(uploaded_file)
if df is None or df.empty:
    st.error("Não foi possível ler o arquivo. Verifique o formato e o separador.")
    st.stop()

regions = df['Regiao'].unique().tolist()
st.sidebar.markdown("Regiões detectadas:")
for r in regions:
    st.sidebar.write(f"- {r}")

sel_regions = st.multiselect("Selecione regiões", options=regions, default=regions)

if run_button:
    results = []
    per_region_csv = {}
    per_region_img = {}
    with st.spinner("Rodando modelos..."):
        for reg in sel_regions:
            df_reg = df[df['Regiao'] == reg].sort_values('Ano').reset_index(drop=True)
            years = df_reg['Ano'].values
            values = df_reg['Soma de Consumo'].values.astype(float)
            if len(values) <= test_years + 3:
                st.warning(f"{reg}: dados insuficientes, pulando.")
                continue
            split_idx = len(values) - test_years
            years_train = years[:split_idx]
            y_train = values[:split_idx]
            years_test = years[split_idx:]
            y_test = values[split_idx:]
            horizon_test = len(y_test)
            last_train_year = int(years_train[-1])
            horizon_future = max(0, int(forecast_end - last_train_year))
      
            rf_pred_test = rf_recursive_forecast(years_train, y_train, n_lags, horizon_test,
                                                 n_estimators=rf_n_estimators, trend_weight=trend_weight, trend_years=trend_years)
            rf_pred_future = rf_recursive_forecast(years_train, y_train, n_lags, horizon_future,
                                                   n_estimators=rf_n_estimators, trend_weight=trend_weight, trend_years=trend_years)

            ar_best = None
            for p in range(0, p_max+1):
                for d in range(0, d_max+1):
                    for q in range(0, q_max+1):
                        try:
                            model = ARIMA(y_train, order=(p,d,q)).fit()
                            pred_test = np.array(model.forecast(steps=horizon_test))
                            r = rmse(y_test, pred_test)
                            if ar_best is None or r < ar_best['rmse']:
                                ar_best = {'order': (p,d,q), 'model': model, 'pred_test': pred_test, 'rmse': r, 'mae': mean_absolute_error(y_test, pred_test)}
                        except Exception:
                            continue
            if ar_best is None:
                st.warning(f"{reg}: ARIMA não convergiu em nenhum (p,d,q), pulando.")
                continue
            sarima_pred_test = ar_best['pred_test']
            sarima_pred_future = np.array(ar_best['model'].forecast(steps=horizon_future)) if horizon_future > 0 else np.array([])
           
            rf_rmse = rmse(y_test, rf_pred_test)
            rf_mae = mean_absolute_error(y_test, rf_pred_test)
            sar_rmse = ar_best['rmse']
            sar_mae = ar_best['mae']
            mean_test = np.mean(y_test) if np.mean(y_test) != 0 else 1.0
            results.append({
                'Regiao': reg,
                'Test_years': ','.join(map(str, years_test.tolist())),
                'RF_RMSE': rf_rmse,
                'RF_MAE': rf_mae,
                'SARIMA_RMSE': sar_rmse,
                'SARIMA_MAE': sar_mae,
                'RF_pct_mean': rf_rmse / mean_test * 100,
                'SARIMA_pct_mean': sar_rmse / mean_test * 100
            })
           
            rows = []
            for i, yv in enumerate(years_test):
                rows.append({'Ano': int(yv), 'Real': float(y_test[i]), 'RF_pred': float(rf_pred_test[i]), 'SARIMA_pred': float(sarima_pred_test[i])})
            for j in range(len(sarima_pred_future)):
                year_f = last_train_year + 1 + j
                rf_val = float(rf_pred_future[j]) if len(rf_pred_future) > j else np.nan
                sar_val = float(sarima_pred_future[j]) if len(sarima_pred_future) > j else np.nan
                rows.append({'Ano': int(year_f), 'Real': np.nan, 'RF_pred': rf_val, 'SARIMA_pred': sar_val})
            df_out = pd.DataFrame(rows)
            buf = io.StringIO()
            df_out.to_csv(buf, index=False, sep=';')
            per_region_csv[reg] = buf.getvalue()
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(years, values, marker='o', label='Real (histórico)')
            ax.plot(years_test, rf_pred_test, marker='x', linestyle='--', label='RF (teste)')
            ax.plot(years_test, sarima_pred_test, marker='s', linestyle='-.', label='SARIMA (teste)')
            if horizon_future > 0:
                fut_years = np.arange(last_train_year + 1, last_train_year + 1 + horizon_future)
                ax.plot(fut_years, rf_pred_future, marker='x', linestyle='--', label='RF (forecast)')
                ax.plot(fut_years, sarima_pred_future, marker='s', linestyle='-.', label='SARIMA (forecast)')
            ax.set_title(reg)
            ax.set_xlabel('Ano')
            ax.set_ylabel('Consumo (MWh)')
            ax.grid(True)
            ax.legend()
            buf_img = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf_img, format='png', dpi=150)
            buf_img.seek(0)
            per_region_img[reg] = buf_img.read()
            plt.close(fig)
    
    if results:
        df_res = pd.DataFrame(results)
        display = df_res.copy()
        for col in ['RF_RMSE','RF_MAE','SARIMA_RMSE','SARIMA_MAE']:
            display[col] = display[col].apply(lambda x: f"{x:,.0f}")
        for col in ['RF_pct_mean','SARIMA_pct_mean']:
            display[col] = display[col].apply(lambda x: f"{x:.2f}%")
        st.subheader("Resumo de desempenho por região (período de teste)")
        st.dataframe(display.set_index('Regiao'))
        st.download_button("Baixar tabela completa (CSV)", df_res.to_csv(index=False, sep=';'), file_name="comparison_summary.csv", mime="text/csv")
        st.subheader("Plots e dados por região")
        cols = st.columns(2)
        i = 0
        for reg in df_res['Regiao'].tolist():
            col = cols[i % 2]
            with col:
                st.image(per_region_img[reg], use_container_width=True, caption=f"{reg}")
                st.download_button(f"Baixar CSV {reg}", per_region_csv[reg], file_name=f"{reg}_preds.csv", mime="text/csv")
            i += 1
    else:
        st.warning("Nenhum resultado gerado.")
