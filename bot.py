import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import requests
import os
import warnings
warnings.filterwarnings('ignore')

# ⚙️ CONFIGURARE TELEGRAM (Preluate securizat din mediu sau setate direct)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8869703770:AAGB7JCox6EZcBQousFvbAWznuiC4tPfRLg")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "526360640")

def trimite_pe_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"Erore trimitere Telegram: {e}")

def ia_fear_greed_reala():
    try:
        url = "https://api.alternative.me/fng/?limit=0"
        response = requests.get(url, timeout=10)
        data = response.json()
        rows = [{'Timestamp': pd.to_datetime(int(i['timestamp']), unit='s'), 'Real_Fear_Greed': int(i['value'])} for i in data['data']]
        return pd.DataFrame(rows).set_index('Timestamp').sort_index()
    except:
        return None

def calculeaza_adx(df, period=14):
    alpha = 1 / period
    df['TR'] = np.maximum(df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift(1)))
    df['H-PH'] = df['High'] - df['High'].shift(1)
    df['PL-L'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['H-PH'] > df['PL-L']) & (df['H-PH'] > 0), df['H-PH'], 0.0)
    df['-DM'] = np.where((df['PL-L'] > df['H-PH']) & (df['PL-L'] > 0), df['PL-L'], 0.0)

    tr_smooth = df['TR'].ewm(alpha=alpha, adjust=False).mean()
    plus_di_smooth = df['+DM'].ewm(alpha=alpha, adjust=False).mean()
    minus_di_smooth = df['-DM'].ewm(alpha=alpha, adjust=False).mean()

    df['+DI'] = 100 * (plus_di_smooth / (tr_smooth + 1e-9))
    df['-DI'] = 100 * (minus_di_smooth / (tr_smooth + 1e-9))

    dx = 100 * np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'] + 1e-9)
    return dx.ewm(alpha=alpha, adjust=False).mean()

def ruleaza_verificarea_live():
    print("⏳ Se descarcă datele recente pentru analiza 4H...")
    
    # 1. Date BTC
    df = yf.download('BTC-USD', period='90d', interval='1h', progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df_4h = df.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()

    # 2. Date Macro & Sentiment
    macro_df = yf.download('^TNX', period='90d', interval='1d', progress=False)
    if isinstance(macro_df.columns, pd.MultiIndex): macro_df.columns = macro_df.columns.get_level_values(0)
    macro_df = macro_df[['Close']].rename(columns={'Close': 'Macro_Yield'})

    fng_df = ia_fear_greed_reala()
    if df_4h.index.tz is not None: df_4h.index = df_4h.index.tz_convert(None)
    if macro_df.index.tz is not None: macro_df.index = macro_df.index.tz_convert(None)
    if fng_df is not None and fng_df.index.tz is not None: fng_df.index = fng_df.index.tz_convert(None)

    df_4h = df_4h.join(macro_df, how='left').ffill().bfill()
    if fng_df is not None: df_4h = df_4h.join(fng_df, how='left').ffill().bfill()
    else: df_4h['Real_Fear_Greed'] = 50

    # 3. Indicatori Tehnici
    df_4h['SMA_50'] = df_4h['Close'].rolling(50).mean()
    df_4h['SMA_200'] = df_4h['Close'].rolling(200).mean()
    
    delta = df_4h['Close'].diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df_4h['RSI_14'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

    stoch_min = df_4h['RSI_14'].rolling(14).min()
    stoch_max = df_4h['RSI_14'].rolling(14).max()
    df_4h['StochRSI'] = (df_4h['RSI_14'] - stoch_min) / (stoch_max - stoch_min + 1e-9) * 100

    macd_fast = df_4h['Close'].ewm(span=12, adjust=False).mean()
    macd_slow = df_4h['Close'].ewm(span=26, adjust=False).mean()
    df_4h['MACD'] = macd_fast - macd_slow
    df_4h['MACD_Signal'] = df_4h['MACD'].ewm(span=9, adjust=False).mean()

    tp = (df_4h['High'] + df_4h['Low'] + df_4h['Close']) / 3
    vwap24 = (tp * df_4h['Volume']).rolling(24).sum() / (df_4h['Volume'].rolling(24).sum() + 1e-9)
    df_4h['Rolling_VWAP_24'] = vwap24
    df_4h['Distanta_POC'] = (df_4h['Close'] - vwap24) / (vwap24 + 1e-9) * 100

    bb_mid = df_4h['Close'].rolling(20).mean()
    bb_std = df_4h['Close'].rolling(20).std()
    df_4h['BB_Upper'] = bb_mid + (bb_std * 2)
    df_4h['BB_Lower'] = bb_mid - (bb_std * 2)
    df_4h['BB_Width'] = (df_4h['BB_Upper'] - df_4h['BB_Lower']) / (bb_mid + 1e-9)
    df_4h['BB_Percent_B'] = (df_4h['Close'] - df_4h['BB_Lower']) / (df_4h['BB_Upper'] - df_4h['BB_Lower'] + 1e-9)

    df_4h['ATR_14'] = (df_4h['High'] - df_4h['Low']).rolling(14).mean()
    df_4h['ADX'] = calculeaza_adx(df_4h)

    # Target pentru antrenare
    randament_viitor = (df_4h['Close'].shift(-3) - df_4h['Close']) / df_4h['Close']
    df_4h['Target'] = 0
    df_4h.loc[(randament_viitor > 0.005) & (df_4h['Close'] > df_4h['SMA_50']), 'Target'] = 1
    df_4h.loc[(randament_viitor < -0.005) & (df_4h['Close'] < df_4h['SMA_50']), 'Target'] = 2

    df_clean = df_4h.dropna()
    features = ['Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI_14', 'StochRSI', 'MACD', 'MACD_Signal', 'Rolling_VWAP_24', 'Distanta_POC', 'BB_Width', 'BB_Percent_B', 'ATR_14', 'Macro_Yield', 'Real_Fear_Greed', 'ADX']

    X = df_clean[features]
    y = df_clean['Target']

    # Antrenăm modelul pe datele istorice recente
    model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.04, random_state=42)
    model.fit(X, y)

    # Evaluăm ULTIMA lumânare închisă disponibilă în piață
    ultima_linie = df_4h.iloc[[-2]] # luăm ultima bară complet închisă
    X_live = ultima_linie[features]

    pred_proba = model.predict_proba(X_live)[0]
    semnal = np.argmax(pred_proba)
    conf = np.max(pred_proba)

    pret_inchidere = float(ultima_linie['Close'].values[0])
    adx_val = float(ultima_linie['ADX'].values[0])
    rsi_val = float(ultima_linie['RSI_14'].values[0])
    macd_val = float(ultima_linie['MACD'].values[0])
    fng_val = int(ultima_linie['Real_Fear_Greed'].values[0])
    atr_val = float(ultima_linie['ATR_14'].values[0])
    timp_bara = ultima_linie.index[0]

    print(f"📊 Analiză finalizată pentru {timp_bara} | Preț: {pret_inchidere} | Semnal: {semnal} | Încredere: {conf*100:.1f}%")

    # Dacă modelul generează semnal valid (LONG = 1 sau SHORT = 2)
    if semnal in [1, 2]:
        in_pos = 'LONG 🟢' if semnal == 1 else 'SHORT 🔴'
        levier_curent = 5 if (adx_val >= 20 and conf >= 0.75) else (4 if conf >= 0.65 else (3 if conf >= 0.55 else (2 if conf >= 0.48 else 1)))
        
        initial_sl_pct = 0.025
        stop_initial_calc = pret_inchidere * (1 - initial_sl_pct) if semnal == 1 else pret_inchidere * (1 + initial_sl_pct)

        mesaj = (
            f"🚨 *SEMNAL NOU 4H: {in_pos}* 🚨\n"
            f"📅 Timp: {timp_bara}\n"
            f"──────────────────\n"
            f"💵 *Preț Intrare:* {pret_inchidere:.2f} USD\n"
            f"🛡️ *Stop-Loss Inițial (2.5%):* {stop_initial_calc:.2f} USD\n"
            f"🎯 *Activare Trailing TP:* {pret_inchidere * 1.03:.2f} USD (Long) / {pret_inchidere * 0.97:.2f} USD (Short)\n"
            f"⚡ *Levier Recomandat:* {levier_curent}x\n"
            f"🎯 *Încredere Model:* {conf*100:.1f}%\n"
            f"──────────────────\n"
            f"📊 *Indicatori Live:*\n"
            f" • RSI (14): {rsi_val:.2f}\n"
            f" • ADX: {adx_val:.2f}\n"
            f" • MACD: {macd_val:.2f}\n"
            f" • Fear & Greed: {fng_val}"
        )
        trimite_pe_telegram(mesaj)
        print("📲 Semnal trimis cu succes pe Telegram!")
    else:
        print("⏸️ Niciun semnal activ la această lumânare. Nu s-a trimis alertă.")

if __name__ == "__main__":
    ruleaza_verificarea_live()
