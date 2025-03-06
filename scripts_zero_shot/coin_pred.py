import ccxt
import openai
import json
import os
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator
import numpy as np
import datetime
import argparse
from dotenv import load_dotenv

load_dotenv()

def get_timedelta(timeframe):
    """
    Convert a timeframe string (e.g. '1m', '1h', '1d') to a timedelta.
    """
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == 'm':
        return datetime.timedelta(minutes=value)
    elif unit == 'h':
        return datetime.timedelta(hours=value)
    elif unit == 'd':
        return datetime.timedelta(days=value)
    else:
        raise ValueError("Unsupported timeframe")

# -------------------------------
# Parse command-line arguments
# -------------------------------
parser = argparse.ArgumentParser(description="Crypto Prediction Script")
parser.add_argument("--model_name", type=str, required=True, help="Model name")
parser.add_argument("--symbol", type=str, required=True, help="Symbol, e.g., TRUMP, BTC")
parser.add_argument("--timeframe", type=str, choices=["1m", "1h", "1d"], required=True,
                    help="Timeframe: 1m, 1h, or 1d")
parser.add_argument("--use_labels", type=int, required=True, help="Use labels (1/0)")
parser.add_argument("--use_examples", type=int, default=0, help="Use labels (1/0)")
parser.add_argument("--use_tech", type=int, default=0, help="Use labels (1/0)")
parser.add_argument("--time_period", type=int, default=None,
                    help="Historical data length (if None, use all available data)")
parser.add_argument("--pred_len", type=int, required=True, help="Number of prediction values")
parser.add_argument("--output", type=str, default="prediction.png", help="Output filename for the plot")
args = parser.parse_args()

# -------------------------------
# Fetch data from the exchange
# -------------------------------
exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv(args.symbol + "/USDT", args.timeframe)
closing_prices = [candle[4] for candle in ohlcv]
# Convert exchange timestamps (ms) to datetime objects:
all_timestamps = [datetime.datetime.fromtimestamp(candle[0] / 1000) for candle in ohlcv]

# -------------------------------
# Slice data according to provided parameters
# -------------------------------

if args.time_period is not None:
    if args.use_labels:
        input_sequence = closing_prices[-(args.time_period + args.pred_len):-args.pred_len]
        labels = closing_prices[-args.pred_len:]
        # Use the last (time_period + pred_len) timestamps for plotting:
        plot_timestamps = all_timestamps[-(args.time_period + args.pred_len):]
        dt_hist = plot_timestamps[:len(input_sequence)]
        dt_labels = plot_timestamps[len(input_sequence):]
    else:
        input_sequence = closing_prices[-args.time_period:]
        dt_hist = all_timestamps[-args.time_period:]
else:
    if args.use_labels:
        input_sequence = closing_prices[:-args.pred_len]
        labels = closing_prices[-args.pred_len:]
        dt_hist = all_timestamps[:-args.pred_len]
        dt_labels = all_timestamps[-args.pred_len:]
    else:
        input_sequence = closing_prices
        dt_hist = all_timestamps

n = len(input_sequence)
# TODO run this
# arr_inp_seq = []
# for i in range(1, n):
#     if input_sequence[i-1] > input_sequence[i]:
#         arr_inp_seq.append(f"Price dropped from {input_sequence[i-1]} to {input_sequence[i]}")
#     else:
#         arr_inp_seq.append(f"Price moved upward from {input_sequence[i-1]} to {input_sequence[i]}")

# input_sequence_str = ', '.join(arr_inp_seq)
time_delta = get_timedelta(args.timeframe)

# For the case when we do NOT have labels, compute future timestamps:
if not args.use_labels:
    dt_pred = [dt_hist[-1] + time_delta * (i + 1) for i in range(args.pred_len)]

if args.use_labels:
    full_dates = dt_hist + dt_labels
else:
    full_dates = dt_hist + dt_pred

# -------------------------------
# Prepare OpenAI prompt and call
# -------------------------------
user_prompts = {
    "TRUMP": f"""TRUMP Coin is a memecoin made by the 47th President of the USA, Donald Trump. It often experiences **high volatility and is sensitive to market whales actions**.
Given this context, simulate the next {args.pred_len} (exactly this amount) values based on the trend in **input_sequence** using your best analytical skills. Output **only** a list of numbers, formatted as [value1, value2, ...], with values separated by commas. Never start your answer with "I'm sorry" or "I can't" etc.
**input_sequence:**
{input_sequence}
""",
    "BTC": f"""Given Bitcoin cryptocurrency graphic. It is the most popular crypto asset, being accumulated by major Finance Funds. However, Bitcoin is **extremely risky and volatile** with **sharp peaks and bottoms**. Price fluctuates frequently across timeframes.
Given this context, simulate the next {args.pred_len} (exactly this amount) values based on the trend in **input_sequence** using your best analytical skills. Keep up with amplitude of previous values, pay attention to the size of the drops and peaks. Output **only** a list of numbers, formatted as [value1, value2, ...], with values separated by commas. Never start your answer with "I'm sorry" or "I can't" etc.
**input_sequence:**
{input_sequence}
"""
}

user_prompt = user_prompts.get(args.symbol, user_prompts["TRUMP"])


if args.use_examples:
    start_date = datetime.datetime(2024, 7, 1)
    limit = args.time_period  # 60 дней данных

    start_sept = datetime.datetime(2024, 9, 1)
    limit_sept = args.pred_len  # 30 дней данных (сентябрь)

    # Переводим начальные даты в timestamp в миллисекундах
    since_time = int(start_date.timestamp() * 1000)
    since_next_time = int(start_sept.timestamp() * 1000)

    ohlcv_example = exchange.fetch_ohlcv(args.symbol + "/USDT", args.timeframe, since=since_time, limit=limit)
    closing_example = [candle[4] for candle in ohlcv_example]
    timestamps_example= [candle[0] for candle in ohlcv_example]

    # Получаем исторические данные для сентября 2024 (30 дней)
    ohlcv_next= exchange.fetch_ohlcv(args.symbol + "/USDT", args.timeframe, since=since_next_time, limit=limit_sept)
    closing_prices_next = [candle[4] for candle in ohlcv_next]
    timestamps_next = [candle[0] for candle in ohlcv_next]

    example_prompt = f"""
    I will give you an example of the same graphic from the past. Pay attention to the following example.
    You can analyze the fluctuations of the price and how the price changed by comparing the two sequences:
        price_before: {closing_example}
        price_after: {closing_prices_next}
    """
    user_prompt += example_prompt

if args.use_tech:
    tech_part = f"""
    ### Time Series Analysis Techniques:
    To simulate the next {args.pred_len} values accurately, consider key forecasting methods:
    - **Moving Averages (SMA, EMA, WMA):** Identify trends while preserving price movement amplitude.
    - **Bollinger Bands & Volatility Models:** Capture rapid price swings and sudden trend reversals.
    - **Fourier & Wavelet Analysis:** Detect periodic patterns and high-frequency fluctuations.
    - **Autoregressive Models (ARIMA, GARCH):** Model trend shifts and price momentum.
    - **Deep Learning (LSTMs, Transformers):** Forecast non-linear dependencies while maintaining volatility.

    ### Simulation Guidelines:
    - **Pay special attention to all values in input_sequence**—each point carries meaningful trend information.
    - **Keep up with the amplitude** of previous fluctuations; **maintain the scale of peaks and drops**. MAKE STEEP UPWARDS AND DOWNS.
    - **Ensure high volatility representation** by incorporating sharp reversals and momentum-driven shifts.
    - **Avoid excessive smoothing**—Bitcoin is known for sudden price spikes and breakdowns.

    Now, simulate the next {args.pred_len} values while following these characteristics.
    """

    # Combine both parts
    user_prompt += tech_part

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

if args.model_name in ['o3-mini-2025-01-31', 'gpt-4o', 'gpt-4o-2024-11-20']:
    ## OPENAI
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    ## DEEPSEEK
    client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

print(user_prompt)
predictions = []
import time

# Максимальное количество попыток
MAX_RETRIES = 3

for i in range(10):
    print(f"Предсказание {i + 1}")
    retry_count = 0
    
    while retry_count < MAX_RETRIES:
        try:
            # Выполняем запрос к API
            response = client.chat.completions.create(
                model=args.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": ("You are a highly skilled time series forecasting assistant with expertise in financial market analysis. Your role is to provide precise and context-aware predictions based solely on the numerical patterns in historical data trends. Assume you are an analyst at Vanguard Total Bond Market, specializing in interpreting financial and operational time series data. Always emulate the behavior of an advanced time series prediction model, considering statistical methods, trend analysis, seasonality, and plausible ranges. When responding, output only a list of predicted numbers in the format `[value1, value2, ...]` with no additional text or explanation.")
                    },
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Проверяем, что ответ содержит данные
            if not response.choices:
                print("Ошибка: Пустой ответ от API.")
                retry_count += 1
                time.sleep(1)  # Пауза перед повторной попыткой
                continue
            
            predicted_values = response.choices[0].message.content
            
            # Проверяем, что predicted_values не пустой
            if not predicted_values:
                print("Ошибка: Пустое предсказание.")
                retry_count += 1
                time.sleep(1)  # Пауза перед повторной попыткой
                continue
            
            # Пытаемся разобрать ответ
            try:
                parsed_values = eval(predicted_values)  # Парсим строку в список
                if not isinstance(parsed_values, list):  # Проверяем, что это список
                    raise ValueError("Ответ не является списком.")
                predictions.append(parsed_values)
                print(f"Успешно добавлено предсказание. Всего предсказаний: {len(predictions)}")
                print("-" * 100)
                break  # Выходим из цикла retry, если успешно
            except (SyntaxError, ValueError) as e:
                print("Syntax Error")
                predicted_values = response.choices[0].message.content
                predicted_values += "]"
                predictions.append(eval(predicted_values))
                print(len(predictions))
                break
        
        except Exception as e:
            print(f"Ошибка при выполнении запроса к API: {e}")
            retry_count += 1
            time.sleep(1)  # Пауза перед повторной попыткой
    
    # Если превышено количество попыток
    if retry_count == MAX_RETRIES:
        print(f"Не удалось получить корректное предсказание после {MAX_RETRIES} попыток.")
    

# Adjust prediction lengths if necessary
for i in range(len(predictions)):
    if len(predictions[i]) < args.pred_len:
        median = np.median(predictions[i])
        predictions[i].extend([median] * (args.pred_len - len(predictions[i])))
    elif len(predictions[i]) > args.pred_len:
        predictions[i] = predictions[i][:args.pred_len]

predicted_values_median = np.median(np.array(predictions), axis=0).tolist()

# -------------------------------
# Unified plotting
# -------------------------------
output_dir = "plots/deepseek/text/few_plots2"
os.makedirs(output_dir, exist_ok=True)

# Отрисовка и сохранение графиков для каждого предсказания
for i, pred in enumerate(predictions):
    plt.figure(figsize=(12, 6))
    
    # Исторические данные
    plt.plot(full_dates[:n], input_sequence, label="Input Sequence", color="blue")
    
    # Предсказанные значения
    combined_pred = [input_sequence[-1]] + pred
    plt.plot(full_dates[n-1:], combined_pred, label="Predicted Values", color="red", linestyle="dashed")
    
    # Если есть фактические значения (labels), отрисовываем их
    if args.use_labels:
        combined_labels = [input_sequence[-1]] + labels
        plt.plot(full_dates[n-1:], combined_labels, label="Actual Values", color="green")
    
    # Форматирование дат на оси x
    locator = AutoDateLocator()
    formatter = DateFormatter('%Y-%m-%d' if args.timeframe == "1d" else '%Y-%m-%d %H:%M')
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gcf().autofmt_xdate()
    
    # Легенда, заголовки и сохранение
    plt.legend()
    plt.xlabel("Дата/Время")
    plt.ylabel("Цена")
    plt.title(f"Прогноз цены для {args.symbol} (Предсказание {i+1})")
    plt.tight_layout()
    
    # Сохраняем график
    plot_filename = os.path.join(output_dir, f"prediction_{i+1}.png")
    plt.savefig(plot_filename)
    plt.close()  # Закрываем график, чтобы освободить память


plt.figure(figsize=(12, 6))
# Строим график исторических данных с использованием реальных дат
plt.plot(full_dates[:n], input_sequence, label="Input Sequence", color="blue")

# Чтобы соединить графики, добавляем последний элемент input_sequence в начало предсказаний
combined_pred = [input_sequence[-1]] + predicted_values_median

if args.use_labels:
    combined_labels = [input_sequence[-1]] + labels
    plt.plot(full_dates[n-1:], combined_labels, label="Actual Values", color="green")
    plt.plot(full_dates[n-1:], combined_pred, label="Predicted Values", color="red", linestyle="dashed")
else:
    plt.plot(full_dates[n-1:], combined_pred, label="Predicted Values", color="red", linestyle="dashed")

# Форматирование дат на оси x
locator = AutoDateLocator()
formatter = DateFormatter('%Y-%m-%d' if args.timeframe == "1d" else '%Y-%m-%d %H:%M')
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)
plt.gcf().autofmt_xdate()

plt.legend()
plt.xlabel("Дата/Время")
plt.ylabel("Цена")
plt.title(f"Прогноз цены для {args.symbol} на {args.model_name} ({args.timeframe})")
plt.tight_layout()
plt.savefig(args.output)
plt.show()