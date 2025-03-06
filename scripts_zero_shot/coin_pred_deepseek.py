import ccxt
import json
import matplotlib.pyplot as plt
import numpy as np
import datetime
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


parser = argparse.ArgumentParser(description="Crypto Prediction Script")
parser.add_argument("--model_name", type=str, required=True, help="Название модели")
parser.add_argument("--symbol", type=str, required=True, help="Название монеты, например TRUMP, BTC")
parser.add_argument("--timeframe", type=str, choices=["1h", "1d"], required=True, help="Таймфрейм: 1h или 1d")
parser.add_argument("--use_labels", type=int, required=True, help="Использовать лейблы (1/0)")
parser.add_argument("--time_period", type=int, default=None, help="Длина исторических данных (по умолчанию весь closing_prices)")
parser.add_argument("--pred_len", type=int, required=True, help="Количество предсказываемых значений")
parser.add_argument("--output", type=str, default="prediction.png", help="Имя файла для сохранения графика")
args = parser.parse_args()

exchange = ccxt.binance()

ohlcv = exchange.fetch_ohlcv(args.symbol + "/USDT", args.timeframe)

closing_prices = [candle[4] for candle in ohlcv]

if args.time_period is None:
    if args.use_labels: 
        input_sequence = closing_prices[:-args.pred_len]
        labels = closing_prices[-args.pred_len:] 
    else:
        input_sequence = closing_prices
else:
    if args.use_labels:
        labels = closing_prices[-args.pred_len:] 
        input_sequence = closing_prices[-(args.time_period+args.pred_len):-args.pred_len]
        timestamps = [candle[0] for candle in ohlcv][-(args.time_period+args.pred_len):]
    else:
        input_sequence = closing_prices[-args.time_period:]
        timestamps = [candle[0] for candle in ohlcv][-args.time_period:]

n = len(input_sequence)

user_prompts = {
    "TRUMP": f"""TRUMP Coin is a memecoin associated with political branding and speculative trading, often experiencing high volatility. 
Given this context, simulate the next {args.pred_len} (exactly this amount) values based on the trend in **input_sequence** using your best analytical skills. Output **only** a list of numbers, formatted as [value1, value2, ...], with values separated by commas. Provide values that are plausible and within the range of the existing data. Never start your answer with "I'm sorry" or "I can't" etc.\n **input_sequence:**\n{input_sequence}\n""",
    "BTC": f"""Given BTC cryptocurrency graphic. It is the most popular crypto asset, that is being accumulated by largest Finance Funds. However it is highly volatile asset, and has high volume. 
Given this context, simulate the next {args.pred_len} (exactly this amount) values based on the trend in **input_sequence** using your best analytical skills. Output **only** a list of numbers, formatted as [value1, value2, ...], with values separated by commas. Provide values that are plausible and within the range of the existing data. Never start your answer with "I'm sorry" or "I can't" etc.\n **input_sequence:**\n{input_sequence}\n"""
}

model_name = args.model_name
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

user_prompt = user_prompts.get(args.symbol, user_prompts["TRUMP"])

predictions = []

messages = [
    {"role": "system", "content": "You are a highly skilled time series forecasting assistant with expertise in financial market analysis. Your role is to provide precise and context-aware predictions based solely on the numerical patterns in historical data trends. Assume you are an analyst at Vanguard Total Bond Market, specializing in interpreting financial and operational time series data. Always emulate the behavior of an advanced time series prediction model, considering statistical methods, trend analysis, seasonality, and plausible ranges. When responding, output only a list of predicted numbers in the format `[value1, value2, ...]` with no additional text or explanation."},
    {"role": "user", "content": user_prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

for _ in range(10):
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=4096
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    predicted_values = eval(response) # предполагаю что вернет чисто массив
    predictions.append(predicted_values)

for i in range(len(predictions)):
    if len(predictions[i]) < args.pred_len:
        median = np.median(predictions[i])
        predictions[i].extend([median] * (args.pred_len - len(predictions[i])))
    elif len(predictions[i]) > args.pred_len:
        predictions[i] = predictions[i][:args.pred_len]

predicted_values_median = np.median(np.array(predictions), axis=0).tolist()

if args.symbol == "TRUMP" and args.use_labels:
    print('here')
    timestamps_input = [datetime.datetime.now() - datetime.timedelta(hours=i) for i in range(n+args.pred_len)]
    timestamps_input.reverse()

    formatted_timestamps = [ts.strftime('%Y-%m-%d') for ts in timestamps_input]
    plt.figure(figsize=(12, 6))
    plt.plot(range(n), input_sequence, label="Input Sequence", color="blue")
    plt.plot(range(n, n + args.pred_len), labels, label=" Labels", color="green")
    plt.plot(range(n, n + args.pred_len), predicted_values_median, label="Predicted Values", color="red", linestyle="dashed")
    plt.xticks(range(0, len(timestamps_input), 24), formatted_timestamps[::24], rotation=90)
    plt.yticks(range(0, 100, 5))
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{args.symbol} Coin Price Prediction")
    plt.tight_layout()

    plt.savefig(args.output)

if args.symbol == "TRUMP" and not args.use_labels:

    timestamps_input = [datetime.datetime.now() - datetime.timedelta(hours=i) for i in range(n)]
    timestamps_input.reverse()

    timestamps_pred = [timestamps_input[-1] + datetime.timedelta(hours=i+1) for i in range(args.pred_len)]

    ts = timestamps_input + timestamps_pred
    formatted_timestamps = [ts.strftime('%Y-%m-%d') for ts in ts]

    plt.figure(figsize=(12, 6))
    plt.plot(range(n), input_sequence, label="Input Sequence", color="blue")
    plt.plot(range(n, n+args.pred_len), predicted_values_median, label="Predicted Values", color="red", linestyle="dashed")
    plt.xticks(range(0, len(formatted_timestamps), 24), formatted_timestamps[::24], rotation=90)
    plt.yticks(range(0, 100, 5))
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{args.symbol} Coin Price Prediction")
    plt.tight_layout()

    plt.savefig(args.output)

if args.symbol == 'BTC' and args.use_labels:
    dates = [datetime.datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d') for ts in timestamps]

    plt.figure(figsize=(12, 6))
    plt.plot(dates[:-args.pred_len], input_sequence, label="Input Sequence", color="blue")
    plt.xticks(range(0, len(dates), 20), [dates[i] for i in range(0, len(dates), 20)], rotation=90)
    plt.plot(dates[-args.pred_len:], labels, label="Actual Values", color="green")
    plt.plot(dates[-args.pred_len:], predicted_values_median, label="Predicted Values", color="red", linestyle="dashed")
    y_range = range(int(np.min(input_sequence)), int(np.max(input_sequence)), int(input_sequence[-1]-input_sequence[0])//20)
    plt.yticks(y_range, [i for i in y_range])
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{args.symbol} Coin Price Prediction")
    plt.tight_layout()

    plt.savefig(args.output)

if args.symbol == 'BTC' and not args.use_labels:
    step = 24 * 3600 * 1000
    timestamps_pred = range(timestamps[-1] + step, timestamps[-1] + args.pred_len * step + 1, step)
    dates = [datetime.datetime.fromtimestamp(t / 1000).strftime('%Y-%m-%d') for t in timestamps]
    pred_dates = [datetime.datetime.fromtimestamp(t / 1000).strftime('%Y-%m-%d') for t in timestamps_pred]
    all_dates = dates + pred_dates
    plt.figure(figsize=(12, 8))
    plt.plot(range(n), input_sequence, label="Input Sequence", color="blue")
    plt.plot(range(n, n + args.pred_len), predicted_values_median, label="Predicted Values", color="red", linestyle="dashed")
    plt.xticks(range(0, len(all_dates), 20), [all_dates[i] for i in range(0, len(all_dates), 20)], rotation=90)
    y_range = range(int(np.min(input_sequence)), int(np.max(input_sequence)), int(input_sequence[-1]-input_sequence[0])//20)
    plt.yticks(y_range, [i for i in y_range])
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{args.symbol} Coin Price Prediction")
    plt.tight_layout()

    plt.savefig(args.output)
