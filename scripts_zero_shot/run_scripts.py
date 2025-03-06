import subprocess

# Список наборов параметров для запуска coin_pred.py
jobs = [
    {
        "model_name": "deepseek-chat",
        "symbol": "BTC",
        "timeframe": "1d",
        "use_labels": "1",
        "time_period": "365",
        "pred_len": "96",
        "output": "plots/deepseek/text/btc_label_v3_d.png"
    },
    {
        "model_name": "deepseek-chat",
        "symbol": "BTC",
        "timeframe": "1h",
        "use_labels": "1",
        "time_period": "512",
        "pred_len": "96",
        "output": "plots/deepseek/text/btc_label_v3_h.png"
    },
    {
        "model_name": "deepseek-chat",
        "symbol": "BTC",
        "timeframe": "1m",
        "use_labels": "1",
        "time_period": "512",
        "pred_len": "96",
        "output": "plots/deepseek/text/btc_label_v3_m.png"
    },
    {
        "model_name": "o3-mini-2025-01-31",
        "symbol": "BTC",
        "timeframe": "1d",
        "use_labels": "1",
        "time_period": "120",
        "pred_len": "30",
        "output": "plots/openai/crypto_charts/H2/with_examples/btc_label_o3_30d.png"
    },
    {
        "model_name": "o3-mini-2025-01-31",
        "symbol": "BTC",
        "timeframe": "1h",
        "use_labels": "1",
        "time_period": "256",
        "pred_len": "48",
        "output": "plots/openai/crypto_charts/H2/with_examples/btc_label_o3_24h.png"
    },
    {
        "model_name": "o3-mini-2025-01-31",
        "symbol": "BTC",
        "timeframe": "1m",
        "use_labels": "1",
        "time_period": "512",
        "pred_len": "60",
        "output": "plots/openai/crypto_charts/H2/with_examples/btc_label_o3_60m.png"
    }
]

for job in jobs:
    cmd = [
        "python3", "coin_pred.py",
        "--model_name", job["model_name"],
        "--symbol", job["symbol"],
        "--timeframe", job["timeframe"],
        "--use_labels", job["use_labels"],
        # "--use_examples", job["use_examples"],
        "--time_period", job["time_period"],
        "--pred_len", job["pred_len"],
        "--output", job["output"]
    ]
    
    print("Запуск команды:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Ошибка при выполнении команды:")
        print(result.stderr)
    else:
        print("Запуск прошёл успешно:")
        print(result.stdout)
