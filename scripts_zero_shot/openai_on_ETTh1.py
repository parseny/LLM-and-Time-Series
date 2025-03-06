import argparse
import torch
import time
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from openai import OpenAI
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models import Autoformer, DLinear, TimeLLM
from data_provider.data_factory import data_provider
from dotenv import load_dotenv

load_dotenv()


os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA-3.2', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768


# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)
# parser.add_argument("--gradient_clip_value", type=float, default=1.0, help="Value for gradient clipping")
# parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")


args = parser.parse_args()

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, ii)
    test_data, test_loader = data_provider(args, 'test')

    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    client = OpenAI(
        api_key = OPENAI_API_KEY,
    )
    # Извлекаем данные из test_data (индекс 0)
    k = random.randint(0, len(test_data))

    # Сохраняем все предсказания
    all_predictions = []
    actual_targets = []

    seq_x, seq_y, seq_x_mark, seq_y_mark = test_data[0]
    # print(seq_x)
    # print("-"*10)
    # print(seq_y)
    # print("-"*10)
    # print(seq_x_mark)
    # print("-"*10)
    # print(seq_y_mark)
    # print("-"*10)
    input_sequence = np.round(np.concatenate([seq_x, seq_y[:args.label_len]]).ravel(), 6).tolist()
    print(len(input_sequence))


    for i in range(10):
        try:
            seq_x, seq_y, seq_x_mark, seq_y_mark = test_data[i]

            seq_x = seq_x.squeeze()
            seq_y = seq_y.squeeze()

            input_sequence = np.concatenate([seq_x, seq_y[:args.label_len]]).tolist()

            formatted_sequence = []
            for number in input_sequence:
                formatted_number = " ".join(char for char in str(number) if char != "-")
                if number < 0:
                    formatted_number = "- " + formatted_number
                formatted_sequence.append(formatted_number)

            formatted_string = ", ".join(formatted_sequence)

            prompt = """The Electricity Transformer Temperature (ETT) dataset contains time-series data with "oil temperature" as the target value and 6 power load features. Data has very pronounced seasonality. Given this context, simulate the next {pred_len} (exactly this amount) values with 16 digits based on the trend in **input_sequence** using your best analytical skills. Output **only** a list of numbers, formatted as [value1, value2, ...], with values separated by commas. Never start your answer with "I'm sorry" or "I can't" etc.\n **input_sequence:**\n{input_sequence}"""
            prompt = prompt.format(pred_len=args.pred_len, input_sequence=input_sequence)

            response = client.chat.completions.create(
                model="o3-mini-2025-01-31",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a highly skilled time series forecasting assistant with expertise in financial market analysis. Your role is to provide precise and context-aware predictions based solely on the numerical patterns in historical data trends. Assume you are an analyst at Vanguard Total Bond Market, specializing in interpreting financial and operational time series data. Always emulate the behavior of an advanced time series prediction model, considering statistical methods, trend analysis, seasonality, and plausible ranges. When responding, output only a list of predicted numbers in the format `[value1, value2, ...]` with no additional text or explanation."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            )

            predicted_values = eval(response.choices[0].message.content)
            if len(predicted_values) < args.pred_len:
                median = np.median(predicted_values[i])
                predicted_values.extend([median] * (args.pred_len - len(predicted_values)))
            elif len(predicted_values) > args.pred_len:
                predicted_values = predicted_values[:args.pred_len]

            mse = mean_squared_error(seq_y[-args.pred_len:], predicted_values)
            mae = mean_absolute_error(seq_y[-args.pred_len:], predicted_values)

            plt.figure(figsize=(12, 6))
            plt.plot(range(len(seq_x) + args.label_len), input_sequence, label="Input Sequence")
            plt.plot(range(len(seq_x) + args.label_len, len(seq_x) + len(seq_y)), seq_y[args.label_len:], label="Target Sequence", color="green")
            plt.plot(range(len(seq_x) + args.label_len, len(seq_x) + len(seq_y)), predicted_values, label="Predicted Sequence", color="red")
            plt.xlabel("Time Steps")
            plt.ylabel("Value")
            plt.title("ETTh1 Data - Input, Target, and Predicted Sequences")
            plt.legend([f"Input Sequence", f"Target Sequence", f"Predicted Sequence\nMSE: {mse:.3f}\nMAE: {mae:.3f}"])

            # Сохраняем и отображаем график
            newpath = f'./plots/openai/experiment_o3'
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            plt.savefig(f'{newpath}/prediction_{i}.png')
            

            all_predictions.append(predicted_values)
            actual_targets.append(seq_y[-args.pred_len:])
        except Exception as e:
            print(e)
            continue

    # Рассчитываем медиану по всем предсказаниям
    median_predictions = np.median(all_predictions, axis=0)

    # Вычисляем метрики для медианного прогноза
    mse_median = mean_squared_error(actual_targets[0], median_predictions)
    mae_median = mean_absolute_error(actual_targets[0], median_predictions)

    # Построение графика
    plt.figure(figsize=(12, 6))
    seq_x = test_data[0][0].squeeze()
    seq_y = test_data[0][1].squeeze()
    plt.plot(range(len(seq_x) + args.label_len), input_sequence, label="Input Sequence")
    # plt.plot(range(len(seq_x), len(seq_x) + args.label_len), seq_y[:args.label_len], label="Label Sequence")
    plt.plot(range(len(seq_x) + args.label_len, len(seq_x) + len(seq_y)), seq_y[args.label_len:], label="Target Sequence", color="green")
    plt.plot(range(len(seq_x) + args.label_len, len(seq_x) + len(seq_y)), median_predictions, label="Median Predicted Sequence", color="red")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.title("ETTh1 Data - Input, Target, and Median Predicted Sequences")
    plt.legend([
        f"Input Sequence", 
        f"Target Sequence", 
        f"Median Predicted Sequence\nMSE: {mse_median:.3f}\nMAE: {mae_median:.3f}"
    ])
    print(f"Median Predicted Sequence\nMSE: {mse_median:.4f}\nMAE: {mae_median:.4f}")

    newpath = f'./plots/openai'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    plt.savefig(f'{newpath}/median-o3_1.png')
    plt.show()

    predicted_values = [0.213024, 0.812225, 0.696961, 0.524237, 0.662382, 0.685434, 0.708487, 0.650855, 0.616448, -0.858585, -2.264289, -3.036384, -3.485742, -4.165626, -3.831533, -2.921121, -2.229709, -1.12352, -0.201753, -0.08649, -0.005805, -0.167174, 0.247604, 0.339815, 0.846804, 1.054279, 0.812225, 0.489658, 0.743066, 0.501184, 0.604922, 0.581869, 0.593395, -0.109542, -0.720269, -1.342522, -1.953248, -2.448711, -2.391079, -2.287341, -1.1581, -1.273363, -0.167174, -0.178701, 0.13234, 0.028774, -0.132595, -0.144122, 0.213024, 0.812225, 0.696961, 0.524237, 0.662382, 0.685434, 0.708487, 0.650855, 0.616448, -0.858585, -2.264289, -3.036384, -3.485742, -4.165626, -3.831533, -2.921121, -2.229709, -1.12352, -0.201753, -0.08649, -0.005805, -0.167174, 0.247604, 0.339815, 0.846804, 1.054279, 0.812225, 0.489658, 0.743066, 0.501184, 0.604922, 0.581869, 0.593395, -0.109542, -0.720269, -1.342522, -1.953248, -2.448711, -2.391079, -2.287341, -1.1581, -1.273363, -0.167174, -0.178701, 0.13234, 0.028774, -0.132595, -0.144122, 0.213024]
    
    predicted_values_r1 = [-0.132595, -0.144122, 0.028774, 0.109287, 0.201498, 0.236077, 0.270656, 0.305236, 0.316762, 0.328288, 0.339815, 0.351341, 0.362867, 0.374394, 0.38592, 0.397447, 0.408973, 0.420499, 0.432026, 0.443552, 0.455079, 0.466605, 0.478131, 0.489658, 0.501184, 0.512711, 0.524237, 0.535763, 0.54729, 0.558816, 0.570342, 0.581869, 0.593395, 0.604922, 0.616448, 0.627974, 0.639501, 0.651027, 0.662553, 0.67408, 0.685606, 0.697132, 0.708659, 0.720185, 0.731711, 0.743238, 0.754764, 0.76629, 0.777817, 0.789343, 0.800869, 0.812396, 0.823922, 0.835448, 0.846975, 0.858501, 0.870027, 0.881554, 0.89308, 0.904606, 0.916133, 0.927659, 0.939185, 0.950712, 0.962238, 0.973764, 0.985291, 0.996817, 1.008343, 1.01987, 1.031396, 1.042922, 1.054449, 1.065975, 1.077501, 1.089028, 1.100554, 1.11208, 1.123607, 1.135133, 1.146659, 1.158186, 1.169712, 1.181238, 1.192765, 1.204291, 1.215817, 1.227344, 1.23887, 1.250396, 1.261923, 1.273449, 1.284975, 1.296502, 1.308028, 1.319554, 1.331081, 1.342607, 1.354133, 1.36566, 1.377186, 1.388712, 1.400239, 1.411765, 1.423291]
    if len(predicted_values) < args.pred_len:
        median = np.median(predicted_values[i])
        predicted_values.extend([median] * (args.pred_len - len(predicted_values)))
    if len(predicted_values_r1) > args.pred_len:
        predicted_values = predicted_values[:args.pred_len]
    
    if len(predicted_values_r1) < args.pred_len:
        median = np.median(predicted_values_r1[i])
        predicted_values_r1.extend([median] * (args.pred_len - len(predicted_values)))
    if len(predicted_values_r1) > args.pred_len:
        predicted_values_r1 = predicted_values_r1[:args.pred_len]
    
    mse_pred = mean_squared_error(seq_y[-args.pred_len:], predicted_values_r1)
    mae_pred = mean_absolute_error(seq_y[-args.pred_len:], predicted_values_r1)

    plt.figure(figsize=(12, 6))
    
    
    plt.plot(range(len(seq_x) + args.label_len), input_sequence, label="Input Sequence")
    plt.plot(range(len(seq_x) + args.label_len, len(seq_x) + len(seq_y)), seq_y[args.label_len:], label="Target Sequence", color="green")
    plt.plot(range(len(seq_x) + args.label_len, len(seq_x) + len(seq_y)), predicted_values_r1, label="Median Predicted Sequence", color="red")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.title("ETTh1 Data - Input, Target, and Median Predicted Sequences")
    plt.legend([
        f"Input Sequence", 
        f"Target Sequence", 
        f"Median Predicted Sequence\nMSE: {mse_pred:.3f}\nMAE: {mae_pred:.3f}"
    ])

    newpath = f'./plots/deepseek_v3'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    plt.savefig(f'{newpath}/r1.png')


    mse_pred = mean_squared_error(seq_y[-args.pred_len:], predicted_values)
    mae_pred = mean_absolute_error(seq_y[-args.pred_len:], predicted_values)

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(seq_x) + args.label_len), input_sequence, label="Input Sequence")
    plt.plot(range(len(seq_x) + args.label_len, len(seq_x) + len(seq_y)), seq_y[args.label_len:], label="Target Sequence", color="green")
    plt.plot(range(len(seq_x) + args.label_len, len(seq_x) + len(seq_y)), predicted_values, label="Median Predicted Sequence", color="red")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.title("ETTh1 Data - Input, Target, and Median Predicted Sequences")
    plt.legend([
        f"Input Sequence", 
        f"Target Sequence", 
        f"Median Predicted Sequence\nMSE: {mse_pred:.3f}\nMAE: {mae_pred:.3f}"
    ])

    newpath = f'./plots/deepseek_v3'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    plt.savefig(f'{newpath}/v3.png')
