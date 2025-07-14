# -*- coding: utf-8 -*-
"""
整合、训练并对比Transformer和LSTM-AdaBoost模型的脚本
[优化]：添加了tqdm进度条和GPU配置
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm # [新增] 导入tqdm库

# PyTorch (Transformer) 相关库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# TensorFlow (LSTM-AdaBoost) 相关库
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Scikit-learn 相关库
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error

# 设置日志级别，减少不必要的输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- 1. 全局配置 ---
class Config:
    FILE_PATH = 'new_data.csv'
    # [修正] 检查并修正列名，如果您的日期列名称不同
    # 如果没有日期列，请将下一行代码删除或注释掉
    DATE_COLUMN = 'date' # <-- 检查此列名是否在您的CSV文件中存在
    SEQ_LEN = 48
    PRED_LEN = 1
    TRAIN_RATIO = 0.8
    EPOCHS = 10
    BATCH_SIZE = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"PyTorch 使用设备: {Config.DEVICE}")

# --- 2. Transformer 模型定义 (来自 PyTorch Notebook) ---
# ... (Transformer和PositionalEncoding, TimeSeriesDataset类的代码与之前相同，此处省略以保持简洁)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.input_embedding(src)
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return output

class TimeSeriesDataset(Dataset):
    def __init__(self, data, target_col_index, seq_len, pred_len):
        self.data = data
        self.target_col_index = target_col_index
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, self.target_col_index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# --- 3. LSTM-AdaBoost 模型定义 (来自 TensorFlow Notebook) ---

class LSTMAdaBoost:
    def __init__(self, n_estimators=5, seq_length=48, learning_rate=0.001):
        self.n_estimators = n_estimators
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.estimators = []
        self.estimator_weights = []
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self._configure_gpu() # [新增] 初始化时配置GPU

    # [新增] 配置TensorFlow的GPU使用方式
    def _configure_gpu(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # 设置内存增长，避免TensorFlow一次性占用所有GPU内存
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"TensorFlow 检测到 {len(gpus)} 个GPU，已配置内存按需增长。")
            except RuntimeError as e:
                print(f"GPU配置失败: {e}")
        else:
            print("TensorFlow 未检测到GPU，将使用CPU。")

    def _create_sequences(self, X, y):
        Xs, ys = [], []
        for i in range(len(X) - self.seq_length):
            Xs.append(X[i:(i + self.seq_length)])
            ys.append(y[i + self.seq_length])
        return np.array(Xs), np.array(ys)

    def _build_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(16),
            Dropout(0.3),
            Dense(1)
        ])
        model.compile(loss='mse', optimizer=Adam(self.learning_rate))
        return model

    def fit(self, X_train, y_train, epochs, batch_size):
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        X_seq, y_seq = self._create_sequences(X_train_scaled, y_train_scaled)
        sample_weights = np.ones(len(X_seq)) / len(X_seq)

        # [新增] 使用tqdm显示集成模型训练的总进度
        for i in tqdm(range(self.n_estimators), desc="训练 LSTM-AdaBoost 集成模型"):
            model = self._build_lstm_model((self.seq_length, X_train.shape[1]))
            early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
            # Keras自带进度条，所以verbose设为1即可，这里tqdm用于外层循环
            model.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size, sample_weight=sample_weights, verbose=0, callbacks=[early_stopping])

            y_pred_train = model.predict(X_seq, verbose=0).flatten()
            error = np.abs(y_pred_train - y_seq)
            weighted_error = np.sum(sample_weights * error)
            if weighted_error >= 1.0: weighted_error = 1.0 - 1e-10
            alpha = 0.5 * np.log((1 - weighted_error) / max(weighted_error, 1e-10))
            sample_weights *= np.exp(alpha * (error / (np.max(error) + 1e-10)))
            sample_weights /= np.sum(sample_weights)

            self.estimators.append(model)
            self.estimator_weights.append(alpha)

    def predict(self, X_test):
        # ... (predict方法与之前相同，此处省略)
        X_test_scaled = self.scaler_X.transform(X_test)
        X_seq, _ = self._create_sequences(X_test_scaled, np.zeros(len(X_test_scaled)))
        if len(X_seq) == 0: return np.array([])
        weighted_predictions = np.zeros(len(X_seq))
        total_weight = np.sum(self.estimator_weights)
        for weight, model in zip(self.estimator_weights, self.estimators):
            pred = model.predict(X_seq, verbose=0).flatten()
            weighted_predictions += weight * pred
        final_predictions_scaled = weighted_predictions / total_weight if total_weight > 0 else weighted_predictions
        final_predictions = self.scaler_y.inverse_transform(final_predictions_scaled.reshape(-1, 1)).flatten()
        return final_predictions

# --- 4. 数据加载与处理 ---
def load_and_prepare_data(config):
    print("开始加载和预处理数据...")
    try:
        # 尝试将配置中指定的列作为日期列解析
        df = pd.read_csv(config.FILE_PATH, parse_dates=[config.DATE_COLUMN])
    except ValueError:
        print(f"警告: 在CSV中未找到名为 '{config.DATE_COLUMN}' 的列。将不解析日期列。")
        df = pd.read_csv(config.FILE_PATH)

    df = df.fillna(method='ffill')
    
    # 动态确定特征列和目标列
    exclude_cols = [config.DATE_COLUMN, 'value']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    target_col = 'value'
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    train_size = int(len(df) * config.TRAIN_RATIO)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"数据划分完成: 训练集 {len(X_train)} 条, 测试集 {len(X_test)} 条")
    return X_train, y_train, X_test, y_test, feature_cols, target_col

# --- 5. 模型训练和预测的主流程 ---
def run_transformer(X_train, y_train, X_test, y_test, config, feature_cols, target_col):
    print("\n--- 开始训练 Transformer 模型 ---")
    start_time = time.time()
    
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_reshaped = y_train.reshape(-1, 1)
    scaler_y.fit(y_train_reshaped)

    train_data_for_torch = np.hstack([X_train_scaled, scaler_y.transform(y_train_reshaped)])
    target_col_index = train_data_for_torch.shape[1] - 1
    
    train_dataset = TimeSeriesDataset(train_data_for_torch, target_col_index, config.SEQ_LEN, config.PRED_LEN)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    model = TransformerModel(
        input_dim=len(feature_cols) + 1, d_model=64, nhead=4, num_layers=2, output_dim=config.PRED_LEN
    ).to(config.DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(config.EPOCHS):
        # [新增] 使用tqdm显示每个epoch的训练进度
        progress_bar = tqdm(train_loader, desc=f"Transformer Epoch {epoch+1}/{config.EPOCHS}")
        epoch_loss = 0
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE) # 确保数据在GPU上
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            epoch_loss += current_loss
            progress_bar.set_postfix({'loss': f'{current_loss:.6f}'})
        print(f"  Epoch {epoch+1} 平均损失: {epoch_loss/len(train_loader):.6f}")

    model.eval()
    X_test_scaled = scaler_x.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
    test_data_for_torch = np.hstack([X_test_scaled, y_test_scaled])
    
    test_dataset = TimeSeriesDataset(test_data_for_torch, target_col_index, config.SEQ_LEN, config.PRED_LEN)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    predictions_scaled = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(config.DEVICE) # 确保预测数据也在GPU上
            output = model(inputs)
            predictions_scaled.append(output.cpu().numpy())
            
    predictions_scaled = np.array(predictions_scaled).squeeze()
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    
    end_time = time.time()
    print(f"Transformer 模型训练和预测完成，耗时: {end_time - start_time:.2f} 秒")
    return predictions

# ... (run_lstm_adaboost 和 plot_comparison 函数与之前相同，此处省略)
def run_lstm_adaboost(X_train, y_train, X_test, y_test, config):
    print("\n--- 开始训练 LSTM-AdaBoost 模型 ---")
    start_time = time.time()
    model = LSTMAdaBoost(n_estimators=5, seq_length=config.SEQ_LEN)
    model.fit(X_train, y_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
    predictions = model.predict(X_test)
    end_time = time.time()
    print(f"LSTM-AdaBoost 模型训练和预测完成，耗时: {end_time - start_time:.2f} 秒")
    return predictions

def plot_comparison(y_true, pred_transformer, pred_lstm_adaboost, seq_len):
    print("\n--- 生成对比可视化图表 ---")
    y_true_aligned = y_true[seq_len:]
    y_prev_day = y_true[seq_len - 24 : -24]
    min_len = min(len(y_true_aligned), len(pred_transformer), len(pred_lstm_adaboost), len(y_prev_day))
    y_true_aligned = y_true_aligned[:min_len]
    pred_transformer = pred_transformer[:min_len]
    pred_lstm_adaboost = pred_lstm_adaboost[:min_len]
    y_prev_day = y_prev_day[:min_len]
    n_points = 100
    plt.figure(figsize=(20, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(y_true_aligned[:n_points], label='实际值', color='black', linewidth=2, marker='o', markersize=4)
    plt.plot(y_prev_day[:n_points], label='前一天的实际值', color='gray', linestyle='--', linewidth=1.5)
    plt.plot(pred_transformer[:n_points], label='Transformer预测值', color='blue', linestyle='--', marker='^', markersize=4, alpha=0.8)
    plt.plot(pred_lstm_adaboost[:n_points], label='LSTM-AdaBoost预测值', color='red', linestyle='-.', marker='x', markersize=4, alpha=0.8)
    plt.title('模型预测结果对比 (前100个测试点)', fontsize=16)
    plt.xlabel('时间步 (Time Step)', fontsize=12)
    plt.ylabel('负荷值 (Value)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    save_path = 'model_comparison_plot_with_progress.png'
    plt.savefig(save_path)
    print(f"对比图表已保存至: {save_path}")
    plt.show()

# --- 主程序入口 ---
if __name__ == "__main__":
    X_train, y_train, X_test, y_test, feature_cols, target_col = load_and_prepare_data(Config)
    transformer_preds = run_transformer(X_train, y_train, X_test, y_test, Config, feature_cols, target_col)
    lstm_adaboost_preds = run_lstm_adaboost(X_train, y_train, X_test, y_test, Config)
    plot_comparison(y_test, transformer_preds, lstm_adaboost_preds, Config.SEQ_LEN)
    mae_transformer = mean_absolute_error(y_test[Config.SEQ_LEN:len(transformer_preds)+Config.SEQ_LEN], transformer_preds)
    mae_lstm_adaboost = mean_absolute_error(y_test[Config.SEQ_LEN:len(lstm_adab_preds)+Config.SEQ_LEN], lstm_adaboost_preds)
    print("\n--- 模型性能量化对比 (MAE) ---")
    print(f"Transformer MAE: {mae_transformer:.4f}")
    print(f"LSTM-AdaBoost MAE: {mae_lstm_adaboost:.4f}")