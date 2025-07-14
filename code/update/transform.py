import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prettytable import PrettyTable
import math
import os
from tqdm import tqdm

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# [优化] 将设备检测和打印封装，更整洁
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    return device

device = get_device()

# 定义MAPE计算函数
def mape(y_true, y_pred):
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# 定义位置编码类 (无变动)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [优化] 增加batch维度以方便广播
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, feature_dim]
        # [优化] pe的维度是 [1, max_len, d_model]，可以直接与x相加，更高效
        return x + self.pe[:, :x.size(1), :]

# 定义Transformer编码器模型
class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, seq_len, dropout=0.1):
        super(TransformerEncoderModel, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # [优化] 使用Flatten层来利用所有时间步的输出，而不是只用最后一个
        self.flatten = nn.Flatten()
        
        # [优化] 调整解码器输入维度，因为我们压平了所有时间步的输出
        self.decoder = nn.Sequential(
            nn.Linear(d_model * seq_len, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout), # [优化] 在解码器中也加入Dropout
            nn.Linear(d_model // 2, output_dim)
        )
        self.d_model = d_model

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        
        # [优化] 压平所有时间步的输出，以利用全部序列信息
        x = self.flatten(x)
        output = self.decoder(x)
        
        return output

# 定义负荷预测数据集类 (无变动)
class LoadForecastDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data = data
        self.length = len(data) - seq_len - pred_len + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, 0]
        return x, y

# [优化] 创建一个新的早停类，用于监控验证损失
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# [优化] 训练和验证函数合并，并加入早停和学习率调度
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, early_stopping):
    model.to(device)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [T]', leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- 验证阶段 ---
        model.eval()
        epoch_val_loss = 0
        val_progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [V]', leave=False)
        with torch.no_grad():
            for inputs, targets in val_progress_bar:
                inputs, targets = inputs.to(device).float(), targets.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')
        
        # [优化] 调用学习率调度器和早停
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
    # 加载性能最好的模型权重
    model.load_state_dict(torch.load(early_stopping.path))
    return train_losses, val_losses


# [优化] 测试函数现在需要scaler来进行反向缩放，以便计算真实指标
def test_model(model, test_loader, device, scaler, target_col_idx=0):
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device).float()
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    predicted_data = np.concatenate(all_predictions)
    Ytest = np.concatenate(all_targets)

    # [优化] 在计算指标前，将数据反标准化回原始尺度
    # 创建一个和原始数据维度相同的空数组用于反向缩放
    num_features = scaler.n_features_in_
    
    # 反向缩放预测值
    pred_inverse_scaled = np.zeros((predicted_data.shape[0], num_features))
    pred_inverse_scaled[:, target_col_idx] = predicted_data[:, 0] # 假设我们只关心第一个预测步
    pred_inverse_scaled = scaler.inverse_transform(pred_inverse_scaled)[:, target_col_idx]
    
    # 反向缩放真实值
    targets_inverse_scaled = np.zeros((Ytest.shape[0], num_features))
    targets_inverse_scaled[:, target_col_idx] = Ytest[:, 0]
    targets_inverse_scaled = scaler.inverse_transform(targets_inverse_scaled)[:, target_col_idx]

    # 返回原始尺度的值和缩放后的值
    return pred_inverse_scaled, targets_inverse_scaled, predicted_data, Ytest


# [优化] 评估函数现在直接在原始尺度的数据上计算
def evaluate_model(predicted_data_orig, Ytest_orig):
    table = PrettyTable(['测试集指标 (原始尺度)', 'Value'])
    
    mse = mean_squared_error(Ytest_orig, predicted_data_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Ytest_orig, predicted_data_orig)
    mape_val = mape(Ytest_orig, predicted_data_orig)
    r2 = r2_score(Ytest_orig, predicted_data_orig)
    
    table.add_row(['MSE', f'{mse:.4f}'])
    table.add_row(['RMSE', f'{rmse:.4f}'])
    table.add_row(['MAE', f'{mae:.4f}'])
    table.add_row(['MAPE', f'{mape_val:.2f}%'])
    table.add_row(['R2-Score', f'{r2*100:.2f}%'])
    
    return table

# 可视化函数 (无大变动, 接收原始尺度数据)
def visualize_predictions(predicted_data, Ytest, title='负荷预测结果对比 (原始尺度)'):
    plt.figure(figsize=(15, 7))
    plt.plot(Ytest, label='真实值', color='black', alpha=0.7)
    plt.plot(predicted_data, label='预测值', color='purple', linestyle='--')
    plt.title(title)
    plt.xlabel('时间步')
    plt.ylabel('负荷值')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig('results/load_forecast_comparison.png')
    plt.show()


# 主函数
def main():
    # --- 参数配置 ---
    config = {
        "file_path": r'../../code/new_data.csv',
        "seq_len": 48,
        "pred_len": 1, # [优化建议] 先从单步预测开始，更容易调试和优化。多步预测可以后续扩展
        "batch_size": 64,
        "epochs": 100, # 可以设置多一点，因为有早停
        "learning_rate": 0.0005, # [优化] 学习率可以稍微调低一些
        "train_split": 0.7, # 70% 训练
        "val_split": 0.15,  # 15% 验证
        # 测试集为 1 - 0.7 - 0.15 = 15%
        "input_dim": 14,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 3, # [优化] 可以尝试增加层数
        "dim_feedforward": 256, # [优化] 增加前馈网络维度
        "dropout": 0.2, # [优化] 增加dropout
        "target_col_name": "value"
    }

    # --- 数据预处理 ---
    print("1. 开始数据预处理...")
    df = pd.read_csv(config['file_path'])
    feature_cols = ['value', 'weather_status', 'temperature', 'humidity', 'wind_speed', 
                    'wind_direction_angle', 'pressure', 'visibility', 'precipitation', 
                    'light', 'holiday', 'minute', 'week', 'year']
    data = df[feature_cols].values
    
    # [优化] 按时间顺序划分数据集
    train_num = int(len(data) * config['train_split'])
    val_num = int(len(data) * config['val_split'])
    test_num = len(data) - train_num - val_num
    
    train_data = data[:train_num]
    val_data = data[train_num : train_num + val_num]
    test_data = data[train_num + val_num:]

    # [优化] Scaler只在训练集上fit
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    val_data_scaled = scaler.transform(val_data)
    test_data_scaled = scaler.transform(test_data)

    # 创建Dataset
    train_dataset = LoadForecastDataset(train_data_scaled, config['seq_len'], config['pred_len'])
    val_dataset = LoadForecastDataset(val_data_scaled, config['seq_len'], config['pred_len'])
    test_dataset = LoadForecastDataset(test_data_scaled, config['seq_len'], config['pred_len'])

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # --- 模型初始化 ---
    print("2. 初始化模型...")
    model = TransformerEncoderModel(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        output_dim=config['pred_len'],
        seq_len=config['seq_len'],
        dropout=config['dropout']
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    # [优化] 添加学习率调度器和早停
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(patience=10, verbose=True, path='models/best_transformer_model.pth')
    
    if not os.path.exists('models'):
        os.makedirs('models')

    # --- 模型训练与验证 ---
    print("3. 开始训练模型...")
    train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, config['epochs'], device, early_stopping)

    # --- 模型测试与评估 ---
    print("4. 开始测试模型...")
    target_col_idx = feature_cols.index(config['target_col_name'])
    pred_orig, targ_orig, _, _ = test_model(model, test_loader, device, scaler, target_col_idx)

    print("\n5. 评估模型性能...")
    eval_table = evaluate_model(pred_orig, targ_orig)
    print(eval_table)
    
    # --- 结果可视化 ---
    print("6. 可视化预测结果...")
    visualize_predictions(pred_orig, targ_orig)

    print("\n模型训练、评估和可视化完成！")

if __name__ == "__main__":
    main()
