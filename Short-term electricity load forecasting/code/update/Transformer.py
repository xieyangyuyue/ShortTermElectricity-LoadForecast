import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prettytable import PrettyTable
import math
import os
from tqdm import tqdm  # 用于显示训练进度

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 检测GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")

# 定义MAPE计算函数
def mape(y_true, y_pred):
    # 处理接近零的值，避免除零错误
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# 定义位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, feature_dim]
        return x + self.pe[:x.size(1), :].unsqueeze(0)

# 定义Transformer编码器模型
class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, dropout=0.1):
        super(TransformerEncoderModel, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # 创建Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # PyTorch 1.9+ 支持batch_first
        )
        
        # 创建多层Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, output_dim)
        )
        
        self.d_model = d_model
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = self.input_embedding(x)  # [batch_size, seq_len, d_model]
        x = self.positional_encoding(x)  # 添加位置编码
        x = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]
        
        # 取序列最后一个时间步的输出作为预测结果
        x = x[:, -1, :]  # [batch_size, d_model]
        output = self.decoder(x)  # [batch_size, output_dim]
        
        return output

# 定义负荷预测数据集类
class LoadForecastDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data = data
        self.length = len(data) - seq_len - pred_len + 1
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 输入序列
        x = self.data[idx:idx+self.seq_len]
        # 目标序列（负荷值）
        y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len, 0]  # 第一列是负荷值
        
        return x, y

# 数据预处理函数
def preprocess_data(file_path, seq_len, pred_len):
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 提取特征列和目标列
    feature_cols = ['value', 'weather_status', 'temperature', 'humidity', 'wind_speed', 
                    'wind_direction_angle', 'pressure', 'visibility', 'precipitation', 
                    'light', 'holiday', 'minute', 'week', 'year']
    
    # 确保所有特征列都存在
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"列 {col} 不存在于数据集中")
    
    # 提取数据
    data = df[feature_cols].values
    
    # 数据标准化
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    # 创建数据集
    dataset = LoadForecastDataset(data, seq_len, pred_len)
    
    return dataset, scaler

# 训练函数
def train_model(model, train_loader, criterion, optimizer, epochs, device):
    model.train()
    model.to(device)  # 将模型移至GPU
    
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}')
        
        for i, (inputs, targets) in progress_bar:
            # 将数据移至GPU
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}')
    
    return train_losses

# 测试函数
def test_model(model, test_loader, device):
    model.eval()
    model.to(device)  # 确保模型在GPU上
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            # 将数据移至GPU
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            
            # 预测
            outputs = model(inputs)
            
            # 将结果移回CPU并收集
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # 转换为numpy数组
    predicted_data = np.concatenate(all_predictions)
    Ytest = np.concatenate(all_targets)
    
    return predicted_data, Ytest

# 评估模型
def evaluate_model(predicted_data, Ytest, n_out):
    # 初始化存储各个评估指标的字典。
    table = PrettyTable(['测试集指标', 'MSE', 'RMSE', 'MAE', 'MAPE', 'R2'])
    mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic = [], [], [], [], []
    
    for i in range(n_out):
        # 遍历每一个预测步长。每一列代表一步预测，现在是在求每步预测的指标
        actual = [float(row[i]) for row in Ytest]  # 一列列提取
        # 从测试集中提取实际值。
        predicted = [float(row[i]) for row in predicted_data]
        # 从预测结果中提取预测值。
        mse = mean_squared_error(actual, predicted)
        # 计算均方误差（MSE）。
        mse_dic.append(mse)
        rmse = math.sqrt(mean_squared_error(actual, predicted))
        # 计算均方根误差（RMSE）。
        rmse_dic.append(rmse)
        mae = mean_absolute_error(actual, predicted)
        # 计算平均绝对误差（MAE）。
        mae_dic.append(mae)
        MApe = mape(np.array(actual), np.array(predicted))
        # 计算平均绝对百分比误差（MAPE）。
        mape_dic.append(MApe)
        r2 = r2_score(actual, predicted)
        # 计算R平方值（R2）。
        r2_dic.append(r2)
        
        if n_out == 1:
            strr = '预测结果指标：'
        else:
            strr = '第' + str(i + 1) + '步预测结果指标：'
        
        table.add_row([strr, mse, rmse, mae, f'{MApe:.2f}%', f'{r2*100:.2f}%'])
    
    return mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic, table

# 可视化预测结果
def visualize_predictions(predicted_data, Ytest, scaler, n_out, feature_idx=0):
    # 为了可视化，我们只展示第一个预测步长的结果
    plt.figure(figsize=(12, 6))
    
    # 还原预测值和实际值的原始尺度
    # 创建一个与原始数据相同形状的数组
    pred_shape = (len(predicted_data), scaler.n_features_in_)
    test_shape = (len(Ytest), scaler.n_features_in_)
    
    pred_data_scaled = np.zeros(pred_shape)
    test_data_scaled = np.zeros(test_shape)
    
    # 将预测值和实际值放入对应位置
    pred_data_scaled[:, feature_idx] = predicted_data[:, 0]
    test_data_scaled[:, feature_idx] = Ytest[:, 0]
    
    # 使用scaler逆变换
    pred_data_original = scaler.inverse_transform(pred_data_scaled)[:, feature_idx]
    test_data_original = scaler.inverse_transform(test_data_scaled)[:, feature_idx]
    
    plt.plot(test_data_original, label='真实值', color='black')
    plt.plot(pred_data_original, label='预测值', color='purple', linestyle='--')
    plt.title('负荷预测结果对比')
    plt.xlabel('时间')
    plt.ylabel('负荷值')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图像
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig('results/load_forecast_results.png')
    plt.show()
    
    # 绘制多步预测结果
    if n_out > 1:
        plt.figure(figsize=(15, 10))
        for i in range(min(n_out, 4)):  # 最多显示4个预测步长
            plt.subplot(2, 2, i+1)
            
            # 还原预测值
            pred_data_scaled = np.zeros(pred_shape)
            pred_data_scaled[:, feature_idx] = predicted_data[:, i]
            pred_data_original = scaler.inverse_transform(pred_data_scaled)[:, feature_idx]
            
            plt.plot(test_data_original, label='真实值', color='black')
            plt.plot(pred_data_original, label=f'预测步长 {i+1}', color=f'C{i}')
            plt.title(f'第 {i+1} 步负荷预测结果')
            plt.xlabel('时间')
            plt.ylabel('负荷值')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/multi_step_forecast_results.png')
        plt.show()

# 主函数
def main():
    # 参数设置
    file_path = r'D:\pythondemo\project-training-2-master\Short-term electricity load forecasting\data\new_data.csv'  # 请替换为实际数据文件路径
    seq_len = 48  # 输入序列长度（小时）
    pred_len = 24  # 预测序列长度（小时）
    batch_size = 64
    epochs = 50
    learning_rate = 0.001
    
    # 模型参数
    input_dim = 14  # 输入特征维度
    d_model = 64  # Transformer模型维度
    nhead = 4  # 注意力头数
    num_layers = 2  # Transformer层数
    dim_feedforward = 128  # 前馈网络维度
    output_dim = pred_len  # 输出维度（预测步长）
    
    # 数据预处理
    print("开始数据预处理...")
    dataset, scaler = preprocess_data(file_path, seq_len, pred_len)
    
    # 划分训练集和测试集
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    print("初始化模型...")
    model = TransformerEncoderModel(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        output_dim=output_dim
    )
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    print("开始训练模型...")
    train_losses = train_model(model, train_loader, criterion, optimizer, epochs, device)
    
    # 测试模型
    print("开始测试模型...")
    predicted_data, Ytest = test_model(model, test_loader, device)
    
    # 评估模型
    print("评估模型性能...")
    mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic, table = evaluate_model(predicted_data, Ytest, pred_len)
    
    # 打印评估结果
    print(table)
    
    # 可视化预测结果
    visualize_predictions(predicted_data, Ytest, scaler, pred_len)
    
    # 保存模型
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), 'models/transformer_load_forecast.pth')
    print("模型训练完成并保存！")

if __name__ == "__main__":
    main()
    
   