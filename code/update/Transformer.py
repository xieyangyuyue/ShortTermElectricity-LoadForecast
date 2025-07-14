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
import json


# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 检测GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")

# 改进的MAPE计算函数 - 添加阈值处理和零值保护
def mape(y_true, y_pred):
    epsilon = 1e-10  # 防止除零错误
    threshold = 0.05  # 忽略非常小的值，避免MAPE爆炸
    
    # 只计算真实值大于阈值的样本的MAPE
    valid_indices = y_true > threshold
    if np.sum(valid_indices) == 0:
        return 0  # 如果没有有效样本，返回0
    
    return np.mean(np.abs((y_true[valid_indices] - y_pred[valid_indices]) / 
                          (y_true[valid_indices] + epsilon))) * 100

# 改进的SMAPE计算函数 - 对称平均绝对百分比误差
def smape(y_true, y_pred):
    epsilon = 1e-10  # 防止除零错误
    return np.mean(2 * np.abs(y_true - y_pred) / 
                  (np.abs(y_true) + np.abs(y_pred) + epsilon)) * 100

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

# 改进的数据预处理函数 - 添加异常值处理
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
    
    # 异常值处理 - 使用IQR方法检测和修正异常值
    for col in range(data.shape[1]):
        # 计算四分位数和IQR
        q1 = np.percentile(data[:, col], 25)
        q3 = np.percentile(data[:, col], 75)
        iqr = q3 - q1
        
        # 定义异常值边界
        lower_bound = q1 - 3 * iqr  # 使用3倍IQR，比标准更严格
        upper_bound = q3 + 3 * iqr
        
        # 检测异常值
        outliers = (data[:, col] < lower_bound) | (data[:, col] > upper_bound)
        print(f"列 {feature_cols[col]} 检测到 {np.sum(outliers)} 个异常值")
        
        # 修正异常值 - 使用上下边界值替换
        data[outliers, col] = np.clip(data[outliers, col], lower_bound, upper_bound)
    
    # 数据标准化
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    # 创建数据集
    dataset = LoadForecastDataset(data, seq_len, pred_len)
    
    return dataset, scaler

# 训练函数
def train_model(model, train_loader, criterion, optimizer, epochs, device, scheduler=None):
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
        
        # 更新学习率（如果有调度器）
        if scheduler:
            scheduler.step(avg_loss)
            print(f'当前学习率: {optimizer.param_groups[0]["lr"]}')
    
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

# 改进的评估模型 - 添加多种评估指标
def evaluate_model(predicted_data, Ytest, n_out):
    # 初始化存储各个评估指标的字典。
    table = PrettyTable(['测试集指标', 'MSE', 'RMSE', 'MAE', 'MAPE', 'SMAPE', 'R2'])
    mse_dic, rmse_dic, mae_dic, mape_dic, smape_dic, r2_dic = [], [], [], [], [], []
    
    for i in range(n_out):
        # 遍历每一个预测步长。每一列代表一步预测，现在是在求每步预测的指标
        actual = np.array([float(row[i]) for row in Ytest])  # 一列列提取
        # 从测试集中提取实际值。
        predicted = np.array([float(row[i]) for row in predicted_data])
        # 从预测结果中提取预测值。
        
        # 计算各种评估指标
        mse = mean_squared_error(actual, predicted)
        mse_dic.append(mse)
        
        rmse = math.sqrt(mse)
        rmse_dic.append(rmse)
        
        mae = mean_absolute_error(actual, predicted)
        mae_dic.append(mae)
        
        MAPE = mape(actual, predicted)
        mape_dic.append(MAPE)
        
        SMAPE = smape(actual, predicted)
        smape_dic.append(SMAPE)
        
        r2 = r2_score(actual, predicted)
        r2_dic.append(r2)
        
        if n_out == 1:
            strr = '预测结果指标：'
        else:
            strr = '第' + str(i + 1) + '步预测结果指标：'
        
        table.add_row([strr, mse, rmse, mae, f'{MAPE:.2f}%', f'{SMAPE:.2f}%', f'{r2*100:.2f}%'])
    
    # 计算平均指标
    avg_mse = sum(mse_dic) / len(mse_dic)
    avg_rmse = sum(rmse_dic) / len(rmse_dic)
    avg_mae = sum(mae_dic) / len(mae_dic)
    avg_mape = sum(mape_dic) / len(mape_dic)
    avg_smape = sum(smape_dic) / len(smape_dic)
    avg_r2 = sum(r2_dic) / len(r2_dic)
    
    # 添加平均指标行
    table.add_row(['平均指标', f'{avg_mse:.6f}', f'{avg_rmse:.6f}', f'{avg_mae:.6f}', 
                   f'{avg_mape:.2f}%', f'{avg_smape:.2f}%', f'{avg_r2*100:.2f}%'])
    
    # 关键指标摘要
    metrics_summary = {
        'avg_mse': avg_mse,
        'avg_rmse': avg_rmse,
        'avg_mae': avg_mae,
        'avg_mape': avg_mape,
        'avg_smape': avg_smape,
        'avg_r2': avg_r2
    }
    
    return mse_dic, rmse_dic, mae_dic, mape_dic, smape_dic, r2_dic, table, metrics_summary

# 可视化训练损失
def visualize_training_loss(train_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失', color='blue')
    plt.title('模型训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失值 (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图像
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig('results/training_loss.png')
    plt.show()

# 可视化5天预测结果 - 适配96点/天格式，共5天
def visualize_predictions_5days(predicted_data, Ytest, scaler, feature_idx=0):
    # 显示5天数据，每天96个点，共480个点
    sample_count = min(5*96, len(predicted_data))
    
    plt.figure(figsize=(20, 8))
    
    # 还原预测值和实际值的原始尺度
    # 创建一个与原始数据相同形状的数组
    pred_shape = (sample_count, scaler.n_features_in_)
    test_shape = (sample_count, scaler.n_features_in_)
    
    pred_data_scaled = np.zeros(pred_shape)
    test_data_scaled = np.zeros(test_shape)
    
    # 将预测值和实际值放入对应位置（只取第一个预测步长）
    pred_data_scaled[:, feature_idx] = predicted_data[:sample_count, 0]
    test_data_scaled[:, feature_idx] = Ytest[:sample_count, 0]
    
    # 使用scaler逆变换
    pred_data_original = scaler.inverse_transform(pred_data_scaled)[:, feature_idx]
    test_data_original = scaler.inverse_transform(test_data_scaled)[:, feature_idx]
    
    # 计算误差
    errors = np.abs(pred_data_original - test_data_original)
    avg_error = np.mean(errors)
    
    # 创建时间轴（以小时为单位，5天共120小时）
    X = np.linspace(0, 5*24, sample_count, endpoint=False)
    
    plt.plot(X, test_data_original, label='真实负荷值', color='black', linewidth=2)
    plt.plot(X, pred_data_original, label='预测负荷值', color='purple', linestyle='--', linewidth=2)
    
    # 用阴影表示预测误差
    plt.fill_between(X, 
                     test_data_original - errors, 
                     test_data_original + errors, 
                     color='gray', alpha=0.2, label='误差范围')
    
    # 标记最大误差点
    max_error_idx = np.argmax(errors)
    plt.scatter([X[max_error_idx]], [test_data_original[max_error_idx]], color='red', s=100, 
                marker='*', label=f'最大误差: {errors[max_error_idx]:.2f}')
    
    # 添加每日分隔线
    for day in range(1, 5):
        plt.axvline(x=day*24, color='gray', linestyle=':', alpha=0.7)
        plt.text(day*24 + 1, plt.ylim()[1]*0.95, f'第{day+1}天', rotation=90)
    
    plt.title(f'电力负荷预测结果对比 (五天内每15分钟一个点, 平均误差: {avg_error:.2f})')
    plt.xlabel('时间 (小时)')
    plt.ylabel('负荷值')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图像
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig('results/load_forecast_5days.png')
    plt.show()
    
    # 准备JSON数据，包含5天的完整数据
    json_data = {
        'time_hours': X.tolist(),
        'actual_load': test_data_original.tolist(),
        'predicted_load': pred_data_original.tolist(),
        'errors': errors.tolist(),
        'average_error': float(avg_error),
        'max_error': {
            'value': float(np.max(errors)),
            'time_hour': float(X[np.argmax(errors)]),
            'actual_value': float(test_data_original[np.argmax(errors)]),
            'predicted_value': float(pred_data_original[np.argmax(errors)])
        }
    }
    
    # 保存为JSON文件
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    json_path = os.path.join(output_dir, 'load_forecast_5days.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"5天预测数据已保存到 {json_path}")
    
    # 打印关键预测结果
    print("\n===== 5天预测结果摘要 =====")
    print(f"• 平均预测误差: {avg_error:.2f}")
    print(f"• 最大预测误差: {np.max(errors):.2f} (发生在 {X[np.argmax(errors)]:.2f} 小时)")
    print(f"• 最小预测误差: {np.min(errors):.2f} (发生在 {X[np.argmin(errors)]:.2f} 小时)")
    
    # 分析异常值
    threshold = avg_error * 3  # 定义异常值阈值为平均误差的3倍
    outliers = errors > threshold
    if np.sum(outliers) > 0:
        print(f"\n检测到 {np.sum(outliers)} 个异常预测值（误差超过平均误差的3倍）")
        for i, is_outlier in enumerate(outliers):
            if is_outlier:
                time_point = X[i]  # 转换为小时
                print(f"  - 时间点 {time_point:.2f}小时: 误差={errors[i]:.2f}, 真实值={test_data_original[i]:.2f}, 预测值={pred_data_original[i]:.2f}")

# 可视化每天的预测结果
def visualize_daily_predictions(predicted_data, Ytest, scaler, feature_idx=0):
    # 每天96个点，展示5天
    for day in range(5):
        start_idx = day * 96
        end_idx = start_idx + 96
        
        # 确保不超过数据范围
        if end_idx > len(predicted_data):
            break
            
        plt.figure(figsize=(15, 6))
        
        # 还原预测值和实际值的原始尺度
        pred_shape = (96, scaler.n_features_in_)
        test_shape = (96, scaler.n_features_in_)
        
        pred_data_scaled = np.zeros(pred_shape)
        test_data_scaled = np.zeros(test_shape)
        
        # 将预测值和实际值放入对应位置
        pred_data_scaled[:, feature_idx] = predicted_data[start_idx:end_idx, 0]
        test_data_scaled[:, feature_idx] = Ytest[start_idx:end_idx, 0]
        
        # 使用scaler逆变换
        pred_data_original = scaler.inverse_transform(pred_data_scaled)[:, feature_idx]
        test_data_original = scaler.inverse_transform(test_data_scaled)[:, feature_idx]
        
        # 计算误差
        errors = np.abs(pred_data_original - test_data_original)
        avg_error = np.mean(errors)
        
        # 创建时间轴（以小时为单位）
        X = np.linspace(0, 24, 96, endpoint=False)
        
        plt.plot(X, test_data_original, label='真实负荷值', color='black', linewidth=2)
        plt.plot(X, pred_data_original, label='预测负荷值', color='blue', linestyle='--', linewidth=2)
        
        # 标记最大误差点
        max_error_idx = np.argmax(errors)
        plt.scatter([X[max_error_idx]], [test_data_original[max_error_idx]], color='red', s=100, 
                    marker='*', label=f'最大误差: {errors[max_error_idx]:.2f}')
        
        plt.title(f'第{day+1}天预测结果 (平均误差: {avg_error:.2f})')
        plt.xlabel('时间 (小时)')
        plt.ylabel('负荷值')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 保存图像
        if not os.path.exists('results'):
            os.makedirs('results')
        plt.savefig(f'results/load_forecast_day_{day+1}.png')
        plt.show()

# 主函数
def main():
    # 参数设置
    file_path = r'D:\pythondemo\project-training-2-master\Short-term electricity load forecasting\data\new_data.csv'  # 请替换为实际数据文件路径
    seq_len = 96  # 输入序列长度（一天，每15分钟一个点）
    pred_len = 96  # 预测序列长度（一天，每15分钟一个点）
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
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # 训练模型
    print("开始训练模型...")
    train_losses = train_model(model, train_loader, criterion, optimizer, epochs, device, scheduler)
    
    # 可视化训练损失
    visualize_training_loss(train_losses)
    
    # 测试模型
    print("开始测试模型...")
    predicted_data, Ytest = test_model(model, test_loader, device)
    
    # 评估模型
    print("评估模型性能...")
    mse_dic, rmse_dic, mae_dic, mape_dic, smape_dic, r2_dic, table, metrics_summary = evaluate_model(predicted_data, Ytest, pred_len)
    
    # 打印评估结果
    print(table)
    
    # 打印关键指标摘要
    print("\n===== 模型性能摘要 =====")
    print(f"• 平均MSE: {metrics_summary['avg_mse']:.6f}")
    print(f"• 平均RMSE: {metrics_summary['avg_rmse']:.6f}")
    print(f"• 平均MAE: {metrics_summary['avg_mae']:.6f}")
    print(f"• 改进的平均MAPE: {metrics_summary['avg_mape']:.2f}%")
    print(f"• 对称平均绝对百分比误差(SMAPE): {metrics_summary['avg_smape']:.2f}%")
    print(f"• 平均R²: {metrics_summary['avg_r2']*100:.2f}%")
    
    # 可视化5天预测结果
    visualize_predictions_5days(predicted_data, Ytest, scaler)
    
    # 可视化每天的预测结果
    visualize_daily_predictions(predicted_data, Ytest, scaler)
    
    # 保存模型
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), 'models/transformer_load_forecast_5days.pth')
    print("\n模型训练完成并保存！")
    print("结果已保存至'results'文件夹")

if __name__ == "__main__":
    main()
    

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from prettytable import PrettyTable
# import math
# import os
# from tqdm import tqdm  # 用于显示训练进度

# # 设置随机种子以确保结果可复现
# torch.manual_seed(42)
# np.random.seed(42)

# # 检测GPU可用性
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"使用设备: {device}")
# if torch.cuda.is_available():
#     print(f"GPU名称: {torch.cuda.get_device_name(0)}")

# # 定义MAPE计算函数
# def mape(y_true, y_pred):
#     # 处理接近零的值，避免除零错误
#     epsilon = 1e-10
#     return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# # 定义位置编码类
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)
        
#     def forward(self, x):
#         # x: [batch_size, seq_len, feature_dim]
#         return x + self.pe[:x.size(1), :].unsqueeze(0)

# # 定义Transformer编码器模型
# class TransformerEncoderModel(nn.Module):
#     def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, dropout=0.1):
#         super(TransformerEncoderModel, self).__init__()
#         self.input_embedding = nn.Linear(input_dim, d_model)
#         self.positional_encoding = PositionalEncoding(d_model)
        
#         # 创建Transformer编码器层
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True  # PyTorch 1.9+ 支持batch_first
#         )
        
#         # 创建多层Transformer编码器
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         # 输出层
#         self.decoder = nn.Sequential(
#             nn.Linear(d_model, d_model//2),
#             nn.ReLU(),
#             nn.Linear(d_model//2, output_dim)
#         )
        
#         self.d_model = d_model
        
#     def forward(self, x):
#         # x shape: [batch_size, seq_len, input_dim]
#         x = self.input_embedding(x)  # [batch_size, seq_len, d_model]
#         x = self.positional_encoding(x)  # 添加位置编码
#         x = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]
        
#         # 取序列最后一个时间步的输出作为预测结果
#         x = x[:, -1, :]  # [batch_size, d_model]
#         output = self.decoder(x)  # [batch_size, output_dim]
        
#         return output

# # 定义负荷预测数据集类
# class LoadForecastDataset(Dataset):
#     def __init__(self, data, seq_len, pred_len):
#         self.seq_len = seq_len
#         self.pred_len = pred_len
#         self.data = data
#         self.length = len(data) - seq_len - pred_len + 1
        
#     def __len__(self):
#         return self.length
    
#     def __getitem__(self, idx):
#         # 输入序列
#         x = self.data[idx:idx+self.seq_len]
#         # 目标序列（负荷值）
#         y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len, 0]  # 第一列是负荷值
        
#         return x, y

# # 数据预处理函数
# def preprocess_data(file_path, seq_len, pred_len):
#     # 读取数据
#     df = pd.read_csv(file_path)
    
#     # 提取特征列和目标列
#     feature_cols = ['value', 'weather_status', 'temperature', 'humidity', 'wind_speed', 
#                     'wind_direction_angle', 'pressure', 'visibility', 'precipitation', 
#                     'light', 'holiday', 'minute', 'week', 'year']
    
#     # 确保所有特征列都存在
#     for col in feature_cols:
#         if col not in df.columns:
#             raise ValueError(f"列 {col} 不存在于数据集中")
    
#     # 提取数据
#     data = df[feature_cols].values
    
#     # 数据标准化
#     scaler = MinMaxScaler()
#     data = scaler.fit_transform(data)
    
#     # 创建数据集
#     dataset = LoadForecastDataset(data, seq_len, pred_len)
    
#     return dataset, scaler

# # 训练函数
# def train_model(model, train_loader, criterion, optimizer, epochs, device):
#     model.train()
#     model.to(device)  # 将模型移至GPU
    
#     train_losses = []
    
#     for epoch in range(epochs):
#         epoch_loss = 0
#         progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}')
        
#         for i, (inputs, targets) in progress_bar:
#             # 将数据移至GPU
#             inputs = inputs.to(device).float()
#             targets = targets.to(device).float()
            
#             # 前向传播
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
            
#             # 反向传播和优化
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             epoch_loss += loss.item()
            
#             # 更新进度条
#             progress_bar.set_postfix({'loss': loss.item()})
        
#         avg_loss = epoch_loss / len(train_loader)
#         train_losses.append(avg_loss)
#         print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}')
    
#     return train_losses

# # 测试函数
# def test_model(model, test_loader, device):
#     model.eval()
#     model.to(device)  # 确保模型在GPU上
    
#     all_predictions = []
#     all_targets = []
    
#     with torch.no_grad():
#         for inputs, targets in test_loader:
#             # 将数据移至GPU
#             inputs = inputs.to(device).float()
#             targets = targets.to(device).float()
            
#             # 预测
#             outputs = model(inputs)
            
#             # 将结果移回CPU并收集
#             all_predictions.append(outputs.cpu().numpy())
#             all_targets.append(targets.cpu().numpy())
    
#     # 转换为numpy数组
#     predicted_data = np.concatenate(all_predictions)
#     Ytest = np.concatenate(all_targets)
    
#     return predicted_data, Ytest

# # 评估模型
# def evaluate_model(predicted_data, Ytest, n_out):
#     # 初始化存储各个评估指标的字典。
#     table = PrettyTable(['测试集指标', 'MSE', 'RMSE', 'MAE', 'MAPE', 'R2'])
#     mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic = [], [], [], [], []
    
#     for i in range(n_out):
#         # 遍历每一个预测步长。每一列代表一步预测，现在是在求每步预测的指标
#         actual = [float(row[i]) for row in Ytest]  # 一列列提取
#         # 从测试集中提取实际值。
#         predicted = [float(row[i]) for row in predicted_data]
#         # 从预测结果中提取预测值。
#         mse = mean_squared_error(actual, predicted)
#         # 计算均方误差（MSE）。
#         mse_dic.append(mse)
#         rmse = math.sqrt(mean_squared_error(actual, predicted))
#         # 计算均方根误差（RMSE）。
#         rmse_dic.append(rmse)
#         mae = mean_absolute_error(actual, predicted)
#         # 计算平均绝对误差（MAE）。
#         mae_dic.append(mae)
#         MApe = mape(np.array(actual), np.array(predicted))
#         # 计算平均绝对百分比误差（MAPE）。
#         mape_dic.append(MApe)
#         r2 = r2_score(actual, predicted)
#         # 计算R平方值（R2）。
#         r2_dic.append(r2)
        
#         if n_out == 1:
#             strr = '预测结果指标：'
#         else:
#             strr = '第' + str(i + 1) + '步预测结果指标：'
        
#         table.add_row([strr, mse, rmse, mae, f'{MApe:.2f}%', f'{r2*100:.2f}%'])
    
#     # 计算平均指标
#     avg_mse = sum(mse_dic) / len(mse_dic)
#     avg_rmse = sum(rmse_dic) / len(rmse_dic)
#     avg_mae = sum(mae_dic) / len(mae_dic)
#     avg_mape = sum(mape_dic) / len(mape_dic)
#     avg_r2 = sum(r2_dic) / len(r2_dic)
    
#     # 添加平均指标行
#     table.add_row(['平均指标', f'{avg_mse:.6f}', f'{avg_rmse:.6f}', f'{avg_mae:.6f}', f'{avg_mape:.2f}%', f'{avg_r2*100:.2f}%'])
    
#     # 关键指标摘要
#     metrics_summary = {
#         'avg_mse': avg_mse,
#         'avg_rmse': avg_rmse,
#         'avg_mae': avg_mae,
#         'avg_mape': avg_mape,
#         'avg_r2': avg_r2
#     }
    
#     return mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic, table, metrics_summary

# # 可视化训练损失
# def visualize_training_loss(train_losses):
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_losses, label='训练损失', color='blue')
#     plt.title('模型训练损失')
#     plt.xlabel('Epoch')
#     plt.ylabel('损失值 (MSE)')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
    
#     # 保存图像
#     if not os.path.exists('results'):
#         os.makedirs('results')
#     plt.savefig('results/training_loss.png')
#     plt.show()

# # 可视化预测结果 - 简化版
# def visualize_predictions_simplified(predicted_data, Ytest, scaler, feature_idx=0):
#     # 为了简化展示，我们只显示前100个样本的第一个预测步长
#     sample_count = min(100, len(predicted_data))
    
#     plt.figure(figsize=(15, 6))
    
#     # 还原预测值和实际值的原始尺度
#     # 创建一个与原始数据相同形状的数组
#     pred_shape = (sample_count, scaler.n_features_in_)
#     test_shape = (sample_count, scaler.n_features_in_)
    
#     pred_data_scaled = np.zeros(pred_shape)
#     test_data_scaled = np.zeros(test_shape)
    
#     # 将预测值和实际值放入对应位置（只取第一个预测步长）
#     pred_data_scaled[:, feature_idx] = predicted_data[:sample_count, 0]
#     test_data_scaled[:, feature_idx] = Ytest[:sample_count, 0]
    
#     # 使用scaler逆变换
#     pred_data_original = scaler.inverse_transform(pred_data_scaled)[:, feature_idx]
#     test_data_original = scaler.inverse_transform(test_data_scaled)[:, feature_idx]
    
#     # 计算误差
#     errors = np.abs(pred_data_original - test_data_original)
#     avg_error = np.mean(errors)
    
#     plt.plot(test_data_original, label='真实负荷值', color='black', linewidth=2)
#     plt.plot(pred_data_original, label='预测负荷值', color='purple', linestyle='--', linewidth=2)
    
#     # 用阴影表示预测误差
#     plt.fill_between(range(sample_count), 
#                      test_data_original - errors, 
#                      test_data_original + errors, 
#                      color='gray', alpha=0.2, label='误差范围')
    
#     plt.title(f'电力负荷预测结果对比 (平均误差: {avg_error:.2f})')
#     plt.xlabel('时间')
#     plt.ylabel('负荷值')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
    
#     # 保存图像
#     plt.savefig('results/simplified_load_forecast.png')
#     plt.show()
    
#     # 打印关键预测结果
#     print("\n===== 预测结果摘要 =====")
#     print(f"• 平均预测误差: {avg_error:.2f}")
#     print(f"• 最大预测误差: {np.max(errors):.2f} (发生在时间点 {np.argmax(errors)})")
#     print(f"• 最小预测误差: {np.min(errors):.2f} (发生在时间点 {np.argmin(errors)})")

# # 主函数
# def main():
#     # 参数设置
#     file_path = r'D:\pythondemo\project-training-2-master\Short-term electricity load forecasting\data\new_data.csv'  # 请替换为实际数据文件路径
#     seq_len = 48  # 输入序列长度（小时）
#     pred_len = 24  # 预测序列长度（小时）
#     batch_size = 64
#     epochs = 50
#     learning_rate = 0.001
    
#     # 模型参数
#     input_dim = 14  # 输入特征维度
#     d_model = 64  # Transformer模型维度
#     nhead = 4  # 注意力头数
#     num_layers = 2  # Transformer层数
#     dim_feedforward = 128  # 前馈网络维度
#     output_dim = pred_len  # 输出维度（预测步长）
    
#     # 数据预处理
#     print("开始数据预处理...")
#     dataset, scaler = preprocess_data(file_path, seq_len, pred_len)
    
#     # 划分训练集和测试集
#     train_size = int(len(dataset) * 0.8)
#     test_size = len(dataset) - train_size
#     train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
#     # 创建数据加载器
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
#     # 初始化模型
#     print("初始化模型...")
#     model = TransformerEncoderModel(
#         input_dim=input_dim,
#         d_model=d_model,
#         nhead=nhead,
#         num_layers=num_layers,
#         dim_feedforward=dim_feedforward,
#         output_dim=output_dim
#     )
    
#     # 定义损失函数和优化器
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
#     # 训练模型
#     print("开始训练模型...")
#     train_losses = train_model(model, train_loader, criterion, optimizer, epochs, device)
    
#     # 可视化训练损失
#     visualize_training_loss(train_losses)
    
#     # 测试模型
#     print("开始测试模型...")
#     predicted_data, Ytest = test_model(model, test_loader, device)
    
#     # 评估模型
#     print("评估模型性能...")
#     mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic, table, metrics_summary = evaluate_model(predicted_data, Ytest, pred_len)
    
#     # 打印评估结果
#     print(table)
    
#     # 打印关键指标摘要
#     print("\n===== 模型性能摘要 =====")
#     print(f"• 平均MSE: {metrics_summary['avg_mse']:.6f}")
#     print(f"• 平均RMSE: {metrics_summary['avg_rmse']:.6f}")
#     print(f"• 平均MAE: {metrics_summary['avg_mae']:.6f}")
#     print(f"• 平均MAPE: {metrics_summary['avg_mape']:.2f}%")
#     print(f"• 平均R²: {metrics_summary['avg_r2']*100:.2f}%")
    
#     # 可视化预测结果（简化版）
#     visualize_predictions_simplified(predicted_data, Ytest, scaler)
    
#     # 保存模型
#     if not os.path.exists('models'):
#         os.makedirs('models')
#     torch.save(model.state_dict(), 'models/transformer_load_forecast.pth')
#     print("\n模型训练完成并保存！")
#     print("结果已保存至'results'文件夹")

# if __name__ == "__main__":
#     main()