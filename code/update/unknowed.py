import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import math
import os

# 确保随机种子设置与训练时一致，以防某些层（如Dropout）的行为受影响，虽然预测时通常会设置为eval模式
torch.manual_seed(42)
np.random.seed(42)

# 检测GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义MAPE计算函数 (预测时可能不需要，但为了完整性可以保留)
def mape(y_true, y_pred):
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# 定义位置编码类 (与原始脚本一致)
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
        return x + self.pe[:x.size(1), :].unsqueeze(0)

# 定义Transformer编码器模型 (与原始脚本一致)
class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, dropout=0.1):
        super(TransformerEncoderModel, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, output_dim)
        )
        self.d_model = d_model

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        output = self.decoder(x)
        return output

        # 模型参数，需要与训练时保持一致
input_dim = 14  # 输入特征维度
d_model = 64  # Transformer模型维度
nhead = 4  # 注意力头数
num_layers = 2  # Transformer层数
dim_feedforward = 128  # 前馈网络维度
pred_len = 24  # 预测序列长度（小时），即 output_dim
output_dim = pred_len

# 实例化模型
model = TransformerEncoderModel(
    input_dim=input_dim,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    dim_feedforward=dim_feedforward,
    output_dim=output_dim
)

# 加载保存的模型权重
model_path = 'models/transformer_load_forecast.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device) # 将模型移动到相应的设备
    model.eval() # 设置模型为评估模式，这将关闭 dropout 等层
    print("模型加载成功！")
else:
    print(f"错误：模型文件未找到，请确保文件路径正确: {model_path}")
    exit()
    

    # 假设您的原始数据文件路径
original_data_file_path = r'../../code/new_data.csv' # 与训练时保持一致
seq_len = 48 # 输入序列长度，与训练时保持一致

# 重新加载原始数据以拟合Scaler，这是为了确保预测时的标准化与训练时一致
# 在实际生产环境中，您应该保存训练好的scaler并直接加载使用
df_original = pd.read_csv(original_data_file_path)
feature_cols = ['value', 'weather_status', 'temperature', 'humidity', 'wind_speed',
                'wind_direction_angle', 'pressure', 'visibility', 'precipitation',
                'light', 'holiday', 'minute', 'week', 'year']
data_original = df_original[feature_cols].values

scaler = MinMaxScaler()
scaler.fit(data_original) # 用整个原始数据集来拟合scaler

print("Scaler 已经基于原始训练数据重新拟合。")

# 准备新的预测数据
# 假设您有最近的 seq_len 个时间步的数据，用于预测未来的 pred_len 个时间步。
# 这里我们创建一个模拟的新数据作为示例
# 实际应用中，您应该从实时数据源或文件中获取最新的 seq_len 个时间步的数据

# 模拟新的输入数据：例如，从原始数据集中取最后 seq_len 个样本作为新输入
# 实际应用中，这里应该是您要预测的真实最新数据
new_raw_data_for_prediction = data_original[-seq_len:] # 获取原始数据中的最后 seq_len 行作为新输入

# 标准化新的输入数据
new_scaled_data = scaler.transform(new_raw_data_for_prediction)

# 转换为 PyTorch tensor
# 注意：模型期望的输入形状是 [batch_size, seq_len, input_dim]
# 这里我们只有一个样本，所以 batch_size=1
input_tensor = torch.tensor(new_scaled_data, dtype=torch.float32).unsqueeze(0).to(device)

print(f"新输入数据的形状: {input_tensor.shape}")

with torch.no_grad(): # 在预测时禁用梯度计算
    predictions_scaled = model(input_tensor)

# predictions_scaled 的形状将是 [1, output_dim] (即 [1, pred_len])
print(f"原始预测结果（标准化后）: {predictions_scaled.cpu().numpy()}")

# 创建一个与原始数据特征数量相同的零数组
# scaler.n_features_in_ 是训练时用于fit的特征数量 (这里是14)
# predictions_scaled.shape[1] 是预测的步长 (这里是 pred_len=24)
dummy_array = np.zeros((predictions_scaled.shape[1], scaler.n_features_in_))

# 将预测的负荷值（第一列）放入零数组的第一列
# 注意：如果您的模型预测的是多步，predictions_scaled 是 [batch_size, pred_len]
# 我们需要处理的是 predictions_scaled[0] 因为只有一个batch
dummy_array[:, 0] = predictions_scaled.cpu().numpy()[0]

# 逆标准化
predictions_original_scale = scaler.inverse_transform(dummy_array)[:, 0]

print("\n最终预测的未来负荷值（原始尺度）:")
print(predictions_original_scale)

# 您现在可以将 predictions_original_scale 用于进一步的分析或展示。
# 例如，如果您想知道未来24小时的负荷预测，这就是结果。