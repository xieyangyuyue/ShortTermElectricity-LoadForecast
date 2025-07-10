import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------
# 配置与初始化
# ----------------------------
# 解决中文显示问题
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False

# 超参数（可根据需求调整）
TIME_STEPS = 24  # 时间窗口大小（用前24小时数据预测）
FEATURE_COLUMNS = ['weather_status', 'temperature', 'humidity', 'wind_speed', 
                   'wind_direction_angle', 'pressure', 'visibility', 
                   'precipitation', 'light', 'holiday', 'minute', 'week', 'year']
TARGET_COLUMN = 'value'  # 目标负荷值列名
TEST_SIZE = 0.2  # 测试集比例（时间序列后20%为测试集）

# ----------------------------
# 1. 数据加载与预处理
# ----------------------------
def load_data(data_path):
    """加载数据并进行基础处理"""
    try:
        # 加载数据（确保路径为纯ASCII字符）
        df = pd.read_csv(data_path)
        print(f"数据集加载成功，形状：{df.shape}")
        
        # 移除无用索引列
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
            print("已移除'Unnamed: 0'列")
        
        # 检查必要列是否存在
        required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要列：{missing_cols}")
        
        # 处理缺失值（用前向填充，适合时间序列）
        df[FEATURE_COLUMNS + [TARGET_COLUMN]] = df[FEATURE_COLUMNS + [TARGET_COLUMN]].fillna(method='ffill')
        print(f"缺失值处理完成，剩余缺失值：{df.isnull().sum().sum()}")
        
        return df
    
    except FileNotFoundError:
        print(f"错误：未找到文件 {data_path}")
        exit()
    except Exception as e:
        print(f"数据加载错误：{str(e)}")
        exit()

def preprocess_data(df):
    """特征标准化与时间序列样本构建"""
    # 提取特征和目标变量
    features = df[FEATURE_COLUMNS]
    target = df[TARGET_COLUMN].values.reshape(-1, 1)
    
    # 特征标准化（0-1缩放）
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    
    features_scaled = scaler_features.fit_transform(features)
    target_scaled = scaler_target.fit_transform(target)
    
    # 构建时间序列样本：[样本数, 时间步长, 特征数]
    def create_sequences(features, target, time_steps):
        X, y = [], []
        for i in range(time_steps, len(features)):
            # 前time_steps个时间步的特征
            X.append(features[i-time_steps:i, :])
            # 当前时间步的目标值
            y.append(target[i, 0])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(features_scaled, target_scaled, TIME_STEPS)
    
    # 按时间顺序划分训练集和测试集（不随机打乱）
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"时间序列样本构建完成：")
    print(f"训练集：{X_train.shape}，测试集：{X_test.shape}")
    
    return (X_train, y_train), (X_test, y_test), scaler_target

# ----------------------------
# 2. GRU模型构建与训练
# ----------------------------
def build_gru_model(input_shape):
    """构建GRU模型"""
    model = Sequential([
        # GRU层：64个单元，返回序列用于堆叠（可选）
        GRU(units=64, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),  # 防止过拟合
        
        # 全连接层
        Dense(32, activation='relu'),
        Dense(1)  # 输出层（预测负荷值）
    ])
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='mean_squared_error'  # 回归任务常用损失
    )
    
    model.summary()
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """训练模型并添加早停策略"""
    early_stopping = EarlyStopping(
        monitor='val_loss',  # 监控验证集损失
        patience=5,  # 5个epoch无改善则停止
        restore_best_weights=True  # 恢复最佳权重
    )
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('GRU模型训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('MSE损失')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return model

# ----------------------------
# 3. 模型评估与可视化
# ----------------------------
def evaluate_model(model, X_test, y_test, scaler_target):
    """多指标评估模型性能"""
    # 预测并反归一化
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_target.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # 计算评估指标
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # 计算MAPE（避免除以0）
    non_zero_mask = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    
    print("\n===== 模型评估指标 =====")
    print(f"平均绝对误差（MAE）：{mae:.4f}")
    print(f"均方根误差（RMSE）：{rmse:.4f}")
    print(f"平均绝对百分比误差（MAPE）：{mape:.2f}%")
    
    return y_true, y_pred

def visualize_results(y_true, y_pred):
    """可视化预测结果"""
    # 1. 随机抽取50个样本对比
    plt.figure(figsize=(12, 6))
    sample_indices = np.random.choice(len(y_true), 50, replace=False)
    plt.scatter(range(50), y_true[sample_indices], label='实际负荷', color='#e63946', alpha=0.7)
    plt.scatter(range(50), y_pred[sample_indices], label='预测负荷', color='#457b9d', alpha=0.7)
    plt.xlabel('样本索引')
    plt.ylabel('电力负荷值')
    plt.title('预测值与实际值对比（随机50样本）')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 2. 前200个测试样本的趋势对比（时间序列特征）
    plt.figure(figsize=(14, 6))
    plt.plot(range(200), y_true[:200], label='实际负荷', color='#e63946', alpha=0.8)
    plt.plot(range(200), y_pred[:200], label='预测负荷', color='#457b9d', alpha=0.8)
    plt.xlabel('时间步（测试集前200个样本）')
    plt.ylabel('电力负荷值')
    plt.title('电力负荷预测趋势对比')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------
# 主函数
# ----------------------------
if __name__ == "__main__":
    # 配置数据路径（请替换为你的实际路径）
    DATA_PATH = "D:/pythondemo/project-training-2-master/Short-term electricity load forecasting/data/new_data.csv"
    
    # 1. 数据流程
    df = load_data(DATA_PATH)
    (X_train, y_train), (X_test, y_test), scaler_target = preprocess_data(df)
    
    # 2. 模型流程
    input_shape = (X_train.shape[1], X_train.shape[2])  # (TIME_STEPS, 特征数)
    model = build_gru_model(input_shape)
    model = train_model(model, X_train, y_train, X_test, y_test)
    
    # 3. 评估与可视化
    y_true, y_pred = evaluate_model(model, X_test, y_test, scaler_target)
    visualize_results(y_true, y_pred)