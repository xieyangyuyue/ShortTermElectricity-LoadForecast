import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# 设置随机种子确保可复现性
np.random.seed(42)
tf.random.set_seed(42)

# ----------------------------
# 配置与初始化
# ----------------------------
# 解决中文显示问题
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False

# 超参数（可根据需求调整）
TIME_STEPS = 48  # 增加时间窗口大小为48小时
FEATURE_COLUMNS = ['weather_status', 'temperature', 'humidity', 'wind_speed', 
                   'wind_direction_angle', 'pressure', 'visibility', 
                   'precipitation', 'light', 'holiday', 'minute', 'week', 'year']
TARGET_COLUMN = 'value'  # 目标负荷值列名
TEST_SIZE = 0.2  # 测试集比例
VAL_SIZE = 0.1   # 验证集比例

# ----------------------------
# 1. 数据加载与预处理
# ----------------------------
def load_data(data_path):
    """加载数据并进行基础处理"""
    try:
        # 加载数据
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
        
        # 处理缺失值（结合前向填充和插值法）
        for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].interpolate(method='time')
                df[col] = df[col].fillna(method='ffill')
        
        print(f"缺失值处理完成，剩余缺失值：{df.isnull().sum().sum()}")
        
        # 提取时间特征
        df['datetime'] = pd.to_datetime(df['datetime']) if 'datetime' in df.columns else pd.to_datetime(df.index)
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        
        # 添加滞后特征
        for i in range(1, 5):  # 添加前4小时的负荷值作为特征
            df[f'lag_{i}'] = df[TARGET_COLUMN].shift(i)
        
        # 处理添加滞后特征后的缺失值
        df = df.dropna()
        
        # 更新特征列
        FEATURE_COLUMNS.extend(['hour', 'day_of_week', 'month'])
        FEATURE_COLUMNS.extend([f'lag_{i}' for i in range(1, 5)])
        
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
    
    # 使用StandardScaler代替MinMaxScaler，更适合GRU模型
    scaler_features = StandardScaler()
    scaler_target = StandardScaler()
    
    features_scaled = scaler_features.fit_transform(features)
    target_scaled = scaler_target.fit_transform(target)
    
    # 构建时间序列样本
    def create_sequences(features, target, time_steps):
        X, y = [], []
        for i in range(time_steps, len(features)):
            X.append(features[i-time_steps:i, :])
            y.append(target[i, 0])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(features_scaled, target_scaled, TIME_STEPS)
    
    # 按时间顺序划分训练集、验证集和测试集
    val_split = int(len(X) * (1 - TEST_SIZE - VAL_SIZE))
    test_split = int(len(X) * (1 - TEST_SIZE))
    
    X_train, X_val, X_test = X[:val_split], X[val_split:test_split], X[test_split:]
    y_train, y_val, y_test = y[:val_split], y[val_split:test_split], y[test_split:]
    
    print(f"时间序列样本构建完成：")
    print(f"训练集：{X_train.shape}，验证集：{X_val.shape}，测试集：{X_test.shape}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_target

# ----------------------------
# 2. 改进的GRU模型构建与训练
# ----------------------------
def build_gru_model(input_shape):
    """构建改进的GRU模型"""
    model = Sequential([
        # 双向GRU层，捕捉时间序列中的双向信息
        GRU(units=128, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        
        GRU(units=64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        
        # 多层全连接层，增加模型复杂度
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        
        Dense(1)  # 输出层
    ])
    
    # 使用Adam优化器并设置学习率
    optimizer = Adam(learning_rate=0.001)
    
    # 编译模型，使用MAE作为额外监控指标
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    model.summary()
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """训练模型并添加早停策略和学习率调整"""
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,  # 增加耐心值
        restore_best_weights=True
    )
    
    # 学习率调整策略
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001
    )
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=100,  # 增加训练轮数
        batch_size=64,  # 增加batch size
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # 绘制训练损失曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='训练MAE')
    plt.plot(history.history['val_mae'], label='验证MAE')
    plt.title('模型训练MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model

# ----------------------------
# 3. 增强的模型评估与可视化
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
    
    # 计算R2分数
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    
    print("\n===== 模型评估指标 =====")
    print(f"平均绝对误差（MAE）：{mae:.4f}")
    print(f"均方根误差（RMSE）：{rmse:.4f}")
    print(f"平均绝对百分比误差（MAPE）：{mape:.2f}%")
    print(f"R2分数：{r2:.4f}")
    
    return y_true, y_pred

def visualize_results(y_true, y_pred):
    """增强的可视化预测结果"""
    # 1. 随机抽取50个样本对比
    plt.figure(figsize=(14, 6))
    sample_indices = np.random.choice(len(y_true), 50, replace=False)
    
    plt.subplot(1, 2, 1)
    plt.scatter(range(50), y_true[sample_indices], label='实际负荷', color='#e63946', alpha=0.7)
    plt.scatter(range(50), y_pred[sample_indices], label='预测负荷', color='#457b9d', alpha=0.7)
    plt.xlabel('样本索引')
    plt.ylabel('电力负荷值')
    plt.title('预测值与实际值对比（随机50样本）')
    plt.legend()
    
    # 2. 预测误差分布
    plt.subplot(1, 2, 2)
    errors = y_true - y_pred
    plt.hist(errors, bins=30, alpha=0.7, color='#1d3557')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel('预测误差')
    plt.ylabel('频率')
    plt.title('预测误差分布')
    
    plt.tight_layout()
    plt.show()
    
    # 3. 前200个测试样本的趋势对比
    plt.figure(figsize=(16, 7))
    plt.plot(range(200), y_true[:200], label='实际负荷', color='#e63946', alpha=0.8, linewidth=2)
    plt.plot(range(200), y_pred[:200], label='预测负荷', color='#457b9d', alpha=0.8, linewidth=2)
    
    # 突出显示预测误差较大的区域
    large_error_indices = np.where(np.abs(y_true[:200] - y_pred[:200]) > np.std(errors) * 2)[0]
    for idx in large_error_indices:
        plt.axvspan(idx, idx+1, color='yellow', alpha=0.3)
    
    plt.xlabel('时间步（测试集前200个样本）')
    plt.ylabel('电力负荷值')
    plt.title('电力负荷预测趋势对比（误差大的区域已高亮）')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # 4. 按时间段的预测误差分析
    plt.figure(figsize=(15, 6))
    
    # 按小时分析误差
    hours = np.arange(24)
    hourly_errors = []
    for h in hours:
        indices = np.where(h % 24 == h)[0]
        if len(indices) > 0:
            hourly_errors.append(np.mean(np.abs(y_true[indices] - y_pred[indices])))
        else:
            hourly_errors.append(0)
    
    plt.bar(hours, hourly_errors, color='#a8dadc', alpha=0.8)
    plt.xlabel('小时')
    plt.ylabel('平均绝对误差')
    plt.title('按小时的预测误差分布')
    plt.xticks(hours)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
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
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_target = preprocess_data(df)
    
    # 2. 模型流程
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_gru_model(input_shape)
    model = train_model(model, X_train, y_train, X_val, y_val)
    
    # 3. 评估与可视化
    y_true, y_pred = evaluate_model(model, X_test, y_test, scaler_target)
    visualize_results(y_true, y_pred)