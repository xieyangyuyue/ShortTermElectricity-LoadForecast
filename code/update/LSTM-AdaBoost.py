# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout
# from tensorflow.keras.optimizers import Adam
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import os

# # 设置日志级别，只显示错误信息
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# class LSTMAdaBoost:
#     def __init__(self, n_estimators=10, seq_length=24, learning_rate=0.001, use_gpu=True):
#         """
#         初始化LSTM-AdaBoost模型
        
#         参数:
#         n_estimators: 弱学习器(LSTM)的数量
#         seq_length: 时间序列长度
#         learning_rate: 优化器学习率
#         use_gpu: 是否使用GPU加速
#         """
#         self.n_estimators = n_estimators
#         self.seq_length = seq_length
#         self.learning_rate = learning_rate
#         self.estimators = []
#         self.estimator_weights = []
#         self.scaler = StandardScaler()
#         self.y_scaler = StandardScaler()  # 新增：用于目标变量的标准化
        
#         # 配置GPU内存使用
#         if use_gpu:
#             self._configure_gpu()
            
#     def _configure_gpu(self):
#         """配置GPU内存使用，防止显存溢出"""
#         try:
#             gpus = tf.config.list_physical_devices('GPU')
#             if gpus:
#                 # 设置GPU内存增长，避免一次性占用所有显存
#                 for gpu in gpus:
#                     tf.config.experimental.set_memory_growth(gpu, True)
#                 print(f"已检测到 {len(gpus)} 个GPU，将用于模型训练")
#             else:
#                 print("未检测到GPU，将使用CPU进行训练")
#         except RuntimeError as e:
#             print(f"GPU配置失败: {e}")
            
#     def _create_dataset(self, X, y, seq_length):
#         """创建时间序列数据集"""
#         Xs, ys = [], []
#         for i in range(len(X) - seq_length):
#             Xs.append(X[i:(i+seq_length)])
#             ys.append(y[i+seq_length])
#         return np.array(Xs), np.array(ys)
    
#     def _build_lstm_model(self, input_shape):
#         """构建LSTM模型"""
#         model = Sequential()
#         model.add(LSTM(32, return_sequences=True, input_shape=input_shape))  # 减少单元数量
#         model.add(Dropout(0.3))  # 增加Dropout比例
#         model.add(LSTM(16))
#         model.add(Dropout(0.3))  # 增加Dropout比例
#         model.add(Dense(1))
        
#         model.compile(
#             loss='mse',
#             optimizer=Adam(learning_rate=self.learning_rate),
#             metrics=['mae']
#         )
        
#         return model
    
#     def fit(self, X, y, epochs=30, batch_size=32, validation_split=0.1, verbose=1):
#         """训练LSTM-AdaBoost模型"""
#         # 数据标准化
#         X = self.scaler.fit_transform(X)
        
#         # 新增：对目标变量进行标准化
#         y_reshaped = y.reshape(-1, 1)
#         y_scaled = self.y_scaler.fit_transform(y_reshaped).flatten()
        
#         # 创建时间序列数据
#         X_seq, y_seq = self._create_dataset(X, y_scaled, self.seq_length)
        
#         # 划分训练集和验证集
#         X_train, X_val, y_train, y_val = train_test_split(
#             X_seq, y_seq, test_size=validation_split, random_state=42
#         )
        
#         # 初始化样本权重
#         sample_weights = np.ones(len(X_train)) / len(X_train)
        
#         for i in range(self.n_estimators):
#             print(f"训练第 {i+1}/{self.n_estimators} 个LSTM模型")
            
#             # 创建并训练LSTM模型
#             model = self._build_lstm_model((self.seq_length, X_train.shape[2]))
            
#             # 使用当前样本权重训练模型
#             history = model.fit(
#                 X_train, y_train,
#                 epochs=epochs,
#                 batch_size=batch_size,
#                 validation_data=(X_val, y_val),
#                 verbose=verbose,
#                 sample_weight=sample_weights
#             )
            
#             # 预测训练集
#             y_pred = model.predict(X_train).flatten()
            
#             # 计算误差
#             error = np.abs(y_pred - y_train)
            
#             # 计算误差率
#             weighted_error = np.sum(sample_weights * error) / np.sum(sample_weights)
            
#             # 计算学习器权重
#             alpha = 0.5 * np.log((1 - weighted_error) / max(weighted_error, 1e-10))
            
#             # 优化：回归问题的样本权重更新（基于误差率）
#             error_rate = error / (np.max(error) + 1e-10)  # 归一化误差
#             sample_weights *= np.exp(alpha * error_rate)  # 误差大的样本权重增加
#             sample_weights /= np.sum(sample_weights)  # 归一化
            
#             # 保存模型和权重
#             self.estimators.append(model)
#             self.estimator_weights.append(alpha)
            
#             print(f"模型 {i+1} 训练完成: 误差率={weighted_error:.4f}, 学习器权重={alpha:.4f}")
            
#         return self
    
#     def predict(self, X):
#         """预测"""
#         # 数据标准化
#         X = self.scaler.transform(X)
        
#         # 创建时间序列数据
#         X_seq, _ = self._create_dataset(X, np.zeros(len(X)), self.seq_length)
        
#         # 集成预测
#         predictions = np.zeros(len(X_seq))
        
#         for i, model in enumerate(self.estimators):
#             pred = model.predict(X_seq).flatten()
#             predictions += self.estimator_weights[i] * pred
            
#         # 归一化预测结果
#         if sum(self.estimator_weights) > 0:
#             predictions /= sum(self.estimator_weights)
            
#         # 新增：反标准化预测结果
#         predictions = self.y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            
#         return predictions
    
#     def evaluate(self, X, y_true):
#         """评估模型性能"""
#         y_pred = self.predict(X)
        
#         # 确保预测和真实值长度匹配
#         y_true = y_true[self.seq_length:]
        
#         mse = mean_squared_error(y_true, y_pred)
#         mae = mean_absolute_error(y_true, y_pred)
#         r2 = r2_score(y_true, y_pred)
        
#         print(f"评估结果: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
#         # 新增：计算相对误差
#         mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#         print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")
        
#         return {
#             'mse': mse,
#             'mae': mae,
#             'r2': r2,
#             'mape': mape
#         }
    
#     def plot_predictions(self, X, y_true, title="LSTM-AdaBoost预测结果"):
#         """绘制预测结果"""
#         y_pred = self.predict(X)
        
#         # 确保预测和真实值长度匹配
#         y_true = y_true[self.seq_length:]
        
#         plt.figure(figsize=(14, 7))
        
#         # 绘制整体预测结果
#         plt.subplot(2, 1, 1)
#         plt.plot(y_true, label='真实值')
#         plt.plot(y_pred, label='预测值')
#         plt.title(title)
#         plt.xlabel('时间')
#         plt.ylabel('值')
#         plt.legend()
#         plt.grid(True)
        
#         # 绘制局部放大图，便于观察细节
#         plt.subplot(2, 1, 2)
#         sample_size = min(100, len(y_true))  # 取前100个点或全部数据
#         plt.plot(y_true[:sample_size], label='真实值')
#         plt.plot(y_pred[:sample_size], label='预测值')
#         plt.title('预测细节（前100个数据点）')
#         plt.xlabel('时间')
#         plt.ylabel('值')
#         plt.legend()
#         plt.grid(True)
        
#         plt.tight_layout()
#         plt.show()

# # 数据加载和预处理函数
# def load_and_preprocess_data(file_path):
#     """加载并预处理数据"""
#     # 加载数据
#     df = pd.read_csv(file_path)
    
#     # 检查缺失值
#     if df.isnull().any().any():
#         print("检测到缺失值，正在进行填充...")
#         df = df.fillna(method='ffill')  # 向前填充
        
#     # 提取特征和目标变量
#     X = df.drop(['value'], axis=1).values
#     y = df['value'].values
    
#     print(f"数据加载完成: 样本数={len(X)}, 特征数={X.shape[1]}")
#     return X, y

# # 主函数
# def main():
#     # 实际使用时请替换为真实文件路径
#     file_path = r'D:\pythondemo\project-training-2-master\Short-term electricity load forecasting\data\new_data.csv'
    
#     # 加载数据
#     X, y = load_and_preprocess_data(file_path)
    
#     # 创建并训练模型
#     model = LSTMAdaBoost(n_estimators=10, seq_length=24, use_gpu=True)
#     model.fit(X, y, epochs=30, batch_size=32, verbose=1)
    
#     # 评估模型
#     metrics = model.evaluate(X, y)
    
#     # 绘制预测结果
#     model.plot_predictions(X, y, title="LSTM-AdaBoost 电力负荷预测结果")
    
#     # 保存模型
#     model_path = r'D:\pythondemo\lstm_adaboost_model'
#     os.makedirs(model_path, exist_ok=True)
    
#     # 保存每个弱学习器
#     for i, estimator in enumerate(model.estimators):
#         estimator.save(f"{model_path}/lstm_estimator_{i}.h5")
    
#     # 保存模型配置
#     with open(f"{model_path}/model_config.txt", 'w') as f:
#         f.write(f"n_estimators: {model.n_estimators}\n")
#         f.write(f"seq_length: {model.seq_length}\n")
#         f.write(f"learning_rate: {model.learning_rate}\n")
#         f.write(f"estimator_weights: {model.estimator_weights}\n")
    
#     print(f"模型已保存至: {model_path}")

# if __name__ == "__main__":
#     main()    

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # 新增导入
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# 设置日志级别，只显示错误信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class LSTMAdaBoost:
    def __init__(self, n_estimators=10, seq_length=24, learning_rate=0.001, use_gpu=True, 
                 early_stopping_patience=5, reduce_lr_patience=3):  # 新增参数
        """
        初始化LSTM-AdaBoost模型
        
        参数:
        n_estimators: 弱学习器(LSTM)的数量
        seq_length: 时间序列长度
        learning_rate: 优化器学习率
        use_gpu: 是否使用GPU加速
        early_stopping_patience: 早停等待轮数
        reduce_lr_patience: 学习率降低等待轮数
        """
        self.n_estimators = n_estimators
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.estimators = []
        self.estimator_weights = []
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()  # 新增：用于目标变量的标准化
        self.early_stopping_patience = early_stopping_patience  # 新增
        self.reduce_lr_patience = reduce_lr_patience  # 新增
        
        # 配置GPU内存使用
        if use_gpu:
            self._configure_gpu()
            
    def _configure_gpu(self):
        """配置GPU内存使用，防止显存溢出"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # 设置GPU内存增长，避免一次性占用所有显存
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"已检测到 {len(gpus)} 个GPU，将用于模型训练")
            else:
                print("未检测到GPU，将使用CPU进行训练")
        except RuntimeError as e:
            print(f"GPU配置失败: {e}")
            
    def _create_dataset(self, X, y, seq_length):
        """创建时间序列数据集"""
        Xs, ys = [], []
        for i in range(len(X) - seq_length):
            Xs.append(X[i:(i+seq_length)])
            ys.append(y[i+seq_length])
        return np.array(Xs), np.array(ys)
    
    def _build_lstm_model(self, input_shape):
        """构建LSTM模型"""
        model = Sequential()
        model.add(LSTM(32, return_sequences=True, input_shape=input_shape))  # 减少单元数量
        model.add(Dropout(0.3))  # 增加Dropout比例
        model.add(LSTM(16))
        model.add(Dropout(0.3))  # 增加Dropout比例
        model.add(Dense(1))
        
        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X, y, epochs=30, batch_size=32, validation_split=0.1, verbose=1):
        """训练LSTM-AdaBoost模型"""
        # 数据标准化
        X = self.scaler.fit_transform(X)
        
        # 新增：对目标变量进行标准化
        y_reshaped = y.reshape(-1, 1)
        y_scaled = self.y_scaler.fit_transform(y_reshaped).flatten()
        
        # 创建时间序列数据
        X_seq, y_seq = self._create_dataset(X, y_scaled, self.seq_length)
        
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_seq, y_seq, test_size=validation_split, random_state=42
        )
        
        # 初始化样本权重
        sample_weights = np.ones(len(X_train)) / len(X_train)
        
        for i in range(self.n_estimators):
            print(f"训练第 {i+1}/{self.n_estimators} 个LSTM模型")
            
            # 创建并训练LSTM模型
            model = self._build_lstm_model((self.seq_length, X_train.shape[2]))
            
            # 新增：早停策略和学习率调度
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.reduce_lr_patience,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            # 使用当前样本权重训练模型
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                verbose=verbose,
                sample_weight=sample_weights,
                callbacks=callbacks  # 新增
            )
            
            # 记录每个模型的最佳验证损失
            best_val_loss = min(history.history['val_loss'])
            print(f"模型 {i+1} 最佳验证损失: {best_val_loss:.6f}")
            
            # 预测训练集
            y_pred = model.predict(X_train).flatten()
            
            # 计算误差
            error = np.abs(y_pred - y_train)
            
            # 计算误差率
            weighted_error = np.sum(sample_weights * error) / np.sum(sample_weights)
            
            # 计算学习器权重
            alpha = 0.5 * np.log((1 - weighted_error) / max(weighted_error, 1e-10))
            
            # 优化：回归问题的样本权重更新（基于误差率）
            error_rate = error / (np.max(error) + 1e-10)  # 归一化误差
            sample_weights *= np.exp(alpha * error_rate)  # 误差大的样本权重增加
            sample_weights /= np.sum(sample_weights)  # 归一化
            
            # 保存模型和权重
            self.estimators.append(model)
            self.estimator_weights.append(alpha)
            
            print(f"模型 {i+1} 训练完成: 误差率={weighted_error:.4f}, 学习器权重={alpha:.4f}")
            
        return self
    
    def predict(self, X):
        """预测"""
        # 数据标准化
        X = self.scaler.transform(X)
        
        # 创建时间序列数据
        X_seq, _ = self._create_dataset(X, np.zeros(len(X)), self.seq_length)
        
        # 集成预测
        predictions = np.zeros(len(X_seq))
        
        for i, model in enumerate(self.estimators):
            pred = model.predict(X_seq).flatten()
            predictions += self.estimator_weights[i] * pred
            
        # 归一化预测结果
        if sum(self.estimator_weights) > 0:
            predictions /= sum(self.estimator_weights)
            
        # 新增：反标准化预测结果
        predictions = self.y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            
        return predictions
    
    def evaluate(self, X, y_true):
        """评估模型性能"""
        y_pred = self.predict(X)
        
        # 确保预测和真实值长度匹配
        y_true = y_true[self.seq_length:]
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"评估结果: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        # 新增：计算相对误差
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    def plot_predictions(self, X, y_true, title="LSTM-AdaBoost预测结果"):
        """绘制预测结果"""
        y_pred = self.predict(X)
        
        # 确保预测和真实值长度匹配
        y_true = y_true[self.seq_length:]
        
        plt.figure(figsize=(14, 7))
        
        # 绘制整体预测结果
        plt.subplot(2, 1, 1)
        plt.plot(y_true, label='真实值')
        plt.plot(y_pred, label='预测值')
        plt.title(title)
        plt.xlabel('时间')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True)
        
        # 绘制局部放大图，便于观察细节
        plt.subplot(2, 1, 2)
        sample_size = min(100, len(y_true))  # 取前100个点或全部数据
        plt.plot(y_true[:sample_size], label='真实值')
        plt.plot(y_pred[:sample_size], label='预测值')
        plt.title('预测细节（前100个数据点）')
        plt.xlabel('时间')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# 数据加载和预处理函数
def load_and_preprocess_data(file_path):
    """加载并预处理数据"""
    # 加载数据
    df = pd.read_csv(file_path)
    
    # 检查缺失值
    if df.isnull().any().any():
        print("检测到缺失值，正在进行填充...")
        df = df.fillna(method='ffill')  # 向前填充
        
    # 提取特征和目标变量
    X = df.drop(['value'], axis=1).values
    y = df['value'].values
    
    print(f"数据加载完成: 样本数={len(X)}, 特征数={X.shape[1]}")
    return X, y

# 主函数
def main():
    # 实际使用时请替换为真实文件路径
    file_path = r'D:\pythondemo\project-training-2-master\Short-term electricity load forecasting\data\new_data.csv'
    
    # 加载数据
    X, y = load_and_preprocess_data(file_path)
    
    # 创建并训练模型，设置早停参数
    model = LSTMAdaBoost(
        n_estimators=1, 
        seq_length=24, 
        use_gpu=True,
        early_stopping_patience=5,  # 验证损失连续5轮不下降则停止
        reduce_lr_patience=3  # 验证损失连续3轮不下降则降低学习率
    )
    
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)  # 可以设置较大的epochs，让早停决定实际训练轮数
    
    # 评估模型
    metrics = model.evaluate(X, y)
    
    # 绘制预测结果
    model.plot_predictions(X, y, title="LSTM-AdaBoost 电力负荷预测结果")
    
    # 保存模型
    model_path = r'D:\pythondemo\lstm_adaboost_model'
    os.makedirs(model_path, exist_ok=True)
    
    # 保存每个弱学习器
    for i, estimator in enumerate(model.estimators):
        estimator.save(f"{model_path}/lstm_estimator_{i}.h5")
    
    # 保存模型配置
    with open(f"{model_path}/model_config.txt", 'w') as f:
        f.write(f"n_estimators: {model.n_estimators}\n")
        f.write(f"seq_length: {model.seq_length}\n")
        f.write(f"learning_rate: {model.learning_rate}\n")
        f.write(f"estimator_weights: {model.estimator_weights}\n")
    
    print(f"模型已保存至: {model_path}")

if __name__ == "__main__":
    main()