# ----------------------------
# 二、随机森林回归（修正版）
# ----------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------
# 解决中文显示问题（Windows系统）
# ----------------------------
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------
# 1. 加载数据集
# ----------------------------
try:
    data_path = os.path.join('Short-term electricity load forecasting', 'data', 'new_data.csv')
    df = pd.read_csv(data_path)
    
    print("数据集加载成功！")
    print(f"数据集形状：{df.shape}（样本数: {df.shape[0]}, 特征数: {df.shape[1]}）")
    print("数据集列名：", df.columns.tolist())
    
except FileNotFoundError:
    print(f"错误：未找到文件 {data_path}")
    exit()
except Exception as e:
    print(f"加载数据时发生错误：{str(e)}")
    exit()

# ----------------------------
# 2. 数据预处理
# ----------------------------
target_column = 'value'  # 目标列名（根据实际数据修改）

if target_column not in df.columns:
    print(f"错误：数据集中未找到目标列'{target_column}'")
    exit()

# 分离特征和目标变量
X = df.drop(columns=[target_column])
y = df[target_column]

# 处理非数值特征
non_numeric_features = X.select_dtypes(exclude=['number']).columns
if len(non_numeric_features) > 0:
    print(f"编码非数值特征：{non_numeric_features.tolist()}")
    X = pd.get_dummies(X, columns=non_numeric_features)

# 处理缺失值
if X.isnull().sum().sum() > 0:
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ----------------------------
# 3. 自定义随机森林回归器（修正max_features参数）
# ----------------------------
class RandomForestRegressorCustom:
    """自定义随机森林回归器"""
    def __init__(self, n_estimators=100, random_state=0):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.trees = []
        self.feature_names = None
    
    def fit(self, X, y):
        """拟合数据集"""
        self.feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f'feature_{i}' for i in range(X.shape[1])]
        n_samples = X.shape[0]
        rs = np.random.RandomState(self.random_state)
        
        for _ in range(self.n_estimators):
            # 修正：将max_features='auto'改为'reg'支持的'sqrt'
            dt = DecisionTreeRegressor(
                random_state=rs.randint(np.iinfo(np.int32).max),
                max_features="sqrt"  # 回归树支持的参数值
            )
            
            # Bootstrap抽样
            sample_indices = rs.randint(0, n_samples, n_samples)
            sample_weight = np.bincount(sample_indices, minlength=n_samples)
            
            # 拟合数据
            dt.fit(X, y, sample_weight=sample_weight)
            self.trees.append(dt)
    
    def predict(self, X):
        """预测数值"""
        # 确保特征匹配
        if isinstance(X, pd.DataFrame):
            X = X.reindex(columns=self.feature_names, fill_value=0)
        else:
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # 累加预测结果
        y_pred = np.zeros(X.shape[0])
        for dt in self.trees:
            y_pred += dt.predict(X)
        
        return y_pred / self.n_estimators

# ----------------------------
# 4. 模型训练与评估
# ----------------------------
# 训练自定义模型
rf_reg_custom = RandomForestRegressorCustom(n_estimators=100, random_state=42)
rf_reg_custom.fit(X_train, y_train)

# 预测
y_pred_custom = rf_reg_custom.predict(X_test)

# 评估
mse_custom = mean_squared_error(y_test, y_pred_custom)
r2_custom = r2_score(y_test, y_pred_custom)
print("\n===== 自定义随机森林回归评估 =====")
print(f"均方误差 (MSE): {mse_custom:.4f}")
print(f"决定系数 (R²): {r2_custom:.4f}")

# ----------------------------
# 5. 可视化（选择一个特征绘制曲线）
# ----------------------------
if X_train.shape[1] > 0:
    # 选择第一个特征进行可视化（可修改为其他特征）
    feature_to_plot = X_train.columns[0]
    print(f"\n使用特征'{feature_to_plot}'绘制回归曲线")
    
    # 准备可视化数据
    X_vis = X_test.copy()
    for col in X_vis.columns:
        if col != feature_to_plot:
            X_vis[col] = X_vis[col].mean()  # 固定其他特征为均值
    X_vis = X_vis.sort_values(by=feature_to_plot)
    
    # 预测可视化数据
    y_vis_custom = rf_reg_custom.predict(X_vis)
    
    # sklearn模型对比
    rf_sklearn = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_sklearn.fit(X_train, y_train)
    y_pred_sklearn = rf_sklearn.predict(X_test)
    y_vis_sklearn = rf_sklearn.predict(X_vis)
    
    # 评估sklearn模型
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
    r2_sklearn = r2_score(y_test, y_pred_sklearn)
    print("\n===== sklearn随机森林回归评估 =====")
    print(f"均方误差 (MSE): {mse_sklearn:.4f}")
    print(f"决定系数 (R²): {r2_sklearn:.4f}")
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].set_facecolor('#f8f9fa')
    axes[1].set_facecolor('#f8f9fa')
    
    # 自定义模型
    axes[0].scatter(X_vis[feature_to_plot], y_test.reindex(X_vis.index), c='#e63946', marker='o', alpha=0.6, label='实际值')
    axes[0].plot(X_vis[feature_to_plot], y_vis_custom, c='#457b9d', label='预测曲线')
    axes[0].set_title('自定义随机森林回归')
    axes[0].set_xlabel(feature_to_plot)
    axes[0].set_ylabel(target_column)
    axes[0].legend()
    axes[0].grid(linestyle='--', alpha=0.7)
    
    # sklearn模型
    axes[1].scatter(X_vis[feature_to_plot], y_test.reindex(X_vis.index), c='#e63946', marker='o', alpha=0.6, label='实际值')
    axes[1].plot(X_vis[feature_to_plot], y_vis_sklearn, c='#1d3557', label='预测曲线')
    axes[1].set_title('sklearn随机森林回归')
    axes[1].set_xlabel(feature_to_plot)
    axes[1].set_ylabel(target_column)
    axes[1].legend()
    axes[1].grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    