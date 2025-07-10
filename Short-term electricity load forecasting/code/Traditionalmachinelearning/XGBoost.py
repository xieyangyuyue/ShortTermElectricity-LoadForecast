import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 解决中文显示问题
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)

# ----------------------------
# 安全MAPE函数
# ----------------------------
def safe_mape(y_true, y_pred, epsilon=1e-8):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)))

# ----------------------------
# 1. 数据加载与预处理
# ----------------------------
def load_and_preprocess_data(data_path, target_column='value'):
    try:
        df = pd.read_csv(data_path)
        print("数据集加载成功！")
        print(f"数据集形状：{df.shape}")

        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
            print("已移除无用的'Unnamed: 0'列")
        
        if target_column not in df.columns:
            raise ValueError(f"目标列'{target_column}'不存在")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]

        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        print(f"\n分类特征：{categorical_features}")
        print(f"数值特征：{numerical_features}")

        # 使用标准化的预处理管道
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        X_processed = preprocessor.fit_transform(X)
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [name.replace('__', '_').replace(' ', '_') for name in feature_names]

        X_processed = pd.DataFrame(
            X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
            columns=feature_names
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.3, random_state=42
        )

        return X_train, X_test, y_train, y_test, feature_names
    
    except FileNotFoundError:
        print(f"错误：文件不存在 - {data_path}")
        exit()
    except Exception as e:
        print(f"数据预处理错误：{str(e)}")
        exit()

# ----------------------------
# 2. 模型训练（单线程 + 参数优化）
# ----------------------------
def train_xgboost_model(X_train, y_train):
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1
    )

    print("\n开始训练XGBoost模型...")
    xgb_model.fit(X_train, y_train)
    return xgb_model

# ----------------------------
# 多步滚动预测（修复R²计算）
# ----------------------------
def multi_step_forecast(model, X_test, y_test, steps=4):
    """
    多步滚动预测：批量计算R²以避免单样本问题
    """
    y_pred_steps = []
    test_data = X_test.copy().reset_index(drop=True)
    true_values = y_test.reset_index(drop=True)
    
    for step in range(steps):
        y_pred_step = model.predict(test_data.iloc[step:step+1])
        y_pred_steps.append(y_pred_step[0])
    
    # 返回所有预测步和对应的真实值，用于批量计算R²
    return np.array(y_pred_steps), true_values.iloc[:steps]

# ----------------------------
# 3. 模型评估与可视化（优化多步指标输出格式）
# ----------------------------
def evaluate_model(model, X_test, y_test, feature_names):
    # 常规单步预测
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = safe_mape(y_test, y_pred)

    print("\n===== 单步预测整体评估结果 =====")
    print(f"均方误差（MSE）：{mse:.4f}")
    print(f"均方根误差（RMSE）：{rmse:.4f}")
    print(f"决定系数（R²）：{r2:.4f}（越接近1越好）")
    print(f"平均绝对百分比误差（MAPE）：{mape:.4f}（越低越好）")

    # 多步滚动预测
    steps = 4  # 可自定义预测步数
    y_pred_steps, y_true_steps = multi_step_forecast(model, X_test, y_test, steps)
    
    # 批量计算整体R²（解决单样本问题）
    overall_r2 = r2_score(y_true_steps, y_pred_steps)
    
    # 构建多步预测指标数据，方便格式化输出
    multi_step_metrics = []
    for i in range(steps):
        step_mse = mean_squared_error([y_true_steps.iloc[i]], [y_pred_steps[i]])
        step_rmse = np.sqrt(step_mse)
        step_mae = np.mean(np.abs(y_true_steps.iloc[i] - y_pred_steps[i]))
        step_mape = safe_mape([y_true_steps.iloc[i]], [y_pred_steps[i]])
        
        multi_step_metrics.append([
            f"第{i+1}步预测结果指标：",
            f"{step_mse:.12f}",
            f"{step_rmse:.12f}",
            f"{step_mae:.12f}",
            f"{step_mape*100:.2f}%",
            f"{overall_r2*100:.2f}%"
        ])
    
    # 优化多步预测指标输出格式，让表格更对齐美观
    print("\n===== 多步预测指标 ======")
    # 表头
    header = ["测试集指标", "MSE", "RMSE", "MAE", "MAPE", "R2"]
    # 先打印表头，用制表符\t对齐
    print("\t".join(header))
    for row in multi_step_metrics:
        # 每行数据用制表符\t分隔，保证对齐
        print("\t".join(row))
    
    # 显示整体多步预测R²
    print(f"\n整体多步预测R²：{overall_r2:.4f}")

    # 1. 特征重要性图（前15个）
    plt.figure(figsize=(10, 6))
    importance = model.feature_importances_
    indices = np.argsort(importance)[-15:]
    values = importance[indices]
    plt.barh(range(len(indices)), values, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(indices))))
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('特征重要性')
    plt.title('XGBoost特征重要性（前15名）')
    for i, v in enumerate(values):
        plt.text(v + 0.005, i, f"{v:.2f}", va='center', fontsize=9)
    plt.tight_layout()
    plt.show()

    # 2. 预测值 vs 实际值（50样本）
    plt.figure(figsize=(12, 6))
    sample_indices = np.random.choice(len(y_test), 50, replace=False)
    plt.scatter(range(50), y_test.iloc[sample_indices], label='实际值', color='#e63946', alpha=0.7)
    plt.scatter(range(50), y_pred[sample_indices], label='预测值', color='#457b9d', alpha=0.7)
    plt.xlabel('样本索引')
    plt.ylabel('电力负荷值')
    plt.title('预测值与实际值对比（随机50样本）')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. 趋势对比（前200测试样本）
    plt.figure(figsize=(12, 6))
    plt.plot(range(200), y_test.iloc[:200].values, label='实际值', color='#e63946')
    plt.plot(range(200), y_pred[:200], label='预测值', color='#457b9d')
    plt.xlabel('时间序列（前200测试样本）')
    plt.ylabel('电力负荷值')
    plt.title('负荷趋势预测对比')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------
# 主函数入口
# ----------------------------
if __name__ == "__main__":
    DATA_PATH = "D:/pythondemo/project-training-2-master/Short-term electricity load forecasting/data/new_data.csv"
    TARGET_COLUMN = "value"

    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(DATA_PATH, TARGET_COLUMN)
    model = train_xgboost_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, feature_names)