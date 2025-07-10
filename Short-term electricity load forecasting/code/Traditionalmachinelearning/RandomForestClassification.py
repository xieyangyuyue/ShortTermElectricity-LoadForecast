# 随机森林分类分析 - 处理new_data.csv数据集

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns




plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ----------------------------
# 1. 加载数据集
# ----------------------------
try:
    # 读取CSV文件
    df = pd.read_csv('Short-term electricity load forecasting/data/new_data.csv')
    print("数据集加载成功！")
    print(f"数据集形状：{df.shape}（行：样本数，列：特征数）")
    print("\n数据集前5行预览：")
    print(df.head())
    
    # 检查数据集基本信息
    print("\n数据集基本信息：")
    print(df.info())
    
    # 检查缺失值
    missing_values = df.isnull().sum()
    print("\n缺失值统计：")
    print(missing_values[missing_values > 0])  # 只显示有缺失值的列
    
except FileNotFoundError:
    print("错误：未找到new_data.csv文件，请确保文件在当前工作目录下")
    exit()
except Exception as e:
    print(f"加载数据时发生错误：{str(e)}")
    exit()

# ----------------------------
# 2. 数据预处理
# ----------------------------
# 假设目标变量是'weather_status'（可根据实际情况修改）
target_column = 'weather_status'

# 检查目标变量是否存在
if target_column not in df.columns:
    print(f"\n错误：数据集中未找到目标变量列'{target_column}'")
    print("请修改代码中的'target_column'变量为实际的目标列名")
    exit()

# 分离特征和目标变量
X = df.drop(columns=[target_column])  # 特征
y = df[target_column]  # 目标变量

# 处理可能的非数值特征（如果有的话）
# 这里假设所有特征都是数值型，如有类别型特征需要进行编码
numeric_features = X.select_dtypes(include=['number']).columns
non_numeric_features = X.columns.difference(numeric_features)

if len(non_numeric_features) > 0:
    print(f"\n警告：检测到非数值特征，将进行One-Hot编码：{non_numeric_features.tolist()}")
    X = pd.get_dummies(X, columns=non_numeric_features)

# 处理缺失值（简单填充）
if df.isnull().sum().sum() > 0:
    print("\n处理缺失值：使用列均值填充数值特征")
    X = X.fillna(X.mean())

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y  # stratify确保分层抽样
)

print(f"\n数据集划分完成：")
print(f"训练集：{X_train.shape[0]}个样本，测试集：{X_test.shape[0]}个样本")

# ----------------------------
# 3. 自定义随机森林分类器
# ----------------------------
class RandomForestClassifierCustom:
    """自定义随机森林分类器"""
    def __init__(self, n_estimators=100, random_state=0):
        self.n_estimators = n_estimators  # 树的数量
        self.random_state = random_state  # 随机种子
        self.trees = []  # 存储决策树集合
    
    def fit(self, X, y):
        """拟合数据集"""
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        for _ in range(self.n_estimators):
            # Bootstrap抽样（有放回）
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X.iloc[sample_indices] if isinstance(X, pd.DataFrame) else X[sample_indices]
            y_boot = y.iloc[sample_indices] if isinstance(y, pd.Series) else y[sample_indices]
            
            # 创建决策树
            tree = DecisionTreeClassifier(
                max_features="sqrt",  # 随机选择特征
                random_state=np.random.randint(1e6)
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
    
    def predict(self, X):
        """预测标签：多数投票"""
        # 转换为numpy数组以便处理
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # 收集所有树的预测结果
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # 多数投票（按列取众数）
        def most_frequent(x):
            return np.bincount(x.astype(int)).argmax()
        
        return np.apply_along_axis(most_frequent, axis=0, arr=predictions)

# ----------------------------
# 4. 训练与评估模型
# ----------------------------
print("\n开始训练自定义随机森林模型...")
rf_custom = RandomForestClassifierCustom(n_estimators=100, random_state=42)
rf_custom.fit(X_train, y_train)

# 预测
y_pred_custom = rf_custom.predict(X_test)

# 评估自定义模型
print("\n===== 自定义随机森林模型评估 =====")
print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred_custom):.4f}")
print("\n混淆矩阵:")
cm_custom = confusion_matrix(y_test, y_pred_custom)
print(cm_custom)
print("\n分类报告:")
print(classification_report(y_test, y_pred_custom))

# 使用sklearn的随机森林进行对比
print("\n开始训练sklearn随机森林模型...")
rf_sklearn = RandomForestClassifier(n_estimators=100, random_state=42)
rf_sklearn.fit(X_train, y_train)

# 评估sklearn模型
y_pred_sklearn = rf_sklearn.predict(X_test)
print("\n===== sklearn随机森林模型评估 =====")
print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred_sklearn):.4f}")
print("\n混淆矩阵:")
cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)
print(cm_sklearn)
print("\n分类报告:")
print(classification_report(y_test, y_pred_sklearn))

# ----------------------------
# 5. 可视化分析
# ----------------------------
# 1. 混淆矩阵可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 自定义模型混淆矩阵
sns.heatmap(cm_custom, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('自定义随机森林混淆矩阵')
axes[0].set_xlabel('预测标签')
axes[0].set_ylabel('真实标签')

# sklearn模型混淆矩阵
sns.heatmap(cm_sklearn, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('sklearn随机森林混淆矩阵')
axes[1].set_xlabel('预测标签')
axes[1].set_ylabel('真实标签')

plt.tight_layout()
plt.show()

# 2. 特征重要性可视化（使用sklearn模型）
if hasattr(rf_sklearn, 'feature_importances_') and len(X.columns) > 0:
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        '特征': X.columns,
        '重要性': rf_sklearn.feature_importances_
    }).sort_values(by='重要性', ascending=False)
    
    sns.barplot(x='重要性', y='特征', data=feature_importance)
    plt.title('随机森林特征重要性')
    plt.tight_layout()
    plt.show()
else:
    print("\n无法生成特征重要性图：特征数量不足或模型不支持")

print("\n分析完成！")