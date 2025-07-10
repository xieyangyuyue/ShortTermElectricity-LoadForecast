import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree

def load_and_preprocess_data(file_path):
    """加载并预处理电力负荷数据"""
    # 加载数据
    new_data = pd.read_csv(file_path)
    new_data = new_data.drop('Unnamed: 0', axis=1)
    
    # 数据类型转换
    new_data = new_data.astype(np.float32)
    
    # 删除异常时间段数据
    drop_index = [i for i in range(1920, 3552)] + [i for i in range(38784, 40320)] + [i for i in range(72480, 74304)]
    new_data = new_data.drop(drop_index, axis=0)
    
    return new_data

def visualize_data(new_data):
    """可视化电力负荷数据"""
    # # 2021-1-1到2021-2-9
    # plt.figure(figsize=(30, 10), dpi=80)
    # X = np.linspace(1, 41, 96*40, endpoint=True)
    # plt.plot(X, np.array(new_data['value'])[: 96*40], color='green')
    # plt.title('2021年1-2月电力负荷')
    # plt.show()
    
    # # 2022-2-1到2022-2-28
    # plt.figure(figsize=(30, 10), dpi=80)
    # X = np.linspace(1, 29, 96*28, endpoint=True)
    # plt.plot(X, np.array(new_data['value'])[38016: 38016+96*28], color='green')
    # plt.title('2022年2月电力负荷')
    # plt.show()
    
    # # 2023-1-20到2023-2-20
    # plt.figure(figsize=(30, 10), dpi=80)
    # X = np.linspace(1, 33, 96*32, endpoint=True)
    # plt.plot(X, np.array(new_data['value'])[71904: 71904+96*32], color='green')
    # plt.title('2023年1-2月电力负荷')
    # plt.show()

     # 2021-1-1到2021-2-9
    plt.figure(figsize=(30, 10), dpi=80)
    X = np.linspace(1, 41, 96*40, endpoint=True)
    y_slice = np.array(new_data['value'])[: 96*40]
    if len(X) != len(y_slice):
        print(f"2021年段：X 长度 {len(X)}，y 长度 {len(y_slice)}，需检查！")
    plt.plot(X, y_slice, color='green')
    plt.title('2021年1-2月电力负荷')
    plt.show()
    
    # 2022-2-1到2022-2-28
    plt.figure(figsize=(30, 10), dpi=80)
    X = np.linspace(1, 29, 96*28, endpoint=True)
    y_slice = np.array(new_data['value'])[38016: 38016+96*28]
    if len(X) != len(y_slice):
        print(f"2022年段：X 长度 {len(X)}，y 长度 {len(y_slice)}，需检查！")
    plt.plot(X, y_slice, color='green')
    plt.title('2022年2月电力负荷')
    plt.show()
    
    # 2023-1-20到2023-2-20
    plt.figure(figsize=(30, 10), dpi=80)

    # 计算正确的起始索引（从数据末尾向前推32天）
    points_per_day = 96
    days = 32  # 假设2023年1-2月数据占32天
    start_idx = max(0, len(new_data['value']) - points_per_day * days)
    end_idx = len(new_data['value'])  # 从末尾向前取32天的数据

    # 取 y 切片
    y_slice = np.array(new_data['value'])[start_idx:end_idx]

    # 打印检查信息
    print(f"2023年段：数据总长度={len(new_data['value'])}，起始索引={start_idx}，结束索引={end_idx}")
    print(f"y_slice长度={len(y_slice)}，是否为空={len(y_slice) == 0}")
    print(f"前5个数据={y_slice[:5]}（若全为NaN或空则异常）")

    # 生成与 y_slice 长度匹配的 X 轴
    X = np.linspace(1, days + 1, len(y_slice), endpoint=True)  # 从1到33天

    # 绘图前检查数据有效性
    if len(y_slice) > 0:
        plt.plot(X, y_slice, color='green')
        plt.title('2023年1-2月电力负荷')
        plt.xlabel('日期')
        plt.ylabel('电力负荷')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
    else:
        print("警告：切片结果为空，无法绘图。请检查数据范围或索引计算。")

def train_full_data_model(new_data):
    """使用所有数据训练决策树模型"""
    Xtrain = np.array(new_data.drop('value', axis=1))
    Ytrain = np.array(new_data['value'])
    
    # 训练两个相同的模型用于对比
    clf1 = tree.DecisionTreeRegressor(criterion="squared_error")
    clf2 = tree.DecisionTreeRegressor(criterion="squared_error")
    
    clf1 = clf1.fit(Xtrain, Ytrain)
    clf2 = clf2.fit(Xtrain, Ytrain)
    
    # 评估模型
    print("模型1在训练集上的得分:", clf1.score(Xtrain, Ytrain))
    print("模型2在训练集上的得分:", clf2.score(Xtrain, Ytrain))
    
    # 特征重要性分析
    feature_name = ['weather_status', 'temperature', 'humidity(湿度)', 'wind_speed', 'wind_direction_angle', 
                    'pressure', 'visibility(可见度)', 'precipitation(降水)', 'light', 'holiday', 
                    'minute', 'week', 'year']
    
    importance_df1 = pd.DataFrame(zip(feature_name, clf1.feature_importances_)).set_index(0).sort_values(by=1, ascending=False)
    importance_df2 = pd.DataFrame(zip(feature_name, clf2.feature_importances_)).set_index(0).sort_values(by=1, ascending=False)
    
    print("模型1特征重要性:\n", importance_df1)
    print("模型2特征重要性:\n", importance_df2)
    
    return clf1, clf2

def train_split_year_model(new_data):
    """使用2021-2022年数据训练，2023年数据测试"""
    train = new_data.loc[:70079]
    test = new_data.loc[70080:]
    
    Xtrain = np.array(train.drop('value', axis=1))
    Ytrain = np.array(train['value'])
    Xtest = np.array(test.drop('value', axis=1))
    Ytest = np.array(test['value'])
    
    print("训练集大小:", len(train))
    print("测试集大小:", len(test))
    
    # 训练模型
    clf3 = tree.DecisionTreeRegressor(criterion="squared_error")
    clf3 = clf3.fit(Xtrain, Ytrain)
    
    # 特征重要性分析
    feature_name = ['weather_status', 'temperature', 'humidity(湿度)', 'wind_speed', 'wind_direction_angle', 
                    'pressure', 'visibility(可见度)', 'precipitation(降水)', 'light', 'holiday', 
                    'minute', 'week', 'year']
    
    importance_df3 = pd.DataFrame(zip(feature_name, clf3.feature_importances_)).set_index(0).sort_values(by=1, ascending=False)
    print("模型3特征重要性:\n", importance_df3)
    
    # 模型评估
    print("模型3在测试集上的得分:", clf3.score(Xtest, Ytest))
    
    # 预测与误差分析
    pred = clf3.predict(Xtest)
    APE = abs(pred - Ytest) / Ytest
    MAPE = sum(APE) / len(APE)
    
    print("平均绝对百分比误差(MAPE):", MAPE)
    print("最小绝对百分比误差:", min(APE), "最大绝对百分比误差:", max(APE))
    
    # 可视化误差分布
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=APE)
    plt.title('模型3误差分布')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=APE)
    plt.ylim(0, 0.4)
    plt.title('模型3误差分布(限制Y轴范围)')
    plt.show()
    
    return clf3

def train_final_model(new_data):
    """使用2021、2022前11个月、2023年数据训练，2022年12月数据测试"""
    # 删除不重要的特征
    new_data = new_data.drop(['light', 'wind_speed', 'wind_direction_angle', 'weather_status', 'precipitation'], axis=1)
    
    # 准备训练集和测试集
    train = pd.concat([new_data.loc[:67103], new_data.loc[70080:]], axis=0)
    test = new_data.loc[67104:70079]
    
    Xtrain = np.array(train.drop('value', axis=1))
    Ytrain = np.array(train['value'])
    Xtest = np.array(test.drop('value', axis=1))
    Ytest = np.array(test['value'])
    
    # 训练模型
    clf4 = tree.DecisionTreeRegressor(criterion="squared_error")
    clf4 = clf4.fit(Xtrain, Ytrain)
    
    # 特征重要性分析
    feature_name = ['temperature', 'humidity', 'pressure', 'visibility', 'holiday', 'minute', 'week', 'year']
    importance_df4 = pd.DataFrame(zip(feature_name, clf4.feature_importances_)).set_index(0).sort_values(by=1, ascending=False)
    print("最终模型特征重要性:\n", importance_df4)
    
    # 模型评估
    print("最终模型在测试集上的得分:", clf4.score(Xtest, Ytest))
    
    # 预测与误差分析
    pred = clf4.predict(Xtest)
    APE = abs(pred - Ytest) / Ytest
    MAPE = sum(APE) / len(APE)
    
    print("平均绝对百分比误差(MAPE):", MAPE)
    print("最小绝对百分比误差:", min(APE), "最大绝对百分比误差:", max(APE))
    
    # 保存误差数据
    np.save('tree_mape', APE)
    
    # 可视化误差分布
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=APE)
    plt.title('最终模型误差分布')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=APE)
    plt.ylim(0, 0.2)
    plt.title('最终模型误差分布(限制Y轴范围)')
    plt.show()
    
    return clf4

def main():
    """主函数，控制整个程序流程"""
    # 文件路径，请根据实际情况修改
    file_path = 'D:/pythondemo/project-training-2-master/Short-term electricity load forecasting/data/new_data.csv'
    
    # 数据加载与预处理
    print("正在加载和预处理数据...")
    new_data = load_and_preprocess_data(file_path)
    new_data.info()
    
    # 数据可视化
    print("正在可视化数据...")
    visualize_data(new_data)
    
    # 模型1: 使用所有数据训练
    print("\n正在训练使用所有数据的模型...")
    clf1, clf2 = train_full_data_model(new_data.copy())
    
    # 模型2: 按年份分割训练集和测试集
    print("\n正在训练按年份分割的模型...")
    clf3 = train_split_year_model(new_data.copy())
    
    # 模型3: 最终模型
    print("\n正在训练最终模型...")
    clf4 = train_final_model(new_data.copy())
    
    print("\n所有模型训练完成！")

if __name__ == "__main__":
    main()