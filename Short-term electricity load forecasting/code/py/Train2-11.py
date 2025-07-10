import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch import optim
from sklearn.preprocessing import MinMaxScaler

# 设置中文字体
plt.rc('font', family='DengXian')

def load_and_preprocess_data(file_path):
    """加载并预处理电力负荷数据"""
    # 加载数据
    new_data = pd.read_csv(file_path)
    
    # 删除不必要的列
    new_data = new_data.drop(['Unnamed: 0', 'light', 'wind_speed', 'wind_direction_angle', 'weather_status', 'precipitation'], axis=1)
    
    # 转换数据类型
    new_data = new_data.astype(np.float32)
    
    # 删除异常时间段数据
    drop_index = [i for i in range(1920, 3552)] + [i for i in range(38784, 40320)] + [i for i in range(72480, 74304)]
    new_data = new_data.drop(drop_index, axis=0)
    
    return new_data

def split_data(new_data):
    """划分训练集和测试集"""
    train1 = new_data.loc[:1919]
    train2 = new_data.loc[3552:38783]
    train3 = new_data.loc[40320:67103]
    train4 = new_data.loc[70080:72479]
    train5 = new_data.loc[74304:]
    test = new_data.loc[67104:70079]
    
    train = pd.concat([new_data.loc[:67103], new_data.loc[70080:]], axis=0)
    
    return train1, train2, train3, train4, train5, test, train

def normalize_data(train1, train2, train3, train4, train5, test, train):
    """数据归一化处理"""
    scaler = MinMaxScaler()
    scaler = scaler.fit(train)
    
    train_result1 = scaler.transform(train1)
    train_result2 = scaler.transform(train2)
    train_result3 = scaler.transform(train3)
    train_result4 = scaler.transform(train4)
    train_result5 = scaler.transform(train5)
    test_result = scaler.transform(test)
    
    return scaler, train_result1, train_result2, train_result3, train_result4, train_result5, test_result

def create_inout_sequences(input_data, tw):
    """生成LSTM模型的输入输出序列"""
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw + 1):
        train_seq = copy.deepcopy(input_data[i:i + tw])
        train_seq[tw - 96:tw, 0] = 1  # 将预测时刻的负荷值设为1
        train_label = input_data[i + tw - 96:i + tw, 0]  # 标签为后96个时刻的负荷值
        inout_seq.append((torch.tensor(train_seq), torch.tensor(train_label)))
    return inout_seq

class LstmNet(nn.Module):
    """LSTM神经网络模型"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.LSTM(9, 100, 1, batch_first=True)  # 输入特征9个，隐藏层100个神经元
        self.layer2 = nn.Linear(100, 1)  # 输出层
        
    def forward(self, x):
        y1, _ = self.layer1(x)
        y1 = y1[:, -96:, :]  # 取最后96个时间步的输出
        y2 = self.layer2(y1)
        return y2.reshape(-1, 96)

def train_model(model, trainloader, epochs=1, lr=0.001):
    """训练LSTM模型"""
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    
    for epoch in range(epochs):
        i = 0
        for xb, yb in trainloader:
            y = model(xb)
            loss = loss_function(y, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
            i += 1
            if i % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {i}, Loss: {loss.data}')
    
    return model

def visualize_prediction(model, data_seq, v_max, v_min, index, title):
    """可视化预测结果"""
    pred = model(data_seq[index][0].reshape(1, 288, 9))
    true_label = data_seq[index][1] * (v_max - v_min) + v_min
    true_label = true_label.detach().numpy()
    true_pred = pred * (v_max - v_min) + v_min
    true_pred = true_pred.reshape(96)
    true_pred = true_pred.detach().numpy()
    
    # 获取前一天数据
    if index - 96 >= 0:
        before_day = data_seq[index - 96][1] * (v_max - v_min) + v_min
        before_day = before_day.detach().numpy()
    
    plt.figure(figsize=(15, 5), dpi=80)
    X = np.linspace(0, 24, 96, endpoint=True)
    plt.plot(X, true_pred, color='blue', label='预测')
    plt.plot(X, true_label, color='green', label='真实')
    if index - 96 >= 0:
        plt.plot(X, before_day, color='yellow', label='前24小时')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()

def calculate_ape_list(model, data_seq, v_max, v_min):
    """计算平均绝对百分比误差(MAPE)"""
    list1 = []
    for i in data_seq:
        pred = model(i[0].reshape(1, 288, 9))
        true_label = i[1] * (v_max - v_min) + v_min
        true_label = true_label.detach().numpy()
        true_pred = pred * (v_max - v_min) + v_min
        true_pred = true_pred.reshape(96)
        true_pred = true_pred.detach().numpy()
        num = sum(abs(true_pred - true_label) / true_label) / 96
        list1.append(num)
    return list1

def evaluate_model(model, testloader, v_max, v_min):
    """评估模型在测试集上的性能"""
    for x, y in testloader:
        pred = model(x)
        true_pred = pred * (v_max - v_min) + v_min
        true_pred = true_pred.detach().numpy()
        true_label = y * (v_max - v_min) + v_min
        true_label = true_label.numpy()
        APE_array = abs(true_pred - true_label) / true_label
        list1 = (sum(APE_array.T)) / 96
    
    return list1

def plot_ape_distribution(ape_list, title, ylim=None):
    """绘制APE分布箱线图"""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=ape_list)
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()

def main():
    # 数据处理
    print("加载并预处理数据...")
    new_data = load_and_preprocess_data('D:/pythondemo/project-training-2-master/Short-term electricity load forecasting/code/new_data.csv')

    # 划分数据集
    print("划分训练集和测试集...")
    train1, train2, train3, train4, train5, test, train = split_data(new_data)
    
    # 数据归一化
    print("数据归一化...")
    scaler, train_result1, train_result2, train_result3, train_result4, train_result5, test_result = normalize_data(
        train1, train2, train3, train4, train5, test, train
    )
    
    # 生成训练序列
    print("生成训练序列...")
    train_seq1 = create_inout_sequences(train_result1, 288)
    train_seq2 = create_inout_sequences(train_result2, 288)
    train_seq3 = create_inout_sequences(train_result3, 288)
    train_seq4 = create_inout_sequences(train_result4, 288)
    train_seq5 = create_inout_sequences(train_result5, 288)
    train_seq = train_seq1 + train_seq2 + train_seq3 + train_seq4 + train_seq5
    
    # 生成测试序列
    test_seq = create_inout_sequences(test_result, 288)
    
    print(f"训练样本数量: {len(train_seq)}")
    print(f"测试样本数量: {len(test_seq)}")
    
    # 创建数据加载器
    trainloader = torch.utils.data.DataLoader(train_seq, batch_size=96, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_seq, batch_size=len(test_seq))
    
    # 初始化模型
    print("初始化LSTM模型...")
    lstmnet = LstmNet()
    
    # 训练模型
    print("开始训练模型...")
    lstmnet = train_model(lstmnet, trainloader, epochs=1, lr=0.001)
    
    # 保存模型
    torch.save(lstmnet.state_dict(), 'lstmnet_2_11')
    print("模型已保存为lstmnet_2_11")
    
    # 获取最大最小值用于反归一化
    v_max = max(train['value'])
    v_min = min(train['value'])
    
    # 可视化预测结果
    visualize_prediction(lstmnet, test_seq, v_max, v_min, 200, "测试集样本预测结果")
    visualize_prediction(lstmnet, train_seq, v_max, v_min, 6000, "训练集样本预测结果")
    
    # 计算并打印MAPE
    print("计算测试集上的MAPE...")
    list1 = calculate_ape_list(lstmnet, test_seq, v_max, v_min)
    print(f"测试集MAPE: {sum(list1) / len(list1) * 100:.4f}%")
    
    # 保存APE列表
    np.save('lstm11_mape', list1)
    
    # 绘制APE分布
    plot_ape_distribution(list1, "测试数据的APE分布箱线图")
    plot_ape_distribution(list1, "测试数据的APE分布箱线图(缩放)", (0, 0.06))
    
    # 绘制APE折线图
    plt.figure(figsize=(30, 10), dpi=80)
    X = np.linspace(1, len(list1), len(list1), endpoint=True)
    plt.plot(X, list1, color='green')
    plt.title("测试数据的APE分布折线图")
    plt.show()
    
    # 绘制测试数据负荷值折线图
    plt.figure(figsize=(30, 10), dpi=80)
    X = np.linspace(1, 31, len(test), endpoint=True)
    plt.plot(X, test['value'], color='green')
    plt.title("测试数据负荷值折线图")
    plt.show()
    
    # 排除异常值后的评估
    list2 = list1[:2500]
    print(f"排除异常值后的MAPE: {sum(list2) / len(list2) * 100:.4f}%")
    plot_ape_distribution(list2, "排除异常值后的APE分布箱线图")
    plot_ape_distribution(list2, "排除异常值后的APE分布箱线图(缩放)", (0, 0.06))
    
    # 绘制排除异常值后的APE折线图
    plt.figure(figsize=(30, 10), dpi=80)
    X = np.linspace(0, 1, len(list2), endpoint=True)
    plt.plot(X, list2, color='green')
    plt.title("排除异常值后的APE分布折线图")
    plt.show()

if __name__ == "__main__":
    main()