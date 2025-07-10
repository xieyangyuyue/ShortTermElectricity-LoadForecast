import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch import optim
from sklearn.preprocessing import MinMaxScaler

# 获取数据
def get_data():
    new_data = pd.read_csv('D:/pythondemo/project-training-2-master/Short-term electricity load forecasting/data/new_data.csv')
    # 剔除指定列
    new_data = new_data.drop(['Unnamed: 0', 'light', 'wind_speed', 'wind_direction_angle', 'weather_status', 'precipitation'], axis=1)
    new_data = new_data.astype(np.float32)
    
    # 删去特定时间段数据
    drop_index = [i for i in range(1920, 3552)] + [i for i in range(38784, 40320)] + [i for i in range(72480, 74304)]
    new_data = new_data.drop(drop_index, axis=0)
    return new_data

# 划分训练集和测试集
def split_train_test(new_data):
    train1 = new_data.loc[:1919]
    train2 = new_data.loc[3552:38783]
    train3 = new_data.loc[40320:67103]
    train4 = new_data.loc[70080:72479]
    train5 = new_data.loc[74304:]
    test = new_data.loc[67104:70079]
    train = pd.concat([new_data.loc[:67103], new_data.loc[70080:]], axis=0)
    return train1, train2, train3, train4, train5, test, train

# 数据归一化
def normalize_data(train1, train2, train3, train4, train5):
    scaler = MinMaxScaler()
    scaler.fit(train1) 
    train_result1 = scaler.transform(train1)
    train_result2 = scaler.transform(train2)
    train_result3 = scaler.transform(train3)
    train_result4 = scaler.transform(train4)
    train_result5 = scaler.transform(train5)
    return scaler, train_result1, train_result2, train_result3, train_result4, train_result5

# 生成训练数据序列
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw + 1):
        m = input_data[i:i + tw - 96].reshape(-1)
        n = input_data[i + tw - 96:i + tw, 1:].reshape(-1)
        seq = np.concatenate((m, n), axis=0)
        label = input_data[i + tw - 96:i + tw, 0]
        inout_seq.append((torch.tensor(seq), torch.tensor(label)))
    return inout_seq

# 建立全连接神经网络模型
class FCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1632, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 96)
        )
        
    def forward(self, x):
        return self.layers(x)

# 训练模型
def train_model(train_seq, epochs=1, lr=0.00001):
    fcnn = FCNN()
    opt = optim.Adam(fcnn.parameters(), lr=lr)
    loss_function = nn.MSELoss()  
    loss_list = []
    trainloader = torch.utils.data.DataLoader(train_seq, batch_size=128, shuffle=True)
    for epoch in range(epochs):
        i = 0
        for xb, yb in trainloader:
            y = fcnn(xb)
            loss = loss_function(y, yb)
            loss_list.append(loss.data)
            loss.backward()
            opt.step()
            opt.zero_grad()
            i += 1
            if i % 10 == 0:
                print(loss.data)
    return fcnn, loss_list

# 预测及可视化
def predict_and_visualize(fcnn, test_seq, train, index, is_test=True):
    v_max = max(train['value'])
    v_min = min(train['value'])
    plt.rc('font', family='DengXian')
    
    def picture(data, idx):
        pred = fcnn(data[idx][0])
        true_label = data[idx][1] * (v_max - v_min) + v_min
        true_label = true_label.detach().numpy()
        true_pred = pred * (v_max - v_min) + v_min
        true_pred = true_pred.detach().numpy()
        if idx - 96 >= 0:
            before_day = data[idx - 96][1] * (v_max - v_min) + v_min
            before_day = before_day.detach().numpy()
        
        plt.figure(figsize=(15, 5), dpi=80)
        X = np.linspace(0, 24, 96, endpoint=True)
        C, S = true_pred, true_label
        plt.plot(X, C, color='blue', label='预测')
        plt.plot(X, S, color='green', label='真实')
        if idx - 96 >= 0:
            plt.plot(X, before_day, color='yellow', label='前24小时')
        plt.legend(loc='upper left')
        plt.title('测试集样本预测' if is_test else '训练集样本预测')
        plt.show()
    
    picture(test_seq if is_test else train_seq, index)

# 计算APE列表
def calculate_ape_list(fcnn, test_seq, train):
    v_max = max(train['value'])
    v_min = min(train['value'])
    def APE_list(data_seq):  
        list1 = []
        for i in data_seq:
            pred = fcnn(i[0])
            true_label = i[1] * (v_max - v_min) + v_min
            true_label = true_label.numpy()
            true_pred = pred * (v_max - v_min) + v_min
            true_pred = true_pred.detach().numpy()
            num = sum(abs(true_pred - true_label) / true_label) / 96
            list1.append(num)
        return list1
    return APE_list(test_seq)

# 绘制损失折线图
def plot_loss(loss_list):
    plt.figure(figsize=(30, 10), dpi=80)
    X = np.linspace(1, len(loss_list), len(loss_list), endpoint=True)
    plt.plot(X, loss_list, color='green')
    plt.title('训练过程损失(MSE)折线图')
    plt.show()

    # 不同 y 轴范围的可视化
    plt.figure(figsize=(30, 10), dpi=80)
    plt.plot(X, loss_list, color='green')
    plt.ylim(0, 0.3500)
    plt.title('训练过程损失(MSE)折线图(ylim 0-0.35)')
    plt.show()

    plt.figure(figsize=(30, 10), dpi=80)
    plt.plot(X, loss_list, color='green')
    plt.ylim(0, 0.0400)
    plt.title('训练过程损失(MSE)折线图(ylim 0-0.04)')
    plt.show()

    plt.figure(figsize=(30, 10), dpi=80)
    plt.plot(X, loss_list, color='green')
    plt.ylim(0, 0.0100)
    plt.title('训练过程损失(MSE)折线图(ylim 0-0.01)')
    plt.show()

if __name__ == "__main__":
    # 数据获取与预处理
    new_data = get_data()
    train1, train2, train3, train4, train5, test, train = split_train_test(new_data)
    scaler, train_result1, train_result2, train_result3, train_result4, train_result5 = normalize_data(train1, train2, train3, train4, train5)

    # 生成数据序列
    train_seq1 = create_inout_sequences(train_result1, 192)
    train_seq2 = create_inout_sequences(train_result2, 192)
    train_seq3 = create_inout_sequences(train_result3, 192)
    train_seq4 = create_inout_sequences(train_result4, 192)
    train_seq5 = create_inout_sequences(train_result5, 192)
    train_seq = train_seq1 + train_seq2 + train_seq3 + train_seq4 + train_seq5
    test_result = scaler.transform(test)
    test_seq = create_inout_sequences(test_result, 192)

    # 模型训练
    fcnn, loss_list = train_model(train_seq)

    # 预测与可视化
    predict_and_visualize(fcnn, test_seq, train, 500, is_test=True)
    predict_and_visualize(fcnn, train_seq, train, 66000, is_test=False)

    # 计算并展示APE相关
    list1 = calculate_ape_list(fcnn, test_seq, train)
    print(f"测试数据上的MAPE：{str(sum(list1) / len(list1) * 100) + '%'}")
    np.save('fcnn_mape', list1)
    sns.boxplot(data=list1)
    plt.ylim(0, 0.06)
    plt.title('测试数据APE分布箱线图')
    plt.show()

    plt.figure(figsize=(30, 10), dpi=80)
    X = np.linspace(1, 2785, len(list1), endpoint=True)
    plt.plot(X, list1, color='green')
    plt.title('测试数据APE分布折线图')
    plt.show()

    # 绘制损失折线图
    plot_loss(loss_list)

    # 保存模型
    torch.save(fcnn.state_dict(), 'fcnn_5')