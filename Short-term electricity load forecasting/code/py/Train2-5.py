import copy
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch import optim
from sklearn.preprocessing import MinMaxScaler

# 数据预处理
def preprocess_data(file_path):
    """加载并预处理电力负荷数据"""
    # 加载数据
    data = pd.read_csv(file_path)
    
    # 处理异常值
    data.loc[33565, 'value'] = (data.loc[33564]['value'] + data.loc[33566]['value']) / 2
    data.loc[60009:60015, 'value'] = (data.loc[60008]['value'] + data.loc[60016]['value']) / 2
    data.loc[60057, 'value'] = (data.loc[60056]['value'] + data.loc[60058]['value']) / 2
    
    # 提取时间特征
    list1 = [i.split('/') for i in data['time']]
    list2 = [i[-1].split(' ') for i in list1]
    list3 = [i[-1].split(':') for i in list2]
    list4 = [int(i[0]) * 60 + int(i[1]) for i in list3]  # 分钟
    minute = np.array(list4).reshape(len(data), 1)
    
    # 生成星期特征
    week = np.empty(len(data))
    for i in range(len(data)//(96*7)):
        m = i * 96 * 7
        week[m:m+96] = 5
        week[m+96: m+96*2] = 6
        week[m+96*2: m+96*3] = 7
        week[m+96*3: m+96*4] = 1
        week[m+96*4: m+96*5] = 2
        week[m+96*5: m+96*6] = 3
        week[m+96*6: m+96*7] = 4
    week[m+96*7: m+96*8] = 5
    week[m+96*8: m+96*9] = 6
    week[m+96*9: m+96*10] = 7
    week = week.reshape(len(data), 1)
    
    # 生成年份特征
    year = np.empty(len(data))
    year[: 96*365] = 1
    year[96*365: 96*365*2] = 2
    year[96*365*2: ] = 3
    year = year.reshape(len(data), 1)
    
    # 拼接新特征
    array1 = np.concatenate((minute, week, year), axis=1)
    df_1 = pd.DataFrame(array1, columns=['minute', 'week', 'year'])
    new_data = pd.concat((data, df_1), axis=1).drop(['Unnamed: 0', 'wind_direction', 'time'], axis=1)
    
    # 处理object类型数据
    df_2 = new_data[['temperature', 'humidity', 'wind_speed', 'wind_level', 'wind_direction_angle', 'pressure', 'visibility', 'precipitation', 'light']]
    array2 = np.array(df_2)
    
    for i in range(9):
        a = [j[1: -1] for j in array2[:, i]]
        a = pd.to_numeric(a)
        array2[:, i] = a
    
    array2 = array2.astype(dtype=np.float32)
    new_data[['temperature', 'humidity', 'wind_speed', 'wind_level', 'wind_direction_angle', 'pressure', 'visibility', 'precipitation', 'light']] = array2
    
    # 数据类型转换
    new_data = new_data.astype(np.float32)
    
    # 删除重复特征
    new_data = new_data.drop('wind_level', axis=1)
    
    return new_data

# 生成训练数据序列
def create_inout_sequences(input_data, tw):
    """生成用于LSTM训练的输入输出序列"""
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw + 1):
        train_seq = copy.deepcopy(input_data[i:i + tw])
        train_seq[96:tw, 0] = 1  # 将预测时刻的负荷值设为1
        train_label = input_data[i + 96:i + tw, 0]  # 标签为后96个时刻的负荷值
        inout_seq.append((torch.tensor(train_seq), torch.tensor(train_label)))
    return inout_seq

# 定义LSTM模型
class LstmNet(nn.Module):
    """LSTM神经网络模型"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.LSTM(14, 60, 1, batch_first=True)  # LSTM层
        self.layer2 = nn.Linear(60, 1)  # 全连接层
        
    def forward(self, x):
        y1, _ = self.layer1(x)
        y1 = y1[:, -96:, :]  # 取最后96个时间步的输出
        y2 = self.layer2(y1)
        return y2.reshape(-1, 96)

# 训练模型
def train_model(model, trainloader, epochs=1, lr=0.001):
    """训练LSTM模型"""
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    loss_list = []
    
    for epoch in range(epochs):
        i = 0
        all_loss = 0
        for xb, yb in trainloader:
            y = model(xb)
            loss = loss_function(y, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
            i += 1
            all_loss += loss.data
            if i % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {i}, Loss: {loss.data}')
        
        epoch_loss = all_loss / len(trainloader)
        loss_list.append(epoch_loss)
        print(f'Epoch {epoch+1} complete. Average loss: {epoch_loss}')
    
    return model, loss_list

# 计算APE列表
def calculate_ape_list(model, data_seq, v_max, v_min):
    """计算平均绝对百分比误差列表"""
    list1 = []
    for i in data_seq:
        pred = model(i[0].reshape(1, 192, 14))
        true_label = i[1] * (v_max - v_min) + v_min
        true_label = true_label.detach().numpy()
        true_pred = pred * (v_max - v_min) + v_min
        true_pred = true_pred.reshape(96)
        true_pred = true_pred.detach().numpy()
        num = sum(abs(true_pred - true_label) / true_label) / 96
        list1.append(num)
    return list1

# 可视化预测结果
def visualize_prediction(model, test_seq, v_max, v_min, index=1000):
    """可视化预测结果与真实值对比"""
    plt.rc('font', family='DengXian')  # 设置中文字体
    
    pred = model(test_seq[index][0].reshape(1, 192, 14))
    true_label = test_seq[index][1] * (v_max - v_min) + v_min
    true_label = true_label.detach().numpy()
    true_pred = pred * (v_max - v_min) + v_min
    true_pred = true_pred.reshape(96)
    true_pred = true_pred.detach().numpy()
    
    plt.figure(figsize=(15, 5), dpi=80)
    X = np.linspace(0, 24, 96, endpoint=True)
    plt.plot(X, true_pred, color='blue', label='预测')
    plt.plot(X, true_label, color='green', label='真实')
    plt.legend(loc='upper left')
    plt.title('电力负荷预测与真实值对比')
    plt.xlabel('时间(小时)')
    plt.ylabel('负荷值')
    plt.show()

# 主函数
def main():
    # 数据预处理
    print("开始数据预处理...")
    new_data = preprocess_data('D:\pythondemo\project-training-2-master\Short-term electricity load forecasting\data\gfdf_with_cloud_1.csv')
    new_data.to_csv('D:/pythondemo/project-training-2-master/Short-term electricity load forecasting/data/new_data.csv')
    print("数据预处理完成，保存为new_data.csv")
    
    # 划分训练集和测试集
    print("划分训练集和测试集...")
    train = new_data.loc[:70079]
    test = new_data.loc[70080:]
    print(f"训练集大小: {len(train)}, 测试集大小: {len(test)}")
    
    # 数据归一化
    print("进行数据归一化...")
    scaler = MinMaxScaler()
    scaler = scaler.fit(train)
    train_result = scaler.transform(train)
    test_result = scaler.transform(test)
    
    # 生成训练序列
    print("生成训练序列...")
    train_inout_seq = create_inout_sequences(train_result, 192)
    print(f"训练样本数量: {len(train_inout_seq)}")
    
    # 创建数据加载器
    trainloader = torch.utils.data.DataLoader(train_inout_seq, batch_size=96, shuffle=True)
    
    # 初始化模型
    print("初始化LSTM模型...")
    lstmnet = LstmNet()
    
    # 训练模型
    print("开始训练模型...")
    lstmnet, loss_list = train_model(lstmnet, trainloader, epochs=1, lr=0.001)
    
    # 保存模型
    torch.save(lstmnet.state_dict(), 'lstmnet_2_5_1.pt')
    print("模型训练完成并保存为lstmnet_2_5_1.pt")
    
    # 获取最大最小值用于反归一化
    v_max = max(train['value'])
    v_min = min(train['value'])
    
    # 可视化预测结果
    print("可视化预测结果...")
    visualize_prediction(lstmnet, create_inout_sequences(test_result, 192), v_max, v_min)
    
    # 计算训练集上的APE
    print("计算训练集上的APE...")
    train_samples = random.sample(train_inout_seq, 5000)
    train_ape = calculate_ape_list(lstmnet, train_samples, v_max, v_min)
    print(f"训练集APE: {sum(train_ape) / len(train_ape) * 100:.4f}%")
    
    # 计算测试集上的APE
    print("计算测试集上的APE...")
    test_seq = create_inout_sequences(test_result, 192)
    test_ape = calculate_ape_list(lstmnet, test_seq, v_max, v_min)
    print(f"测试集APE: {sum(test_ape) / len(test_ape) * 100:.4f}%")
    
    # 可视化APE分布
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=test_ape)
    plt.ylim(0, 0.4)
    plt.title('测试集APE分布')
    plt.show()

if __name__ == "__main__":
    main()