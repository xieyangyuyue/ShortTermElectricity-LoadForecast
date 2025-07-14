# import copy
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import torch
# from torch import nn
# from torch import optim
# from sklearn.preprocessing import MinMaxScaler
# from torch.optim.lr_scheduler import ReduceLROnPlateau

# # 设置中文字体
# plt.rc('font', family='DengXian')
# plt.rcParams['figure.dpi'] = 300

# # 数据加载和预处理
# def load_and_preprocess_data(file_path):
#     """加载并预处理电力负荷数据"""
#     # 加载数据
#     new_data = pd.read_csv(file_path)
    
#     # 删除不必要的列
#     new_data = new_data.drop(['Unnamed: 0', 'light', 'wind_speed', 'wind_direction_angle', 
#                              'weather_status', 'precipitation'], axis=1)
    
#     # 转换数据类型
#     new_data = new_data.astype(np.float32)
    
#     # 删除异常时间段数据
#     drop_index = [i for i in range(1920, 3552)] + [i for i in range(38784, 40320)] + [i for i in range(72480, 74304)]
#     new_data = new_data.drop(drop_index, axis=0)
    
#     return new_data

# # 划分数据集
# def split_data(new_data):
#     """划分训练集和测试集"""
#     train1 = new_data.loc[:1919]
#     train2 = new_data.loc[3552:38783]
#     train3 = new_data.loc[40320:67103]
#     train4 = new_data.loc[70080:72479]
#     train5 = new_data.loc[74304:]
#     test = new_data.loc[67104:70079]
    
#     train = pd.concat([new_data.loc[:67103], new_data.loc[70080:]], axis=0)
    
#     return train1, train2, train3, train4, train5, test, train

# # 数据归一化
# def normalize_data(train1, train2, train3, train4, train5, test, train):
#     """数据归一化处理"""
#     scaler = MinMaxScaler()
#     scaler = scaler.fit(train)
    
#     train_result1 = scaler.transform(train1)
#     train_result2 = scaler.transform(train2)
#     train_result3 = scaler.transform(train3)
#     train_result4 = scaler.transform(train4)
#     train_result5 = scaler.transform(train5)
#     test_result = scaler.transform(test)
    
#     return scaler, train_result1, train_result2, train_result3, train_result4, train_result5, test_result

# # 创建序列数据
# def create_inout_sequences(input_data, tw):
#     """生成LSTM模型的输入输出序列"""
#     inout_seq = []
#     L = len(input_data)
#     for i in range(L - tw + 1):
#         train_seq = np.empty((96, 15), dtype=np.float32)
#         for j in range(96):
#             seq1 = input_data[i+j:i+tw-96:96, 0].reshape(-1)  # 前7天同一时刻的负荷
#             seq2 = input_data[i+tw-96+j, 1:].reshape(-1)      # 预测日同一时刻的气象数据
#             seq = np.concatenate((seq1, seq2), axis=0)
#             train_seq[j] = seq
#         train_label = input_data[i+tw-96:i+tw, 0]  # 预测日的负荷值
#         inout_seq.append((torch.tensor(train_seq), torch.tensor(train_label)))
#     return inout_seq

# # LSTM模型定义
# class LstmNet(nn.Module):
#     """LSTM神经网络模型"""
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.LSTM(15, 100, 1, batch_first=True)  # 输入特征15个，隐藏层100个神经元
#         self.layer2 = nn.Linear(100, 1)  # 输出层
        
#     def forward(self, x):
#         y1, _ = self.layer1(x)
#         y2 = self.layer2(y1)
#         return y2.reshape(-1, 96)

# # 模型训练
# def train_model(model, trainloader, testloader, epochs, lr, v_max, v_min, model_path, previous_mape=None):
#     """训练LSTM模型并在验证集上评估"""
#     opt = optim.Adam(model.parameters(), lr=lr)
#     loss_function = nn.MSELoss()
#     scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2, verbose=True)
    
#     loss_list = []
#     mape_list = []
#     best_mape = previous_mape if previous_mape is not None else float('inf')
#     early_stopping_counter = 0
#     early_stopping_patience = 6
    
#     for epoch in range(epochs):
#         # 训练阶段
#         model.train()
#         epoch_loss = 0
#         for i, (xb, yb) in enumerate(trainloader):
#             y = model(xb)
#             loss = loss_function(y, yb)
#             loss_list.append(loss.item())
#             epoch_loss += loss.item()
            
#             loss.backward()
#             opt.step()
#             opt.zero_grad()
            
#             if i % 10 == 0:
#                 print(f'Epoch {epoch+1}/{epochs}, Batch {i}/{len(trainloader)}, Loss: {loss.item():.6f}')
        
#         avg_train_loss = epoch_loss / len(trainloader)
#         print(f'Epoch {epoch+1}/{epochs}, Average Training Loss: {avg_train_loss:.6f}')
        
#         # 评估阶段
#         model.eval()
#         with torch.no_grad():
#             for x, y in testloader:
#                 pred = model(x)
#                 true_pred = pred * (v_max - v_min) + v_min
#                 true_pred = true_pred.detach().numpy()
#                 true_label = y * (v_max - v_min) + v_min
#                 true_label = true_label.numpy()
#                 APE_array = abs(true_pred - true_label) / true_label
#                 list1 = (sum(APE_array.T)) / 96
        
#         mape = sum(list1) / len(list1) * 100
#         mape_list.append(mape)
#         print(f'Epoch {epoch+1}/{epochs}, Test MAPE: {mape:.4f}%')
        
#         # 更新学习率
#         scheduler.step(mape)
        
#         # 保存最佳模型
#         if mape < best_mape:
#             best_mape = mape
#             torch.save(model.state_dict(), model_path)
#             print(f'Model saved with improved MAPE: {mape:.4f}%')
#             early_stopping_counter = 0
#         else:
#             early_stopping_counter += 1
#             print(f'Early stopping counter: {early_stopping_counter}/{early_stopping_patience}')
        
#         # 早停检查
#         if early_stopping_counter >= early_stopping_patience:
#             print(f'Early stopping after {epoch+1} epochs')
#             break
    
#     return loss_list, mape_list, best_mape

# # 可视化预测结果
# def visualize_prediction(model, data_seq, v_max, v_min, index, title, show_previous_day=True):
#     """可视化预测结果与真实值对比"""
#     model.eval()
#     with torch.no_grad():
#         pred = model(data_seq[index][0].reshape(1, 96, 15))
#         true_label = data_seq[index][1] * (v_max - v_min) + v_min
#         true_label = true_label.numpy()
#         true_pred = pred * (v_max - v_min) + v_min
#         true_pred = true_pred.reshape(96)
#         true_pred = true_pred.detach().numpy()
        
#         plt.figure(figsize=(15, 6))
#         X = np.linspace(0, 24, 96, endpoint=True)
#         plt.plot(X, true_pred, 'b-', label='预测')
#         plt.plot(X, true_label, 'g-', label='真实')
        
#         if show_previous_day and index - 96 >= 0:
#             before_day = data_seq[index - 96][1] * (v_max - v_min) + v_min
#             before_day = before_day.numpy()
#             plt.plot(X, before_day, 'y-', label='前一天')
        
#         plt.legend(loc='upper left')
#         plt.title(title)
#         plt.xlabel('时间 (小时)')
#         plt.ylabel('电力负荷')
#         plt.grid(True, linestyle='--', alpha=0.7)
#         plt.tight_layout()
#         plt.show()

# # 计算APE和MAPE
# def calculate_ape_and_mape(model, data_loader, v_max, v_min):
#     """计算平均绝对百分比误差(MAPE)"""
#     model.eval()
#     all_ape = []
    
#     with torch.no_grad():
#         for x, y in data_loader:
#             pred = model(x)
#             true_pred = pred * (v_max - v_min) + v_min
#             true_pred = true_pred.detach().numpy()
#             true_label = y * (v_max - v_min) + v_min
#             true_label = true_label.numpy()
            
#             # 计算每个样本的APE
#             for i in range(len(true_pred)):
#                 ape = np.abs(true_pred[i] - true_label[i]) / true_label[i]
#                 all_ape.append(ape)
    
#     # 计算每个时间点的平均APE
#     ape_array = np.array(all_ape)
#     mean_ape_per_time_point = np.mean(ape_array, axis=0)
    
#     # 计算整体MAPE
#     mape = np.mean(mean_ape_per_time_point) * 100
    
#     return mape, mean_ape_per_time_point, ape_array

# # 可视化APE分布
# def visualize_ape_distribution(ape_array, title, ylim=None):
#     """可视化APE分布"""
#     # 计算每个样本的平均APE
#     mean_ape_per_sample = np.mean(ape_array, axis=1)
    
#     plt.figure(figsize=(12, 6))
    
#     # 箱线图
#     plt.subplot(1, 2, 1)
#     sns.boxplot(data=mean_ape_per_sample)
#     if ylim:
#         plt.ylim(ylim)
#     plt.title('APE分布箱线图')
#     plt.ylabel('绝对百分比误差 (APE)')
    
#     # 直方图
#     plt.subplot(1, 2, 2)
#     sns.histplot(mean_ape_per_sample, kde=True, bins=30)
#     plt.title('APE分布直方图')
#     plt.xlabel('绝对百分比误差 (APE)')
#     plt.ylabel('频率')
    
#     plt.tight_layout()
#     plt.suptitle(title, y=1.02)
#     plt.show()
    
#     # 打印统计信息
#     print(f"最小APE: {np.min(mean_ape_per_sample) * 100:.2f}%")
#     print(f"最大APE: {np.max(mean_ape_per_sample) * 100:.2f}%")
#     print(f"平均APE: {np.mean(mean_ape_per_sample) * 100:.2f}%")
#     print(f"中位数APE: {np.median(mean_ape_per_sample) * 100:.2f}%")

# # 主函数
# def main():
#     # 数据处理
#     print("加载并预处理数据...")
#     new_data = load_and_preprocess_data('D:/pythondemo/project-training-2-master/Short-term electricity load forecasting/code/new_data.csv')

#     # 划分数据集
#     print("划分训练集和测试集...")
#     train1, train2, train3, train4, train5, test, train = split_data(new_data)
    
#     # 数据归一化
#     print("数据归一化...")
#     scaler, train_result1, train_result2, train_result3, train_result4, train_result5, test_result = normalize_data(
#         train1, train2, train3, train4, train5, test, train
#     )
    
#     # 生成训练序列和测试序列
#     print("生成训练和测试序列...")
#     train_seq1 = create_inout_sequences(train_result1, 768)
#     train_seq2 = create_inout_sequences(train_result2, 768)
#     train_seq3 = create_inout_sequences(train_result3, 768)
#     train_seq4 = create_inout_sequences(train_result4, 768)
#     train_seq5 = create_inout_sequences(train_result5, 768)
#     train_seq = train_seq1 + train_seq2 + train_seq3 + train_seq4 + train_seq5
#     test_seq = create_inout_sequences(test_result, 768)
    
#     print(f"训练样本数量: {len(train_seq)}")
#     print(f"测试样本数量: {len(test_seq)}")
    
#     # 创建数据加载器
#     print("创建数据加载器...")
#     trainloader = torch.utils.data.DataLoader(train_seq, batch_size=128, shuffle=True)
#     testloader = torch.utils.data.DataLoader(test_seq, batch_size=len(test_seq))
    
#     # 获取最大最小值用于反归一化
#     v_max = max(train['value'])
#     v_min = min(train['value'])
    
#     # 初始化模型
#     print("初始化LSTM模型...")
#     lstmnet = LstmNet()
    
#     # 训练模型
#     model_path = 'lstmnet_2_19.pt'
#     print(f"开始训练模型，最佳模型将保存为: {model_path}")
    
#     # 初始训练
#     print("第一阶段训练 (学习率=0.001)...")
#     loss_list1, mape_list1, best_mape = train_model(
#         lstmnet, trainloader, testloader, epochs=5, lr=0.001, 
#         v_max=v_max, v_min=v_min, model_path=model_path
#     )
    
#     # 继续训练，降低学习率
#     print("第二阶段训练 (学习率=0.0001)...")
#     loss_list2, mape_list2, best_mape = train_model(
#         lstmnet, trainloader, testloader, epochs=10, lr=0.0001, 
#         v_max=v_max, v_min=v_min, model_path=model_path, previous_mape=best_mape
#     )
    
#     # 合并损失和MAPE列表
#     loss_list = loss_list1 + loss_list2
#     mape_list = mape_list1 + mape_list2
    
#     # 加载最佳模型
#     lstmnet.load_state_dict(torch.load(model_path))
    
#     # 可视化训练损失
#     plt.figure(figsize=(15, 6))
#     plt.plot(loss_list, 'g-')
#     plt.title('训练损失变化')
#     plt.xlabel('批次')
#     plt.ylabel('损失 (MSE)')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.show()
    
#     # 可视化训练过程中的MAPE变化
#     plt.figure(figsize=(15, 6))
#     plt.plot(range(1, len(mape_list) + 1), mape_list, 'b-')
#     plt.title('测试集MAPE变化')
#     plt.xlabel('轮次')
#     plt.ylabel('MAPE (%)')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.show()
    
#     # 可视化预测结果
#     visualize_prediction(lstmnet, test_seq, v_max, v_min, 500, '测试集样本预测结果')
#     visualize_prediction(lstmnet, train_seq, v_max, v_min, 6000, '训练集样本预测结果')
    
#     # 计算并可视化APE分布
#     print("计算并分析APE分布...")
#     mape, mean_ape_per_time_point, ape_array = calculate_ape_and_mape(lstmnet, testloader, v_max, v_min)
#     print(f"整体测试集 MAPE: {mape:.4f}%")
    
#     # 可视化APE分布
#     visualize_ape_distribution(ape_array, '测试集APE分布分析', ylim=(0, 0.1))
    
#     # 可视化24小时内不同时间点的APE
#     plt.figure(figsize=(15, 6))
#     X = np.linspace(0, 24, 96, endpoint=True)
#     plt.plot(X, mean_ape_per_time_point * 100, 'r-')
#     plt.title('不同时间点的平均APE')
#     plt.xlabel('时间 (小时)')
#     plt.ylabel('平均APE (%)')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.show()
    
#     # 保存结果
#     np.save('lstm19_mape.npy', ape_array)
#     np.save('lstm19_loss.npy', np.array(loss_list))
#     np.save('lstm19_mape_list.npy', np.array(mape_list))
    
#     print("电力负荷预测模型训练和评估完成!")

# if __name__ == "__main__":
#     main()

import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch import optim
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from prettytable import PrettyTable  # 用于生成指标表格

# 设置中文字体
plt.rc('font', family='DengXian')
plt.rcParams['figure.dpi'] = 300

# 数据加载和预处理
def load_and_preprocess_data(file_path):
    """加载并预处理电力负荷数据"""
    new_data = pd.read_csv(file_path)
    # 删除不必要的列
    new_data = new_data.drop(['Unnamed: 0', 'light', 'wind_speed', 'wind_direction_angle', 
                             'weather_status', 'precipitation'], axis=1)
    # 转换数据类型
    new_data = new_data.astype(np.float32)
    # 删除异常时间段数据
    drop_index = [i for i in range(1920, 3552)] + [i for i in range(38784, 40320)] + [i for i in range(72480, 74304)]
    new_data = new_data.drop(drop_index, axis=0)
    return new_data

# 划分数据集
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

# 数据归一化
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

# 创建序列数据
def create_inout_sequences(input_data, tw):
    """生成LSTM模型的输入输出序列"""
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw + 1):
        train_seq = np.empty((96, 15), dtype=np.float32)
        for j in range(96):
            seq1 = input_data[i+j:i+tw-96:96, 0].reshape(-1)  # 前7天同一时刻的负荷
            seq2 = input_data[i+tw-96+j, 1:].reshape(-1)      # 预测日同一时刻的气象数据
            seq = np.concatenate((seq1, seq2), axis=0)
            train_seq[j] = seq
        train_label = input_data[i+tw-96:i+tw, 0]  # 预测日的负荷值
        inout_seq.append((torch.tensor(train_seq), torch.tensor(train_label)))
    return inout_seq

# LSTM模型定义
class LstmNet(nn.Module):
    """LSTM神经网络模型"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.LSTM(15, 100, 1, batch_first=True)  # 输入特征15个，隐藏层100个神经元
        self.layer2 = nn.Linear(100, 1)  # 输出层
        
    def forward(self, x):
        y1, _ = self.layer1(x)
        y2 = self.layer2(y1)
        return y2.reshape(-1, 96)

# 模型训练
def train_model(model, trainloader, testloader, epochs, lr, v_max, v_min, model_path, previous_mape=None):
    """训练LSTM模型并在验证集上评估"""
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2, verbose=True)
    
    loss_list = []
    mape_list = []
    best_mape = previous_mape if previous_mape is not None else float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 6
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0
        for i, (xb, yb) in enumerate(trainloader):
            y = model(xb)
            loss = loss_function(y, yb)
            loss_list.append(loss.item())
            epoch_loss += loss.item()
            
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            if i % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {i}/{len(trainloader)}, Loss: {loss.item():.6f}')
        
        avg_train_loss = epoch_loss / len(trainloader)
        print(f'Epoch {epoch+1}/{epochs}, Average Training Loss: {avg_train_loss:.6f}')
        
        # 评估阶段
        model.eval()
        with torch.no_grad():
            all_true_pred = []
            all_true_label = []
            for x, y in testloader:
                pred = model(x)
                true_pred = pred * (v_max - v_min) + v_min
                true_pred = true_pred.detach().numpy()
                true_label = y * (v_max - v_min) + v_min
                true_label = true_label.numpy()
                all_true_pred.append(true_pred)
                all_true_label.append(true_label)
        
        # 拼接所有预测和真实值，计算多步MAPE
        all_true_pred = np.concatenate(all_true_pred, axis=0)
        all_true_label = np.concatenate(all_true_label, axis=0)
        APE_array = np.abs(all_true_pred - all_true_label) / all_true_label
        mean_ape_per_step = np.mean(APE_array, axis=0)  # 按预测步长取平均
        mape = np.mean(mean_ape_per_step) * 100
        mape_list.append(mape)
        print(f'Epoch {epoch+1}/{epochs}, Test MAPE: {mape:.4f}%')
        
        # 更新学习率
        scheduler.step(mape)
        
        # 保存最佳模型
        if mape < best_mape:
            best_mape = mape
            torch.save(model.state_dict(), model_path)
            print(f'Model saved with improved MAPE: {mape:.4f}%')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f'Early stopping counter: {early_stopping_counter}/{early_stopping_patience}')
        
        # 早停检查
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    return loss_list, mape_list, best_mape, all_true_pred, all_true_label  # 返回预测和真实值用于指标计算

# 可视化预测结果
def visualize_prediction(model, data_seq, v_max, v_min, index, title, show_previous_day=True):
    """可视化预测结果与真实值对比"""
    model.eval()
    with torch.no_grad():
        pred = model(data_seq[index][0].reshape(1, 96, 15))
        true_label = data_seq[index][1] * (v_max - v_min) + v_min
        true_label = true_label.numpy()
        true_pred = pred * (v_max - v_min) + v_min
        true_pred = true_pred.reshape(96)
        true_pred = true_pred.detach().numpy()
        
        plt.figure(figsize=(15, 6))
        X = np.linspace(0, 24, 96, endpoint=True)
        plt.plot(X, true_pred, 'b--', label='预测值 (predict)')  # 虚线区分
        plt.plot(X, true_label, 'k-', label='真实值 (Real)')     # 实线
        if show_previous_day and index - 96 >= 0:
            before_day = data_seq[index - 96][1] * (v_max - v_min) + v_min
            before_day = before_day.numpy()
            plt.plot(X, before_day, 'y-.', label='前一天')
        
        plt.legend(loc='upper left')
        plt.title(title)
        plt.xlabel('时间 (小时)')
        plt.ylabel('电力负荷')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

# 计算多步预测指标（MSE、RMSE、MAE、MAPE、R2）
def calculate_multistep_metrics(true_pred, true_label):
    """计算多步预测的各项指标"""
    metrics_table = PrettyTable(['测试集指标', 'MSE', 'RMSE', 'MAE', 'MAPE', 'R2'])
    mse_list, rmse_list, mae_list, mape_list, r2_list = [], [], [], [], []
    
    for step in range(96):  # 假设预测96步（0~95对应第1~96步）
        actual = true_label[:, step]
        predicted = true_pred[:, step]
        
        # 计算指标
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual - predicted))
        # 避免除以0，加小epsilon
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100  
        r2 = 1 - np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
        
        mse_list.append(mse)
        rmse_list.append(rmse)
        mae_list.append(mae)
        mape_list.append(mape)
        r2_list.append(r2)
        
        metrics_table.add_row([
            f'第{step+1}步预测结果指标：', 
            f'{mse:.6e}', 
            f'{rmse:.6f}', 
            f'{mae:.6f}', 
            f'{mape:.2f}%', 
            f'{r2*100:.2f}%'
        ])
    
    print(metrics_table)
    return mse_list, rmse_list, mae_list, mape_list, r2_list

# 计算APE和MAPE（整体）
def calculate_ape_and_mape(model, data_loader, v_max, v_min):
    """计算平均绝对百分比误差(MAPE)"""
    model.eval()
    all_ape = []
    
    with torch.no_grad():
        for x, y in data_loader:
            pred = model(x)
            true_pred = pred * (v_max - v_min) + v_min
            true_pred = true_pred.detach().numpy()
            true_label = y * (v_max - v_min) + v_min
            true_label = true_label.numpy()
            
            # 计算每个样本的APE
            for i in range(len(true_pred)):
                ape = np.abs(true_pred[i] - true_label[i]) / (true_label[i] + 1e-8)
                all_ape.append(ape)
    
    # 计算每个时间点的平均APE
    ape_array = np.array(all_ape)
    mean_ape_per_time_point = np.mean(ape_array, axis=0)
    
    # 计算整体MAPE
    mape = np.mean(mean_ape_per_time_point) * 100
    
    return mape, mean_ape_per_time_point, ape_array

# 可视化APE分布
def visualize_ape_distribution(ape_array, title, ylim=None):
    """可视化APE分布"""
    # 计算每个样本的平均APE
    mean_ape_per_sample = np.mean(ape_array, axis=1)
    
    plt.figure(figsize=(12, 6))
    
    # 箱线图
    plt.subplot(1, 2, 1)
    sns.boxplot(data=mean_ape_per_sample)
    if ylim:
        plt.ylim(ylim)
    plt.title('APE分布箱线图')
    plt.ylabel('绝对百分比误差 (APE)')
    
    # 直方图
    plt.subplot(1, 2, 2)
    sns.histplot(mean_ape_per_sample, kde=True, bins=30)
    plt.title('APE分布直方图')
    plt.xlabel('绝对百分比误差 (APE)')
    plt.ylabel('频率')
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()
    
    # 打印统计信息
    print(f"最小APE: {np.min(mean_ape_per_sample) * 100:.2f}%")
    print(f"最大APE: {np.max(mean_ape_per_sample) * 100:.2f}%")
    print(f"平均APE: {np.mean(mean_ape_per_sample) * 100:.2f}%")
    print(f"中位数APE: {np.median(mean_ape_per_sample) * 100:.2f}%")

# 主函数
def main():
    # 数据处理
    print("加载并预处理数据...")
    file_path = 'D:/pythondemo/project-training-2-master/Short-term electricity load forecasting/code/new_data.csv'
    new_data = load_and_preprocess_data(file_path)

    # 划分数据集
    print("划分训练集和测试集...")
    train1, train2, train3, train4, train5, test, train = split_data(new_data)
    
    # 数据归一化
    print("数据归一化...")
    scaler, train_result1, train_result2, train_result3, train_result4, train_result5, test_result = normalize_data(
        train1, train2, train3, train4, train5, test, train
    )
    
    # 生成训练序列和测试序列
    print("生成训练和测试序列...")
    train_seq1 = create_inout_sequences(train_result1, 768)
    train_seq2 = create_inout_sequences(train_result2, 768)
    train_seq3 = create_inout_sequences(train_result3, 768)
    train_seq4 = create_inout_sequences(train_result4, 768)
    train_seq5 = create_inout_sequences(train_result5, 768)
    train_seq = train_seq1 + train_seq2 + train_seq3 + train_seq4 + train_seq5
    test_seq = create_inout_sequences(test_result, 768)
    
    print(f"训练样本数量: {len(train_seq)}")
    print(f"测试样本数量: {len(test_seq)}")
    
    # 创建数据加载器
    print("创建数据加载器...")
    trainloader = torch.utils.data.DataLoader(train_seq, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_seq, batch_size=len(test_seq))
    
    # 获取最大最小值用于反归一化
    v_max = max(train['value'])
    v_min = min(train['value'])
    
    # 初始化模型
    print("初始化LSTM模型...")
    lstmnet = LstmNet()
    
    # 训练模型
    model_path = 'lstmnet_2_19.pt'
    print(f"开始训练模型，最佳模型将保存为: {model_path}")
    
    # 初始训练
    print("第一阶段训练 (学习率=0.001)...")
    loss_list1, mape_list1, best_mape, _, _ = train_model(
        lstmnet, trainloader, testloader, epochs=5, lr=0.001, 
        v_max=v_max, v_min=v_min, model_path=model_path
    )
    
    # 继续训练，降低学习率
    print("第二阶段训练 (学习率=0.0001)...")
    loss_list2, mape_list2, best_mape, all_true_pred, all_true_label = train_model(
        lstmnet, trainloader, testloader, epochs=10, lr=0.0001, 
        v_max=v_max, v_min=v_min, model_path=model_path, previous_mape=best_mape
    )
    
    # 合并损失和MAPE列表
    loss_list = loss_list1 + loss_list2
    mape_list = mape_list1 + mape_list2
    
    # 加载最佳模型
    lstmnet.load_state_dict(torch.load(model_path))
    
    # 可视化训练损失
    plt.figure(figsize=(15, 6))
    plt.plot(loss_list, 'g-')
    plt.title('训练损失变化')
    plt.xlabel('批次')
    plt.ylabel('损失 (MSE)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # 可视化训练过程中的MAPE变化
    plt.figure(figsize=(15, 6))
    plt.plot(range(1, len(mape_list) + 1), mape_list, 'b-')
    plt.title('测试集MAPE变化')
    plt.xlabel('轮次')
    plt.ylabel('MAPE (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # 可视化预测结果（测试集和训练集各一个样本）
    visualize_prediction(lstmnet, test_seq, v_max, v_min, 500, '测试集样本预测结果')
    visualize_prediction(lstmnet, train_seq, v_max, v_min, 6000, '训练集样本预测结果')
    
    # 计算并可视化APE分布
    print("计算并分析APE分布...")
    mape, mean_ape_per_time_point, ape_array = calculate_ape_and_mape(lstmnet, testloader, v_max, v_min)
    print(f"整体测试集 MAPE: {mape:.4f}%")
    
    # 可视化APE分布
    visualize_ape_distribution(ape_array, '测试集APE分布分析', ylim=(0, 0.1))
    
    # 可视化24小时内不同时间点的APE
    plt.figure(figsize=(15, 6))
    X = np.linspace(0, 24, 96, endpoint=True)
    plt.plot(X, mean_ape_per_time_point * 100, 'r-')
    plt.title('不同时间点的平均APE')
    plt.xlabel('时间 (小时)')
    plt.ylabel('平均APE (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # 计算多步预测指标
    print("开始测试模型...")
    print("评估模型性能...")
    mse_list, rmse_list, mae_list, mape_list, r2_list = calculate_multistep_metrics(all_true_pred, all_true_label)
    
    # 可视化多步预测指标变化
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(range(1, 97), mse_list, 'b-')
    plt.title('MSE随预测步长变化')
    plt.xlabel('预测步长')
    plt.ylabel('MSE')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 2)
    plt.plot(range(1, 97), rmse_list, 'g-')
    plt.title('RMSE随预测步长变化')
    plt.xlabel('预测步长')
    plt.ylabel('RMSE')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 3)
    plt.plot(range(1, 97), mape_list, 'r-')
    plt.title('MAPE随预测步长变化')
    plt.xlabel('预测步长')
    plt.ylabel('MAPE (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 4)
    plt.plot(range(1, 97), r2_list, 'm-')
    plt.title('R2随预测步长变化')
    plt.xlabel('预测步长')
    plt.ylabel('R2')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # 保存结果
    np.save('lstm19_mape.npy', ape_array)
    np.save('lstm19_loss.npy', np.array(loss_list))
    np.save('lstm19_mape_list.npy', np.array(mape_list))
    
    print("电力负荷预测模型训练和评估完成!")

if __name__ == "__main__":
    main()

