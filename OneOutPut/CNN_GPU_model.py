#%% md
# # 数据读取
#%%
import pandas as pd

BH_tif_400_400_after_processing = pd.read_csv('../useful_data_after_processing/BH_tif_400_400_after_processing.csv',
                                              sep=',', header=None)
BV_tif_400_400_after_processing = pd.read_csv('../useful_data_after_processing/BV_tif_400_400_after_processing.csv',
                                              sep=',', header=None)
CNM_tif_400_400_after_processing = pd.read_csv('../useful_data_after_processing/CNM_tif_400_400_after_processing.csv',
                                               sep=',', header=None)
LAI_tif_400_400_after_processing = pd.read_csv('../useful_data_after_processing/LAI_tif_400_400_after_processing.csv',
                                               sep=',', header=None)
DSM_tif_400_400_after_processing = pd.read_csv('../useful_data_after_processing/DSM_tif_400_400_after_processing.csv',
                                               sep=',', header=None)
Weather_1795_6_after_processing = pd.read_csv('../useful_data_after_processing/weather_1795_6.csv', sep=',')

#%%
Weather_1795_6_after_processing
#%% md
# # 数据处理
#%%
import numpy as np
#%%
X = np.stack([BH_tif_400_400_after_processing, BV_tif_400_400_after_processing, CNM_tif_400_400_after_processing,
              LAI_tif_400_400_after_processing, DSM_tif_400_400_after_processing], axis=-1)
#%%
X.shape
#%%
Y = Weather_1795_6_after_processing['tempMax']
#%%
Y.shape
#%% md
# # 模型构建
#%%
X_original = X
Y_original = Y
X_original.shape, Y_original.shape
#%%
import torch
#%%
# 设置设备：GPU 如果可用，否则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
#%%
# 初始设定 n = 5
n = 5
seq_length = 20

# 将时间序列拆分为输入序列和预测值
def create_sequences(data, seq_length):
    X_seq = []
    Y_seq = []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i + seq_length])
        Y_seq.append(data[i + seq_length])
    return np.array(X_seq), np.array(Y_seq)


# 拆分时间序列数据
Y_seq, Y_target = create_sequences(Y_original, seq_length)
Y_seq = np.expand_dims(Y_seq, axis=-1)  # [980, 20, 1]
Y_target = np.expand_dims(Y_target, axis=-1)  # [980, 1]

#%%
# 转换为 PyTorch 张量并移动到设备上
X_land = torch.tensor(X_original, dtype=torch.float32).unsqueeze(0).repeat(Y_seq.shape[0], 1, 1, 1).to(device)
Y_seq = torch.tensor(Y_seq, dtype=torch.float32).to(device)
Y_target = torch.tensor(Y_target, dtype=torch.float32).to(device)
#%%
X_land.shape, Y_seq.shape, Y_target.shape
#%%
batch_size = 128
lr = 0.001 * batch_size / 32 * 2
#%%
from torch.utils.data import TensorDataset, DataLoader, random_split

# 数据集大小
total_size = X_land.shape[0]
train_size = int(0.8 * total_size)
val_size = total_size - train_size

# 创建使用完整 X_land 的 DataLoader
train_dataset_full, val_dataset_full = random_split(
    TensorDataset(X_land.to(device), Y_seq.to(device), Y_target.to(device)),
    [train_size, val_size]
)

train_loader_full = DataLoader(train_dataset_full, batch_size=batch_size, shuffle=True)
val_loader_full = DataLoader(val_dataset_full, batch_size=batch_size)

# 创建使用单独通道的 DataLoader
channels = [0, 1, 2, 3, 4]
train_loaders = {}
val_loaders = {}

for channel in channels:
    X_land_channel = X_land[:, :, :, channel:channel + 1].to(device)  # 选择单个通道并移动到设备
    train_dataset_channel, val_dataset_channel = random_split(
        TensorDataset(X_land_channel, Y_seq.to(device), Y_target.to(device)),
        [train_size, val_size]
    )

    train_loaders[channel] = DataLoader(train_dataset_channel, batch_size=batch_size, shuffle=True)
    val_loaders[channel] = DataLoader(val_dataset_channel, batch_size=batch_size, shuffle=False)

# 结果展示
print("train_loader_full length:", len(train_loader_full))
print("val_loader_full length:", len(val_loader_full))

for channel in channels:
    print(f"train_loader_channel_{channel} length:", len(train_loaders[channel]))
    print(f"val_loader_channel_{channel} length:", len(val_loaders[channel]))

#%%
# 输出dataloader的信息, 输出第一次和最后一次的信息
def print_dataloader_info(loader):
    for i, (land_data, temp_seq, target) in enumerate(loader):

        if i == 0 or i == len(loader) - 1:
            print(f'lenth of loader: {len(loader)}')
            print(f'Batch {i + 1}')
            print(f'Land Data Shape: {land_data.shape}')
            print(f'Temporal Sequence Shape: {temp_seq.shape}')
            print(f'Target Shape: {target.shape}')
            print()
#%%
print_dataloader_info(train_loader_full)
#%%
print_dataloader_info(val_loader_full)
#%%
print_dataloader_info(train_loaders[0])
#%%
print_dataloader_info(val_loaders[0])
#%%
# 输出一个dataloader每批次的最值
def print_dataloader_min_max(loader):
    for i, (land_data, temp_seq, target) in enumerate(loader):
        print(f'Batch {i + 1}')
        print(f'Land Data Min: {land_data.min():.4f}, Max: {land_data.max():.4f}')
        print(f'Temporal Sequence Min: {temp_seq.min():.4f}, Max: {temp_seq.max():.4f}')
        print(f'Target Min: {target.min():.4f}, Max: {target.max():.4f}')
        print()
#%%
print_dataloader_min_max(train_loader_full)
#%%
print_dataloader_min_max(val_loader_full)
#%%
print_dataloader_min_max(train_loaders[0])
#%%
print_dataloader_min_max(val_loaders[0])
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#%%
# 定义CNN特征提取器和LSTM模型
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * 50 * 50, 128)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.reshape(-1, 128 * 50 * 50)
        x = torch.relu(self.fc(x))
        return x


class LSTMWithSpatialFeatures(nn.Module):
    def __init__(self, seq_length, in_channels=1):
        super(LSTMWithSpatialFeatures, self).__init__()  # 确保最开始调用 super().__init__()
        # 判断是否使用CNN
        if in_channels > 0:
            self.use_cnn = True
            self.cnn = CNNFeatureExtractor(in_channels=in_channels)
            lstm_input_size = 128 + 1  # CNN输出的128维特征 + 1维时间序列数据
        else:
            self.use_cnn = False
            lstm_input_size = 1  # 只有时间序列数据，没有CNN输出

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, land_data, temp_seq):
        if self.use_cnn:

            # 调整输入形状为 [batch_size, in_channels, height, width]
            land_data = land_data.permute(0, 3, 1,
                                          2)  # 之前的形状 [batch_size, 400, 400, 4] -> 新形状 [batch_size, 4, 400, 400]

            cnn_out = self.cnn(land_data)
            # 经过 CNN 处理后
            # land_data -> [batch_size, 4, 400, 400]
            # CNN 的输出 cnn_out -> [batch_size, 128]  # 经过卷积和全连接层后，输出为 128 维的特征向量

            cnn_out = cnn_out.unsqueeze(1).repeat(1, temp_seq.size(1), 1)
            # 在第二个维度（时间步长维度）增加一个维度，然后沿着这个维度重复
            # cnn_out -> [batch_size, 1, 128] -> [batch_size, seq_length, 128]  # 这里 seq_length = temp_seq.size(1)

            combined_input = torch.cat((cnn_out, temp_seq), dim=2)
            # 将 CNN 输出的特征向量和温度序列数据结合
            # temp_seq -> [batch_size, seq_length, 1]
            # combined_input -> [batch_size, seq_length, 128 + 1] -> [batch_size, seq_length, 129]
        else:
            combined_input = temp_seq
            # combined_input -> [batch_size, seq_length, 1]  只有时间序列数据

        lstm_out, _ = self.lstm(combined_input)
        # 经过 LSTM 层处理
        # combined_input -> [batch_size, seq_length, 129]
        # lstm_out -> [batch_size, seq_length, 128]  # LSTM 的输出是 128 维的特征向量        

        output = self.fc(lstm_out[:, -1, :])
        # 取 LSTM 最后一层输出，并通过全连接层进行预测
        # lstm_out[:, -1, :] -> [batch_size, 128]
        # output -> [batch_size, 1]  # 最终输出一个标量，表示下一时间步长的预测值

        return output

#%%
# 训练和验证函数
def train_and_evaluate(model, train_loader, val_loader, num_epochs=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    val_losses = []

    # 在CUDA上启用AMP（混合精度）
    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None

    for epoch in range(num_epochs):
        print(f'----------Epoch {epoch + 1}/{num_epochs}----------')
        model.train()
        train_loss = 0.0
        print('----------Training----------')
        for i, (land_data, temp_seq, target) in enumerate(train_loader):
            land_data, temp_seq, target = land_data.to(device), temp_seq.to(device), target.to(device)
            
            optimizer.zero_grad()

            # 使用新的 AMP API 的 autocast
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                output = model(land_data, temp_seq)
                loss = criterion(output, target)

            if scaler:
                # 使用混合精度缩放梯度
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 正常的反向传播
                loss.backward()
                optimizer.step()
            
            # output = model(land_data, temp_seq)
            # loss = criterion(output, target)
            # 
            # loss.backward()
            # optimizer.step()
            
            train_loss += loss.item()
            print(f'EPOCH {epoch + 1} / {num_epochs} - Batch {i + 1} / {len(train_loader)} - Loss: {loss.item()}')
        print(f'Training Loss: {train_loss / len(train_loader)}')

        model.eval()
        val_loss = 0.0
        print('----------Validation----------')
        with torch.no_grad():
            for i, (land_data, temp_seq, target) in enumerate(val_loader):
                land_data, temp_seq, target = land_data.to(device), temp_seq.to(device), target.to(device)

                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    output = model(land_data, temp_seq)
                    loss = criterion(output, target)
                    
                # output = model(land_data, temp_seq)
                # loss = criterion(output, target)
                
                val_loss += loss.item()
                print(f'EPOCH {epoch + 1} / {num_epochs} - Batch {i + 1} / {len(val_loader)} - Loss: {loss.item()}')

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f'Validation Loss: {avg_val_loss}')

    return val_losses
#%%
# 比较不同模型的函数
specific_indices = None
#%%
num_epochs = 10
#%%
results = {}
#%%
print_dataloader_info(train_loader_full)
#%%
%%time
# Baseline
print("Training baseline model...")
model_baseline = LSTMWithSpatialFeatures(seq_length=seq_length, in_channels=0).to(device)  # 无空间数据
results["Baseline"] = train_and_evaluate(model_baseline, train_loader_full, val_loader_full, num_epochs=num_epochs)
print(f"Baseline MSE: {results['Baseline']}")
#%%
results
#%%

#%%
%%time
# Baseline + n种所有空间数据
print(f"\nTraining baseline + all {n} spatial features model...")
model_baseline_n = LSTMWithSpatialFeatures(seq_length=seq_length, in_channels=n).to(device)
results[f"Baseline + all {n} features"] = train_and_evaluate(model_baseline_n, train_loader_full, val_loader_full,
                                                             num_epochs=num_epochs)
print(f"Baseline + all {n} features MSE: {results[f'Baseline + all {n} features']}")

#%%
results
#%%
# Baseline + 单个空间数据
#%%
# Baseline + BH 层
model_baseline_BH = LSTMWithSpatialFeatures(seq_length=seq_length, in_channels=1).to(device)
#%%
%%time
results[f"Baseline + feature BH"] = train_and_evaluate(model_baseline_BH, train_loaders[0], val_loaders[0],
                                                       num_epochs=num_epochs)
#%%
results
#%%
# Baseline + BV 层
model_baseline_BV = LSTMWithSpatialFeatures(seq_length=seq_length, in_channels=1).to(device)
#%%
%%time
results[f"Baseline + feature BV"] = train_and_evaluate(model_baseline_BV, train_loaders[1], val_loaders[1],
                                                       num_epochs=num_epochs)
#%%
results
#%%
# Baseline + CNM 层
model_baseline_CNM = LSTMWithSpatialFeatures(seq_length=seq_length, in_channels=1).to(device)
#%%
%%time
results[f"Baseline + feature CNM"] = train_and_evaluate(model_baseline_CNM, train_loaders[2], val_loaders[2],
                                                        num_epochs=num_epochs)
#%%
results
#%%
# Baseline + LAI 层
model_baseline_LAI = LSTMWithSpatialFeatures(seq_length=seq_length, in_channels=1).to(device)
#%%
%%time
results[f"Baseline + feature LAI"] = train_and_evaluate(model_baseline_LAI, train_loaders[3], val_loaders[3],
                                                        num_epochs=num_epochs)
#%%
results
#%%
# Baseline + DSM 层
model_baseline_DSM = LSTMWithSpatialFeatures(seq_length=seq_length, in_channels=1).to(device)
#%%
%%time
results[f"Baseline + feature DSM"] = train_and_evaluate(model_baseline_DSM, train_loaders[4], val_loaders[4],
                                                        num_epochs=num_epochs)
#%%
results
#%%
# 储存results

import pickle

with open('Results/RF_results.pkl', 'wb') as f:
    pickle.dump(results, f)
    
#%%
import matplotlib.pyplot as plt

# 模型名称
model_names = list(results.keys())

# 模型结果
model_results = [results[model_name] for model_name in model_names]

plt.figure(figsize=(10, 6))

name_print = ["Baseline", "B + 5 layers", "B + BH", "B + BV", "B + CNM", "B + LAI", "B + DEM"]

# 绘制条形图
plt.bar(name_print, model_results, color='skyblue')

# 倾斜45度显示模型名称
plt.xticks(rotation=45)

# 条形图上显示数值
for i, result in enumerate(model_results):
    plt.text(i, result + 0.01, f"{result:.4f}", ha="center", va="bottom")

plt.ylabel('Mean Squared Error')
plt.title('Model Comparison')
plt.show()

#%%
