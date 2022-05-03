from cgi import test
import pandas as pd
import numpy as np
import glob
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import accuracy_score


data_recording_frequency = 10 # Hz
time_series = 60 # Secconds
samples_per_time_series = data_recording_frequency * time_series

x_features = [
    'abs',
    'tcs',
    'airspeed',
    'wheelspeed',
    'rpm',
    'throttle input',
    'throttle',
    'brake input',
    'brake',
    'steering input',
    'steering',
    'gear',
]

y_feature = 'BAC'

## Data loading functions
def load_data_from_csv(fname, x_features, y_feature, driver=None, course=None):
    x_features_ = x_features.copy()
    # Read file
    print("Reading data from {} ...".format(fname))
    df = pd.read_csv(fname)

    # Remove data recording before car is in motion
    in_motion = np.where(df['wheelspeed'] >= 1)[0]
    df = df.drop(df.index[0:in_motion[0]])

    # Remove data recording after car is in motion
    df = df.drop(df.index[in_motion[-1]:])
    
    # Process features
    if 'gear' in x_features_:
        x_features_.remove('gear')
        df['gear'] = [float(gear.split('x')[-1][:-1]) for gear in df['gear']]
        dummies = pd.get_dummies(df.gear, prefix='gear')
        for i in range(8):
            if 'gear_{}.0'.format(i) not in dummies.columns:
                dummies['gear_{}.0'.format(i)] = np.zeros((df.shape[0]), dtype=np.int32)
        x_features_ += list(dummies.columns)
        df = df.drop('gear', axis=1)
        df = pd.concat([df, dummies], axis=1)
    if 'airspeed' in x_features_:
        df['airspeed'] = df['airspeed'] / 44
    if 'wheelspeed' in x_features_:
        df['wheelspeed'] = df['wheelspeed'] / 44
    if 'rpm' in x_features_:
        df['rpm'] = df['rpm'] / 7000
    if 'steering' in x_features_:
        df['steering'] = (df['steering'] + 1080) / (1080 * 2)# Noramlize to max steering input
    if 'steering input' in x_features_:
        df['steering input'] = (df['steering input'] + 1) / 2

    if df[x_features_].to_numpy().max() > 1 or df[x_features_].to_numpy().min() < 0:
        import pudb; pu.db

    df_x = df[x_features_]
    df_y = df[y_feature]

    return df_x, df_y

def create_time_series_data(dir, samples_per_set, x_features, y_feature, driver=None, course=None):
    fnames = glob.glob(os.path.join(dir, '*.csv'))
    x = np.array([])
    y = np.array([])

    for fname in fnames:
        df_x, df_y = load_data_from_csv(fname, x_features, y_feature, driver, course)
        
        if x.shape == (0,):
            x = np.empty((0, df_x.shape[1], samples_per_set))

        i = 0
        while (i + samples_per_set) < df_x.shape[0]:
            x = np.append(x, [df_x.iloc[i:i+samples_per_set].to_numpy().T], axis=0)
            y = np.append(y, df_y.iloc[i])

            # Start next time series half way through the last time series
            i += int(samples_per_set / 2)

    x = x.astype(np.float64)
    y = y.astype(np.float64)
    indices = np.random.permutation(x.shape[0])
    train_indices, validate_indices, test_indices = \
        indices[:int(x.shape[0]*0.8)], \
        indices[int(x.shape[0]*0.8):int(x.shape[0]*0.9)], \
        indices [int(x.shape[0]*0.9):]

    return x[train_indices], y[train_indices], x[validate_indices], y[validate_indices], x[test_indices], y[test_indices]

## PyTorch Network
class Net(nn.Module):
    def __init__(self, num_features, device):
        super().__init__()
        self.device = device
        # Define layers
        self.conv1 = nn.Conv1d(num_features, 30, 3, stride=1, padding=2, device=self.device)
        self.conv2 = nn.Conv1d(30, 90, 5, stride=2, padding=3, device=self.device)
        self.conv3 = nn.Conv1d(90, 1000, 7, stride=5, padding=4, device=self.device)

        # self.pool = nn.MaxPool1d(5)

        self.dout1 = nn.Dropout(p=0.2)
        self.dout2 = nn.Dropout(p=0.2)
        self.dout3 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(61000, 1800, device=self.device)
        self.fc2 = nn.Linear(1800, 200, device=self.device)
        self.fc3 = nn.Linear(200, 30, device=self.device)
        self.fc4 = nn.Linear(30, 1, device=self.device)

        self.dout4 = nn.Dropout(p=0.2)
        self.dout5 = nn.Dropout(p=0.2)
        self.dout6 = nn.Dropout(p=0.2)

    def forward(self, x):
        # Convolution
        x = F.relu(self.conv1(x))
        x = self.dout1(x)
        x = F.relu(self.conv2(x))
        x = self.dout3(x)
        x = F.relu(self.conv3(x))
        x = self.dout3(x)

        # Fully Connected
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dout3(x)
        x = F.relu(self.fc2(x))
        x = self.dout4(x)
        x = F.relu(self.fc3(x))
        x = self.dout5(x)
        x = self.fc4(x)
        # x = torch.sigmoid(x)
        x = torch.squeeze(x)
        
        return x

## Main
if __name__ == "__main__":
    # Hyperparameters
    epochs = 12
    batch_size = 100

    device = torch.device("cuda:0") 
    # device = torch.device("cpu")

    x_train, y_train, x_val, y_val, x_test, y_test = create_time_series_data('data/', samples_per_time_series, x_features, y_feature)
    
    # TODO: train test
    x_train = torch.from_numpy(x_train).type(torch.float32).to(device)
    y_train = torch.from_numpy(y_train).to(device).to(torch.float32)
    x_val = torch.from_numpy(x_val).type(torch.float32).to(device)
    y_val = torch.from_numpy(y_val).to(device).to(torch.float32)
    x_test = torch.from_numpy(x_test).type(torch.float32).to(device)
    y_test = torch.from_numpy(y_test).to(device).to(torch.float32)
    
    # Create CNN, Loss Function, and Optomizer
    model = Net(x_train.shape[1], device)
    criterion = nn.BCEWithLogitsLoss() # Was cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # from torchsummary import summary
    # print(summary(model, x[0].shape))


    for epoch in range(epochs):  # loop over the dataset multiple times

        num_batches = int(np.floor(x_train.shape[0] / batch_size)) + 1
        loss_sum = 0
        for batch in range(num_batches):
            # Create slice for batches data
            data_slice = slice(batch * batch_size, (batch + 1) * batch_size)
            if batch == (num_batches - 1):
                data_slice = slice(batch * batch_size, x_train.shape[0]) 
            
            # Slice data
            x_train_ = x_train[data_slice]
            y_train_ = y_train[data_slice]
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_hat_ = model(x_train_)
            loss = criterion(y_hat_, y_train_)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        # print statistics
        prediction = model(x_train).cpu().detach().numpy()
        train_accuracy = accuracy_score(y_train.cpu().detach().numpy(), (prediction >= 0.5).astype(np.float32))
        prediction = model(x_val).cpu().detach().numpy()
        val_accuracy = accuracy_score(y_val.cpu().detach().numpy(), (prediction >= 0.5).astype(np.float32))

        y_hat = model(x_val)
        val_loss = criterion(y_hat, y_val)
        
        print("Epoch {}/{}, Train Accuracy: {}, Training Loss: {}, Validation Accuracy: {}, Validation Loss: {}".format( \
            epoch+1, epochs, train_accuracy, loss_sum, val_accuracy, val_loss)
        )

    # Print test performance
    test_prediction = model(x_test).cpu().detach().numpy()
    test_accuracy = accuracy_score(y_test.cpu().detach().numpy(), (test_prediction >= 0.5).astype(np.float32))
    print("Testing Accuracy: {}".format(test_accuracy))