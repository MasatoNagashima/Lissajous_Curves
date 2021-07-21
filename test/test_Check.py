import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import test_Data

#import
train_dataloader = test_Data.train_dataloader
test_dataloader =test_Data.test_dataloader
device = test_Data.device
model = test_Data.model
model.load_state_dict(torch.load('test/model.pth'))

#check train data
with torch.no_grad():
    for xy_data in train_dataloader:
        predicted_xy_data_list = []
        rnn_hidden = None
        for step_idx in range(100-1):
            if step_idx == 0:
                t_xy_data = xy_data[:, step_idx].to(device)
            else:
                t_xy_data = predicted_tplus1_xy_data
            predicted_tplus1_xy_data, rnn_hidden = model(t_xy_data, rnn_hidden)
            predicted_xy_data_list.append(predicted_tplus1_xy_data[0].cpu().detach().numpy())
        break

    predicted_xy_data_list = np.array(predicted_xy_data_list)
    input_xy_data_list = xy_data[0].detach().numpy()

train_fig = plt.figure()
plt.title('Train data')
plt.scatter(input_xy_data_list[:, 0], input_xy_data_list[:, 1], label='input')
plt.scatter(predicted_xy_data_list[:, 0], predicted_xy_data_list[:, 1], label='predicted')
plt.legend()
train_fig.savefig("test/train_data.png")

#check test data
with torch.no_grad():
    for xy_data in test_dataloader:
        predicted_xy_data_list = []
        rnn_hidden = None
        for step_idx in range(100-1):
            if step_idx == 0:
                t_xy_data = xy_data[:, step_idx].to(device)
            else:
                t_xy_data = predicted_tplus1_xy_data
            predicted_tplus1_xy_data, rnn_hidden = model(t_xy_data, rnn_hidden)
            predicted_xy_data_list.append(predicted_tplus1_xy_data[0].cpu().detach().numpy())
        break

    predicted_xy_data_list = np.array(predicted_xy_data_list)
    input_xy_data_list = xy_data[0].detach().numpy()

test_fig = plt.figure()
plt.title('Test data')
plt.scatter(input_xy_data_list[:, 0], input_xy_data_list[:, 1], label='input')
plt.scatter(predicted_xy_data_list[:, 0], predicted_xy_data_list[:, 1], label='predicted')
plt.legend()
test_fig.savefig("test/test_data.png")
