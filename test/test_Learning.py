import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import test_Data

#import
epochs = test_Data.epochs
train_dataloader = test_Data.train_dataloader
device = test_Data.device
model = test_Data.model
criterion = test_Data.criterion
optimizer = test_Data.optimizer

#learning
for epoch_idx in range(epochs):
  for xy_data in train_dataloader:
    rnn_hidden = None
    loss = 0
    for step_idx in range(100-1):
      t_xy_data =xy_data[:,step_idx].to(device)
      tplus1_xy_data = xy_data[:,step_idx+1].to(device)
      predicted_tplus1_xy_data, rnn_hidden = model(t_xy_data, rnn_hidden)
      step_loss = criterion(tplus1_xy_data, predicted_tplus1_xy_data)
      loss += step_loss
    loss /= 99

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  if epoch_idx % 100 == 0:
      if epoch_idx == 0:
          t = open("test/step_loss.txt", "w")
          t.write('Epoch: {}, Loss: {}\n'.format(epoch_idx,loss.item()))
          print('Epoch: {}, Loss: {}'.format(epoch_idx,loss.item()))
          t.close()
      else:
          t = open("test/step_loss.txt", "a")
          t.write('Epoch: {}, Loss: {}\n'.format(epoch_idx,loss.item()))
          print('Epoch: {}, Loss: {}'.format(epoch_idx,loss.item()))
          t.close()


#save Model
model_path = 'test/model.pth'
torch.save(model.state_dict(), model_path)
