import numpy as np
import matplotlib.pyplot as plt
import torch
import Data
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

#import
device = Data.device
criterion = Data.criterion
optimizer = Data.optimizer
epochs = Data.epochs
model = Data.model
train_dataloader = Data.train_dataloader

train_loss_list = []
test_loss_list = []
for epoch_idx in range(epochs):
  for train_img, train_seq, train_typ in train_dataloader:
    train_rnn_hidden = model.get_hcs(train_typ)
    train_img_loss, train_seq_loss, train_loss = 0, 0, 0
    for step_idx in range(100-1):
      if step_idx == 0:
        t_train_img = train_img[:,step_idx].reshape(3,1,3,32,32).to(device)
        t_train_seq = train_seq[:,step_idx].reshape(3,1,2).to(device)
        train_predicted_img, train_predicted_seq, train_rnn_hidden = model(t_train_img,t_train_seq,train_rnn_hidden)
      else:
        train_predicted_img, train_predicted_seq, train_rnn_hidden = model(train_predicted_img,train_predicted_seq,train_rnn_hidden)
      tplus1_train_seq = train_seq[:,step_idx+1].reshape(3,1,2).to(device)
      tplus1_train_img = train_img[:,step_idx+1].reshape(3,1,3,32,32).to(device)
      train_step_img_loss = criterion(tplus1_train_img, train_predicted_img)/2
      train_step_seq_loss = criterion(tplus1_train_seq, train_predicted_seq)
      train_img_loss += train_step_img_loss
      train_seq_loss += train_step_seq_loss
    train_img_loss /= 99
    train_seq_loss /= 99
    train_loss = train_img_loss + train_seq_loss
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
  if epoch_idx % (epochs/50) == 0:
    train_loss_list.append(train_loss.item())

  if epoch_idx % (epochs/5) == 0:
    print('Epoch: {}, train_Loss: {}, train_seq_Loss: {}, train_img_Loss: {}'.format(epoch_idx,train_loss.item(),train_seq_loss,train_img_loss))

fig = plt.figure()
plt.yscale('log')
plt.plot(np.arange(len(train_loss_list)), train_loss_list, label="train")
fig.savefig("Loss.png")

model_path = "model.pth"
torch.save(model.state_dict(), model_path)
