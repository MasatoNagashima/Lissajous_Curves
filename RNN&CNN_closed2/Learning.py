import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import Data

model = Data.model
device = Data.device
criterion = Data.criterion
optimizer = Data.optimizer
epochs = Data.epochs
train_dataloader = Data.train_dataloader
test_dataloader = Data.test_dataloader

train_loss_list = []
test_loss_list = []
for epoch_idx in range(epochs):
  for train_img, train_seq, train_pb in train_dataloader:
    train_rnn_hidden = None
    train_pb = train_pb.to(device)
    train_img_loss, train_seq_loss, train_loss = 0, 0, 0
    for step_idx in range(60-1):
      t_train_img = train_img[:,step_idx].reshape(50,1,3,50,50).to(device)
      t_train_seq = train_seq[:,step_idx].reshape(50,1,2).to(device)
      train_predicted_img, train_predicted_seq, train_rnn_hidden = model(t_train_img,t_train_seq,train_pb,train_rnn_hidden)
      tplus1_train_img = train_img[:,step_idx+1].reshape(50,1,3,50,50).to(device)
      tplus1_train_seq = train_seq[:,step_idx+1].reshape(50,1,2).to(device)
      train_step_img_loss = criterion(tplus1_train_img, train_predicted_img)/10
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
    for test_img, test_seq, test_pb in test_dataloader:
      test_rnn_hidden = None
      test_pb = test_pb.to(device)
      test_loss , test_seq_loss, test_img_loss = 0, 0, 0
      for step_idx in range(50-1):
        t_test_img = test_img[:,step_idx].reshape(1,1,3,50,50).to(device)
        t_test_seq = test_seq[:,step_idx].reshape(1,1,2).to(device)
        test_predicted_img, test_predicted_seq, test_rnn_hidden = model(t_test_img,t_test_seq,test_pb,test_rnn_hidden)
        tplus1_test_image = test_img[:,step_idx+1].reshape(1,1,3,50,50).to(device)
        tplus1_test_seq = test_seq[:,step_idx+1].reshape(1,1,2).to(device)
        test_step_seq_loss = criterion(tplus1_test_seq, test_predicted_seq)
        test_step_img_loss = criterion(tplus1_test_image, test_predicted_img)/10
        test_seq_loss += test_step_seq_loss
        test_img_loss += test_step_img_loss
      test_img_loss /= 99
      test_seq_loss /= 99
      test_loss = test_img_loss + test_seq_loss
    test_loss_list.append(test_loss)

  if epoch_idx % (epochs/10) == 0:
    print('Epoch: {}, train_Loss: {}, train_seq_Loss: {}, train_img_Loss: {}'.format(epoch_idx,train_loss.item(),train_seq_loss,train_img_loss))
    print('Epoch: {}, test_Loss:{}, test_seq_Loss: {}, test_img_Loss: {}'.format(epoch_idx,test_loss,test_seq_loss,test_img_loss))

fig = plt.figure()
plt.yscale('log')
plt.plot(np.arange(len(train_loss_list)), train_loss_list, label="train")
plt.plot(np.arange(len(test_loss_list)), test_loss_list, label="test")
fif.savefig("loss.png")

#save Model
model_path = 'test/model.pth'
torch.save(model.state_dict(), model_path)
