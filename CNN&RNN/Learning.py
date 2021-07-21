import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import Data

#import
model = Data.model
criterion = Data.criterion
optimizer = Data.optimizer
epochs = Data.epochs
train_dataloader = Data.train_dataloader
test_dataloader = Data.test_dataloader
device = Data.device


#learning
train_seq_loss_list = []
train_img_loss_list = []
train_loss_list = []
test_loss_list = []
for epoch_idx in range(epochs):
  for train_image, train_xy in train_dataloader:
    train_rnn_hidden = None
    train_img_loss = 0
    train_seq_loss = 0
    train_loss = 0
    for step_idx in range(100-1):
      t_train_image = train_image[:,step_idx].to(device)
      t_train_image = t_train_image.reshape(9,1,3,32,32)
      tplus1_train_image = train_image[:,step_idx+1].to(device)
      tplus1_train_image = tplus1_train_image.reshape(9,1,3,32,32)
      t_train_xy = train_xy[:,step_idx].to(device)
      t_train_xy = t_train_xy.reshape(9,1,2)
      tplus1_train_xy = train_xy[:,step_idx+1].to(device)
      tplus1_train_xy =tplus1_train_xy.reshape(9,1,2)
      train_predicted_img, train_predicted_xy, train_rnn_hidden = model(t_train_image,t_train_xy,train_rnn_hidden)
      train_step_img_loss = 0.3*criterion(tplus1_train_image, train_predicted_img)
      train_step_seq_loss = 0.7*criterion(tplus1_train_xy, train_predicted_xy)
      train_img_loss += train_step_img_loss
      train_seq_loss += train_step_seq_loss
      train_loss += train_step_img_loss + train_step_seq_loss
    train_img_loss /= 99
    train_seq_loss /= 99
    train_loss /= 99
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
  if epoch_idx % 100 == 0:
    train_seq_loss_list.append(train_seq_loss)
    train_img_loss_list.append(train_img_loss)
    train_loss_list.append(train_loss.item())
    for test_img, test_xy in test_dataloader:
      test_rnn_hidden = None
      test_loss = 0
      test_seq_loss = 0
      test_img_loss = 0
      for _step_idx in range(100-1):
        t_test_image = test_img[:,_step_idx].to(device)
        t_test_image = t_test_image.reshape(1,1,3,32,32)
        tplus1_test_image = test_img[:,_step_idx+1].to(device)
        tplus1_test_image = tplus1_test_image.reshape(1,1,3,32,32)
        t_test_xy = test_xy[:,step_idx].to(device)
        t_test_xy = t_test_xy.reshape(1,1,2)
        tplus1_test_xy = test_xy[:,step_idx+1].to(device)
        tplus1_test_xy =tplus1_test_xy.reshape(1,1,2)
        test_predicted_img, test_predicted_xy, test_rnn_hidden = model(t_test_image,t_test_xy,test_rnn_hidden)
        test_step_seq_loss = 0.7 * criterion(tplus1_test_xy, test_predicted_xy)
        test_step_img_loss =  0.3 * criterion(tplus1_test_image, test_predicted_img)
        test_loss += test_step_seq_loss + test_step_img_loss
        test_img_loss += test_step_img_loss
        test_seq_loss += test_step_seq_loss
      test_loss /= 99
      test_seq_loss /= 99
      test_img_loss /= 99
    test_loss_list.append(test_loss)

  if epoch_idx % 1000 == 0:
    print('Epoch: {}, train_Loss: {}, train_seq_Loss: {}, train_img_Loss: {}'.format(epoch_idx,train_loss.item(),train_seq_loss,train_img_loss))
    print('Epoch: {}, test_Loss:{}, test_seq_Loss: {}, test_img_Loss: {}'.format(epoch_idx,test_loss,test_seq_loss,test_img_loss))


fig = plt.figure()
plt.yscale('log')
plt.plot(np.arange(len(train_loss_list)), train_loss_list, label="train")
plt.plot(np.arange(len(test_loss_list)), test_loss_list, label="test")
plt.plot(np.arange(len(train_img_loss_list)), train_img_loss_list, label="train_img")
plt.plot(np.arange(len(train_seq_loss_list)), train_seq_loss_list, label="train_seq")
fig.savefig("Loss.png")

#save Model
model_path = 'model.pth'
torch.save(model.state_dict(), model_path)
