import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from IPython.display import HTML
import Data

#import
model = Data.model
model.load_state_dict(torch.load('model.pth'))
model.eval()
train_dataloader = Data.train_dataloader
test_dataloader = Data.test_dataloader
device = Data.device

with torch.no_grad():#train
  for train_image, train_xy in train_dataloader:
    ptrain_xy_list = []
    ptrain_img_list = []
    train_rnn_hidden = None
    for step_idx in range(100-1):
      t_train_img = train_image[0,step_idx].to(device)
      t_train_img = t_train_img.reshape(1,1,3,32,32)
      t_train_xy = train_xy[0,step_idx].to(device)
      t_train_xy = t_train_xy.reshape(1,1,2)
      predicted_train_img, predicted_train_xy, train_rnn_hidden = model(t_train_img, t_train_xy, train_rnn_hidden)
      ptrain_xy_list.append(predicted_train_xy[0][0].cpu().detach().numpy())
      ptrain_img_list.append(predicted_train_img[0][0].cpu().detach().numpy())
    rtrain_xy_list = train_xy.detach().numpy()
    rtrain_img_list = train_image.detach().numpy()

  ptrain_xy_list = np.array(ptrain_xy_list)
  ptrain_xy_list = ptrain_xy_list.reshape(1,99,2)
  ptrain_img_list = np.array(ptrain_img_list)
  ptrain_img_list = ptrain_img_list.reshape(1,99,3,32,32)
ptrain_img_list = ptrain_img_list.transpose(0,1,3,4,2)
rtrain_img_list = rtrain_img_list.transpose(0,1,3,4,2)

with torch.no_grad():#test
  for test_image, test_xy in test_dataloader:
    ptest_xy_list = []
    ptest_img_list = []
    test_rnn_hidden = None
    for step_idx in range(100-1):
      t_test_img = test_image[:,step_idx].to(device)
      t_test_img = t_test_img.reshape(1,1,3,32,32)
      t_test_xy = test_xy[:,step_idx].to(device)
      t_test_xy = t_test_xy.reshape(1,1,2)
      predicted_test_img, predicted_test_xy, test_rnn_hidden = model(t_test_img, t_test_xy, test_rnn_hidden)
      ptest_xy_list.append(predicted_test_xy[0][0].cpu().detach().numpy())
      ptest_img_list.append(predicted_test_img[0][0].cpu().detach().numpy())
    rtest_xy_list = test_xy.detach().numpy()
    rtest_img_list = test_image.detach().numpy()

  ptest_xy_list = np.array(ptest_xy_list)
  ptest_xy_list = ptest_xy_list.reshape(1,99,2)
  ptest_img_list = np.array(ptest_img_list)
  ptest_img_list = ptest_img_list.reshape(1,99,3,32,32)

#train_xy
train_fig = plt.figure()
plt.title('Train data')
plt.scatter(rtrain_xy_list[0,:,0], rtrain_xy_list[0,:,1], label='input')
plt.scatter(ptrain_xy_list[0,:,0], ptrain_xy_list[0,:,1], label='predicted')
plt.legend()
train_fig.savefig("Train.png")

#test_xy
test_fig = plt.figure()
plt.title('Test data')
plt.scatter(rtest_xy_list[0,:,0], rtest_xy_list[0,:,1], label='input')
plt.scatter(ptest_xy_list[0,:,0], ptest_xy_list[0,:,1], label='predicted')
plt.legend()
test_fig.savefig("Test.png")

#train_video
ptrain_img_list[ptrain_img_list < 0] = 0
ptrain_img_list[ptrain_img_list > 1] = 1
vrtrain_img_list = rtrain_img_list[0]
vptrain_img_list = ptrain_img_list[0]
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
def update(i):
  if i != 0:
    for ax in axes:
      ax.cla()
    axes[0].set_title('Train Input Image')
    axes[0].imshow(vrtrain_img_list[i])
    axes[1].set_title('Train Predicted Image')
    axes[1].imshow(vptrain_img_list[i])

a = anim.FuncAnimation(fig, update, interval=100, frames=vptrain_img_list.shape[0])
a.save("train.gif", writer="imagemagick")

#test_video
ptest_img_list[ptest_img_list < 0] = 0
ptest_img_list[ptest_img_list > 1] = 1
vrtest_img_list = rtest_img_list[0]
vptest_img_list = ptest_img_list[0]
vrtest_img_list = vrtest_img_list.transpose(0,2,3,1)
vptest_img_list = vptest_img_list.transpose(0,2,3,1)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
def update(i):
  if i !=0:
    for ax in axes:
      ax.cla()
    axes[0].set_title('Test Input Image')
    axes[0].imshow(vrtest_img_list[i])
    axes[1].set_title('Test Predicted Image')
    axes[1].imshow(vptest_img_list[i])

a = anim.FuncAnimation(fig, update, interval=100, frames=vptest_img_list.shape[0])
a.save("test.gif", writer="imagegick")
