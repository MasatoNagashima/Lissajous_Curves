import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset7
import Data

#import
model = Data.model
torch.load_state_dict(torch.load("model.pth"))
device = Data.device
train_dataloader = Data.train_dataloader
test_dataloader = Data.test_dataloader

#train
with torch.no_grad():
  for train_img, train_seq, train_pb in train_dataloader:
    predicted_train_seq_list = []
    predicted_train_img_list = []
    train_rnn_hidden = None

    for step_idx in range(60-1):
      if step_idx == 0:
        t_train_img = train_img[:, step_idx].reshape(50,1,3,50,50)
        t_train_seq = train_seq[:,step_idx].reshape(50,1,2)
      else:
        t_train_img, t_train_seq = predicted_train_img, predicted_train_seq
      predicted_train_img, predicted_train_seq, train_rnn_hidden = model(t_train_img, t_train_seq, train_pb, train_rnn_hidden)
      predicted_train_seq_list.append(predicted_train_seq[:,0].detach().numpy())
      predicted_train_img_list.append(predicted_train_img[:,0].detach().numpy())
    real_train_seq_list = train_seq.detach().numpy()
    real_train_seq_list = real_train_seq_list.reshape(50,60,2)
    real_train_img_list = train_img.detach().numpy()
    real_train_img_list = real_train_img_list.reshape(50,60,3,50,50)
    break
  predicted_train_seq_list = np.array(predicted_train_seq_list).reshape(50,59,2)
  predicted_train_img_list = np.array(predicted_train_img_list).reshape(50,59,3,50,50)
predicted_train_img_list = predicted_train_img_list.transpose(0,1,3,4,2)
real_train_img_list = real_train_img_list.transpose(0,1,3,4,2)

#test
with torch.no_grad():
  for test_img, test_seq, test_pb in test_dataloader:
    predicted_test_seq_list = []
    predicted_test_img_list = []
    test_rnn_hidden = None
    for step_idx in range(50-1):
      if step_idx == 0:
        t_test_img = test_img[0,step_idx].reshape(1,1,3,50,50)
        t_test_seq = test_seq[0,step_idx].reshape(1,1,2)
        test_pb = test_pb[0].reshape(1,2)
        predicted_test_img, predicted_test_seq, test_rnn_hidden = model(t_test_img, t_test_seq, test_pb, test_rnn_hidden)
        predicted_test_seq_list.append(predicted_test_seq[0][0].detach().numpy())
        predicted_test_img_list.append(predicted_test_img[0][0].detach().numpy())
      else:
        t_test_img, t_test_seq = predicted_test_img, predicted_test_seq
        predicted_test_img, predicted_test_seq, test_rnn_hidden = model(t_test_img, t_test_seq, test_pb, test_rnn_hidden)
        predicted_test_seq_list.append(predicted_test_seq[0][0].detach().numpy())
        predicted_test_img_list.append(predicted_test_img[0][0].detach().numpy())

    real_test_seq_list = test_seq.detach().numpy()
    real_test_img_list = test_img.detach().numpy()

  predicted_test_seq_list = np.array(predicted_test_seq_list).reshape(1,59,2)
  predicted_test_img_list = np.array(predicted_test_img_list).reshape(1,59,3,50,50)
predicted_test_img_list = predicted_test_img_list.transpose(0,1,3,4,2)
real_test_img_list = real_test_img_list.transpose(0,1,3,4,2)

#train_xy
idx = 5
train_fig = plt.figure()
plt.title('Train data')
plt.scatter(real_train_seq_list[idx,:,0], real_train_seq_list[idx,:,1], label='input')
plt.scatter(predicted_train_seq_list[idx,:,0], predicted_train_seq_list[idx,:,1], label='predicted')
plt.legend()
train_fig.savefig("Train.png")

#test_xy
test_fig = plt.figure()
plt.title('Test data')
plt.scatter(real_test_seq_list[0,:,0], real_test_seq_list[0,:,1], label='input')
plt.scatter(predicted_test_seq_list[0,:,0], predicted_test_seq_list[0,:,1], label='predicted')
plt.legend()
test_fig.savefig("Test.png")

#train_video
import matplotlib.animation as anim
from IPython.display import HTML
predicted_train_img_list[predicted_train_img_list < 0] = 0
predicted_train_img_list[predicted_train_img_list > 1] = 1
real_train_video = real_train_img_list[0]
predicted_train_video = predicted_train_img_list[0]
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
def update(i):
  if i !=0:
    for ax in axes:
      ax.cla()
    axes[0].set_title('Train Input Image')
    axes[0].imshow(real_train_video[i])
    axes[1].set_title('Train Predicted Image')
    axes[1].imshow(predicted_train_video[i-1])

a = anim.FuncAnimation(fig, update, interval=100, frames=real_train_video.shape[0])
a.save("train.gif", writer="imagegick")

#test_video
predicted_test_img_list[predicted_test_img_list < 0] = 0
predicted_test_img_list[predicted_test_img_list > 1] = 1
real_test_video = real_test_img_list[0]
predicted_test_video = predicted_test_img_list[0]
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
def update(i):
  if i !=0:
    for ax in axes:
      ax.cla()
    axes[0].set_title('Test Input Image')
    axes[0].imshow(real_test_video[i])
    axes[1].set_title('Test Predicted Image')
    axes[1].imshow(predicted_test_video[i-1])

a = anim.FuncAnimation(fig, update, interval=100, frames=predicted_test_video.shape[0])
a.save("test.gif", writer="imagegick")
