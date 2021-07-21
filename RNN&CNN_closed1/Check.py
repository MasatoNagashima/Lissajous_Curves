import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import Data

#import
device = Data.device
train_dataloader = Data.train_dataloader
model = Data.model
model.load_state_dict(torch.load("model.pth"))
model.eval()

with torch.no_grad():#train
  for train_img, train_seq, train_typ in train_dataloader:
    predicted_train_seq_list = []
    predicted_train_img_list = []
    train_rnn_hidden = model.get_hcs(train_typ)
    for step_idx in range(100-1):
      if step_idx == 0:
        t_train_img = train_img[:, step_idx].reshape(3,1,3,32,32).to(device)
        t_train_seq = train_seq[:,step_idx].reshape(3,1,2).to(device)
        predicted_train_img, predicted_train_seq, train_rnn_hidden = model(t_train_img, t_train_seq, train_rnn_hidden)
      else:
        predicted_train_img, predicted_train_seq, train_rnn_hidden = model(predicted_train_img, predicted_train_seq, train_rnn_hidden)
      predicted_train_seq_list.append(predicted_train_seq[0][0].cpu().detach().numpy())
      predicted_train_img_list.append(predicted_train_img[0][0].cpu().detach().numpy())
    real_train_seq_list = train_seq.detach().numpy()
    real_train_seq_list = real_train_seq_list[0].reshape(1,100,2)
    real_train_img_list = train_img.detach().numpy()
    real_train_img_list = real_train_img_list[0].reshape(1,100,3,32,32)
  predicted_train_seq_list = np.array(predicted_train_seq_list).reshape(1,99,2)
  predicted_train_img_list = np.array(predicted_train_img_list).reshape(1,99,3,32,32)
predicted_train_img_list = predicted_train_img_list.transpose(0,1,4,3,2)
real_train_img_list = real_train_img_list.transpose(0,1,4,3,2)

#train_xy
train_fig = plt.figure()
plt.title('Train data')
plt.scatter(real_train_seq_list[0,:,0], real_train_seq_list[0,:,1], label='input')
plt.scatter(predicted_train_seq_list[0,:,0], predicted_train_seq_list[0,:,1], label='predicted')
plt.legend()
train_fig.savefig("Train_data")

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
    axes[1].imshow(predicted_train_video[i])

a = anim.FuncAnimation(fig, update, interval=100, frames=predicted_train_video.shape[0])
a.save("train.gif", writer="imagegick")
