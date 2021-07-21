import numpy as np
import matplotlib.pyplot as plt
import torch

#data
def normalization(data, indataRange, outdataRange):
    data = data.astype(np.float32)
    data = ( data - indataRange[0] ) / ( indataRange[1] - indataRange[0] )
    data = data * ( outdataRange[1] - outdataRange[0] ) + outdataRange[0]
    return data

def getLissajousMovie(total_step, num_cycle, imsize, circle_r, x_mag, y_mag, vmin=-0.9, vmax=0.9):
    from PIL import Image, ImageDraw
    #total_step: 点の数, num_cycle: tの大きさ, imsize: 画像の大きさ, circle_r: 円の大きさ,x_mag: xの周期
    #y_mag: yの周期, vmin:正規化の最小値? vmax:正規化の最大値?
    t = np.linspace(0, 2.0*np.pi*num_cycle, total_step)
    x = np.cos(t*x_mag)
    y = np.sin(t*y_mag)
    imgs = []
    for _t in range(total_step):
        _x = (x[_t] * (imsize * 0.4))+imsize/2
        _y = (y[_t] * (imsize * 0.4))+imsize/2
        img = Image.new("RGB", (imsize, imsize), "white")
        draw = ImageDraw.Draw(img)
        draw.ellipse((_x-circle_r, _y-circle_r, _x+circle_r, _y+circle_r), fill=128)
        imgs.append(np.expand_dims(np.asarray(img), 0))
    imgs = np.vstack(imgs)
    imgs = normalization(imgs.astype(np.float32), [0, 255], [vmin, vmax])#画像データ
    seq = normalization(np.c_[x,y].astype(np.float32), [-1.0, 1.0], [vmin, vmax])#座標データ
    return imgs, seq

Image, Sequence = getLissajousMovie(total_step=300, num_cycle = 1, imsize=32, circle_r=3, x_mag=1, y_mag=3,vmin=-1,vmax=1)

Image_sample = Image.transpose(0,3,1,2)
train_dataset, val_dataset = torch.utils.data.random_split(Image_sample, [250, 50])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=True)


#learning
import torch
import torch.nn as nn
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

class ConvAutoencoder(nn.Module):
  def __init__(self):
    super(ConvAutoencoder, self).__init__()
    #enc_Conv2d
    self.enc_conv = nn.Sequential(
         nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size =4, stride = 2), nn.LeakyReLU(),
         nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 2), nn.LeakyReLU(),
         nn.Conv2d(in_channels = 16, out_channels = 20, kernel_size= 3, stride = 1), nn.LeakyReLU(),
    )
    #enc_Linear
    self.enc_lnr = nn.Sequential(
         nn.Linear(20*5*5, 100), nn.Tanh(),
         nn.Linear(100, 50), nn.Tanh(),
         nn.Linear(50, 20), nn.Tanh()
    )
    #dec_Linear
    self.dec_lnr = nn.Sequential(
         nn.Linear(20,50), nn.Tanh(),
         nn.Linear(50,100), nn.Tanh(),
         nn.Linear(100,20*5*5), nn.Tanh()
    )
    #dec_Conv2d
    self.dec_conv = nn.Sequential(
        nn.ConvTranspose2d(in_channels = 20, out_channels =16, kernel_size = 3, stride = 1), nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels = 16, out_channels =8, kernel_size = 3, stride = 2), nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels = 8, out_channels =3, kernel_size = 4, stride = 2), nn.LeakyReLU(),
    )

  def forward(self, in_img):
    hid = self.enc_conv(in_img)
    batch, ch, h, w = hid.shape
    hid = hid.view(batch, ch*h*w)
    hid = self.enc_lnr(hid)
    hid = self.dec_lnr(hid)
    hid = hid.view(batch, ch, h, w)
    out_img = self.dec_conv(hid)
    return out_img

def train_net(n_epochs, train_loader, net, optimizer_cls = optim.Adam, loss_fn = nn.MSELoss(), device = "cuda:0"):
  """
  n_epochs…訓練の実施回数
  net …ネットワーク
  device …　"cpu" or "cuda:0"
  """
  losses = []
  optimizer = optimizer_cls(net.parameters(), lr = 0.001)
  net.to(device)

  for epoch in range(n_epochs):
    running_loss = 0.0
    net.train()

    for i, XX in enumerate(train_loader):
      XX=XX.to(device)
      optimizer.zero_grad()
      XX_pred = net(XX)
      loss = loss_fn(XX,XX_pred)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

    losses.append(running_loss/i)
    if epoch % 1000 == 0:
        if epoch == 0:
            t = open("step_loss.txt", "w")
        else:
            t = open("step_loss.txt", "a")
        t.write("epoch:{}, train_loss:{}\n".format(epoch, running_loss/i))
        print("epoch", epoch, ": ", running_loss / i)
        t.close()
  return losses

net = ConvAutoencoder()
losses = train_net(n_epochs = 10000, train_loader = train_loader, net = net)


#check
img_num = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
idx = int(np.random.random() * Image.shape[0])
for train_image in train_loader:
  train_Image = train_image.to(device)
  pred_Image = net.forward(train_Image)
  train_Image = train_Image.permute(0,2,3,1)
  pred_Image = pred_Image.permute(0,2,3,1)
  train_Image = train_Image.cpu().detach().numpy()
  pred_Image = pred_Image.cpu().detach().numpy()
  break

plt.figure(figsize=(5,5))
train_Img = normalization(train_Image[img_num].astype(np.float32), [-0.9, 0.9], [0.1, 0.9])
plt.imsave("train_data.png", train_Img)

plt.figure(figsize=(5,5))
pred_Img = normalization(pred_Image[img_num].astype(np.float32), [-0.9, 0.9], [0.1, 0.9])
plt.imsave("test_data.png", pred_Img)
