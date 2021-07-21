import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

def normalization(data, indataRange, outdataRange):
    data = data.astype(np.float32)
    data = ( data - indataRange[0] ) / ( indataRange[1] - indataRange[0] )
    data = data * ( outdataRange[1] - outdataRange[0] ) + outdataRange[0]
    return data

def getLissajousMovie(first_pos, total_step, num_cycle, imsize, circle_r, x_mag, y_mag, vmin=-0.9, vmax=0.9):
    from PIL import Image, ImageDraw
    #total_step: 点の数, num_cycle: tの大きさ, imsize: 画像の大きさ, circle_r: 円の大きさ,x_mag: xの周期
    #y_mag: yの周期, vmin:正規化の最小値? vmax:正規化の最大値?
    t = np.linspace(first_pos, first_pos+2.0*np.pi*num_cycle, total_step)
    x = np.cos(t*x_mag)
    y = np.sin(t*y_mag-np.pi/4)
    imgs = []
    for _t in range(total_step):
        _x = (x[_t] * (imsize * 0.4))+imsize/2
        _y = (y[_t] * (imsize * 0.4))+imsize/2
        img = Image.new("RGB", (imsize, imsize), "white")
        draw = ImageDraw.Draw(img)
        draw.ellipse((_x-circle_r, _y-circle_r, _x+circle_r, _y+circle_r), fill=128)
        imgs.append(np.expand_dims(np.asarray(img), 0))
    imgs = np.vstack(imgs)
    ### normalization
    imgs = normalization(imgs.astype(np.float32), [0, 255], [vmin, vmax])
    seq = normalization(np.c_[x,y].astype(np.float32), [-1.0, 1.0], [vmin, vmax])
    pb = normalization(np.c_[x_mag,y_mag].astype(np.float32), [0.5, 2.5], [vmin, vmax])
    return imgs, seq, pb

#train_data
train_Image = []
train_Sequence = []
train_PB = []
for x_ in range(2):
  for y_ in range(2):
    for f_ in range(100):
      first_position = 2 * np.pi * f_ / 100
      img, seq, pb = getLissajousMovie(first_position, total_step=60, num_cycle = 1, imsize=50, circle_r=1, x_mag=(x_+1),  y_mag=(y_+1), vmin=-0.9, vmax=0.9)
      if x_ == 0 and y_ == 0 and f_ == 0:
        train_Image , train_Sequence, train_PB = img, seq, pb
      else:
        train_Image = np.append(train_Image, img, axis = 0)
        train_Sequence = np.append(train_Sequence,seq, axis = 0)
        train_PB = np.append(train_PB, pb)

train_Image = train_Image.reshape(400, 60, 50, 50, 3)
train_Image =train_Image.transpose(0,1,4,2,3)
train_Sequence = train_Sequence.reshape(400,60,2)
train_PB = train_PB.reshape(400, 2)
train_Image = torch.Tensor(train_Image)
train_Sequence = torch.Tensor(train_Sequence)
train_PB = torch.Tensor(train_PB)
train_img_seq_pb = TensorDataset(train_Image,train_Sequence,train_PB)
train_dataloader = torch.utils.data.DataLoader(train_img_seq_pb, batch_size=50, shuffle=True)

#test_data
test_Image, test_Sequence, test_PB = getLissajousMovie(first_pos=0, total_step=60, num_cycle = 1, imsize=50, circle_r=1, x_mag=2,  y_mag=3, vmin=-0.9, vmax=0.9)
test_Image = test_Image.reshape(1, 60, 50, 50, 3)
test_Image =test_Image.transpose(0,1,4,2,3)
test_Sequence = test_Sequence.reshape(1, 60, 2)
test_PB = test_PB.reshape(1, 2)
test_Image = torch.Tensor(test_Image)
test_Sequence = torch.Tensor(test_Sequence)
test_PB = torch.Tensor(test_PB)
test_img_seq_pb = TensorDataset(test_Image,test_Sequence,test_PB)
test_dataloader = torch.utils.data.DataLoader(test_img_seq_pb, batch_size=1, shuffle=False)


class CNN_RNN(nn.Module):
  def __init__(self):
    super(CNN_RNN, self).__init__()
    self.enc_conv = nn.Sequential(
         nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size =3, stride = 2),nn.LeakyReLU(),
         nn.Conv2d(in_channels = 32, out_channels = 24, kernel_size = 3, stride = 2),nn.LeakyReLU(),
         nn.Conv2d(in_channels = 24, out_channels = 10, kernel_size= 3, stride = 2), nn.LeakyReLU(),
    )
    self.enc_lnr = nn.Sequential(
        nn.Linear(10*5*5, 100), nn.Tanh(),
        nn.Linear(100, 20), nn.Tanh()
    )
    self.lstm = nn.LSTMCell(24, 22)
    self.dec_lnr = nn.Sequential(
         nn.Linear(20, 50), nn.Tanh(),
         nn.Linear(50, 100), nn.Tanh(),
         nn.Linear(100,10*5*5)
    )
    self.dec_conv = nn.Sequential(
        nn.ConvTranspose2d(in_channels = 10, out_channels =24, kernel_size = 3, stride = 2), nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels = 24, out_channels =32, kernel_size = 4, stride = 2), nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels = 32, out_channels =3, kernel_size = 4, stride = 2)
    )
    self.xy_lnr = nn.Sequential(
        nn.Linear(2,10), nn.Tanh(),
        nn.Linear(10,2)
    )


  def __call__(self, in_img, in_xy, in_pb, lstm_hid):
    batch, num, ch, h, w = in_img.shape
    batch, num, xy =  in_xy.shape
    in_img = in_img.view(batch*num, ch, h, w)
    in_xy = in_xy.view(batch*num, xy)
    #in_img
    img_hid = self.enc_conv(in_img)
    batch, _ch, _h, _w = img_hid.shape
    img_hid = img_hid.view(batch, _ch*_h*_w)
    img_hid = self.enc_lnr(img_hid)
    #in_img, in_xy, in_pos
    hid = torch.cat([img_hid,in_xy,in_pb], dim=-1)
    new_hid1, new_hid2 = self.lstm(hid, lstm_hid)
    img_hid, xy_hid = torch.split(new_hid1, [20,2], dim=-1)
    #out_img
    img_hid = self.dec_lnr(img_hid)
    img_hid = img_hid.view(batch, _ch, _h, _w)
    out_img = self.dec_conv(img_hid)
    out_img = out_img.view(batch, num, ch, h, w)
    #out_xy
    out_xy = self.xy_lnr(xy_hid)
    out_xy = out_xy.view(batch, num, xy)
    return out_img, out_xy, (new_hid1, new_hid2)

device = torch.device("cuda:0")
model = CNN_RNN().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
epochs = 1500
