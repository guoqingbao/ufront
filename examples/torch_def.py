import torch
import torch.nn as nn
from torch.onnx import TrainingMode
import math

class SimpleCNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(4096, 512), 
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
            nn.Softmax())

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ComplexCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32, 3, 1)
    self.conv2 = nn.Conv2d(64, 64, 3, 1)
    self.pool1 = nn.MaxPool2d(2, 2)
    self.conv3 = nn.Conv2d(64, 64, 3, 1)
    self.conv4 = nn.Conv2d(64, 64, 3, 1)
    self.pool2 = nn.MaxPool2d(2, 2)

    self.batch_norm = nn.BatchNorm2d(64)

    self.flat1 = nn.Flatten()
    self.linear1 = nn.Linear(1600, 512)
    self.linear2 = nn.Linear(512, 10)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, input1, input2):
    y1 = self.conv1(input1)
    y1 = self.relu(y1)
    y2 = self.conv1(input2)
    y2 = self.relu(y2)
    y = torch.cat((y1, y2), 1)
    (y1, y2) = torch.split(y, [32, 32], dim=1) # split into [32, 32] in axis 1
    y = torch.cat((y1, y2), 1)
    y = self.conv2(y)
    y = self.relu(y)
    y = self.pool1(y)
    y = self.conv3(y)
    y = self.relu(y)
    y = self.conv4(y)
    y = self.relu(y)
    y = self.pool2(y)

    y = self.batch_norm(y)

    y = self.flat1(y)
    y = self.linear1(y)
    y = self.relu(y)
    yo = self.linear2(y)
    return (yo, y)
  
class SimpleCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32, 3, 1)
    self.conv2 = nn.Conv2d(64, 64, 3, 1)
    self.pool1 = nn.MaxPool2d(4, 4)
    self.batch_norm = nn.BatchNorm2d(64)
    self.flat1 = nn.Flatten()
    self.linear1 = nn.Linear(3136, 512)
    self.linear2 = nn.Linear(512, 10)
    self.relu = nn.ReLU()

  def forward(self, input1, input2):
    y1 = self.relu(self.conv1(input1))
    y2 = self.relu(self.conv1(input2))
    y = torch.cat((y1, y2), 1)
    (y1, y2) = torch.split(y, [32, 32], dim=1) # split into [32, 32] in axis 1
    y = torch.cat((y1, y2), 1)
    y = self.relu(self.conv2(y))
    y = self.pool1(y)
    y = self.batch_norm(y)
    y = self.flat1(y)
    y = self.relu(self.linear1(y))
    return self.linear2(y)
  
class TestCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Conv2d(3, 32, 3, 4, bias=False)
    self.pool = nn.AvgPool2d(2)
    self.batch_norm = nn.BatchNorm2d(32)
    self.linear = nn.Linear(32, 10, bias=False)
    self.flat = nn.Flatten()
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.flat(self.batch_norm(self.pool(self.conv(x))))
    
    x = self.linear(x)
    return self.relu(x)

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, seq_size):
        super().__init__()
        self.input_sz = input_size
        self.hidden_size = hidden_size
        self.seq_sz = seq_size
        self.W = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x, 
                h_t, c_t):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
         
        HS = self.hidden_size
        for t in range(self.seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)