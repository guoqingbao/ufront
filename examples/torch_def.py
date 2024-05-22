import torch
import io
import os
import json
import pathlib
import torch.nn as nn
import math
from urllib.request import urlopen
from PIL import Image
import numpy as np

def mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return torch.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

# also known as cod
def r_square(y_true, y_pred):
    y_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    return 1 - ss_res / ss_tot

def mpe(y_true, y_pred):
    return torch.mean((y_true - y_pred) / y_true) * 100

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

def load_sample_image():
    url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
    path = str(pathlib.Path(__file__).parent.resolve()) + "/resources/YellowLabradorLooking_new.jpg"
    if os.path.exists(path):
        with open(path, "rb") as f:
            img = Image.open(io.BytesIO(f.read()))
    else:
        response = urlopen(url)
        img = Image.open(io.BytesIO(response.read()))
    image = np.array(img.resize((224, 224)), dtype=np.float32)
    image = image[np.newaxis, :]
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    clast_image = image/ 255.0
    clast_image = (clast_image - mean) / std #normlization 
    clast_image = clast_image.astype(np.float32)
    return clast_image, np.moveaxis(clast_image, -1, 1)

def decode_result(preds, top=5):
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError(
            "`decode_predictions` expects "
            "a batch of predictions "
            "(i.e. a 2D array of shape (samples, 1000)). "
            "Found array with shape: " + str(preds.shape)
        )
    url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
    path = str(pathlib.Path(__file__).parent.resolve()) + "/resources/imagenet_class_index.json"
    if os.path.exists(path):
        with open(path) as f:
            data_json = json.loads(f.read())
    else:
        response = urlopen(url)
        data_json = json.loads(response.read())
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(data_json[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results