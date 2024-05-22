import torch
import torch.nn as nn
from ufront.onnx.model import ONNXModel, ONNXModelKeras, UFrontONNX
import io
from ufront.pytorch.model import UFrontTorch
import iree.compiler as ireec
from iree import runtime
import time
import torch
from torch_def import *
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

GPU = False
batch_size = 8
hidden_size = 128
seq_size = 32
input_size = 256
t1_start = time.perf_counter()
input = np.random.randn(batch_size, seq_size,hidden_size).astype(np.float32)
h0 = np.zeros((batch_size, hidden_size), dtype=np.float32)
c0 = np.zeros((batch_size, hidden_size), dtype=np.float32)
input, h0, c0 = torch.Tensor(input), torch.Tensor(h0), torch.Tensor(c0)
lstm = SimpleLSTM(input_size = 10, hidden_size = hidden_size, seq_size=seq_size)
model = UFrontTorch(lstm, batch_size=batch_size, pass_weights=True)
output_tensors = model(inputs = [input, h0, c0])

#This will trigger model compilation, i.e., convert Rust computation graph to a unified high-level IR and lower it to TOSA IR
model.compile(optimizer={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"},
                      loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])


# modelir= model.dump_ir()
# print(modelir)

tosa_ir = model.dump_tosa_ir()
# print(modelir)
t1_stop = time.perf_counter()

if GPU:
    binary = ireec.compile_str(tosa_ir,
                    target_backends=["cuda"], 
                    input_type=ireec.InputType.TOSA)
    module = runtime.load_vm_flatbuffer(binary, driver="cuda")
else:
    binary = ireec.compile_str(tosa_ir,
                    target_backends=["llvm-cpu"], 
                    input_type=ireec.InputType.TOSA)
    module = runtime.load_vm_flatbuffer(binary,backend="llvm-cpu") 
t2_stop = time.perf_counter()

print("Time taken: ", t2_stop - t1_start, "TOSA ", t1_stop - t1_start, " Binary ", t2_stop - t1_stop)
for i in range(10):
    ufront_ret = module.forward(input, h0, c0)
t3_stop = time.perf_counter()
print("Inference time taken: ", (t3_stop - t2_stop)/10)

output, (hn,cn) = lstm(input, h0, c0)
ufront_out = ufront_ret[0].to_host()
compared_out = output.detach().numpy()

dif = compared_out - ufront_out
mae = np.mean(abs(dif))
print("MAE:", mae)
print("RMSE:", rmse(torch.Tensor(compared_out), torch.Tensor(ufront_out)).numpy())
print("COD:", r_square(torch.Tensor(compared_out), torch.Tensor(ufront_out)).numpy())
print("MPE:", mpe(torch.Tensor(compared_out), torch.Tensor(ufront_out)).numpy(), "%")
