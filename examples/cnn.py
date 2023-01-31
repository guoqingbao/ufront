import torch.nn as nn
import numpy as np
import torch

from ufront.pytorch.model import PyTorchModel #Flexflow-like PytorchModel wrapper
from ufront import Model, PyOperator, TensorF32, Optimizer, LossType, MetricsType #Rust frontend

# A sample pytorch model definition
class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 32, 3, 1)
    self.pool1 = nn.MaxPool2d(2, 2)
    self.conv3 = nn.Conv2d(32, 64, 3, 1)
    self.conv4 = nn.Conv2d(64, 64, 3, 1)
    self.pool2 = nn.MaxPool2d(2, 2)
    self.flat1 = nn.Flatten()
    self.linear1 = nn.Linear(512, 512)
    self.linear2 = nn.Linear(512, 10)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, input1, input2):
    y1 = self.conv1(input1)
    y1 = self.relu(y1)
    y2 = self.conv1(input2)
    y2 = self.relu(y2)
    y = torch.cat((y1, y2), 1)
    (y1, y2) = torch.split(y, 1) 
    y = torch.cat((y1, y2), 1)
    y = self.conv2(y)
    y = self.relu(y)
    y = self.pool1(y)
    y = self.conv3(y)
    y = self.relu(y)
    y = self.conv4(y)
    y = self.relu(y)
    y = self.pool2(y)
    y = self.flat1(y)
    y = self.linear1(y)
    y = self.relu(y)
    yo = self.linear2(y)
    return (yo, y)


if __name__ == "__main__":

    torch_model = PyTorchModel(CNN()) # Intermedium torch model
    ufront_model = Model() # Ufront Rust model

    
    batch_size = 32
    
    operators = []
    arr = np.zeros((batch_size, 3, 32, 32), dtype=np.float32)
    input_tensor = TensorF32(arr) # convert to Rust f32 tensor

    #save model to file (compatible with flexflow)
    # output_tensors, operators = torch_model.torch_to_ff(ufront_model, [input_tensor, input_tensor])

    #load model from file (compatible with flexflow)
    # output_tensors, operators = PyTorchModel.file_to_ff('cnn.ff', ufront_model, [input_tensor, input_tensor])

    #torch model to ufront model, this will trigger Rust frontend for building computation graph
    #operators can also be managed by python side (each operator here corresponding to an operator in the Rust computation graph)
    output_tensors, operators = torch_model.torch_to_ff(ufront_model, [input_tensor, input_tensor])

    softmax_op = ufront_model.softmax(input=output_tensors[0], name="softmax")
    operators.append(softmax_op)

    #The Rust frontend will build computation graph and initialize temporary inputs and outputs for each operator
    total_memory = 0
    for operator in operators:
      sz = operator.memory_usage()
      total_memory += sz
      print("{0} > name: {1}, raw_ptr: {2:#06x}, No. of outputs: {3}, memory used:{4:.5f}MB".format(operator.op_type, operator.params['name'], 
      operator.raw_ptr, operator.num_of_outputs(),  sz/1024/1024))
    
    #Total memory cached for inputs/outputs of all operators (in Rust)
    print("\r\nTotal memory cached for operators {:.2f}MB".format(total_memory/1024/1024))

    #The output of the model (forward pass have not been triggered at the moment!)
    output = softmax_op.get_output(0)
    print(output.shape)
    
    #optimizer
    ufront_model.optimizer = Optimizer(params={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"})
    
    #This will trigger model compilation, i.e., convert Rust computation graph to a unified high-level IR and lower it to TOSA IR
    ufront_model.compile(loss_type=LossType.SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[MetricsType.ACCURACY, MetricsType.SPARSE_CATEGORICAL_CROSSENTROPY])
    
    #This will be supported later
    #ufront_model.forward()
    
    #This will be supported later
    #ufront_model.backward()