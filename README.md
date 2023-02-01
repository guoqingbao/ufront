# ufront
Unified Computing Frontend for Deep Learning 

_(This project is under development)_


## Project discription
1. The objective of this project is to create a **unified frontend** for deep learning computing.

2. The frontend imports Pytorch, Keras, ONNX (and possiblely Tensorflow) models using the FlexFlow-like python scripts and then translate them into the **Unified High-level IR** based on Rust.

3. The frontend was built based on **Rust** and the Rust computing interfaces were exposed to python through PyO3. 

4. The computing nodes (operators) in the Pytorch, Keras, Tensorflow, and ONNX models can be mapped to Rust computing interfaces, in which the Rust frontend will maintain a high-level computation graph.

5. The Rust frontend will then translate the high-level graph into IR and then lower it into TOSA dialect (a standard IR for deep learning computing).

6. In addition to translating Pytorch, Keras, Tensorflow, and ONNX models into the standard computing IR (TOSA), the Rust frontend also provide standard computing workflows including operators, forward, backward, gradient update, etc. for training.

## Sample usage for Pytorch Models
``` Python
import torch.nn as nn
import numpy as np
import torch

from ufront.pytorch.model import PyTorchModel #Flexflow-like PytorchModel wrapper
from ufront import Model, PyOperator, TensorF32, Optimizer, LossType, MetricsType #Rust frontend

# A sample pytorch model definition
class ComplexCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32, 3, 1)
    self.conv2 = nn.Conv2d(64, 64, 3, 1)
    self.pool1 = nn.MaxPool2d(2, 2)
    self.conv3 = nn.Conv2d(64, 64, 3, 1)
    self.conv4 = nn.Conv2d(64, 64, 3, 1)
    self.pool2 = nn.MaxPool2d(2, 2)
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
    y = self.flat1(y)
    y = self.linear1(y)
    y = self.relu(y)
    yo = self.linear2(y)
    return (yo, y)


if __name__ == "__main__":
    torch_model = PyTorchModel(ComplexCNN()) # Intermedium torch model
    ufront_model = Model() # Ufront Rust model

    batch_size = 32
    operators = []
    arr = np.zeros((batch_size, 3, 128, 128), dtype=np.float32)
    input_tensor1 = TensorF32(arr, "input1") # convert to Rust f32 tensor
    input_tensor2 = TensorF32(arr, "input2") # convert to Rust f32 tensor

    #save model to file (compatible with flexflow)
    # torch_model.torch_to_file('cnn.ff')

    #load model from file (compatible with flexflow)
    # output_tensors, operators = PyTorchModel.file_to_ff('cnn.ff', ufront_model, [input_tensor, input_tensor])

    #torch model to ufront model, this will trigger Rust frontend for building computation graph
    #operators can also be managed by python side (each operator here corresponding to an operator in the Rust computation graph)
    output_tensors, operators = torch_model.apply(ufront_model, [input_tensor1, input_tensor2])

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
    
    print("\r\n\r\n")

    # for operator in operators:
    #   print(operator.ir) #show ir for each operator

    print(ufront_model.dump_ir()) # see below results

    #This will be supported later
    #ufront_model.forward()
    
    #This will be supported later
    #ufront_model.backward()
```

## IR Dump Results (for Pytorch)
### (Converted from Pytorch model to High-level IR after calling dump_ir(), i.e., TOSA-like dialect IR)
``` Python
func.func @forward(%input1: tensor<32x3x32x32>, %input2: tensor<32x3x32x32>) -> tensor<32x10>  { 
        %conv1="ufront.conv2d"(%input1){"groups": "1", "padding": "[0, 0]", "out_channels": "32", "kernel_size": "[3, 3]", "stride": "[1, 1]"}:(tensor<32x3x32x32>) -> tensor<32x32x30x30>
        %relu_1="ufront.relu"(%conv1):(tensor<32x32x30x30>) -> tensor<32x32x30x30>
        %conv1_1="ufront.conv2d"(%input2){"groups": "1", "stride": "[1, 1]", "padding": "[0, 0]", "out_channels": "32", "kernel_size": "[3, 3]"}:(tensor<32x3x32x32>) -> tensor<32x32x30x30>
        %relu_2="ufront.relu"(%conv1_1):(tensor<32x32x30x30>) -> tensor<32x32x30x30>
        %cat_1="ufront.concat"(%relu_1, %relu_2){"axis": "1"}:(tensor<32x32x30x30>, tensor<32x32x30x30>) -> tensor<32x64x30x30>
        %split_1, %split_1_1="ufront.split"(%cat_1){"axis": "1", "sizes": "[32, 32]"}:(tensor<32x64x30x30>) -> tensor<32x32x30x30>, tensor<32x32x30x30>
        %cat_2="ufront.concat"(%split_1, %split_1_1){"axis": "1"}:(tensor<32x32x30x30>, tensor<32x32x30x30>) -> tensor<32x64x30x30>
        %conv2="ufront.conv2d"(%cat_2){"kernel_size": "[3, 3]", "stride": "[1, 1]", "out_channels": "64", "padding": "[0, 0]", "groups": "1"}:(tensor<32x64x30x30>) -> tensor<32x64x28x28>
        %relu_3="ufront.relu"(%conv2):(tensor<32x64x28x28>) -> tensor<32x64x28x28>
        %pool1="ufront.pool2d"(%relu_3){"pool_type": "PoolType.POOL_MAX", "stride": "[2, 2]", "padding": "[0, 0]", "kernel_size": "[2, 2]"}:(tensor<32x64x28x28>) -> tensor<32x64x14x14>
        %conv3="ufront.conv2d"(%pool1){"out_channels": "64", "groups": "1", "stride": "[1, 1]", "kernel_size": "[3, 3]", "padding": "[0, 0]"}:(tensor<32x64x14x14>) -> tensor<32x64x12x12>
        %relu_4="ufront.relu"(%conv3):(tensor<32x64x12x12>) -> tensor<32x64x12x12>
        %conv4="ufront.conv2d"(%relu_4){"padding": "[0, 0]", "groups": "1", "stride": "[1, 1]", "kernel_size": "[3, 3]", "out_channels": "64"}:(tensor<32x64x12x12>) -> tensor<32x64x10x10>
        %relu_5="ufront.relu"(%conv4):(tensor<32x64x10x10>) -> tensor<32x64x10x10>
        %pool2="ufront.pool2d"(%relu_5){"pool_type": "PoolType.POOL_MAX", "stride": "[2, 2]", "padding": "[0, 0]", "kernel_size": "[2, 2]"}:(tensor<32x64x10x10>) -> tensor<32x64x5x5>
        %flat1="ufront.flat"(%pool2):(tensor<32x64x5x5>) -> tensor<32x1600>
        %linear1="ufront.linear"(%flat1):(tensor<32x1600>) -> tensor<32x512>
        %relu_6="ufront.relu"(%linear1):(tensor<32x512>) -> tensor<32x512>
        %linear2="ufront.linear"(%relu_6):(tensor<32x512>) -> tensor<32x10>
        %softmax="ufront.softmax"(%linear2):(tensor<32x10>) -> tensor<32x10>
        return %softmax: tensor<32x10>
}
```

## Sample usage for ONNX Models
``` Python
from ufront.onnx.model import ONNXModel, ONNXModelKeras
from ufront import Model, PyOperator, TensorF32, Optimizer, LossType, MetricsType #Rust frontend

import argparse
import numpy as np
import onnx
import torch
import torch.nn as nn
from torch.onnx import TrainingMode

if __name__ == "__main__":
    batch_size = 32
    
    operators = []
    arr = np.zeros((batch_size, 3, 32, 32), dtype=np.float32)
    input_tensor1 = TensorF32(arr, "input1") # convert to Rust f32 tensor
    input_tensor2 = TensorF32(arr, "input2") # convert to Rust f32 tensor

    torch.onnx.export(model=ComplexCNN(), args=(torch.from_numpy(arr), torch.from_numpy(arr)), f="cifar10_cnn_pt.onnx", export_params=False, training=TrainingMode.TRAINING)
    
    onnx_model = onnx.load("cifar10_cnn_pt.onnx")

    dims_input = [batch_size, 3, 32, 32]
    ufront_model = Model() # Ufront Rust model
    onnx_model = ONNXModel("cifar10_cnn_pt.onnx")

    #torch model to ufront model, this will trigger Rust frontend for building computation graph
    #operators can also be managed by python side (each operator here corresponding to an operator in the Rust computation graph)
    output_tensors, operators = onnx_model.apply(ufront_model, input_tensors=[input_tensor1, input_tensor2])

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
    output = output_tensors[0]
    print(output.shape)
    
    #optimizer
    ufront_model.optimizer = Optimizer(params={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"})
    
    #This will trigger model compilation, i.e., convert Rust computation graph to a unified high-level IR and lower it to TOSA IR
    ufront_model.compile(loss_type=LossType.SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[MetricsType.ACCURACY, MetricsType.SPARSE_CATEGORICAL_CROSSENTROPY])
    
    print("\r\n\r\n")

    # for operator in operators:
    #   print(operator.ir) #show ir for each operator

    print(ufront_model.dump_ir())

    #This will be supported later
    #ufront_model.forward()
    
    #This will be supported later
    #ufront_model.backward()

```

## IR Dump Results (for ONNX)
### (Converted from ONNX model to High-level IR after calling dump_ir(), i.e., TOSA-like dialect IR)
``` Python
func.func @forward(%input2: tensor<32x3x32x32>, %input1: tensor<32x3x32x32>) -> tensor<32x10>  { 
        %Conv_0="ufront.conv2d"(%input1){"out_channels": "32", "stride": "[1, 1]", "kernel_size": "[3, 3]", "groups": "1", "padding": "[0, 0]"}:(tensor<32x3x32x32>) -> tensor<32x32x30x30>
        %Relu_1="ufront.relu"(%Conv_0):(tensor<32x32x30x30>) -> tensor<32x32x30x30>
        %Conv_2="ufront.conv2d"(%input2){"stride": "[1, 1]", "out_channels": "32", "groups": "1", "kernel_size": "[3, 3]", "padding": "[0, 0]"}:(tensor<32x3x32x32>) -> tensor<32x32x30x30>
        %Relu_3="ufront.relu"(%Conv_2):(tensor<32x32x30x30>) -> tensor<32x32x30x30>
        %Concat_4="ufront.concat"(%Relu_1, %Relu_3){"axis": "1"}:(tensor<32x32x30x30>, tensor<32x32x30x30>) -> tensor<32x64x30x30>
        %Split_5, %Split_5_1="ufront.split"(%Concat_4){"sizes": "[32, 32]", "axis": "1"}:(tensor<32x64x30x30>) -> tensor<32x32x30x30>, tensor<32x32x30x30>
        %Concat_6="ufront.concat"(%Split_5, %Split_5_1){"axis": "1"}:(tensor<32x32x30x30>, tensor<32x32x30x30>) -> tensor<32x64x30x30>
        %Conv_7="ufront.conv2d"(%Concat_6){"out_channels": "64", "kernel_size": "[3, 3]", "stride": "[1, 1]", "groups": "1", "padding": "[0, 0]"}:(tensor<32x64x30x30>) -> tensor<32x64x28x28>
        %Relu_8="ufront.relu"(%Conv_7):(tensor<32x64x28x28>) -> tensor<32x64x28x28>
        %MaxPool_9="ufront.pool2d"(%Relu_8){"padding": "[0, 0]", "kernel_size": "[2, 2]", "stride": "[2, 2]"}:(tensor<32x64x28x28>) -> tensor<32x64x14x14>
        %Conv_10="ufront.conv2d"(%MaxPool_9){"stride": "[1, 1]", "groups": "1", "out_channels": "64", "padding": "[0, 0]", "kernel_size": "[3, 3]"}:(tensor<32x64x14x14>) -> tensor<32x64x12x12>
        %Relu_11="ufront.relu"(%Conv_10):(tensor<32x64x12x12>) -> tensor<32x64x12x12>
        %Conv_12="ufront.conv2d"(%Relu_11){"stride": "[1, 1]", "out_channels": "64", "padding": "[0, 0]", "kernel_size": "[3, 3]", "groups": "1"}:(tensor<32x64x12x12>) -> tensor<32x64x10x10>
        %Relu_13="ufront.relu"(%Conv_12):(tensor<32x64x10x10>) -> tensor<32x64x10x10>
        %MaxPool_14="ufront.pool2d"(%Relu_13){"kernel_size": "[2, 2]", "stride": "[2, 2]", "padding": "[0, 0]"}:(tensor<32x64x10x10>) -> tensor<32x64x5x5>
        %Flatten_15="ufront.flat"(%MaxPool_14):(tensor<32x64x5x5>) -> tensor<32x1600>
        %Dense_1="ufront.linear"(%Flatten_15):(tensor<32x1600>) -> tensor<32x512>
        %Relu_17="ufront.relu"(%Dense_1):(tensor<32x512>) -> tensor<32x512>
        %Dense_2="ufront.linear"(%Relu_17):(tensor<32x512>) -> tensor<32x10>
        %softmax="ufront.softmax"(%Dense_2):(tensor<32x10>) -> tensor<32x10>
        return %softmax: tensor<32x10>
}
```