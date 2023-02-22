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
``` python
import torch.nn as nn
import numpy as np
import torch

# from ufront import Model, PyOperator, TensorF32, Optimizer, LossType, MetricsType #Rust frontend
from ufront.pytorch.model import UFrontTorch #Pytorch wrapper

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
    batch_size = 32
    input = np.zeros((batch_size, 3, 32, 32), dtype=np.float32)
    model = UFrontTorch(ComplexCNN(), batch_size=batch_size) # convert torch model to ufront model

    #save model to file (compatible with flexflow)
    # model.torch_to_file('cnn.ff')

    #load model from file (compatible with flexflow)
    # output_tensors = UFrontTorch.file_to_ff('cnn.ff', [input_tensor, input_tensor])

    #This will trigger Rust frontend for actual model conversion and graph building
    #operators can also be managed by python side (each operator here corresponding to an operator in the Rust computation graph)
    output_tensors = model(inputs = [input, input])

    #The output of the model (forward pass have not been triggered at the moment!)
    output = model.softmax(input=output_tensors[0], name="softmax")
    print(output.shape)

    #The Rust frontend will build computation graph and initialize temporary inputs and outputs for each operator
    total_memory = 0
    for operator in model.operators: # access operators in the ufront computation graph
      sz = operator.memory_usage()
      total_memory += sz
      print("{0} > name: {1}, raw_ptr: {2:#06x}, No. of outputs: {3}, memory used:{4:.5f}MB".format(operator.op_type, operator.params['name'], 
      operator.raw_ptr, operator.num_of_outputs(),  sz/1024/1024))
    
    #Total memory cached for inputs/outputs of all operators (in Rust)
    print("\r\nTotal memory cached for operators {:.2f}MB".format(total_memory/1024/1024))

    #This will trigger model compilation, i.e., convert Rust computation graph to a unified high-level IR and lower it to TOSA IR
    model.compile(optimizer={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"},
                         loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
    
    print("\r\n\r\n")

    # for operator in operators:
    #   print(operator.ir) #show ir for each operator

    print(model.dump_ir()) # TOSA IR

    #This will be supported later
    #model.forward()
    
    #This will be supported later
    #model.backward()


  ## IR Dump Results (for Pytorch)
  ### (Converted from Pytorch model to TOSA-like IR after calling dump_ir())

func.func @forward(%input2: tensor<1x3x32x32xf32>, %input1: tensor<1x3x32x32xf32>) -> tensor<1x10xf32>  { 
   	%1="ufront.conv2d"(%input1){kernel: [3, 3], pad: [0, 0], groups: 1, stride: [1, 1]}:(tensor<1x3x32x32xf32>) -> tensor<1x32x30x30xf32>
   	%2="ufront.relu"(%1):(tensor<1x32x30x30xf32>) -> tensor<1x32x30x30xf32>
   	%3="ufront.conv2d"(%input2){stride: [1, 1], kernel: [3, 3], pad: [0, 0], groups: 1}:(tensor<1x3x32x32xf32>) -> tensor<1x32x30x30xf32>
   	%4="ufront.relu"(%3):(tensor<1x32x30x30xf32>) -> tensor<1x32x30x30xf32>
   	%5="ufront.concat"(%2, %4){axis: 1}:(tensor<1x32x30x30xf32>, tensor<1x32x30x30xf32>) -> tensor<1x64x30x30xf32>
   	%6, %7="ufront.split"(%5){axis: 1, sizes: [32, 32]}:(tensor<1x64x30x30xf32>) -> tensor<1x32x30x30xf32>, tensor<1x32x30x30xf32>
   	%8="ufront.concat"(%6, %7){axis: 1}:(tensor<1x32x30x30xf32>, tensor<1x32x30x30xf32>) -> tensor<1x64x30x30xf32>
   	%9="ufront.conv2d"(%8){pad: [0, 0], groups: 1, kernel: [3, 3], stride: [1, 1]}:(tensor<1x64x30x30xf32>) -> tensor<1x64x28x28xf32>
   	%10="ufront.relu"(%9):(tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32>
   	%11="ufront.pool2d"(%10){pad: [0, 0], kernel: [2, 2], pool_type: PoolType.POOL_MAX, stride: [2, 2]}:(tensor<1x64x28x28xf32>) -> tensor<1x64x14x14xf32>
   	%12="ufront.conv2d"(%11){kernel: [3, 3], groups: 1, stride: [1, 1], pad: [0, 0]}:(tensor<1x64x14x14xf32>) -> tensor<1x64x12x12xf32>
   	%13="ufront.relu"(%12):(tensor<1x64x12x12xf32>) -> tensor<1x64x12x12xf32>
   	%14="ufront.conv2d"(%13){stride: [1, 1], groups: 1, pad: [0, 0], kernel: [3, 3]}:(tensor<1x64x12x12xf32>) -> tensor<1x64x10x10xf32>
   	%15="ufront.relu"(%14):(tensor<1x64x10x10xf32>) -> tensor<1x64x10x10xf32>
   	%16="ufront.pool2d"(%15){pool_type: PoolType.POOL_MAX, kernel: [2, 2], stride: [2, 2], pad: [0, 0]}:(tensor<1x64x10x10xf32>) -> tensor<1x64x5x5xf32>
   	%17="ufront.flat"(%16):(tensor<1x64x5x5xf32>) -> tensor<1x1600xf32>
   	%18="ufront.linear"(%17):(tensor<1x1600xf32>) -> tensor<1x512xf32>
   	%19="ufront.relu"(%18):(tensor<1x512xf32>) -> tensor<1x512xf32>
   	%20="ufront.linear"(%19):(tensor<1x512xf32>) -> tensor<1x10xf32>
   	%21="ufront.softmax"(%20):(tensor<1x10xf32>) -> tensor<1x10xf32>
   	return %21: tensor<1x10xf32>
   }
```

## Sample usage for ONNX Models
``` python
# from ufront import Model, PyOperator, TensorF32, Optimizer, LossType, MetricsType #Rust frontend
from ufront.onnx.model import ONNXModel, ONNXModelKeras, UFrontONNX #ONNX wrapper

import numpy as np
import torch
from torch.onnx import TrainingMode

if __name__ == "__main__":
    batch_size = 32
    input = np.zeros((batch_size, 3, 32, 32), dtype=np.float32)
    torch.onnx.export(model=ComplexCNN(), args=(torch.from_numpy(input), torch.from_numpy(input)), f="cifar10_cnn_pt.onnx", export_params=False, training=TrainingMode.TRAINING)
    
    model = UFrontONNX(onnx_model="cifar10_cnn_pt.onnx", batch_size=batch_size)

    #This will trigger Rust frontend for model conversion and graph building
    #operators can also be managed by python side (each operator here corresponding to an operator in the Rust computation graph)
    output_tensors = model(inputs=[input, input])

    #The output of the model (forward pass have not been triggered at the moment!)
    output = model.softmax(input=output_tensors[0], name="softmax")
    print(output.shape)

    #The Rust frontend will build computation graph and initialize temporary inputs and outputs for each operator
    total_memory = 0
    for operator in model.operators:
      sz = operator.memory_usage()
      total_memory += sz
      print("{0} > name: {1}, raw_ptr: {2:#06x}, No. of outputs: {3}, memory used:{4:.5f}MB".format(operator.op_type, operator.params['name'], 
      operator.raw_ptr, operator.num_of_outputs(),  sz/1024/1024))
    
    #Total memory cached for inputs/outputs of all operators (in Rust)
    print("\r\nTotal memory cached for operators {:.2f}MB".format(total_memory/1024/1024))

    #This will trigger model compilation, i.e., convert Rust computation graph to a unified high-level IR and lower it to TOSA IR
    model.compile(optimizer={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"},
                         loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
    print("\r\n\r\n")

    # for operator in operators:
    #   print(operator.ir) #show ir for each operator

    print(model.dump_ir())

    #This will be supported later
    #model.forward()
    
    #This will be supported later
    #model.backward()

  ## IR Dump Results (for ONNX)
  ### (Converted from ONNX model to High-level IR after calling dump_ir(), i.e., TOSA-like dialect IR)
  func.func @forward(%input.5: tensor<32x3x32x32xf32>, %input.1: tensor<32x3x32x32xf32>) -> tensor<32x10xf32>  { 
   	%1="ufront.conv2d"(%input.1){groups: 1, kernel: [3, 3], pad: [0, 0], stride: [1, 1]}:(tensor<32x3x32x32xf32>) -> tensor<32x32x30x30xf32>
   	%2="ufront.relu"(%1):(tensor<32x32x30x30xf32>) -> tensor<32x32x30x30xf32>
   	%3="ufront.conv2d"(%input.5){pad: [0, 0], stride: [1, 1], groups: 1, kernel: [3, 3]}:(tensor<32x3x32x32xf32>) -> tensor<32x32x30x30xf32>
   	%4="ufront.relu"(%3):(tensor<32x32x30x30xf32>) -> tensor<32x32x30x30xf32>
   	%5="ufront.concat"(%2, %4){axis: 1}:(tensor<32x32x30x30xf32>, tensor<32x32x30x30xf32>) -> tensor<32x64x30x30xf32>
   	%6, %7="ufront.split"(%5){sizes: [32, 32], axis: 1}:(tensor<32x64x30x30xf32>) -> tensor<32x32x30x30xf32>, tensor<32x32x30x30xf32>
   	%8="ufront.concat"(%6, %7){axis: 1}:(tensor<32x32x30x30xf32>, tensor<32x32x30x30xf32>) -> tensor<32x64x30x30xf32>
   	%9="ufront.conv2d"(%8){groups: 1, kernel: [3, 3], pad: [0, 0], stride: [1, 1]}:(tensor<32x64x30x30xf32>) -> tensor<32x64x28x28xf32>
   	%10="ufront.relu"(%9):(tensor<32x64x28x28xf32>) -> tensor<32x64x28x28xf32>
   	%11="ufront.pool2d"(%10){kernel: [2, 2], pad: [0, 0], stride: [2, 2], pool_type: PoolType.POOL_MAX}:(tensor<32x64x28x28xf32>) -> tensor<32x64x14x14xf32>
   	%12="ufront.conv2d"(%11){stride: [1, 1], kernel: [3, 3], pad: [0, 0], groups: 1}:(tensor<32x64x14x14xf32>) -> tensor<32x64x12x12xf32>
   	%13="ufront.relu"(%12):(tensor<32x64x12x12xf32>) -> tensor<32x64x12x12xf32>
   	%14="ufront.conv2d"(%13){kernel: [3, 3], groups: 1, stride: [1, 1], pad: [0, 0]}:(tensor<32x64x12x12xf32>) -> tensor<32x64x10x10xf32>
   	%15="ufront.relu"(%14):(tensor<32x64x10x10xf32>) -> tensor<32x64x10x10xf32>
   	%16="ufront.pool2d"(%15){pad: [0, 0], stride: [2, 2], kernel: [2, 2], pool_type: PoolType.POOL_MAX}:(tensor<32x64x10x10xf32>) -> tensor<32x64x5x5xf32>
   	%17="ufront.flat"(%16):(tensor<32x64x5x5xf32>) -> tensor<32x1600xf32>
   	%18="ufront.linear"(%17):(tensor<32x1600xf32>) -> tensor<32x512xf32>
   	%19="ufront.relu"(%18):(tensor<32x512xf32>) -> tensor<32x512xf32>
   	%20="ufront.linear"(%19):(tensor<32x512xf32>) -> tensor<32x10xf32>
   	%21="ufront.softmax"(%20):(tensor<32x10xf32>) -> tensor<32x10xf32>
   	return %21: tensor<32x10xf32>
   }
```
## Sample usage for Keras Models
``` python
from tensorflow.keras import backend
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Input, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.datasets import cifar10

from ufront.keras.model import UFrontKeras

def NestedCNN(shape=(3, 32, 32), dtype="float32", num_classes = 10):
    input_tensor1 = Input(shape=shape, dtype=dtype)
    output_tensor1 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu")(input_tensor1)
    output_tensor1 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu")(output_tensor1)
    output_tensor1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor1)
    model1 = Model(input_tensor1, output_tensor1)
    
    input_tensor2 = Input(shape=(32, 14, 14), dtype="float32")
    output_tensor2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu")(input_tensor2)
    output_tensor2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu")(output_tensor2)
    output_tensor2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor2)
    output_tensor2 = Flatten()(output_tensor2)
    output_tensor2 = Dense(512, activation="relu")(output_tensor2)
    output_tensor2 = Dense(num_classes)(output_tensor2)
    output_tensor2 = Activation("softmax")(output_tensor2)
    model2 = Model(input_tensor2, output_tensor2)
    
    input_tensor3 = Input(shape=(3, 32, 32), dtype="float32")
    output_tensor3 = model1(input_tensor3)
    output_tensor3 = model2(output_tensor3)
    
    return {3: input_tensor3}, output_tensor3, "NestedCNN"

if __name__ == "__main__":
    backend.set_image_data_format('channels_first')
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    y_train = y_train.astype('int32')
    print("shape: ", x_train.shape)
    
    # inputs, outputs, model_name = SequentialCNN(shape=(3, 32, 32), dtype="float32", num_classes=10)
    # inputs, outputs, model_name = ConcatenatedCNN(shape=(3, 32, 32), dtype="float32", num_classes=10)
    inputs, outputs, model_name = NestedCNN(shape=(3, 32, 32), dtype="float32", num_classes=10)

    model = UFrontKeras(inputs = inputs, outputs = outputs, batch_size = 32)

    model.compile(optimizer={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"},
                         loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])

    # opt = optimizers.SGD(learning_rate=0.01)
    # model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
    
    print(model.summary()) 

    # for operator in operators:
    #   print(operator.ir) #show ir for each operator
    print(model.dump_ir())

    #This will be supported later
    #model.forward()
    
    #This will be supported later
    #model.backward()

  ## IR Dump Results (for Keras)
  ### (Converted from Keras model to High-level IR after calling dump_ir(), i.e., TOSA-like dialect IR)
  func.func @forward(%input1: tensor<32x3x32x32xf32>) -> tensor<32x10xf32>  { 
   	%1="ufront.conv2d"(%input1){kernel: [3, 3], stride: [1, 1], pad: [0, 0], groups: 1}:(tensor<32x3x32x32xf32>) -> tensor<32x32x30x30xf32>
   	%2="ufront.relu"(%1):(tensor<32x32x30x30xf32>) -> tensor<32x32x30x30xf32>
   	%3="ufront.conv2d"(%2){stride: [1, 1], groups: 1, pad: [0, 0], kernel: [3, 3]}:(tensor<32x32x30x30xf32>) -> tensor<32x32x28x28xf32>
   	%4="ufront.relu"(%3):(tensor<32x32x28x28xf32>) -> tensor<32x32x28x28xf32>
   	%5="ufront.pool2d"(%4){pool_type: PoolType.POOL_MAX, stride: [2, 2], pad: [0, 0], kernel: [2, 2]}:(tensor<32x32x28x28xf32>) -> tensor<32x32x14x14xf32>
   	%6="ufront.conv2d"(%5){pad: [0, 0], groups: 1, kernel: [3, 3], stride: [1, 1]}:(tensor<32x32x14x14xf32>) -> tensor<32x64x12x12xf32>
   	%7="ufront.relu"(%6):(tensor<32x64x12x12xf32>) -> tensor<32x64x12x12xf32>
   	%8="ufront.conv2d"(%7){groups: 1, kernel: [3, 3], pad: [0, 0], stride: [1, 1]}:(tensor<32x64x12x12xf32>) -> tensor<32x64x10x10xf32>
   	%9="ufront.relu"(%8):(tensor<32x64x10x10xf32>) -> tensor<32x64x10x10xf32>
   	%10="ufront.pool2d"(%9){pad: [0, 0], pool_type: PoolType.POOL_MAX, kernel: [2, 2], stride: [2, 2]}:(tensor<32x64x10x10xf32>) -> tensor<32x64x5x5xf32>
   	%11="ufront.flat"(%10):(tensor<32x64x5x5xf32>) -> tensor<32x1600xf32>
   	%12="ufront.linear"(%11):(tensor<32x1600xf32>) -> tensor<32x512xf32>
   	%13="ufront.relu"(%12):(tensor<32x512xf32>) -> tensor<32x512xf32>
   	%14="ufront.linear"(%13):(tensor<32x512xf32>) -> tensor<32x10xf32>
   	%15="ufront.softmax"(%14):(tensor<32x10xf32>) -> tensor<32x10xf32>
   	return %15: tensor<32x10xf32>
   }   
```

## Sample native usage
``` python
import ufront
import numpy as np;
from ufront import OpType, PoolType, LossType, MetricsType, Optimizer

if __name__ == "__main__":
   model = ufront.Model()
   batch_size = 1

   input = np.ones((batch_size,3,32,32), dtype=np.float32)

   tensor_input1 = ufront.TensorF32(input, name="input1")
   tensor_input2 = ufront.TensorF32(input, name="input2")

   x = model.conv2d(input=tensor_input1, out_channels=32, kernel=[3, 3], stride=[1, 1], pad=[0, 0], groups=1)
   x1 = model.relu(input=x.get_output(0))

   x = model.conv2d(input=tensor_input2, out_channels=32, kernel=[3, 3], stride=[1, 1], pad=[0, 0], groups=1)
   x2 = model.relu(input=x.get_output(0))

   x = model.concat(tensors=[x1.get_output(0), x2.get_output(0)], axis=1)
   x = model.split(input=x.get_output(0), sizes = [32, 32], axis=1)
   x = model.concat(tensors=[x.get_output(0), x.get_output(1)], axis=1)

   x = model.conv2d(input=x.get_output(0), out_channels=64, kernel=[3, 3], stride=[1, 1], pad=[0, 0], groups=1)
   x = model.relu(input=x.get_output(0))
   x = model.pool2d(input=x.get_output(0), kernel=[2, 2], stride=[2, 2], pad=[0, 0], pool_type=PoolType.POOL_MAX)

   x = model.conv2d(input=x.get_output(0), out_channels=64, kernel=[3, 3], stride=[1, 1], pad=[0, 0], groups=1)
   x = model.relu(input=x.get_output(0))

   x = model.conv2d(input=x.get_output(0), out_channels=64, kernel=[3, 3], stride=[1, 1], pad=[0, 0], groups=1)
   x = model.relu(input=x.get_output(0))
   x = model.pool2d(input=x.get_output(0), kernel=[2, 2], stride=[2, 2], pad=[0, 0], pool_type=PoolType.POOL_MAX)

   x = model.flat(input=x.get_output(0))
   x = model.dense(input=x.get_output(0), out_dim=512)

   x = model.relu(input=x.get_output(0))
   x = model.dense(input=x.get_output(0), out_dim=10)
   x = model.softmax(input=x.get_output(0))

   model.optimizer = Optimizer(params={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"})

   model.compile(loss=LossType.CATEGORICAL_CROSSENTROPY, metrics=[MetricsType.ACCURACY, MetricsType.SPARSE_CATEGORICAL_CROSSENTROPY])
   
   print(model.dump_ir())
   
   #This will be supported later
   #model.forward()
    
   #This will be supported later
   #model.backward()

  ## IR Dump Results (for Native usage)
  ### (Converted from natively defined model to High-level IR after calling dump_ir(), i.e., TOSA-like dialect IR)
  func.func @forward(%input1: tensor<1x3x32x32xf32>, %input2: tensor<1x3x32x32xf32>) -> tensor<1x10xf32>  { 
   	%1="ufront.conv2d"(%input1){kernel: [3, 3], stride: [1, 1], pad: [0, 0], groups: 1}:(tensor<1x3x32x32xf32>) -> tensor<1x32x30x30xf32>
   	%2="ufront.relu"(%1):(tensor<1x32x30x30xf32>) -> tensor<1x32x30x30xf32>
   	%3="ufront.conv2d"(%input2){kernel: [3, 3], stride: [1, 1], groups: 1, pad: [0, 0]}:(tensor<1x3x32x32xf32>) -> tensor<1x32x30x30xf32>
   	%4="ufront.relu"(%3):(tensor<1x32x30x30xf32>) -> tensor<1x32x30x30xf32>
   	%5="ufront.concat"(%2, %4){axis: 1}:(tensor<1x32x30x30xf32>, tensor<1x32x30x30xf32>) -> tensor<1x64x30x30xf32>
   	%6, %7="ufront.split"(%5){axis: 1, sizes: [32, 32]}:(tensor<1x64x30x30xf32>) -> tensor<1x32x30x30xf32>, tensor<1x32x30x30xf32>
   	%8="ufront.concat"(%6, %7){axis: 1}:(tensor<1x32x30x30xf32>, tensor<1x32x30x30xf32>) -> tensor<1x64x30x30xf32>
   	%9="ufront.conv2d"(%8){groups: 1, pad: [0, 0], kernel: [3, 3], stride: [1, 1]}:(tensor<1x64x30x30xf32>) -> tensor<1x64x28x28xf32>
   	%10="ufront.relu"(%9):(tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32>
   	%11="ufront.pool2d"(%10){kernel: [2, 2], pad: [0, 0], stride: [2, 2], pool_type: PoolType.POOL_MAX}:(tensor<1x64x28x28xf32>) -> tensor<1x64x14x14xf32>
   	%12="ufront.conv2d"(%11){kernel: [3, 3], stride: [1, 1], pad: [0, 0], groups: 1}:(tensor<1x64x14x14xf32>) -> tensor<1x64x12x12xf32>
   	%13="ufront.relu"(%12):(tensor<1x64x12x12xf32>) -> tensor<1x64x12x12xf32>
   	%14="ufront.conv2d"(%13){stride: [1, 1], kernel: [3, 3], pad: [0, 0], groups: 1}:(tensor<1x64x12x12xf32>) -> tensor<1x64x10x10xf32>
   	%15="ufront.relu"(%14):(tensor<1x64x10x10xf32>) -> tensor<1x64x10x10xf32>
   	%16="ufront.pool2d"(%15){pool_type: PoolType.POOL_MAX, stride: [2, 2], kernel: [2, 2], pad: [0, 0]}:(tensor<1x64x10x10xf32>) -> tensor<1x64x5x5xf32>
   	%17="ufront.flat"(%16):(tensor<1x64x5x5xf32>) -> tensor<1x1600xf32>
   	%18="ufront.linear"(%17):(tensor<1x1600xf32>) -> tensor<1x512xf32>
   	%19="ufront.relu"(%18):(tensor<1x512xf32>) -> tensor<1x512xf32>
   	%20="ufront.linear"(%19):(tensor<1x512xf32>) -> tensor<1x10xf32>
   	%21="ufront.softmax"(%20):(tensor<1x10xf32>) -> tensor<1x10xf32>
   	return %21: tensor<1x10xf32>
   }
}
```