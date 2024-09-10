# ufront
Unified MLIR Computing Frontend for Deep Learning 

## How it works?
Convert Pytorch, Tensorflow, Keras, ONNX models to UFront IR and then lower them into standard MLIR dialect (TOSA IR)

## Project discription
1. The objective of this project is to create a `unified MLIR frontend` for deep learning computing.

2. The frontend imports Pytorch, Keras, ONNX (and possiblely Tensorflow) models using the FlexFlow-like python scripts and then translate them into the `Unified High-level IR` based on Rust.

3. The frontend was built based on `Rust` and the Rust computing interfaces were exposed to Python through `PyO3`. 

4. The computing nodes (operators) in the Pytorch, Keras, Tensorflow, and ONNX models can be mapped to Rust computing interfaces, in which the Rust frontend will maintain a high-level `computation graph`.

5. The Rust frontend will then translate the computation graph into `high-level IR` (intermediate representation, sample generated IRs can be found in folder `examples/output_ir`) and then lower it into `TOSA dialect` (a standard IR in MLIR, using the subproject `UFront2TOSA`).

6. In addition to translating Pytorch, Keras, Tensorflow, and ONNX models into the standard MLIR IR (TOSA), the Rust frontend also provide standard computing workflows including operators, forward, and backward (gradient update for training, future work).

## Citation
```
Guoqing Bao, Heng Shi, Chengyi Cui, Yalin Zhang, and Jianguo Yao. 2024. UFront: Toward A Unified MLIR Frontend for Deep Learning. In 39th IEEE/ACM International Conference on Automated Software Engineering (ASE ’24), October 27-November 1, 2024, Sacramento, CA, USA. ACM, New York, NY, USA, 13 pages. https://doi.org/10.1145/3691620.3695002
```

## Experiencing UFront without build

Experiencing UFront on Kaggle (for model compilation, performance comparison, ImageNet inference, accuracy validation, etc.)

Run the anonymous UFront online tests in Kaggle using the links below, **be sure to login** to use full functionality and free GPU (T4x2) resources.

https://www.kaggle.com/code/anomyuser/ufront-test/

https://www.kaggle.com/code/anomyuser/test-torch

https://www.kaggle.com/code/anomyuser/test-tf

**Note**: the results on Kaggle may slightly different from the paper reported because of different CPU and GPU used.

**Important:** Access GPU at no cost or turn on an internet connection. Need to **login** and **Get phone verified** in Kaggle.

**The Internet Connection** in the Kaggle notebook need to be **turned on** to allow package download.

## How to build?

### Option 1: Docker image (**recommended**)
```sh
git clone git@github.com:guoqingbao/ufront.git
cd ufront
git submodule update --init --recursive
docker build -t ufront:latest .
docker run --name <your container name> ufront:latest /bin/bash
```

### Option 2: Manual build
#### Install tools for building main project
```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh #Rust

pip install maturin==0.15.1 #Rust cross-language build tool

pip install maturin[patchelf] #for packaging dependencies
```

#### Install dependencies for subproject (UFront2TOSA)
```sh
apt update && apt install -y wget cmake ninja-build gnupg #C++ build tools
apt install zlib1g zlib1g-dev #zlib

#LLVM-16 for Ubuntu 20.04, you may change this for Ubuntu 22.04 or 18.04 (see https://apt.llvm.org/)
echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" >> /etc/apt/sources.list && \
    echo "deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" >> /etc/apt/sources.list && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add

apt install libomp-16-dev #openmp

apt update && apt install -y clang-16 lldb-16 lld-16 libmlir-16-dev mlir-16-tools #LLVM/MLIR version 16
```

#### Build the subproject first
```sh
cd cpp/UFront2TOSA && mkdir build && cd build

cmake .. -G Ninja -DMLIR_DIR=/usr/lib/llvm-16/lib/cmake/mlir && \
    ninja && \
```

#### Build the main project for development
```sh
maturin develop #Debug mode
```

#### Run the examples
```sh
cd examples
python native_test.py
python torch_test.py #make sure torch-cpu installed, other options include onnx_test, keras_test (tf-cpu required), bert_test, etc.
```

#### Build the release package (wheel file)
```sh
maturin build --release -i python3.7 #for python3.7
maturin build --release -i python3.8 #for python3.8
maturin build --release -i python3.9 #for python3.9
maturin build --release -i python3.10 #for python3.10
maturin build --release -i python3.11 #for python3.11
```
#### Install compiler backend & runtime for execution
```sh
pip install iree-compiler==20230815.614 iree-runtime==20230815.614
```

Trouble shootings can be found in: https://github.com/guoqingbao/anonyufront


## Supplymentary Material
Coming soon

## End-to-end demo

``` python
import torch
import numpy as np
from torchvision.models import resnet18, resnet50, squeezenet1_1, regnet_x_32gf, maxvit_t, shufflenet_v2_x1_5, inception_v3, mobilenet_v3_small, efficientnet_v2_s, densenet121, convnext_small
import torchvision.models as models
from ufront.pytorch.model import UFrontTorch 
import iree.compiler as ireec
from iree.compiler import tools
from iree import runtime

batch_size = 1
import tensorflow as tf

def decode_result(result):
  return tf.keras.applications.resnet50.decode_predictions(result, top=5)[0]
    
def load_read_image():
    content_path = tf.keras.utils.get_file(
    'YellowLabradorLooking_new.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

    image = tf.io.read_file(content_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = image[tf.newaxis, :]
    return np.moveaxis(image.numpy(), -1, 1)/ 255.0

if __name__ == "__main__":
    # net = resnet18(pretrained=True)
    net = resnet50(pretrained=True)
    # net = densenet121(pretrained=True)
    # net = inception_v3(pretrained=True) 
    # net = squeezenet1_1(pretrained=True)
    # net = shufflenet_v2_x1_5(pretrained=True)
    # net = mobilenet_v3_small(pretrained=True, dropout=0.0)
    # net = models.vision_transformer.vit_b_16(weights=True) 
    # net = BertModel(config=config) #refer to bert_torch_test.py
    # net = YOUR Models

    net = resnet50(pretrained=True)
    net.train(False) 
    input = load_read_image()
    print("Pytorch: ", decode_result(net.forward(torch.Tensor(input)).detach().numpy()))

    model = UFrontTorch(net, batch_size=batch_size, pass_weights=True) # convert torch model to ufront model
    #This will trigger Rust frontend for actual model conversion and graph building
    #operators can also be managed by python side (each operator here corresponding to an operator in the Rust computation graph)
    output_tensors = model(inputs = [input])

    #The output of the model (forward pass have not been triggered at the moment!)
    # if model.model.__class__.__name__ not in ["MaxVit", "SwinTransformer", "VisionTransformer", "MultiHeadAttention"]:
    #     output = model.softmax(input=output_tensors[0], name="softmax_out")

    #This will trigger model compilation, i.e., convert Rust computation graph to a unified high-level IR and lower it to TOSA IR
    model.compile(optimizer={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"},
                        loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
    tosa_ir= model.dump_tosa_ir()

    print("Compiling TOSA model...")
    if torch.cuda.is_available():
        binary = ireec.compile_str(tosa_ir,
                        target_backends=["cuda"], 
                        input_type=ireec.InputType.TOSA)
        module = runtime.load_vm_flatbuffer(binary, driver="cuda")
    else:
        binary = ireec.compile_str(tosa_ir,
                        target_backends=["llvm-cpu"], 
                        input_type=ireec.InputType.TOSA)
        module = runtime.load_vm_flatbuffer(binary,backend="llvm-cpu") 

    print("UFront: ", decode_result(module.forward(input).to_host()))
```

output:

Pytorch:  [('n02099712', 'Labrador_retriever', 8.617878), ('n02099849', 'Chesapeake_Bay_retriever', 8.343), ('n02092339', 'Weimaraner', 7.604711), ('n15075141', 'toilet_tissue', 7.396191), ('n02808304', 'bath_towel', 6.9576306)]

Compiling TOSA model...

UFront:  [('n02099712', 'Labrador_retriever', 8.617871), ('n02099849', 'Chesapeake_Bay_retriever', 8.342996), ('n02092339', 'Weimaraner', 7.6047263), ('n15075141', 'toilet_tissue', 7.396185), ('n02808304', 'bath_towel', 6.9576297)]

## Sample usage for Pytorch Models
``` python
import torch.nn as nn
import numpy as np
import torch

# from ufront import Model, PyOperator, Tensor, Optimizer, LossType, MetricsType #Rust frontend
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

    print(model.dump_ir()) # UFront IR
    #print(model.dump_tosa_ir()) # TOSA IR

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
# from ufront import Model, PyOperator, Tensor, Optimizer, LossType, MetricsType #Rust frontend
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

    print(model.dump_ir()) # UFront IR

    #model.dump_tosa_ir() #TOSA IR

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
    
    inputs, outputs, model_name = NestedCNN(shape=(3, 32, 32), dtype="float32", num_classes=10)

    model = UFrontKeras(inputs = inputs, outputs = outputs, batch_size = 32)

    model.compile(optimizer={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"},
                         loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])

    print(model.summary()) 

    # for operator in operators:
    #   print(operator.ir) #show ir for each operator
    print(model.dump_ir()) #UFront IR
    #print(model.dump_tosa_ir()) #TOSA IR

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

   tensor_input1 = ufront.Tensor(input, name="input1")
   tensor_input2 = ufront.Tensor(input, name="input2")

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
   
   print(model.dump_ir()) #UFront IR
   #print(model.dump_tosa_ir()) #TOSA IR


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
