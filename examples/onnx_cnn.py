from ufront.onnx.model import ONNXModel, ONNXModelKeras
from ufront import Model, PyOperator, TensorF32, Optimizer, LossType, MetricsType #Rust frontend

import argparse
import numpy as np
import onnx
import torch
import torch.nn as nn
from torch.onnx import TrainingMode
from cnn_definition import SimpleCNN, ComplexCNN

if __name__ == "__main__":
    batch_size = 32
    
    operators = []
    arr = np.zeros((batch_size, 3, 32, 32), dtype=np.float32)
    input_tensor = TensorF32(arr) # convert to Rust f32 tensor

    torch.onnx.export(model=ComplexCNN(), args=(torch.from_numpy(arr), torch.from_numpy(arr)), f="cifar10_cnn_pt.onnx", export_params=False, training=TrainingMode.TRAINING)
    
    onnx_model = onnx.load("cifar10_cnn_pt.onnx")

    dims_input = [batch_size, 3, 32, 32]
    ufront_model = Model() # Ufront Rust model
    onnx_model = ONNXModel("cifar10_cnn_pt.onnx")

    #torch model to ufront model, this will trigger Rust frontend for building computation graph
    #operators can also be managed by python side (each operator here corresponding to an operator in the Rust computation graph)
    output_tensors, operators = onnx_model.apply(ufront_model, input_tensors=[input_tensor, input_tensor])

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
    
    #This will be supported later
    #ufront_model.forward()
    
    #This will be supported later
    #ufront_model.backward()


    
