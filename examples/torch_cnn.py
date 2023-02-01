import torch.nn as nn
import numpy as np
import torch

from ufront.pytorch.model import PyTorchModel #Flexflow-like PytorchModel wrapper
from ufront import Model, PyOperator, TensorF32, Optimizer, LossType, MetricsType #Rust frontend
from cnn_definition import SimpleCNN, ComplexCNN

if __name__ == "__main__":

    torch_model = PyTorchModel(ComplexCNN()) # Intermedium torch model
    ufront_model = Model() # Ufront Rust model

    
    batch_size = 32
    
    operators = []
    arr = np.zeros((batch_size, 3, 32, 32), dtype=np.float32)
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

    print(ufront_model.dump_ir())

    #This will be supported later
    #ufront_model.forward()
    
    #This will be supported later
    #ufront_model.backward()
    
