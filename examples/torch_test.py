# import torch.nn as nn
import numpy as np
import torch
from torchvision.models import resnet18, resnet50
import torchvision.models as models


from ufront.pytorch.model import UFrontTorch #Flexflow-like PytorchModel wrapper
from torch_def import SimpleCNN, ComplexCNN

if __name__ == "__main__":
    batch_size = 1
    # input = np.zeros((batch_size, 3, 32, 32), dtype=np.float32)
    input = np.zeros((batch_size, 3, 224, 224), dtype=np.float32)

    net = resnet50(pretrained=False)
    # net = models.vision_transformer.vit_b_16(pretrained=False)
    # net.train(False)

    # resnet.train(mode=False)
    model = UFrontTorch(net, batch_size=batch_size) # convert torch model to ufront model

    # model = UFrontTorch(ComplexCNN(), batch_size=batch_size) # convert torch model to ufront model

    #save model to file (compatible with flexflow)
    # model.torch_to_file('cnn.ff')

    #load model from file (compatible with flexflow)
    # output_tensors = UFrontTorch.file_to_ff('cnn.ff', [input_tensor, input_tensor])

    #This will trigger Rust frontend for actual model conversion and graph building
    #operators can also be managed by python side (each operator here corresponding to an operator in the Rust computation graph)
    output_tensors = model(inputs = [input])

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

    print(model.dump_ir())

    #This will be supported later
    #model.forward()
    
    #This will be supported later
    #model.backward()
    