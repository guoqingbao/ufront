from ufront.onnx.model import ONNXModel, ONNXModelKeras, UFrontONNX

import numpy as np
import torch
from torch.onnx import TrainingMode
from torch_def import SimpleCNN, ComplexCNN

if __name__ == "__main__":
    batch_size = 32
    input = np.zeros((batch_size, 3, 32, 32), dtype=np.float32)
    model_name = "ComplexCNN"
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

    print("\r\n\r\nIR for ", model_name)

    # for operator in operators:
    #   print(operator.ir) #show ir for each operator
    modelir= model.dump_ir()
    print(modelir)

    import pathlib
    path = str(pathlib.Path(__file__).parent.resolve()) + "/output_ir/onnx_" + model_name + ".ir"
    f = open(path, "w")
    f.write(modelir)
    f.close()


    #This will be supported later
    #model.forward()
    
    #This will be supported later
    #model.backward()

    
