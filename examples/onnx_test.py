from ufront.onnx.model import ONNXModel, ONNXModelKeras, UFrontONNX
import io
import numpy as np
import torch  #pytorch cpu
import onnx
from torch.onnx import TrainingMode
from torch_def import SimpleCNN, ComplexCNN
from torchvision.models import resnet18, resnet50, squeezenet1_1, regnet_x_32gf, maxvit_t, shufflenet_v2_x1_5, inception_v3, mobilenet_v3_small, efficientnet_v2_s, densenet121, convnext_small
from torchvision import models
import onnxsim
import numpy as np
import iree.compiler as ireec
from iree import runtime
import tensorflow as tf #tensorflow cpu

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
    batch_size = 1
    GPU = True #if NV GPU and driver available
    input = load_read_image()

    # torch_model = ComplexCNN()
    # torch_model = resnet18(pretrained=True)
    # torch_model = resnet50(pretrained=True)
    # torch_model = squeezenet1_1(pretrained=True)
    # torch_model = regnet_x_32gf(pretrained=True)
    # torch_model = mobilenet_v3_small(pretrained=True)
    torch_model = densenet121(pretrained=True)
    # torch_model = models.vision_transformer.vit_b_16(weights=True)
    # torch_model = inception_v3(pretrained=True) #training=TrainingMode.EVAL important!
    # torch_model = shufflenet_v2_x1_5(pretrained=False) 
    # torch_model = efficientnet_v2_s(pretrained=False) 
    # torch_model = convnext_small(pretrained=False) #not supported at the moment 
    # torch_model = models.swin_transformer.swin_t(weights=None) #not supported at the moment 
    torch_model.train(False) 

    f = io.BytesIO()
    model_name = torch_model.__class__.__name__ #"ComplexCNN"
    torch.onnx.export(model=torch_model, args=(torch.from_numpy(input)), f=f, export_params=True, 
                      training=TrainingMode.EVAL if model_name=="Inception3" else TrainingMode.TRAINING, opset_version=17)
    onnx_model = onnx.load_model_from_string(f.getvalue())

    transformer = True if model_name in ["MaxVit", "SwinTransformer", "VisionTransformer", "MultiHeadAttention"] else False
    model = UFrontONNX(onnx_model=onnx_model, batch_size=batch_size, simplify=True, pass_weights=True, transformer=transformer)

    #This will trigger Rust frontend for model conversion and graph building
    #operators can also be managed by python side (each operator here corresponding to an operator in the Rust computation graph)
    output_tensors = model(inputs=[input])

    #The output of the model (forward pass have not been triggered at the moment!)
    # if model_name not in ["MaxVit", "SwinTransformer", "VisionTransformer", "MultiHeadAttention"]:
    #   output = model.softmax(input=output_tensors[0], name="softmax_out")
    #   print(output.shape)
    

    #The Rust frontend will build computation graph and initialize temporary inputs and outputs for each operator
    # total_memory = 0
    # for operator in model.operators:
    #   sz = operator.memory_usage()
    #   total_memory += sz
    #   print("{0} > name: {1}, raw_ptr: {2:#06x}, No. of outputs: {3}, memory used:{4:.5f}MB".format(operator.op_type, operator.params['name'], 
    #   operator.raw_ptr, operator.num_of_outputs(),  sz/1024/1024))
    
    #Total memory cached for inputs/outputs of all operators (in Rust)
    # print("\r\nTotal memory cached for operators {:.2f}MB".format(total_memory/1024/1024))

    #This will trigger model compilation, i.e., convert Rust computation graph to a unified high-level IR and lower it to TOSA IR
    model.compile(optimizer={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"},
                         loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
    print("\r\n\r\n")

    # for operator in operators:
    #   print(operator.ir) #show ir for each operator
    print("Pytorch: ", decode_result(torch_model.forward(torch.from_numpy(input)).detach().numpy()))

    print("\r\n\r\nIR for ", model_name)
    # for operator in model.operators:
    #   print(operator.ir)
      
    # for operator in operators:
    #   print(operator.ir) #show ir for each operator
    modelir= model.dump_ir()
    # print(modelir)

    # import pathlib
    # path = str(pathlib.Path(__file__).parent.resolve()) + "/output_ir/onnx_" + model_name + ".ir"
    # f = open(path, "w")
    # f.write(modelir)
    # f.close()

    tosa_ir= model.dump_tosa_ir()
    # f = open("onnx_resnet50.mlir", "w")
    # f.write(tosa_ir)
    # f.close()
    #This will be supported later
    #model.forward()
    
    #This will be supported later
    #model.backward()
    print("Compiling TOSA model...")
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

    print("UFront: ", decode_result(module.forward(input).to_host()))
    
