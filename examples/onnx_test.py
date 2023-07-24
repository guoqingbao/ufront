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
import iree.compiler as ireec
from iree import runtime
from urllib.request import urlopen
import json
from PIL import Image

def load_read_image():
    url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
    response = urlopen(url)
    img = Image.open(io.BytesIO(response.read()))
    image = np.array(img.resize((224, 224)), dtype=np.float32)
    image = image[np.newaxis, :]
    clast_image = image/ 255.0
    return clast_image, np.moveaxis(clast_image, -1, 1)

def decode_result(preds, top=5):
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError(
            "`decode_predictions` expects "
            "a batch of predictions "
            "(i.e. a 2D array of shape (samples, 1000)). "
            "Found array with shape: " + str(preds.shape)
        )
    url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"

    response = urlopen(url)
    data_json = json.loads(response.read())
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(data_json[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

if __name__ == "__main__":
    batch_size = 1
    GPU = False
    input_last, input = load_read_image()
    input = np.vstack([input, input])
    # net = resnet18(weights="DEFAULT")

    net = resnet50(weights="DEFAULT")
    # net = densenet121(weights="DEFAULT")
    # net = inception_v3(weights="DEFAULT", dropout=0.0) 
    # net = squeezenet1_1(weights="DEFAULT")
    # net = shufflenet_v2_x1_5(weights="DEFAULT")
    # net = mobilenet_v3_small(weights="DEFAULT", dropout=0.0)
    # net = models.vision_transformer.vit_b_16(weights="DEFAULT") 
    net.eval()

    f = io.BytesIO()
    model_name = net.__class__.__name__ 
    torch.onnx.export(model=net, args=(torch.from_numpy(input)), f=f, export_params=True, #do_constant_folding=True,
                      training=TrainingMode.EVAL if model_name=="Inception3" else TrainingMode.TRAINING, opset_version=17)
    onnx_model = onnx.load_model_from_string(f.getvalue())

    transformer = True if model_name in ["MaxVit", "SwinTransformer", "VisionTransformer", "MultiHeadAttention"] else False
    model = UFrontONNX(onnx_model=onnx_model, batch_size=batch_size, simplify=True, pass_weights=True, transformer=transformer)

    #This will trigger Rust frontend for model conversion and graph building
    #operators can also be managed by python side (each operator here corresponding to an operator in the Rust computation graph)
    output_tensors = model(inputs=[input])

    output = model.softmax(input=output_tensors[0], name="softmax_out")

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

    print("\r\n\r\nIR for ", model_name)
    # for operator in model.operators:
    #   print(operator.ir)
      
    modelir= model.dump_ir()
    with open("onnx_vit.mlir", "w") as f:
        f.write(modelir)
    tosa_ir= model.dump_tosa_ir()

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
    
