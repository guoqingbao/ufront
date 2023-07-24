import torch
import numpy as np
from torchvision.models import resnet18, resnet50, squeezenet1_1, regnet_x_32gf, maxvit_t, shufflenet_v2_x1_5, inception_v3, mobilenet_v3_small, efficientnet_v2_s, densenet121, convnext_small
import torchvision.models as models
from ufront.pytorch.model import UFrontTorch 
# pip install iree-compiler iree-runtime iree-tools-tf -f https://openxla.github.io/iree/pip-release-links.html
import iree.compiler as ireec
from iree.compiler import tools
from iree import runtime
from torch.onnx import TrainingMode
import io
import onnxruntime
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

    # net = resnet50(weights="DEFAULT")
    # net = densenet121(weights="DEFAULT")
    # net = inception_v3(weights="DEFAULT", dropout=0.0) 
    # net = squeezenet1_1(weights="DEFAULT")
    # net = shufflenet_v2_x1_5(weights="DEFAULT")
    net = mobilenet_v3_small(weights="DEFAULT", dropout=0.0)
    # net = models.vision_transformer.vit_b_16(weights="DEFAULT") 
    net.eval()

    f = io.BytesIO()
    torch.onnx.export(model=net, args=(torch.from_numpy(input)), f=f, export_params=True, do_constant_folding=False,
                      training=TrainingMode.EVAL, opset_version=17)

    ort_session = onnxruntime.InferenceSession(f.getvalue())  
    
    ort_output = ort_session.run(None, {ort_session._inputs_meta[0].name: input} )[0]
    ort_output = torch.softmax(torch.Tensor(ort_output), dim=1).detach().numpy()
    print("ONNX Runtime: ", decode_result(ort_output))

    modelname = net.__class__.__name__
    
    torch_ret = torch.softmax(net.forward(torch.Tensor(input)), dim=1).detach().numpy()
    print("Pytorch: ", decode_result(torch_ret))

    model = UFrontTorch(net, batch_size=batch_size, pass_weights=True) # convert torch model to ufront model
    #This will trigger Rust frontend for actual model conversion and graph building
    #operators can also be managed by python side (each operator here corresponding to an operator in the Rust computation graph)
    output_tensors = model(inputs = [input])

    #The output of the model (forward pass have not been triggered at the moment!)
    # if model.model.__class__.__name__ not in ["MaxVit", "SwinTransformer", "VisionTransformer", "MultiHeadAttention"]:
    output = model.softmax(input=output_tensors[0], name="softmax_out")

    #This will trigger model compilation, i.e., convert Rust computation graph to a unified high-level IR and lower it to TOSA IR
    model.compile(optimizer={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"},
                        loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])

    modelir = model.dump_ir()
    print(modelir)

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

    ufront_ret = module.forward(input).to_host()
    print("UFront: ", decode_result(ufront_ret))
    
    dif = torch_ret - ufront_ret
    mae = np.mean(abs(dif))
    print("Model: ", modelname, ", MAE with Pytorch: ", mae)

    dif = ort_output - ufront_ret
    mae = np.mean(abs(dif))
    print("Model: ", modelname, ", MAE with ONNXRuntime: ", mae)
    