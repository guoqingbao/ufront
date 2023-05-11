import time
import torch
from torchvision.models import resnet18, resnet50, squeezenet1_1, regnet_x_32gf, maxvit_t, shufflenet_v2_x1_5, inception_v3, mobilenet_v3_small, efficientnet_v2_s, densenet121, convnext_small
import torchvision.models as models
from ufront.pytorch.model import UFrontTorch 
# pip install iree-compiler iree-runtime iree-tools-tf -f https://openxla.github.io/iree/pip-release-links.html

import torch
from iree.compiler import tools
from iree.runtime.benchmark import benchmark_module
from iree import runtime

batch_size = 1
input = torch.ones((batch_size, 3, 224, 224), dtype=torch.float32)

model_list = {"MobileNetV3":mobilenet_v3_small(pretrained=False), "ShuffleNetV2":shufflenet_v2_x1_5(pretrained=False),
            "ResNet18":resnet18(pretrained=False), "ResNet50":resnet50(pretrained=False), "SqueezeNet":squeezenet1_1(pretrained=False),
            "DenseNet121":densenet121(pretrained=False), "InceptionV3":inception_v3(pretrained=False), "ViT_B16":models.vision_transformer.vit_b_16(weights=False, dropout=0.1)}

for modelname, net in model_list.items():
    net.train(False) 
    t1_start = time.perf_counter()
    model = UFrontTorch(net, batch_size=batch_size) # convert torch model to ufront model
    #This will trigger Rust frontend for actual model conversion and graph building
    #operators can also be managed by python side (each operator here corresponding to an operator in the Rust computation graph)
    output_tensors = model(inputs = [input])

    #The output of the model (forward pass have not been triggered at the moment!)
    if model.model.__class__.__name__ not in ["MaxVit", "SwinTransformer", "VisionTransformer", "MultiHeadAttention"]:
        output = model.softmax(input=output_tensors[0], name="softmax_out")

    #This will trigger model compilation, i.e., convert Rust computation graph to a unified high-level IR and lower it to TOSA IR
    model.compile(optimizer={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"},
                        loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
    tosa_ir= model.dump_tosa_ir()

    print("Compiling TOSA model...")

    binary = tools.compile_str(
        tosa_ir, input_type="tosa", target_backends=["cuda"])
    module = runtime.load_vm_flatbuffer(binary, driver="cuda")
    print("Performance benchmark:\n  ", end="")

    ret = benchmark_module(module.vm_module, entry_functiong="forward", inputs=["1x3x224x224xf32=1"], device="cuda")

    print("{} - {} ".format(modelname, ret))