import torch
import numpy as np
from torchvision.models import resnet18, resnet50, squeezenet1_1, regnet_x_32gf, maxvit_t, shufflenet_v2_x1_5, inception_v3, mobilenet_v3_small, efficientnet_v2_s, densenet121, convnext_small
import torchvision.models as models
from ufront.pytorch.model import UFrontTorch 
import iree.compiler as ireec
from iree import runtime
from torch_def import mae, mse, r_square, rmse, mpe, load_sample_image, decode_result

if __name__ == "__main__":
    batch_size = 1
    GPU = True
    input_last, input = load_sample_image()

    # net = resnet18(weights="DEFAULT")
    # net = resnet50(weights="DEFAULT")
    # net = densenet121(weights="DEFAULT")
    # net = inception_v3(weights="DEFAULT", dropout=0.0) 
    # net = squeezenet1_1(weights="DEFAULT")
    # net = shufflenet_v2_x1_5(weights="DEFAULT")
    # net = mobilenet_v3_small(weights="DEFAULT", dropout=0.0)
    net = models.vision_transformer.vit_b_16(weights="DEFAULT") 
    net.eval()

    modelname = net.__class__.__name__
    
    torch_ret = torch.softmax(net.forward(torch.Tensor(input)), dim=1).detach().numpy()
    print("\nPytorch Inference Results: ", decode_result(torch_ret))

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
    # print(modelir) # show high-level IR

    tosa_ir= model.dump_tosa_ir() # lower-level IR

    # for operator in operators:
    #   print(operator.ir) #show ir for each operator

    # path = str(pathlib.Path(__file__).parent.resolve()) + "/output_ir/torch_" + model.model.__class__.__name__ + ".ir"

    # f = open(path, "w")
    # f.write(modelir)
    # f.close()

    # print("\r\n\r\nIR for ", model.model.__class__.__name__, " generated: ", path)

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
    print("\r\nUFront Inference Results: ", decode_result(ufront_ret))
    
    dif = torch_ret - ufront_ret
    mae = np.mean(abs(dif))
    print("UFront vs Pytorch for Model - ", modelname)
    print("MAE: ", mae)
    print("RMSE:", rmse(torch.Tensor(torch_ret), torch.Tensor(ufront_ret)).numpy())
    print("COD:", r_square(torch.Tensor(torch_ret), torch.Tensor(ufront_ret)).numpy())
    print("MPE:", mpe(torch.Tensor(torch_ret), torch.Tensor(ufront_ret)).numpy(), "%")
    