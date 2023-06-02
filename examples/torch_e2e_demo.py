import torch
import numpy as np
from torchvision.models import resnet18, resnet50, squeezenet1_1, regnet_x_32gf, maxvit_t, shufflenet_v2_x1_5, inception_v3, mobilenet_v3_small, efficientnet_v2_s, densenet121, convnext_small
import torchvision.models as models
from ufront.pytorch.model import UFrontTorch 
# pip install iree-compiler iree-runtime iree-tools-tf -f https://openxla.github.io/iree/pip-release-links.html
import iree.compiler as ireec
from iree.compiler import tools
from iree import runtime
import tensorflow as tf
from multihead_attention import MultiHeadAttention, MultiHeadAttentionNet, EncoderNet

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
    GPU = True
    input = load_read_image()
    # net = resnet18(pretrained=True)# ok
    # net = resnet50(pretrained=True)# ok
    # net = densenet121(pretrained=True)# ok
    # net = inception_v3(pretrained=True) # ok
    # net = squeezenet1_1(pretrained=True) #note
    # net = shufflenet_v2_x1_5(pretrained=True)# ok
    # net = mobilenet_v3_small(pretrained=True)#note
    net = models.vision_transformer.vit_b_16(weights=True) #ok
    # input = torch.empty(1, 197, 768).normal_(std=0.02)
    # mask = MultiHeadAttention.gen_history_mask(input)
    # net = MultiHeadAttention(128, 16)
    # net = MultiHeadAttentionNet()
    # net = EncoderNet()

    net.train(False) 
    
    # ret = net.forward(torch.Tensor(input)).detach().numpy()
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

    modelir = model.dump_ir()

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