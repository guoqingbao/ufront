# import torch, torchtext
from ufront.pytorch.model import UFrontTorch 

from bert import BertModel, BertConfig
import torch

input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
    num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

net = BertModel(config=config)
net.eval()
# all_encoder_layers, pooled_output = net(input_ids, token_type_ids, input_mask)

model = UFrontTorch(net, batch_size=1, pass_weights=True) # convert torch model to ufront model
#This will trigger Rust frontend for actual model conversion and graph building
#operators can also be managed by python side (each operator here corresponding to an operator in the Rust computation graph)
output_tensors = model(inputs = [input_ids, token_type_ids, input_mask])

#The output of the model (forward pass have not been triggered at the moment!)
# if model.model.__class__.__name__ not in ["MaxVit", "SwinTransformer", "VisionTransformer", "MultiHeadAttention"]:
output = model.softmax(input=output_tensors[0], name="softmax_out")

#This will trigger model compilation, i.e., convert Rust computation graph to a unified high-level IR and lower it to TOSA IR
model.compile(optimizer={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"},
                    loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])

modelir = model.dump_ir()

print(modelir)

import pathlib
path = str(pathlib.Path(__file__).parent.resolve()) + "/output_ir/torch_" + model.model.__class__.__name__ + ".ir"
# path = str(pathlib.Path(__file__).parent.resolve()) + "/output_ir/torch_Resnet18.ir"

f = open(path, "w")
f.write(modelir)
f.close()

print("\r\n\r\nIR for ", model.model.__class__.__name__, " generated: ", path)

print("Compiling TOSA model...")
# tosa_ir= model.dump_tosa_ir()