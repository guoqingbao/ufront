# import torch, torchtext
from ufront.pytorch.model import UFrontTorch 
import iree.compiler as ireec
from iree import runtime
from torch_bert import BertModel, BertConfig
import torch
import time
GPU = False
import numpy as np
input_ids = torch.from_numpy(np.array([[31, 51, 99], [15, 5, 0]], dtype="int32"))
input_mask = torch.from_numpy(np.array([[1, 1, 1], [1, 1, 0]], dtype="int32"))
token_type_ids = torch.from_numpy(np.array([[0, 0, 1], [0, 1, 0]], dtype="int32"))

config = BertConfig(vocab_size_or_config_json_file=16000, hidden_size=768,
    num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

net = BertModel(config=config)
net.eval()
# all_encoder_layers, pooled_output = net(input_ids, token_type_ids, input_mask)
t1_start = time.perf_counter()
model = UFrontTorch(net, batch_size=1, pass_weights=True) # convert torch model to ufront model
#This will trigger Rust frontend for actual model conversion and graph building
#operators can also be managed by python side (each operator here corresponding to an operator in the Rust computation graph)
output_tensors = model(inputs = [input_ids, token_type_ids, input_mask])

#The output of the model (forward pass have not been triggered at the moment!)
# if model.model.__class__.__name__ not in ["MaxVit", "SwinTransformer", "VisionTransformer", "MultiHeadAttention"]:
# output = model.softmax(input=output_tensors[0], name="softmax_out")

#This will trigger model compilation, i.e., convert Rust computation graph to a unified high-level IR and lower it to TOSA IR
model.compile(optimizer={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"},
                    loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])

# modelir = model.dump_ir()

# print(modelir)

import pathlib
path = str(pathlib.Path(__file__).parent.resolve()) + "/output_ir/torch_BertModel.ir"

# f = open(path, "w")
# f.write(modelir)
# f.close()

# print("\r\n\r\nIR for ", model.model.__class__.__name__, " generated: ", path)

print("Compiling TOSA model...")
tosa_ir= model.dump_tosa_ir()
t1_stop = time.perf_counter()

print(len(tosa_ir))

# f = open("bert.tosa.mlir", "w")
# f.write(tosa_ir)
# f.close()

print("Compiling Binary...")

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

t2_stop = time.perf_counter()

print("Bert****Ufront->TOSA Time: {:.3f}s, TOSA->Binary Time: {:.3f}s, Total Time: {:.3f}s".format(t1_stop - t1_start, t2_stop - t1_stop, t2_stop - t1_start)) # print performance indicator

ufront_ret = module.forward(input_ids, token_type_ids, input_mask).to_host()

print(ufront_ret.shape)
