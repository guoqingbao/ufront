import torch
from ufront.pytorch.model import UFrontTorch
# checkpoint = "THUDM/chatglm-6b-int4"
checkpoint = "/root/llm/chatglm-6b" #set the path for model weights
# import onnx
# from torch.onnx import TrainingMode
# from ufront.onnx.model import ONNXModel, ONNXModelKeras, UFrontONNX
from chatglm_q.loader import load_model_and_tokenizer

_, net, tokenizer = load_model_and_tokenizer(checkpoint)

input_ids, prefix_mask = tokenizer.encode("[Round 0] Please proofread the following sentence: convert torch model to ufront model\n")

model = UFrontTorch(net.half(), batch_size=1, pass_weights=True) # convert torch model to ufront model

output_tensors = model( inputs = 
    [torch.FloatTensor([input_ids]).half(),
    torch.FloatTensor([prefix_mask]).half()]
)

total_memory = 0.0
for operator in model.operators: # access operators in the ufront computation graph
    sz = operator.memory_usage()
    total_memory += sz
    print("{0} > name: {1}, raw_ptr: {2:#06x}, No. of outputs: {3}, memory used:{4:.5f}MB".format(operator.op_type, operator.params['name'], 
    operator.raw_ptr, operator.num_of_outputs(),  sz/1024/1024))

print("\r\nTotal memory cached for operators {:.2f}GB".format(total_memory/1024/1024/1024))

model.compile(optimizer={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"},
                        loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])


# for operator in operators:
#   print(operator.ir) #show ir for each operator
modelir= model.dump_ir()

modelname = "chatglm-6b"
import pathlib
path = str(pathlib.Path(__file__).parent.resolve()) + "/output_ir/torch_" + modelname + ".ir"

f = open(path, "w")
f.write(modelir)
f.close()

print("\r\n\r\nIR for ", modelname, " generated: ", path)
