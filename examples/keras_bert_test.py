from bert import BertModelLayer
from tensorflow import keras
import numpy as np
from ufront.keras.model import UFrontKeras
import iree.compiler as ireec
from iree import runtime
import time
GPU = False

l_bert = BertModelLayer(**BertModelLayer.Params(
  vocab_size               = 16000,        # embedding params
  use_token_type           = True,
  use_position_embeddings  = True,
  token_type_vocab_size    = 2,

  num_layers               = 12,           # transformer encoder params
  hidden_size              = 768,
  hidden_dropout           = 0.1,
  intermediate_size        = 4*768,
  intermediate_activation  = "gelu",

  adapter_size             = None,         # see arXiv:1902.00751 (adapter-BERT)

  shared_layer             = False,        # True for ALBERT (arXiv:1909.11942)
  embedding_size           = None,         # None for BERT, wordpiece embedding size for ALBERT
  num_heads = 12,
  # name                     = "bert"        # any other Keras layer params
))

input_ids = np.array([[31, 51, 99], [15, 5, 0]], dtype='int64')
input_mask = np.array([[1, 1, 1], [1, 1, 0]], dtype='int64')
token_type_ids = np.array([[0, 0, 1], [0, 1, 0]], dtype='int64')

max_seq_len = 128
l_input_ids      = keras.layers.Input(shape=(max_seq_len,), dtype='int64')
l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int64')

output = l_bert([l_input_ids, l_token_type_ids])          # [batch_size, max_seq_len, hidden_size]
net = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)

t1_start = time.perf_counter()
#build UFront model
model = UFrontKeras(net, inputs = [input_ids, token_type_ids], batch_size = 1, transformer=True, pass_weights=True)


model.compile(optimizer={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"},
                        loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])

modelir = model.dump_ir()

print(modelir)

import pathlib
path = str(pathlib.Path(__file__).parent.resolve()) + "/output_ir/keras_BertModel.ir"

f = open(path, "w")
f.write(modelir)
f.close()

# print("\r\n\r\nIR for ", model.model.__class__.__name__, " generated: ", path)

print("Compiling TOSA model...")
tosa_ir= model.dump_tosa_ir()
t1_stop = time.perf_counter()

# f = open("tosa_tf.mlir", "w")
# f.write(tosa_ir)
# f.close()

print(len(tosa_ir))

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

ufront_ret = module.forward(input_ids, token_type_ids).to_host()

print(ufront_ret.shape)