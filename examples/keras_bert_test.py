from bert import BertModelLayer
from tensorflow import keras
import numpy as np
from ufront.keras.model import UFrontKeras

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

#build UFront model
ufront_model = UFrontKeras(net, inputs = [input_ids, token_type_ids], batch_size = 1, transformer=True, pass_weights=True)


ufront_model.compile(optimizer={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"},
                        loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])