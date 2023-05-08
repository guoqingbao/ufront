# Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical), Enflame Tech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

try:
    from tensorflow.keras.models import Model as tf_keras_Model
    from tensorflow.keras import optimizers as tf_keras_optimizer
    import tf2onnx
except:
    print("You need to first install tensorflow and tf2onnx before using keras models!")

#from tensorflow.keras import losses as tf_keras_losses
#from tensorflow.keras import metrics as tf_keras_metrics
# from tensorflow.keras.callbacks import Callback, LearningRateScheduler, VerifyMetrics, EpochVerifyMetrics

# Revised for Unified Computing Frontend (UFront)
# Enflame Tech. (ERA)
from ..onnx.model import ONNXModelKeras
from ..ufront import (OpType, ActiMode, AggrMode, PoolType, TensorF32, DataType, ParamSyncType, Initializer)
from ..ufront import Model, PyOperator, TensorF32, Optimizer, LossType, MetricsType #Rust frontend

tracing_id = 100

class Tensor(object):
  def __init__(self, batch_size,
               key,
               shape,
               name,
               batch_shape=None,
               dtype=None):

    self._ffhandle = None
    if type(dtype) == DataType:
      self.dtype = dtype
    else:
      if dtype == None or dtype == "float32" or dtype == "f32":
          self.dtype =  DataType.Float
      elif dtype == "float64" or dtype == "f64":
          self.dtype =  DataType.Double
      elif dtype == "int32" or dtype == "i32":
          self.dtype =  DataType.Int32
      elif dtype == "int64" or dtype == "i64":
          self.dtype =  DataType.Int64
      elif dtype == "float16" or dtype == "f16":
          self.dtype =  DataType.Half
      else:
          assert 0, "Unsupported datatype"

    if batch_shape != None:
      self.batch_shape = batch_shape
    else:
      self.batch_shape = (batch_size,) + tuple(shape[1:])
    #print(self.batch_shape)
    self.num_dims = len(self.batch_shape)
    self.key = key
    self.name = name

  @property
  def ffhandle(self):
    return self._ffhandle
    
  @ffhandle.setter
  def ffhandle(self, handle):
    assert isinstance(handle, Tensor) == True, "[Tensor]: ffhandle is not the correct type"
    assert self._ffhandle == None, "[Tensor]: check handle, already set"
    self._ffhandle = handle
    self.__verify_ffhandle_shape()
    self.__verify_ffhandle_dtype()

  @property
  def dtype_str(self):
    if self.dtype == DataType.Float:
      return "float32"
    elif self.dtype == DataType.Double:
      return "float64"
    elif self.dtype == DataType.Half:
      return "float16"
    elif self.dtype == DataType.Int32:
      return "int32"
    elif self.dtype == DataType.Int64:
      return "int64"

  def create_ff_tensor(self, ffmodel, name):
    assert self.batch_shape[0] != 0, "[Tensor]: batch size is not set"
    if (self.num_dims == 2 or self.num_dims == 4):
      #print(self.batch_shape, type(self.batch_shape))
      self._ffhandle = ffmodel.create_tensor(self.batch_shape, self.dtype, True, name)
    else:
      assert 0, "un-supported dims"
    self.__verify_ffhandle_shape()
    self.__verify_ffhandle_dtype()

  def set_batch_size(self, size):
    lst = list(self.batch_shape)
    lst[0] = size
    self.batch_shape = tuple(lst)

  def __verify_ffhandle_shape(self):
    assert self.num_dims == self._ffhandle.dims, "[Tensor]: check tensor shape"
    for i in range(0, self.num_dims):
      assert self.batch_shape[i] == self._ffhandle.shape[i], "[Tensor]: please check shape dim %d (%d == %d)" %(i, self.batch_shape[i], self._ffhandle.dims[i])

  def __verify_ffhandle_dtype(self):
    assert self.dtype == self._ffhandle.dtype

class BaseModel(object):
  def __init__(self, inputs, onnx_model, batch_size, transformer):
    self.ufront_model = Model()
    self.transformer=transformer
    self._onnx_model = onnx_model
    self._input_tensors = []
    for key in inputs:
      input_tensor = inputs[key]
      t = Tensor(batch_size=batch_size, key=key, name=input_tensor.name, shape=input_tensor.shape, dtype=input_tensor.dtype)
      self._input_tensors.append(t)
    
    self._loss = None
    self._metrics = []
    self._label_type = DataType.Float
    self._my_onnx_model = None
    self._output_tensor = None
    self._num_samples = 0
    self._input_dataloaders = []
    self._input_dataloaders_dim = []
    self._label_dataloader = 0
    self._label_dataloader_dim = 0
    self.batch_size = batch_size
    
    global tracing_id
    self.__tracing_id = tracing_id
    tracing_id += 1
    self._create_input_tensors()
    self._create_layers()
    
  def compile(self,
              optimizer,
              loss=None,
              metrics=None,
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              comp_mode=None,
              **kwargs):
    if loss_weights != None:
      assert 0, "loss_weights is not supported"
    if weighted_metrics != None:
      assert 0, "weighted_metrics is not supported"
    if run_eagerly != None:
      assert 0, "run_eagerly is not supported"

    assert loss != None, "loss is None"
    if loss == 'categorical_crossentropy':
      self._loss = LossType.CATEGORICAL_CROSSENTROPY
    elif loss == 'sparse_categorical_crossentropy':
      self._loss = LossType.SPARSE_CATEGORICAL_CROSSENTROPY
      self._label_type = DataType.Int32
    elif loss == 'mean_squared_error':
      self._loss = LossType.MEAN_SQUARED_ERROR_AVG_REDUCE
    else:
      assert 0, 'Unsupported loss'

    assert metrics != None, "metrics is None"
    assert isinstance(metrics, list) == True, 'Metrics should be a list'
    for metric in metrics:
      if metric == 'accuracy':
        self._metrics.append(MetricsType.ACCURACY)
      elif metric == 'categorical_crossentropy':
        self._metrics.append(MetricsType.CATEGORICAL_CROSSENTROPY)
      elif metric == 'sparse_categorical_crossentropy':
        self._metrics.append(MetricsType.SPARSE_CATEGORICAL_CROSSENTROPY)
      elif metric == 'mean_squared_error':
        self._metrics.append(MetricsType.MEAN_SQUARED_ERROR)
      elif metric == 'root_mean_squared_error':
        self._metrics.append(MetricsType.ROOT_MEAN_SQUARED_ERROR)
      elif metric == 'mean_absolute_error':
        self._metrics.append(MetricsType.MEAN_ABSOLUTE_ERROR)
      else:
        assert 0, 'Unsupported metric'
      
    
    if isinstance(optimizer, tf_keras_optimizer.Optimizer) == True:
      if isinstance(optimizer, tf_keras_optimizer.SGD) == True:
        self._ffoptimizer = Optimizer(params={"type":"sgd", "lr":str(optimizer.learning_rate.numpy()), "momentum":str(optimizer.momentum.numpy()), "nesterov":str(optimizer.nesterov), "weight_decay":str(optimizer.decay.numpy())})
      elif isinstance(optimizer, tf_keras_optimizer.Adam) == True:
        self._ffoptimizer = Optimizer(params={"type":"adam", "lr":str(optimizer.learning_rate.numpy()), "beta_1":str(optimizer.beta_1.numpy()), "beta_2":str(optimizer.beta_2.numpy()), "epsilon":str(optimizer.epsilon.numpy())})
      else:
        assert 0, "Unsupported optimizer"
    elif type(optimizer) == str:
      if optimizer == 'SGD':
        self._ffoptimizer = Optimizer(params={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"})
      elif optimizer == 'Adam':
        self._ffoptimizer = Optimizer(params={"type":"adam", "lr":"0.01"})
      else:
        assert 0, "Unsupported optimizer"
    elif type(optimizer) == dict:
      self._ffoptimizer = Optimizer(params=optimizer)
    else:
      assert 0, "Unsupported optimizer"

    self.ufront_model.optimizer = self._ffoptimizer
    self.ufront_model.compile(loss_type=self._loss, metrics=self._metrics, comp_mode=comp_mode)
    self._create_label_tensor()
    
  #TODO: finish API
  def fit(self,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.0,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_batch_size=None,
          validation_freq=1,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False):
    if batch_size != None:
      assert self._ffconfig.batch_size == batch_size, "batch size is not correct use -b to set it"
    if validation_split != 0.0:
      assert 0, "validation_split is not supported"
    if validation_data != None:
      assert 0, "validation_data is not supported"
    if shuffle != True:
      assert 0, "shuffle is not supported"
    if class_weight != None:
      assert 0, "class_weight is not supported"
    if sample_weight != None:
      assert 0, "sample_weight is not supported"
    if initial_epoch != 0:
      assert 0, "initial_epoch is not supported"
    if steps_per_epoch != None:
      assert 0, "steps_per_epoch is not supported"
    if validation_steps != None:
      assert 0, "validation_steps is not supported"
    if validation_batch_size != None:
      assert 0, "validation_batch_size is not supported"
    if validation_freq != 1:
      assert 0, "validation_freq is not supported"
    if max_queue_size != 10:
      assert 0, "max_queue_size is not supported"
    if workers != 1:
      assert 0, "workers is not supported"
    if use_multiprocessing != False:
      assert 0, "use_multiprocessing is not supported"

    assert self._output_tensor != None, "tensor is not init"
    if (isinstance(x, list) == False):
      input_tensors = [x]
    else:
      input_tensors = x
    label_tensor = y
    self._verify_tensors(input_tensors, label_tensor)
    self._create_data_loaders(input_tensors, label_tensor)
    self.ufront_model.init_layers()
    self._train(epochs, callbacks, eval=False)

  def _create_label_tensor(self):
    self._label_tensor = Tensor(batch_size=self.batch_size, key="label", name="label", shape=(self.batch_size, 1), dtype=self._label_type)

  def _create_input_tensors(self):
    idx = 0
    for input_tensor in self._input_tensors:
      self._input_tensors[idx].create_ff_tensor(self.ufront_model, "input"+str(idx+1))
      idx += 1
      
  def _create_layers(self):
    self._my_onnx_model = ONNXModelKeras(self._onnx_model, self.ufront_model, self.transformer)
    input_dict = {}
    for input_tensor in self._input_tensors:
      input_dict[input_tensor.name] = input_tensor.ffhandle
    self._output_tensor = self._my_onnx_model.apply(input_dict)

  def get_output_operator(self):
    return self._my_onnx_model.get_output_operator()
    
  def _verify_tensors(self, input_arrays, label_array):
    assert len(input_arrays) == len(self._input_tensors), "check len of input tensors"
    # TODO: move check shape into another function
    for np_array, t in zip(input_arrays, self._input_tensors):
      np_shape = np_array.shape
      assert len(np_shape) == t.num_dims, "check input shape"
      for i in range(1, len(np_shape)):
        assert np_shape[i] == t.batch_shape[i], "check input dims"
      assert np_array.dtype == t.dtype_str, "check input dtype"

    np_shape = label_array.shape
    assert len(np_shape) == self._label_tensor.num_dims, "check label shape"
    for i in range(1, len(np_shape)):
      assert np_shape[i] == self._label_tensor.batch_shape[i], "check label dims"
    assert label_array.dtype == self._label_tensor.dtype_str
  
  # TODO support dataloader in UFront
  def _create_data_loaders(self, x_trains, y_train):
    # Todo: check all num_samples, should be the same
    input_shape = x_trains[0].shape
    self._num_samples = input_shape[0]

    assert len(self._input_tensors) != 0, "input_tensor is not set"
    assert self._label_tensor != 0, "label_tensor is not set"

    idx = 0
    for x_train in x_trains:
      dataloader = self.ufront_model.create_data_loader(self._input_tensors[idx].ffhandle, x_train)
      self._input_dataloaders.append(dataloader)
      self._input_dataloaders_dim.append(len(input_shape))
      idx += 1
    dataloader = self.ufront_model.create_data_loader(self._label_tensor.ffhandle, y_train)
    self._label_dataloader = dataloader
    self._label_dataloader_dim = len(input_shape)

  def _train(self, epochs, callbacks, eval=False):
    if callbacks != None:
      for callback in callbacks:
        callback.set_model(self)

    if callbacks != None:
      for callback in callbacks:
        callback.on_train_begin()

    ts_start = self._ffconfig.get_current_time()
    epoch = 0
    epoch_flag = True
    self.__tracing_id += 1
    while (epoch < epochs) and (epoch_flag == True):
      if callbacks != None:
        for callback in callbacks:
          callback.on_epoch_begin(epoch)

      for dataloader in self._input_dataloaders:
        dataloader.reset()
      self._label_dataloader.reset()
      self.ufront_model.reset_metrics()
      iterations = self._num_samples / self._ffconfig.batch_size

      for iter in range(0, int(iterations)):
        if callbacks != None:
          for callback in callbacks:
            callback.on_batch_begin(iter)

        for dataloader in self._input_dataloaders:
          dataloader.next_batch(self.ufront_model)
        self._label_dataloader.next_batch(self.ufront_model)

        self._ffconfig.begin_trace(self.__tracing_id)
        self.ufront_model.forward()
        # for layer in self._layers:
        #   layer.ffhandle.forward(self.ufront_model)
        if eval == False:
          self.ufront_model.zero_gradients()
          self.ufront_model.backward()
          self.ufront_model.update()
        else:
          self.ufront_model.compute_metrics()
        self._ffconfig.end_trace(self.__tracing_id)

        if callbacks != None:
          for callback in callbacks:
            callback.on_batch_end(iter)

      if callbacks != None:
        for callback in callbacks:
          early_stop = callback.on_epoch_end(epoch)
          if early_stop == True:
            print("Accuracy reaches, now early stop, epoch: %d" %(epoch))
            epoch_flag = False

      epoch += 1

    ts_end = self._ffconfig.get_current_time()
    run_time = 1e-6 * (ts_end - ts_start);
    print("epochs %d, ELAPSED TIME = %.4fs, interations %d, samples %d, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, int(iterations), self._num_samples, self._num_samples * epochs / run_time));

    if callbacks != None:
      for callback in callbacks:
        callback.on_train_end()
   
class UFrontKeras(tf_keras_Model):
  def __init__(self, inputs, outputs, 
            batch_size, verbose=False, name=None, transformer=False):
    super(UFrontKeras, self).__init__(inputs=inputs, outputs=outputs, name=name)
    if (isinstance(inputs, dict) == True):
      # import onnx
      # onnx_model = onnx.load_model("Keras_VIT.onnx")
      # self._base_model = BaseModel(inputs=inputs, onnx_model=onnx_model, batch_size=batch_size, transformer=transformer)
      onnx_model = tf2onnx.convert.from_keras(self, opset=18 if transformer else 17)
      self._base_model = BaseModel(inputs=inputs, onnx_model=onnx_model[0], batch_size=batch_size, transformer=transformer)
    else:
      assert 0, "Inputs must be in dict format, e.g., {1: input_tensor1}"

  def umodel(self):
    return self._base_model._my_onnx_model.umodel
  
  def dump_ir(self):
    return self._base_model.ufront_model.dump_ir()
  
  def dump_tosa_ir(self):
    return self._base_model.ufront_model.dump_tosa_ir()
  

  def get_output_operator(self):
    return self._base_model.get_output_operator()
  
  def compile(self,
              optimizer,
              loss=None,
              metrics=None,
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              **kwargs):
    self._base_model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights, weighted_metrics=weighted_metrics, run_eagerly=run_eagerly, **kwargs)
    
  def fit(self,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.0,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_batch_size=None,
          validation_freq=1,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False):
    self._base_model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks,
      validation_split=validation_split, validation_data=validation_data, shuffle=shuffle, class_weight=class_weight,
      sample_weight=sample_weight, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
      validation_batch_size=validation_batch_size, validation_freq=validation_freq, max_queue_size=max_queue_size, workers=workers,
      use_multiprocessing=use_multiprocessing)
      
class KerasSequential(tf_keras_Model):
  def __init__(self, inputs, outputs, batch_size, name=None, transformer=False):
    super(KerasSequential, self).__init__(inputs=inputs, outputs=outputs, name=name)
    
    if (isinstance(inputs, dict) == True):
      onnx_model = tf2onnx.convert.from_keras(self)
      self._base_model = BaseModel(inputs=inputs, onnx_model=onnx_model, batch_size=batch_size, transformer=transformer)

  def compile(self,
              optimizer,
              loss=None,
              metrics=None,
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              **kwargs):
    self._base_model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights, weighted_metrics=weighted_metrics, run_eagerly=run_eagerly, **kwargs)
    
  def fit(self,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.0,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_batch_size=None,
          validation_freq=1,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False):
    self._base_model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks,
      validation_split=validation_split, validation_data=validation_data, shuffle=shuffle, class_weight=class_weight,
      sample_weight=sample_weight, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
      validation_batch_size=validation_batch_size, validation_freq=validation_freq, max_queue_size=max_queue_size, workers=workers,
      use_multiprocessing=use_multiprocessing)
