# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
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

# Revised for Unified Computing Frontend (UFront)
# Enflame Tech. (ERA)
from collections import OrderedDict
import logging
try:
    import onnx
except:
    print("You need to first install onnx package before using onnx models!")

import struct
from ..ufront import (OpType, ActiMode, AggrMode, PoolType, TensorF32, DataType, ParamSyncType, Initializer)
from ..ufront import Model, PyOperator, TensorF32, Optimizer, LossType, MetricsType #Rust frontend

def onnx_to_ff_dt(datatype):
    if datatype == onnx.TensorProto.FLOAT:
        return DataType.FLOAT
    elif datatype == onnx.TensorProto.DOUBLE:
        return DataType.DOUBLE
    elif datatype == onnx.TensorProto.INT32:
        return DataType.INT32
    elif datatype == onnx.TensorProto.INT64:
        return DataType.INT64
    elif datatype == onnx.TensorProto.FLOAT16:
        return DataType.HALF
    else:
        assert 0, "Unsupported datatype"

class ONNXTensor(object):
    def __init__(self, name, dims, flag):
        self.name = name
        self.dims = [0] * len(dims)
        if flag == 1:
            self._set_dims_from_input(dims)
        else:
            self._set_dims_from_initializer(dims)
    
    def _set_dims_from_input(self, dims):
        for i in range(len(dims)):
            if hasattr(dims, 'dim_param'):
                self.dims[i] = dims[i].dim_param # "N"
            else:
                self.dims[i] = dims[i].dim_value
        
    def _set_dims_from_initializer(self, dims):
        for i in range(len(dims)):
            self.dims[i] = dims[i]

class ONNXModel(object):
    def __init__(self, onnx_model, ufront_model=None):
        if ufront_model != None:
            self.ufront_model = ufront_model
        else:
            self.ufront_model = Model()
        if type(onnx_model) == str:
            model = onnx.load(onnx_model)
        else:
            model = onnx_model

        self.inputs = {}
        self.operators = []
        for input in model.graph.input:
            tensor = ONNXTensor(input.name, input.type.tensor_type.shape.dim, 1)
            self.inputs[input.name] = tensor
        self.outputs = {}
        for output in model.graph.output:
            self.outputs[output.name] = output
        self.model = model
        self.symbol_table = {}

    def handleAdd(self, node, node_to_output):
        input0 = node_to_output[node.input[0]]
        input1 = node_to_output[node.input[1]]
        return self.ufront_model.add(x=input0, y=input1, name=node.name)
        
    def handleSub(self, node, node_to_output):
        print(node)
        input0 = node_to_output[node.input[0]]
        input1 = node_to_output[node.input[1]]
        return self.ufront_model.subtract(x=input0, y=input1, name=node.name)
        
    def handleMul(self, node, node_to_output):
        input0 = node_to_output[node.input[0]]
        input1 = node_to_output[node.input[1]]
        return self.ufront_model.multiply(x=input0, y=input1, name=node.name)

    def handleConcat(self, node, node_to_output):
        inputs = [node_to_output[i] for i in node.input]
        attribute = {x.name: x for x in node.attribute}
        return self.ufront_model.concat(tensors=inputs, axis=attribute['axis'].i, name=node.name)

    def handleSplit(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        split = list(attribute['split'].ints)
        if 'axis' in attribute:
            axis = attribute['axis'].i
        else:
            axis = 0
        return self.ufront_model.split(input=input, sizes=split, axis=axis, name=node.name)

    def handleAveragePool(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        kernel = attribute["kernel_shape"].ints
        stride = attribute["strides"].ints
        if "pads" in attribute:
            padding = attribute["pads"].ints
        elif "auto_pad" in attribute:
            if attribute["auto_pad"].s == b'VALID':
                padding = [0, 0]
            elif attribute["auto_pad"].s == b'SAME':
                # TODO
                assert 0
            else:
                assert 0, "Unknown auto_pad"
        else:
            padding = [0, 0]
        return self.ufront_model.pool2d(input=input, kernel_h=kernel[0], kernel_w=kernel[1], 
                        stride_h=stride[0], stride_w=stride[1], padding_h=padding[0], padding_w=padding[1], 
                        pool_type=PoolType.POOL_AVG, name=node.name)

    def handleGlobalAveragePool(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        return self.ufront_model.pool2d(input=input, kernel_h=input.dims[2], kernel_w=input.dims[3], 
                            stride_h=1, stride_w=1, padding_h=0, padding_w=0, pool_type=PoolType.POOL_AVG, name=node.name)

    def handleBatchNormalization(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        return self.ufront_model.batch_norm(input=input)

    def handleConv(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        kernel = attribute["kernel_shape"].ints
        stride = attribute["strides"].ints
        if "pads" in attribute:
            padding = attribute["pads"].ints
        elif "auto_pad" in attribute:
            if attribute["auto_pad"].s == b'VALID':
                padding = [0, 0]
            elif attribute["auto_pad"].s == b'SAME':
                # TODO
                assert 0
            else:
                assert 0, "Unknown auto_pad"
        else:
            padding = [0, 0]
        group = attribute["group"].i
        out_channels = self.inputs[node.input[1]].dims[0]
        return self.ufront_model.conv2d(input=input, out_channels=out_channels, kernel_h=kernel[0], 
                kernel_w=kernel[1], stride_h=stride[0], stride_w=stride[1], 
                padding_h=padding[0], padding_w=padding[1], 
                activation=ActiMode.AC_MODE_NONE, groups=group, name=node.name)

    def handleDropout(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        rate = attribute["ratio"].f
        seed = 0
        return self.ufront_model.dropout(input=input, rate=rate, seed=0, name=node.name)

    def handleFlatten(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        return self.ufront_model.flat(input=input, name=node.name)

    # def handleGemm(self, node):
    #     input = self.symbol_table[node.input[0]]
    #     dim = self.inputs[node.input[1]].dims[0]
    #     output = self.ufront_model.dense(input, dim, name=node.name)
    #     self.symbol_table[node.output[0]] = output
    #     logging.debug("self.ufront_model.dense({}, {}, name={})".format(node.input[0], dim, node.name))
        
    def handleDense(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        dim = attribute["out_dim"].i
        return self.ufront_model.dense(input=input, out_dim=dim, name=node.name)

    def handleMaxPool(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        kernel = attribute["kernel_shape"].ints
        stride = attribute["strides"].ints
        if "pads" in attribute:
            padding = attribute["pads"].ints
        elif "auto_pad" in attribute:
            if attribute["auto_pad"].s == b'VALID':
                padding = [0, 0]
            elif attribute["auto_pad"].s == b'SAME':
                # TODO
                assert 0
            else:
                assert 0, "Unknown auto_pad"
        else:
            padding = [0, 0]
        return self.ufront_model.pool2d(input=input, kernel_h=kernel[0], kernel_w=kernel[1], 
                            stride_h=stride[0], stride_w=stride[1], 
                            padding_h=padding[0], padding_w=padding[1], name=node.name)

    def handleRelu(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        return self.ufront_model.relu(input=input, name=node.name)

    def handlePad(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        output = input
        return output

    def handleSoftmax(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        return self.ufront_model.softmax(input=input, name=node.name)

    def handleReshape(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        shape = node_to_output[node.input[1]]
        return self.ufront_model.reshape(input=input, shape=list(shape.int64_data), name=node.name)
    
    def handleCast(self, node, node_to_output):
        # TODO: add cast
        input = node_to_output[node.input[0]]
        return input
        
    def handleUnsqueeze(self, node, node_to_output):
        # TODO: add unsqueeze
        input = node_to_output[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        axes = attribute["axes"].ints
        return input

    #TODO fix constant  
    def handleConstant(self, node, node_to_output):
        attribute = {x.name: x for x in node.attribute}
        tensor = attribute["value"].t
        data_type = onnx_to_ff_dt(tensor.data_type)
        raw_data = tensor.raw_data
        if data_type == DataType.FLOAT:
            value = struct.unpack('f', raw_data)
        else:
            assert 0, "not implemented"
        if len(tensor.dims) != 0:
            #TODO: this path has not tested
            output = self.ufront_model.create_constant(tensor,dims, value[0], data_type)
            logging.warning("self.ufront_model.create_constant: {}, {}, {}".format(dims, value[0], data_type))
        else:
            output = value[0]
        return output
        
    def handleRange(self, node, node_to_output):
        # TODO: add range
        start = node_to_output[node.input[0]]
        limit = node_to_output[node.input[1]]
        delta = node_to_output[node.input[2]]
        return start

    def apply(self, input_tensors):
        self._fusion()
        node_to_output = OrderedDict()

        inputs = {}
        for i in range(len(self.model.graph.input)):
            input = self.model.graph.input[i]
            if input.name.find("input") >=0:
                if type(input_tensors) == list:
                    input_tensor = input_tensors[i]
                    if type(input_tensor) != TensorF32:
                        input_tensor = TensorF32(input_tensor, input.name) # convert to Rust f32 tensor
                    inputs[input.name] = input_tensor
                elif type(input_tensors) == dict:
                    input_tensor = input_tensors[input.name]
                    if type(input_tensor) != TensorF32:
                        input_tensor = TensorF32(input_tensor, input.name) # convert to Rust f32 tensor
                    inputs[input.name] = input_tensor
                else:
                    assert 0, "Not a valid input type!"

        outputs = OrderedDict()

        for output in self.model.graph.output:
            outputs[output.name] = output

        node_to_output.update(inputs)
        
        for node in self.model.graph.node:
            handler_name = 'handle' + node.op_type
            if hasattr(self, handler_name):
                handler = getattr(self, handler_name)
                operator = handler(node, node_to_output)
                
                if node.op_type == "Transpose":
                    node_output = operator
                elif node.op_type == "Split":
                    self.operators.append(operator)
                    node_output = []
                    for i in range(operator.num_of_outputs()):
                        node_output.append(operator.get_output(i))
                else:
                    self.operators.append(operator)
                    node_output = operator.get_output(0)
            else:
                logging.warning("Can't handle: {}".format(node.op_type))

            if node_output is not None:
                if node.op_type == "Split":
                    for i in range(len(node_output)):
                        node_to_output[node.output[i]] = node_output[i]
                else:
                    node_to_output[node.output[0]] = node_output
                    
        tensor_outputs = []
        for name in outputs.keys():
             tensor_outputs.append(node_to_output[name])
        return tensor_outputs
        
    def _fusion(self):
        flag = True
        dense_idx = 1
        while flag == True:
            idx = 0
            flag_found = False
            for node in self.model.graph.node:
                if node.op_type == 'MatMul':
                    output = node.output[0]
                    for add_node in self.model.graph.node:
                        if add_node.op_type == 'Add' and (add_node.input[0] == output or add_node.input[1] == output):
                            #print(node, add_node)
                            flag_found = True
                            dim = self.inputs[node.input[1]].dims[1]
                            dense_node = onnx.helper.make_node('Dense', inputs=[node.input[0]], outputs=[add_node.output[0]], out_dim=dim, name="Dense_"+str(dense_idx))
                            dense_idx += 1
                            #print(dense_node)
                            break
                    if flag_found:
                        self.model.graph.node.insert(idx, dense_node)
                        self.model.graph.node.remove(add_node)
                        self.model.graph.node.remove(node)
                        break
                
                elif node.op_type == 'Gemm':
                    flag_found = True
                    dim = self.inputs[node.input[1]].dims[0]
                    dense_node = onnx.helper.make_node('Dense', inputs=[node.input[0]], outputs=[node.output[0]], out_dim=dim, name="Dense_"+str(dense_idx))
                    dense_idx += 1
                    self.model.graph.node.insert(idx, dense_node)
                    self.model.graph.node.remove(node)
                    break
                    
                idx += 1
            flag = flag_found
        
        
class ONNXModelKeras(ONNXModel):
    def __init__(self, onnx_model, ufront_model=None):
        super(ONNXModelKeras, self).__init__(onnx_model=onnx_model, ufront_model=ufront_model)
        for node in onnx_model.graph.node:
            if node.name.find("u_front_keras/") != -1:
                node.name = node.name[node.name.find("keras_model/")+len("u_front_keras/")+1:]
            print(node.name)
        for initializer in self.model.graph.initializer:
            if ('/bias' in initializer.name or '/BiasAdd/ReadVariableOp' in initializer.name )and 'dense' in initializer.name:
                # self.symbol_table[initializer.name] = self._create_initializer_tensor(ffconfig, ffmodel, initializer)
                pass
            else:
                tensor = ONNXTensor(initializer.name, initializer.dims, 2)
                self.inputs[initializer.name] = tensor
        
    def handleMatMul(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        dim = self.inputs[node.input[1]].dims[1]
        return self.ufront_model.dense(input = input, out_dim=dim, use_bias=False, name=node.name)
        
    def handleTranspose(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        output = input
        return output
        
    def handleReshape(self, node, node_to_output):
        return self.handleFlatten(node, node_to_output)

    #TODO fix constant
    def _create_initializer_tensor(self, ffconfig, input):
        if len(input.dims) == 1:
            dims = [ffconfig.batch_size, input.dims[0]]
            print("dims", dims)
        else:
            assert 0
        tensor = self.ufront_model.create_constant(dims, 0.0, DataType.FLOAT)
        print("create constant", input.name)
        return tensor

class UFrontONNX(ONNXModel):
    def __init__(
        self,
        onnx_model,
        batch_size,
        verbose=False,
        seq_length=None,
    ): 
        self.ufront_model = Model() # Ufront Rust model
        super(UFrontONNX, self).__init__(onnx_model, self.ufront_model)
        # self.input_names = input_names
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.operators = []
        self._metrics = []
        self._loss = LossType.SPARSE_CATEGORICAL_CROSSENTROPY
        self._label_type = DataType.Int32
        self.ufront_model.optimizer = Optimizer(params={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"})

    def __call__(self, inputs):
        return self.apply(inputs)

    def softmax(self, input, name="softmax"):
        softmax_op = self.ufront_model.softmax(input=input, name=name)
        self.operators.append(softmax_op)
        return softmax_op.get_output(0)

    def dump_ir(self):
        return self.ufront_model.dump_ir()

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
            
        if type(optimizer) == str:
            if optimizer == 'SGD':
                self.ufront_model.optimizer = Optimizer(params={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"})
            elif optimizer == 'Adam':
                self.ufront_model.optimizer = Optimizer(params={"type":"adam", "lr":"0.01"})
            else:
                assert 0, "Unsupported optimizer"
        elif type(optimizer) == dict:
            self.ufront_model.optimizer = Optimizer(params=optimizer)
        else:
            assert 0, "Unsupported optimizer"
        self.ufront_model.compile(loss_type=self._loss, metrics=self._metrics, comp_mode=comp_mode)

