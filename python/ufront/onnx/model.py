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
import io
import logging
import numpy as np
from functools import reduce
try:
    import onnx
except:
    print("You need to first install onnx package before using onnx models!")

try:
    import onnxsim
except:
    print("Some of the onnx models requires onnxsim library, please install onnxsim before usage!")

import struct
from ..ufront import (OpType, ActiMode, AggrMode, PoolType, TensorF32, DataType, ParamSyncType, Initializer)
from ..ufront import Model, PyOperator, TensorF32, Optimizer, LossType, MetricsType #Rust frontend
from ..utils import list_product, onnx_to_ufront_dtype, numpy_to_ufront_dtype, ufront_to_numpy_dtype

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
    const_tensor_idx = 1
    def __init__(self, onnx_model, umodel=None):
        if umodel != None:
            self.umodel = umodel
        else:
            self.umodel = Model()
        if type(onnx_model) == str:
            model = onnx.load(onnx_model)
        else:
            try:
                    # simply onnx models, for example, merge sub operators in onnx for chunk, remove redundant operators
                    onnx_model_, check = onnxsim.simplify(onnx_model) 
                    if check:
                        onnx_model = onnx_model_
                        # import onnx 
                        # onnx.save(onnx_model, "keras_vit.onnx")
                    else:
                        print("Simplified ONNX model could not be validated")
            except:
                    print("Some of the ONNX models requires onnxsim library!")
        
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
        # return self.umodel.add(x=input0, y=input1, name=node.name)
        if type(input0) == TensorF32 and type(input1) == TensorF32:
            return self.umodel.add(x=input0, y=input1, name=node.name)
        elif type(input0) == TensorF32:
            if type(input1) in [float, int, np.float32, np.float64, np.int32, np.int64, np.half]:
                return self.umodel.sadd(
                    input=input0, scalar=input1, name=node.name
                )
            else: 
                assert type(input1) == np.ndarray, "The given input is not an ndarray!"
                return self.umodel.sadd(
                    input=input0, operand=input1.tolist(), name=node.name
                )
        elif type(input1) == TensorF32:
            if type(input0) in [float, int, np.float32, np.float64, np.int32, np.int64, np.half]:
                return self.umodel.sadd(
                    input=input1, scalar=input0, name=node.name
                )
            else:
                assert type(input0) == np.ndarray, "The given input is not an ndarray!"
                return self.umodel.sadd(
                    input=input1, operand=input0.tolist(), name=node.name
                )
        else:
            return input0 + input1
    
    def handleSub(self, node, node_to_output):
        print(node)
        input0 = node_to_output[node.input[0]]
        input1 = node_to_output[node.input[1]]
    
        if type(input0) == TensorF32 and type(input1) == TensorF32:
            return self.umodel.subtract(x=input0, y=input1, name=node.name)
        elif type(input0) == TensorF32:
            if type(input1) in [float, int, np.float32, np.float64, np.int32, np.int64, np.half]:
                return self.umodel.ssub(input=input0, scalar=input1, name=node.name)
            else:
                assert type(input1) == np.ndarray, "The given input is not an ndarray!"
                return self.umodel.ssub(input=input0, operand=input1.tolist(), name=node.name)
            
        elif type(input1) == TensorF32:
            if type(input0) in [float, int, np.float32, np.float64, np.int32, np.int64, np.half]:
                return self.umodel.ssub(input=input1, scalar=input0, name=node.name)
            else:
                assert type(input0) == np.ndarray, "The given input is not an ndarray!"
                return self.umodel.ssub(input=input1, operand=input0.tolist(), name=node.name)   
        else:
            return input0 - input1
        
    def handleMul(self, node, node_to_output):
        input0 = node_to_output[node.input[0]]
        input1 = node_to_output[node.input[1]]
        if type(input0) == TensorF32 and type(input1) == TensorF32:
            return self.umodel.multiply(x=input0, y=input1, name=node.name)
        elif type(input0) == TensorF32:
            if type(input1) in [float, int, np.float32, np.float64, np.int32, np.int64, np.half]:
                return self.umodel.smultiply(
                    input=input0, scalar=input1, name=node.name
                )
            else:
                assert type(input1) == np.ndarray, "The given input is not an ndarray!"
                operator = self.addTensor(input1, False, node.input[1])
                self.operators.append(operator)
                input1 = operator.get_output(0)

                return self.umodel.multiply(
                    x=input0, y=input1, name=node.name
                )
        elif type(input1) == TensorF32:
            if type(input0) in [float, int, np.float32, np.float64, np.int32, np.int64, np.half]:
                return self.umodel.smultiply(
                    input=input1, scalar=input0, name=node.name
                )
            else:
                assert type(input0) == np.ndarray, "The given input is not an ndarray!"
                operator = self.addTensor(input0, False, node.input[0])
                self.operators.append(operator)
                input0 = operator.get_output(0)

                return self.umodel.multiply(
                    x=input1, y=input0, name=node.name
                )
        else:
            return input0 * input1

    def handleConcat(self, node, node_to_output):
        inputs = [node_to_output[i] for i in node.input]
        attribute = {x.name: x for x in node.attribute}
        allTensors = True
        for input in inputs:
            if type(input) != TensorF32:
                allTensors = False
        if allTensors:
            return self.umodel.concat(tensors=inputs, axis=attribute['axis'].i, name=node.name) # tensor concat
        else:
            return inputs #scalar concat
        
    def handleExpand(self, node, node_to_output):
        input_tensor = node_to_output[node.input[0]]
        output_shape = node_to_output[node.input[1]]
        if type(input_tensor) == np.ndarray:
            operator = self.addTensor(input_tensor, False, node.input[0])
            self.operators.append(operator)
            input_tensor = operator.get_output(0)

        return self.umodel.expand(
            input=input_tensor, sizes=output_shape, name=node.name,
        )
    
    def handleSplit(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        attribute = {x.name: x for x in node.attribute}

        if "split" in attribute:
            split = list(attribute['split'].ints)
        elif len(node.input) > 1:
            split = node_to_output[node.input[1]]
        if 'axis' in attribute:
            axis = attribute['axis'].i
        else:
            axis = 0
        return self.umodel.split(input=input, sizes=list(split), axis=axis, name=node.name)

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
        return self.umodel.pool2d(input=input, kernel=[kernel[0], kernel[1]], 
                        stride=[stride[0], stride[1]], pad=[padding[0], padding[1]], 
                        pool_type=PoolType.POOL_AVG, name=node.name)

    def handleGlobalAveragePool(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        return self.umodel.pool2d(input=input, output_size=[1, 1] if len(input.shape) > 3 else [1], 
                            stride=[1, 1] if len(input.shape) > 3 else [1], pad=[0, 0] if len(input.shape) > 3 else [0], pool_type=PoolType.POOL_ADAPTIVE, name=node.name)

    def handleBatchNormalization(self, node, node_to_output):
        attribute = {x.name: x for x in node.attribute}

        momentum=0.99,
        eps=1e-3,
        training_mode = True

        if "epsilon" in attribute:
            eps = attribute["epsilon"].f

        if "momentum" in attribute:
            momentum = attribute["momentum"].f

        if "training_mode" in attribute:
            training_mode = attribute["training_mode"].i == 1

        input = node_to_output[node.input[0]]
        return self.umodel.batch_norm(input=input, affine=True, eps=eps, momentum=momentum[0] if type(momentum)== list or type(momentum)==tuple else momentum, track_running_stats=training_mode)

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
        return self.umodel.conv2d(input=input, out_channels=out_channels, 
                kernel=[kernel[0], kernel[1]],
                stride=[stride[0], stride[1]], 
                pad=[padding[0], padding[1]], 
                activation=ActiMode.AC_MODE_NONE, groups=group, name=node.name)

    def handleDropout(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        inplace = False
        if len(node.attribute) > 0:
            attribute = {x.name: x for x in node.attribute}
            rate = attribute["ratio"].f
        else:
            rate = node_to_output[node.input[1]]
            if len(node.input) > 2:
                inplace = node_to_output[node.input[2]]
        seed = 0
        return self.umodel.dropout(input=input, rate=rate, seed=0, name=node.name)

    def handleFlatten(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        if len(node.attribute) > 0:
            attribute = {x.name: x for x in node.attribute}
            axis = attribute["axis"].i
        else:
            axis = 1

        return self.umodel.flat(input=input, start_dim=axis, end_dim=-1, name=node.name)
    
    def handleSqueeze(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        axes = node_to_output[node.input[1]]
        return self.umodel.flat(input=input, start_dim=axes[0] - 1, end_dim=-1, name=node.name)

    # def handleGemm(self, node):
    #     input = self.symbol_table[node.input[0]]
    #     dim = self.inputs[node.input[1]].dims[0]
    #     output = self.umodel.dense(input, dim, name=node.name)
    #     self.symbol_table[node.output[0]] = output
    #     logging.debug("self.umodel.dense({}, {}, name={})".format(node.input[0], dim, node.name))
        
    def handleDense(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        dim = attribute["out_dim"].i
        return self.umodel.dense(input=input, out_dim=dim, name=node.name)

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
        return self.umodel.pool2d(input=input, kernel=[kernel[0], kernel[1]], 
                            stride=[stride[0], stride[1]], pool_type=PoolType.POOL_MAX,
                            pad=[padding[0], padding[1]], name=node.name)

    def handleRelu(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        return self.umodel.relu(input=input, name=node.name)

    def handlePad(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        output = input
        return output

    def handleSoftmax(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        return self.umodel.softmax(input=input, name=node.name)

    def handleReshape(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        size = list_product(input.shape)
        shape = node_to_output[node.input[1]]
        for i in range(len(shape)):
            if shape[i] == -1:
                shape[i] = 1
                shape[i] = int(size/list_product(shape))
        return self.umodel.reshape(input=input, shape=list(shape), name=node.name)
    
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

    def unpack_rawdata(self, raw_data, data_type, shape):
        if len(shape) > 0:
            length = list_product(shape)
        else:
            length = 1

        if data_type == DataType.Float:
            fmt = "<%df" % (length)
        elif data_type == DataType.Int32:
            fmt = "<%di" % (length)
        elif data_type == DataType.Int64:
            fmt = "<%dq" % (length)
        elif data_type == DataType.Double:
            fmt = "<%dd" % (length)
        elif data_type == DataType.Half:
            fmt = "<%de" % (length)
        elif data_type == DataType.Bool:
            fmt = "<%d?" % (length)
        
        output = np.array(struct.unpack(fmt, raw_data))

        if len(shape) == 0 or (len(shape) == 1 and shape[0]==1):
            output = output[0] #scalar
        else:
            output = output.reshape(shape) # ndarray

        return output
    
    def handleConstant(self, node, node_to_output):
        attribute = {x.name: x for x in node.attribute}
        tensor = attribute["value"].t
        data_type = onnx_to_ufront_dtype(tensor.data_type)
        raw_data = tensor.raw_data
        
        if len(tensor.dims) > 1: #TODO set raw_data array to constant tensor
            np_tensor = self.unpack_rawdata(raw_data, data_type, tensor.dims)
            output = self.umodel.create_tensor(tensor.dims, DataType.Float, True, "constant_tensor" + str(ONNXModel.const_tensor_idx))
            output.set_ndarray(np_tensor)
            ONNXModel.const_tensor_idx += 1
        else:
            output = self.unpack_rawdata(raw_data, data_type, tensor.dims)

        return output
        
    def handleRange(self, node, node_to_output):
        # TODO: add range
        start = node_to_output[node.input[0]]
        limit = node_to_output[node.input[1]]
        delta = node_to_output[node.input[2]]
        return start

    def handleMatMul(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        dim = self.inputs[node.input[1]].dims[1]
        return self.umodel.dense(input = input, out_dim=dim, use_bias=False, name=node.name)
        
    def handleTranspose(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        perm = attribute["perm"].ints
        # output = input
        # return output
        return self.umodel.transpose(
            input=input, perms=perm, name=node.name,
        )
    
    def handleShape(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        return input.shape
    
    def handleIdentity(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        return self.umodel.identify(input = input, name=node.name)
    
    def handleConstantOfShape(self, node, node_to_output):
        value = node_to_output[node.input[0]]
        return value
    
    def handleSigmoid(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        return self.umodel.sigmoid(input = input, name=node.name)
    
    def addTensor(self, np_tensor, requires_grad, name):
        output = io.BytesIO()
        np.savez_compressed(output, x=np_tensor)
        raw_bytes = str(output.getvalue().hex())
        
        operator = self.umodel.tensor(np_tensor=np_tensor.astype(np.float32), dtype=numpy_to_ufront_dtype(np_tensor.dtype), requires_grad=requires_grad, initializer=raw_bytes, name=name)
        return operator
    
    def handleRandomUniformLike(self, node, node_to_output):
        np_tensor = node_to_output[node.input[0]]
        assert type(np_tensor) == np.ndarray, "The given input is not an ndarray!"
        operator = self.addTensor(np_tensor, False, node.input[0])
        self.operators.append(operator)

        return self.umodel.uniform_like(input = operator.get_output(0), dtype=numpy_to_ufront_dtype(np_tensor.dtype), name=node.name)

    def handleLess(self, node, node_to_output):
        input1 = node_to_output[node.input[0]]
        input2 = node_to_output[node.input[1]]
        if type(input2) != TensorF32:
            assert type(input2) == np.ndarray, "The given input is not an ndarray!"
            operator = self.addTensor(input2, False, node.input[1])
            self.operators.append(operator)
            input2 = operator.get_output(0)
        elif type(input1) != TensorF32:
            assert type(input1) == np.ndarray, "The given input is not an ndarray!"
            operator = self.addTensor(input1, False, node.input[0])
            self.operators.append(operator)
            input1 = operator.get_output(0)

        return self.umodel.less(x = input1, y=input2, name=node.name)
    
    def handleCast(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        dtype = attribute["to"].i
        dtype = onnx_to_ufront_dtype(dtype)
        if type(input) == TensorF32:
            return self.umodel.cast(input = input, dtype=dtype, name=node.name)
        else:
            return input
    
    def handleDiv(self, node, node_to_output):
        input1 = node_to_output[node.input[0]]
        scalar = node_to_output[node.input[1]]
        if type(scalar) in [float, int, np.float32, np.float64, np.int32, np.int64, np.half]:
            return self.umodel.divide(input = input1, scalar=scalar, name=node.name)
        else:
            assert type(scalar) == np.ndarray, "The given input is not an ndarray!"
            return self.umodel.divide(input = input1, operand=scalar.tolist(), name=node.name)

    
    def handleReduceMean(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        dims = attribute["axes"].ints

        return self.umodel.mean(
            input=input, dims=dims, keepdims=True, name=node.name,
        )
    
    def handleSlice(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        starts = node_to_output[node.input[1]]
        ends = node_to_output[node.input[2]]
        axis = node_to_output[node.input[3]]
        if type(input) != list and type(input) != tuple:
            output_shape = input.shape
            output_shape[axis] = ends - starts
            return self.umodel.slice_tensor(input=input, output_shape=list(output_shape), axis=axis, start=starts, end=ends, name=node.name)
        else:
            output_shape = input
            return ends - starts
    
    def handleLayerNormalization(self, node, node_to_output):
        input_tensor = node_to_output[node.input[0]]
        weight_tensor = node_to_output[node.input[1]]
        attribute = {x.name: x for x in node.attribute}
        eps = attribute["epsilon"].f
        shape = weight_tensor.type.tensor_type.shape.dim
        weight_shape = []
        for i in range(len(shape)):
            weight_shape.append(shape[i].dim_value)
        return self.umodel.layer_norm(input=input_tensor, normalized_shape=weight_shape, 
                                  eps=eps, elementwise_affine=True, name=node.name)
    
    def process_initializer(self, inputs, initializer):
        print("Processing initializer...")
        for item in initializer:
            print(item.dims, " ", item.data_type, " ", item.name, ": ", len(item.raw_data))
            datatype = onnx_to_ufront_dtype(item.data_type)
            if len(item.raw_data) > 0:
                values = self.unpack_rawdata(item.raw_data, datatype, item.dims)
            else:
                if datatype == DataType.Float:
                    values = np.array(item.float_data, dtype=np.float32)
                elif datatype == DataType.Int32:
                    values = np.array(item.int32_data, dtype=np.int32)
                elif datatype == DataType.Int64:
                    values = np.array(item.int64_data, dtype=np.int64)
                else:
                    assert 0, "Not supported data type!"

            inputs[item.name] = values
        
    def apply(self, input_tensors):
        self._fusion()
        node_to_output = OrderedDict()

        inputs = {}
        for i in range(len(self.model.graph.input)):
            input = self.model.graph.input[i]
            if input.name.find("input") >=0 or (i==0 and input.name.find("x") >=0):
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
            else:
                inputs[input.name] = input
            print("input ", i, ": ", input.name)

        print("\r\n")
        outputs = OrderedDict()
        
        # self.node_value_info = OrderedDict()
        # # print(self.model.graph.value_info)
        # for info in self.model.graph.value_info:
        #     self.node_value_info[info.name] = info.type
        # for i in range(3):
        #     item = self.model.graph.initializer[i]
        #     print(item.dims, " ", item.data_type, " ", item.name, ": ", len(item.raw_data))
        self.process_initializer(inputs, self.model.graph.initializer)

        for output in self.model.graph.output:
            outputs[output.name] = output

        node_to_output.update(inputs)
        for node in self.model.graph.node:
            print(node.name)

        for node in self.model.graph.node:
            handler_name = 'handle' + node.op_type
            if hasattr(self, handler_name):
                handler = getattr(self, handler_name)
                operator = handler(node, node_to_output)
                
                if type(operator) == PyOperator:
                    self.operators.append(operator)
                    if node.op_type == "Split":
                        node_output = []
                        for i in range(operator.num_of_outputs()):
                            node_output.append(operator.get_output(i))
                    else:
                        node_output = operator.get_output(0)
                else:
                    node_output = operator
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
    def __init__(self, onnx_model, umodel=None):
        super(ONNXModelKeras, self).__init__(onnx_model=onnx_model, umodel=umodel)
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
        return self.umodel.dense(input = input, out_dim=dim, use_bias=False, name=node.name)
        
    # def handleTranspose(self, node, node_to_output):
    #     input_tensor = node_to_output[node.input[0]]

    #     attribute = {x.name: x for x in node.attribute}
    #     perms = attribute["perm"].ints

    #     perms = node_to_output[node.input[1]] #TODO

    #     return self.umodel.transpose(
    #         input=input_tensor, perms=perms, name=node.name,
    #     )
    
        
    def handleReshape(self, node, node_to_output):
        input_tensor = node_to_output[node.input[0]]
        shape = node_to_output[node.input[1]]
        valid_shape = list(filter(lambda x: x > 0, shape))

        for i in range(len(shape)):
            if shape[i] == -1:
                shape[i] = list_product(input_tensor.shape) // list_product(valid_shape)
                break

        return self.umodel.reshape(input=input_tensor, shape=list(shape), name=node.name)

    #TODO fix constant
    def _create_initializer_tensor(self, ffconfig, input):
        if len(input.dims) == 1:
            dims = [ffconfig.batch_size, input.dims[0]]
            print("dims", dims)
        else:
            assert 0
        tensor = self.umodel.create_tensor(dims, DataType.Float, True, "constant_tensor"+str(ONNXModel.const_tensor_idx))
        ONNXModel.const_tensor_idx += 1
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
        self.umodel = Model() # Ufront Rust model
        super(UFrontONNX, self).__init__(onnx_model, self.umodel)
        # self.input_names = input_names
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.operators = []
        self._metrics = []
        self._loss = LossType.SPARSE_CATEGORICAL_CROSSENTROPY
        self._label_type = DataType.Int32
        self.umodel.optimizer = Optimizer(params={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"})

    def __call__(self, inputs):
        return self.apply(inputs)

    def softmax(self, input, name="softmax"):
        softmax_op = self.umodel.softmax(input=input, name=name)
        self.operators.append(softmax_op)
        return softmax_op.get_output(0)

    def dump_ir(self):
        return self.umodel.dump_ir()

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
                self.umodel.optimizer = Optimizer(params={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"})
            elif optimizer == 'Adam':
                self.umodel.optimizer = Optimizer(params={"type":"adam", "lr":"0.01"})
            else:
                assert 0, "Unsupported optimizer"
        elif type(optimizer) == dict:
            self.umodel.optimizer = Optimizer(params=optimizer)
        else:
            assert 0, "Unsupported optimizer"
        self.umodel.compile(loss_type=self._loss, metrics=self._metrics, comp_mode=comp_mode)

