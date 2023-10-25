# Copyright (c) 2023 Enflame Tech, Facebook, Inc. and its affiliates (legacy code). All rights reserved.
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
from ..ufront import (OpType, ActiMode, AggrMode, PoolType, Tensor, DataType, WeightType, ParamSyncType, Initializer)
from ..ufront import Model, PyOperator, Tensor, Optimizer, LossType, MetricsType #Rust frontend
from ..utils import list_product, onnx_to_ufront_dtype, numpy_to_ufront_dtype, ufront_to_numpy_dtype

class ONNXTensor(object):
    def __init__(self, name, dims, flag):
        self.name = name
        self.shape = [0] * len(dims)
        if flag == 1:
            self._set_dims_from_input(dims)
        else:
            self._set_dims_from_initializer(dims)
    
    def _set_dims_from_input(self, dims):
        for i in range(len(dims)):
            if hasattr(dims, 'dim_param'):
                self.shape[i] = dims[i].dim_param # "N"
            else:
                self.shape[i] = dims[i].dim_value
        
    def _set_dims_from_initializer(self, dims):
        for i in range(len(dims)):
            self.shape[i] = dims[i]


class ONNXModel(object):
    const_tensor_idx = 1
    def __init__(self, onnx_model, umodel=None, simplify = False, pass_weights=False):
        if umodel != None:
            self.umodel = umodel
        else:
            self.umodel = Model()

        self.umodel.weight_type = WeightType.EXTERNAL if pass_weights else WeightType.INTERNAL
        
        if type(onnx_model) == str:
            model = onnx.load(onnx_model)
        elif simplify:
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
        # return self.umodel.add(x=input0, y=input1, name=node.name)
        ss = 0.0001
        if type(input0) == Tensor and type(input1) == Tensor:
            return self.umodel.add(x=input0, y=input1, name=node.name)
        elif type(input0) == Tensor:
            if type(input1) in [float, int, np.float32, np.float64, np.int32, np.int64, np.half]:
                return self.umodel.sadd(
                    input=input0, scalar=ss, name=node.name
                )
            else: 
                assert type(input1) == np.ndarray, "The given input is not an ndarray!"
                operator = self.addTensor(input1, False, node.input[1])
                self.operators.append(operator)
                input1 = operator.get_output(0)
                return self.umodel.add(
                    x=input0, y=input1, name=node.name
                )
        elif type(input1) == Tensor:
            if type(input0) in [float, int, np.float32, np.float64, np.int32, np.int64, np.half]:
                return self.umodel.sadd(
                    input=input1, scalar=ss, name=node.name
                )
            else:
                assert type(input0) == np.ndarray, "The given input is not an ndarray!"
                operator = self.addTensor(input0, False, node.input[0])
                self.operators.append(operator)
                input0 = operator.get_output(0)
                return self.umodel.sadd(
                    x=input0, y=input1, name=node.name
                )
        else:
            return input0 + input1
    
    def handleSub(self, node, node_to_output):
        # print(node)
        input0 = node_to_output[node.input[0]]
        input1 = node_to_output[node.input[1]]
    
        if type(input0) == Tensor and type(input1) == Tensor:
            return self.umodel.subtract(x=input0, y=input1, name=node.name)
        elif type(input0) == Tensor:
            if type(input1) in [float, int, np.float32, np.float64, np.int32, np.int64, np.half]:
                return self.umodel.ssub(input=input0, scalar=input1, scalar_position="\"RIGHT\"", name=node.name)
            else:
                assert type(input1) == np.ndarray, "The given input is not an ndarray!"
                return self.umodel.ssub(input=input0, operand=input1.tolist(), scalar_position="\"RIGHT\"", name=node.name)
            
        elif type(input1) == Tensor:
            if type(input0) in [float, int, np.float32, np.float64, np.int32, np.int64, np.half]:
                return self.umodel.ssub(input=input1, scalar=input0, scalar_position="\"LEFT\"", name=node.name)
            else:
                assert type(input0) == np.ndarray, "The given input is not an ndarray!"
                return self.umodel.ssub(input=input1, operand=input0.tolist(), scalar_position="\"LEFT\"", name=node.name)   
        else:
            return input0 - input1
        
    def handleMul(self, node, node_to_output):
        input0 = node_to_output[node.input[0]]
        input1 = node_to_output[node.input[1]]
        if type(input0) == Tensor and type(input1) == Tensor:
            return self.umodel.multiply(x=input0, y=input1, name=node.name)
        elif type(input0) == Tensor:
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
        elif type(input1) == Tensor:
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
        for i in range(len(inputs)):
            input = inputs[i]
            if type(input).__name__ == "ValueInfoProto":
                dims = [x.dim_value for x in input.type.tensor_type.shape.dim]
                inputs[i] = self.umodel.create_tensor(dims, DataType.Float, True, input.name)
            elif type(input) == np.ndarray:
                param_op = self.umodel.parameter(np_tensor=input, dtype=numpy_to_ufront_dtype(input.dtype), requires_grad=True, name=node.input[i])
                inputs[i] = param_op.get_output(0)
            elif type(input) != Tensor:
                allTensors = False
        if allTensors:
            return self.umodel.concat(tensors=inputs, axis=attribute['axis'].i, name=node.name) # tensor concat
        else:
            ret = []
            for input in inputs:
                if type(input) == list: #corner case [[1, 2], 3]
                    ret.extend(input)
                else:
                    ret.append(input)
            return ret #scalar concat
        
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
                        stride=[stride[0], stride[1]], 
                        # pad=[padding[0], padding[1]], 
                        pad=[padding[2], padding[3]] if len(padding) == 4 else [padding[0], padding[1]],
                        pool_type=PoolType.POOL_AVG, name=node.name)

    def handleGlobalAveragePool(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        return self.umodel.pool2d(input=input, output_size=[1, 1] if len(input.shape) > 3 else [1], 
                            stride=[1, 1] if len(input.shape) > 3 else [1], pad=[0, 0] if len(input.shape) > 3 else [0], pool_type=PoolType.POOL_ADAPTIVE, name=node.name)

    def handleGlobalMaxPool(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        return self.umodel.pool2d(input=input, output_size=[1, 1] if len(input.shape) > 3 else [1], 
                            stride=[1, 1] if len(input.shape) > 3 else [1], pad=[0, 0] if len(input.shape) > 3 else [0], pool_type=PoolType.POOL_ADAPTIVE_MAX, name=node.name)

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

        # if self.umodel.weight_type == WeightType.INTERNAL or not training_mode or len(node.input) < 2:
        # if self.umodel.weight_type == WeightType.INTERNAL or len(node.input) < 2:
        if training_mode and len(node.input) > 4:
            weight = node_to_output[node.input[1]]

            weight = weight.reshape((1, weight.shape[0], 1, 1))
            weight_op = self.umodel.parameter(np_tensor=weight, dtype=numpy_to_ufront_dtype(weight.dtype), requires_grad=True, name=node.input[1])

            bias = node_to_output[node.input[2]]
            bias = bias.reshape((1, bias.shape[0], 1, 1))
            bias_op = self.umodel.parameter(np_tensor=bias, dtype=numpy_to_ufront_dtype(bias.dtype), requires_grad=True, name=node.input[2])

            running_mean = node_to_output[node.input[3]]
            running_mean = running_mean.reshape((1, running_mean.shape[0], 1, 1))
            running_mean_op = self.umodel.parameter(np_tensor=running_mean, dtype=numpy_to_ufront_dtype(running_mean.dtype), requires_grad=True, name=node.input[3])
            
            running_var = node_to_output[node.input[4]]
            running_var = running_var.reshape((1, running_var.shape[0], 1, 1))
            running_var_op = self.umodel.parameter(np_tensor=running_var, dtype=numpy_to_ufront_dtype(running_var.dtype), requires_grad=True, name=node.input[4])
            return self.umodel.batch_norm(input=input, weight=weight_op.get_output(0),
                                        bias=bias_op.get_output(0),
                                        mean=running_mean_op.get_output(0),
                                        variance=running_var_op.get_output(0),
                                        affine=True, eps=eps, momentum=momentum[0] if type(momentum)== list or type(momentum)==tuple else momentum, track_running_stats=training_mode)
        else:
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
                padding = [0, 0, 0, 0]
            elif attribute["auto_pad"].s == b'SAME':
                # TODO
                assert 0
            else:
                assert 0, "Unknown auto_pad"
        else:
            padding = [0, 0, 0, 0]
        padding[1], padding[2] = padding[2], padding[1]
        if "dilations" in attribute:
            dilation = attribute["dilations"].ints

        group = attribute["group"].i
        out_channels = self.inputs[node.input[1]].shape[0]
        if self.umodel.weight_type == WeightType.INTERNAL or len(node.input) < 2:
            return self.umodel.conv2d(input=input, out_channels=out_channels, 
                    kernel=[kernel[0], kernel[1]],
                    stride=[stride[0], stride[1]], 
                    # pad=[padding[2], padding[3]] if len(padding) == 4 else [padding[0], padding[1]], 
                    pad = padding,
                    dilation = dilation,
                    activation=ActiMode.AC_MODE_NONE, groups=group, name=node.name)
        else:
            weight = node_to_output[node.input[1]]
            weight_op = self.umodel.parameter(np_tensor=weight, dtype=numpy_to_ufront_dtype(weight.dtype), requires_grad=True, name=node.input[1])
            if len(node.input) > 2:
                bias = node_to_output[node.input[2]]
                bias_op = self.umodel.parameter(np_tensor=bias, dtype=numpy_to_ufront_dtype(bias.dtype), requires_grad=True, name=node.input[2])

                return self.umodel.conv2d(input=input, weight=weight_op.get_output(0), bias=bias_op.get_output(0),
                                          out_channels=out_channels, 
                        kernel=[kernel[0], kernel[1]],
                        stride=[stride[0], stride[1]], 
                        # pad=[padding[2], padding[3]] if len(padding) == 4 else [padding[0], padding[1]], 
                        pad = padding,
                        dilation = dilation,
                        activation=ActiMode.AC_MODE_NONE, groups=group, name=node.name)
            else:
                return self.umodel.conv2d(input=input, weight=weight_op.get_output(0), 
                                          out_channels=out_channels, 
                        kernel=[kernel[0], kernel[1]],
                        stride=[stride[0], stride[1]], 
                        # pad=[padding[2], padding[3]] if len(padding) == 4 else [padding[0], padding[1]], 
                        pad = padding,
                        dilation = dilation,
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
        return self.umodel.dropout(input=input, rate=rate, seed=0, training=False, inplace=inplace, name=node.name)

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
        axis = axes[0] - 1 if type(axes)==tuple or type(axes)==list or type(axes)==np.ndarray else int(axes)
        if axis < 0: axis = 0
        if axis >= len(input.shape): axis = len(input.shape) - 1
        return self.umodel.flat(input=input, start_dim=axis, end_dim=-1, name=node.name)
    
    def handleSqrt(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        return self.umodel.sqrt(input=input, name=node.name)
    
    def handleReciprocal(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        return self.umodel.reciprocal(input=input, name=node.name)
    
    def handleNeg(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        return self.umodel.smultiply(
            input=input, scalar=-1.0, name=node.name
        )
        # return self.umodel.neg(input=input, name=node.name)

    # def handleGemm(self, node):
    #     input = self.symbol_table[node.input[0]]
    #     dim = self.inputs[node.input[1]].dims[0]
    #     output = self.umodel.dense(input, dim, name=node.name)
    #     self.symbol_table[node.output[0]] = output
    #     logging.debug("self.umodel.dense({}, {}, name={})".format(node.input[0], dim, node.name))
        
    def handleDense(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        if "out_dim" in attribute:
            dim = attribute["out_dim"].i 
        # elif len(node.input) > 1:
        #     weight = node_to_output[node.input[1]]
        #     dim = weight.shape[-2] if weight.shape[-1] == input.shape[1] else weight.shape[-1]
            
        if self.umodel.weight_type == WeightType.INTERNAL or len(node.input) < 2:
            weight = node_to_output[node.input[1]]
            dim = weight.shape[-2] if weight.shape[-1] == input.shape[1] else weight.shape[-1]
            return self.umodel.dense(input=input, out_dim=dim, name=node.name)
        else:
            weight = node_to_output[node.input[1]]
            dim = weight.shape[-2] if weight.shape[-1] == input.shape[1] else weight.shape[-1]
            weight_op = self.umodel.parameter(np_tensor=weight, dtype=numpy_to_ufront_dtype(weight.dtype), requires_grad=True, name=node.input[1])
            if len(node.input) > 2:
                bias = node_to_output[node.input[2]]
                bias_op = self.umodel.parameter(np_tensor=bias, dtype=numpy_to_ufront_dtype(bias.dtype), requires_grad=True, name=node.input[2])
                return self.umodel.dense(input=input, weight=weight_op.get_output(0), bias=bias_op.get_output(0), out_dim=dim, weight_transposed=False, name=node.name)
            else:
                return self.umodel.dense(input=input, weight=weight_op.get_output(0), out_dim=dim, weight_transposed=False, name=node.name)


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
                            # pad=[padding[0], padding[1]], 
                            pad=[padding[2], padding[3]] if len(padding) == 4 else [padding[0], padding[1]],
                            name=node.name)

    def handleRelu(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        return self.umodel.relu(input=input, name=node.name)

    def handleGelu(self, node, node_to_output):
        input1 = node_to_output[node.input[0]]
        if len(node.input) > 1:
            input2 = node_to_output[node.input[1]]

        return self.umodel.gelu(input=input1 if type(input1)==Tensor else input2, approximate=True, name=node.name)
    
    def handleTanh(self, node, node_to_output):
        input1 = node_to_output[node.input[0]]
        return self.umodel.tanh(input=input1, name=node.name)

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
        out_size = list_product(list(filter(lambda x: x > 0, shape)))

        for i in range(len(shape)):
            if shape[i] == -1:
                shape[i] = int(size/out_size)
                break
        return self.umodel.reshape(input=input, shape=list(shape), name=node.name)
    
    # def handleCast(self, node, node_to_output):
    #     input = node_to_output[node.input[0]]
    #     attribute = {x.name: x for x in node.attribute}
    #     if hasattr(attribute, "dtype"):
    #         dtype = attribute["dtype"]
    #     elif len(node.input) > 1 and type(node.input[1]) == np.dtype:
    #         dtype = node.input[1]
    #     else:
    #         assert 0, "Invalid dtype to cast!"
    #     return self.umodel.cast(input=input, dtype=numpy_to_ufront_dtype(dtype), name=node.name)
        
    def handleUnsqueeze(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        if hasattr(attribute, "axes"):
            axes = attribute["axes"].ints
        elif len(node.input) > 1:
            axes = node_to_output[node.input[1]]
        else:
            assert 0, "Invalid unsqueeze call!"

        shape = list(input.shape)
        shape.insert(axes, 1)
        return self.umodel.reshape(
            input=input, shape=list(shape), name=node.name,
        )

    def unpack_rawdata(self, raw_data, data_type, shape, name):
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
        
        output = np.array(struct.unpack(fmt, raw_data), dtype=ufront_to_numpy_dtype(data_type))

        if len(shape) == 0 or (len(shape) == 1 and shape[0]==1):
            output = output[0] #scalar
        elif len(shape) > 1:
            output = output.reshape(shape) # ndarray
            if name =="class_token" or name =="encoder.pos_embedding":
                weight_op = self.umodel.parameter(np_tensor=output, dtype=numpy_to_ufront_dtype(output.dtype), requires_grad=True, name=name)
                output = weight_op.get_output(0)
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
        input1 = node_to_output[node.input[0]]
        if node.input[1] in node_to_output:
            input2 = node_to_output[node.input[1]]
        elif node.input[1] in self.inputs:
            input2 = self.inputs[node.input[1]]
        else:
            assert 0, "Unable to obtain input2!"
            
        if type(input2) != Tensor:
            assert type(input2) == np.ndarray, "The given input is not an ndarray!"
            # operator = self.addTensor(input2, False, node.input[1])
            weight_op = self.umodel.parameter(np_tensor=input2.astype(np.float32), dtype=numpy_to_ufront_dtype(input2.dtype), requires_grad=True, name=node.input[1])
            # self.operators.append(operator)
            input2 = weight_op.get_output(0)

        if len(input1.shape) < 3 and len(input2.shape) < 3:
            return self.umodel.matmul(x = input1, y=input2, name=node.name)
        else:
            return self.umodel.batch_matmul(x = input1, y=input2, name=node.name)
        
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
        # output = io.BytesIO()
        # np.savez_compressed(output, x=np_tensor)
        # raw_bytes = str(output.getvalue().hex())
        
        operator = self.umodel.parameter(np_tensor=np_tensor, dtype=numpy_to_ufront_dtype(np_tensor.dtype), requires_grad=requires_grad, name=name)
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
        if type(input2) != Tensor:
            assert type(input2) == np.ndarray, "The given input is not an ndarray!"
            operator = self.addTensor(input2, False, node.input[1])
            self.operators.append(operator)
            input2 = operator.get_output(0)
        elif type(input1) != Tensor:
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
        if type(input) == Tensor:
            return self.umodel.cast(input = input, dtype=dtype, name=node.name)
        else:
            return input
    
    def handleDiv(self, node, node_to_output):
        input1 = node_to_output[node.input[0]]
        scalar = node_to_output[node.input[1]]
        if type(scalar) in [float, int, np.float32, np.float64, np.int32, np.int64, np.half]:
            return self.umodel.struediv(input = input1, scalar=scalar, name=node.name)
        else:
            assert type(scalar) == np.ndarray, "The given input is not an ndarray!"
            return self.umodel.divide(input = input1, operand=scalar.tolist(), name=node.name)

    
    def handleReduceMean(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        dims = []

        if len(node.input) > 1:
            if type(node_to_output[node.input[1]]) != np.ndarray:
                dims = [int(node_to_output[node.input[1]])]
            else:
                dims = list(node_to_output[node.input[1]])

        attribute = {x.name: x for x in node.attribute}
        keepdims = attribute["keepdims"].i > 0

        if "axes" in attribute:
            dims = attribute["axes"].ints
        elif "noop_with_empty_axes" in attribute:
            if attribute["noop_with_empty_axes"].i == 0 and len(dims) == 0:
                dims = list(range(0, len(input.shape)))
        elif len(dims) < 1:
            assert 0, "invalid axes for reduce mean!"

        return self.umodel.mean(
            input=input, dims=dims, keepdims=keepdims, name=node.name,
        )

    
    def handleSlice(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        starts = node_to_output[node.input[1]]
        ends = node_to_output[node.input[2]]
        axis = node_to_output[node.input[3]]
        if type(axis) == np.ndarray:
            output_shape = input.shape
            for i in axis:
                if ends[i] > output_shape[i]:
                    ends[i] = output_shape[i]

                output_shape[i] = ends[i] - starts[i]

            return self.umodel.slice_tensor(input=input, output_shape=list(output_shape), axis=list(axis), start=list(starts), end=list(ends), name=node.name)
    
        elif type(input) != list and type(input) != tuple:
            output_shape = input.shape
            output_shape[axis] = ends - starts
            if axis == -1:
                axis = len(input.shape) - 1
            return self.umodel.slice_tensor(input=input, output_shape=list(output_shape), axis=[axis], start=[starts], end=[ends], name=node.name)
        else:
            output_shape = input
            return ends - starts
        
    def handleGather(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        start = node_to_output[node.input[1]]
        attribute = {x.name: x for x in node.attribute}
        axis = attribute["axis"].i
        if type(axis) == int and (type(start) == int or type(start) == np.int32 or type(start) == np.int64) :
            output_shape = input.shape
            output_shape = output_shape[0:axis] + output_shape[axis+1:]
            return self.umodel.slice_tensor(input=input, output_shape=list(output_shape), axis=[axis], start=[start], end=[start+1], name=node.name)
        elif type(start) == np.ndarray and type(input) == list:
            return [input[i] for i in list(start)]
        else:
            assert 0, "Multidimentional gather not supported!"
    
    
    def handleLayerNormalization(self, node, node_to_output):
        input_tensor = node_to_output[node.input[0]]
        if node.input[1] in node_to_output:
            weight_tensor = node_to_output[node.input[1]]
            if type(weight_tensor) == np.int64 or type(weight_tensor) == np.int32:
                weight_shape = [input_tensor.shape[int(weight_tensor)]]
                eps=0.000001
            elif type(weight_tensor) == ONNXTensor or type(weight_tensor) == np.ndarray:
                weight_shape = weight_tensor.shape
                eps=0.000001
            else:
                shape = weight_tensor.type.tensor_type.shape.dim
                weight_shape = []
                for i in range(len(shape)):
                    weight_shape.append(shape[i].dim_value)
                attribute = {x.name: x for x in node.attribute}
                eps = attribute["epsilon"].f

        else:
            weight_shape = [input_tensor.shape[-1]]
            eps=0.000001

        return self.umodel.layer_norm(input=input_tensor, normalized_shape=list(weight_shape), 
                                  eps=eps, elementwise_affine=True, name=node.name)
    
    def handleLayerNorm(self, node, node_to_output):
        return self.handleLayerNormalization(node, node_to_output)
    
    def handleMultiHeadAttention(self, node, node_to_output):
        # inputs -> [x, weight_v, weight_q, weight_k, weight_o]
        q = node_to_output[node.input[0]]
        if len(node.input) > 5 and type(node.input[1]) != np.ndarray:
            k = node_to_output[node.input[1]]
            embed_dim=k[-1]
            weight_v =  np.array(node_to_output[node.input[2]])
            weight_q = np.array(node_to_output[node.input[3]])
            weight_k = np.array(node_to_output[node.input[4]])
            weight_o = np.array(node_to_output[node.input[5]])
        else:
            embed_dim = q.shape[-1]
            weight_v =  np.array(node_to_output[node.input[1]])
            weight_q = np.array(node_to_output[node.input[2]])
            weight_k = np.array(node_to_output[node.input[3]])
            weight_o = np.array(node_to_output[node.input[4]])

        operator_q = self.umodel.parameter(np_tensor=weight_q, dtype=numpy_to_ufront_dtype(weight_q.dtype), requires_grad=True, name=node.name + "_weight_q")
        operator_k = self.umodel.parameter(np_tensor=weight_k, dtype=numpy_to_ufront_dtype(weight_k.dtype), requires_grad=True, name=node.name + "_weight_k")
        operator_v = self.umodel.parameter(np_tensor=weight_v, dtype=numpy_to_ufront_dtype(weight_v.dtype), requires_grad=True, name=node.name + "_weight_v")
        operator_o = self.umodel.parameter(np_tensor=weight_o, dtype=numpy_to_ufront_dtype(weight_o.dtype), requires_grad=True, name=node.name + "_weight_o")
    
        batch_first=True
        dropout=0.0
        
        num_heads=12

        return self.umodel.multihead_attention(
            q=q,
            k=q,
            v=q,

            weight_q=operator_q.get_output(0),
            weight_k=operator_k.get_output(0),
            weight_v=operator_v.get_output(0),
            weight_o=operator_o.get_output(0),
            # bias_q=operator_bias_q.get_output(0),
            # bias_k=operator_bias_k.get_output(0),
            # bias_v=operator_bias_v.get_output(0),

            # bias_o=operator_bias_o.get_output(0),
            
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
            weight_transposed=False,
            name=node.name,
        )

    def handleMultiHeadDotProductAttention(self, node, node_to_output):
        return self.handleMultiHeadAttention(node, node_to_output)  
    
    def handleself_attention(self, node, node_to_output):
        return self.handleMultiHeadAttention(node, node_to_output) 
    
    def handleattention(self, node, node_to_output):
        return self.handleMultiHeadAttention(node, node_to_output) 
    
    def handleHardSigmoid(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        return self.umodel.hardsigmoid(input = input, name=node.name)

    def handleHardSwish(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        return self.umodel.hardswish(input = input, name=node.name)
    
    def handleClip(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        minimum = node_to_output[node.input[1]]
        maximum = node_to_output[node.input[2]]
        return self.umodel.clip(input = input, minimum=minimum, maximum=maximum, name=node.name)
    
    def handleErf(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        return self.umodel.erf(input=input, approximate=True, name=node.name)
    
    def handlePow(self, node, node_to_output):
        input = node_to_output[node.input[0]]
        pow = node_to_output[node.input[1]]
        return self.umodel.pow(input=input, pow=pow, name=node.name)
    
    def handleembeddings(self, node, node_to_output):
        weight = node_to_output[node.input[0]]
        input = node_to_output[node.input[1]]
        num_embeddings = weight.shape[0]
        embedding_dim = weight.shape[1]

        if type(weight) != Tensor:
            assert type(weight) == np.ndarray, "The given input is not an ndarray!"
            weight_op = self.umodel.parameter(np_tensor=weight, dtype=numpy_to_ufront_dtype(weight.dtype), requires_grad=True, name=node.input[0])
            weight = weight_op.get_output(0)
        
        return self.umodel.embedding(
            input=input,
            weight=weight,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            name=node.name,
        )
        
    def process_initializer(self, inputs, initializer):
        # print("Processing initializer...")
        for item in initializer:
            # print(item.dims, " ", item.data_type, " ", item.name, ": ", len(item.raw_data))
            datatype = onnx_to_ufront_dtype(item.data_type)
            if len(item.raw_data) > 0:
                values = self.unpack_rawdata(item.raw_data, datatype, item.dims, item.name)
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
        
    def convert_to_tensor(self, input):
        item = input.type
        dims = [x.dim_value for x in item.tensor_type.shape.dim]
        return self.umodel.create_tensor(dims, DataType.Float, True, input.name)
    
    def apply(self, input_tensors):
        node_to_output = OrderedDict()
        inputs_nodes = []
        if type(input_tensors) != list and type(input_tensors) != dict:
            assert 0, "Not a valid input type!"

        for i in range(len(self.model.graph.input)):
            input = self.model.graph.input[i]
            if input.name.find("input") >=0 or (i==0 and input.name.find("x") >=0) or input.name.find("onnx::") >=0:
                input_tensor = input_tensors[i] if type(input_tensors) == list else input_tensors[input.name]
                if type(input_tensor) != Tensor:
                    dtype=input_tensor.dtype
                    if input_tensor.__module__ == "torch":
                        dtype = input_tensor.numpy().dtype
                    input1 = np.ones(shape=input_tensor.shape, dtype=dtype)
                    input1[:] = input_tensor
                    inputname = input.name.replace("::", "") # "::" not valid in MLIR for name
                    input_tensor = Tensor(np_tensor=input1, dtype=numpy_to_ufront_dtype(input1.dtype), name=inputname) # convert to Rust f32 tensor
                self.inputs[input.name] = input_tensor
                inputs_nodes.append(input_tensor)
            elif input.name.find(".weight") < 0 and input.name.find(".bias") < 0 and input.name.find("onnx::") < 0:
                    self.inputs[input.name] = self.convert_to_tensor(input)
                    if input.name =="class_token" or input.name =="encoder.pos_embedding":
                        operator = self.addTensor(self.inputs[input.name].ndarray, True, input.name)
                        out = operator.get_output(0)
                        self.inputs[input.name] = out
            else:
                if input.name not in self.inputs:
                    self.inputs[input.name] = input
            # print("input ", i, ": ", input.name)

        if len(inputs_nodes) > 0:
            self.umodel.input(tensors=inputs_nodes, num_of_inputs=len(inputs_nodes))
            inputs_nodes = []

        self.process_initializer(self.inputs, self.model.graph.initializer)

        # for node in self.model.graph.node:
        #     print(node.name)

        self._fusion()
        op_fusion = ["MultiHeadDotProductAttention", ("attention", "output/Add"), "embeddings", "LayerNorm", "self_attention", "Gelu", "Dense"] if self.transformer else []
        for f in op_fusion:
            self._fusion_layer(f)

        outputs = OrderedDict()
        
        for output in self.model.graph.output:
            outputs[output.name] = output

        node_to_output.update(self.inputs)

        # for node in self.model.graph.node:
        #     print(node.name)

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
             if name in node_to_output.keys():
                tensor_outputs.append(node_to_output[name])
        return tensor_outputs if len(tensor_outputs) > 0 else node_to_output[next(reversed(node_to_output))]
    
    def get_output_operator(self):
        if len(self.operators) > 0:
            return self.operators[-1]
        else:
            return None
        
    def _fusion(self):
        flag = True
        dense_idx = 1
        while flag == True:
            idx = 0
            flag_found = False
            dense_node = None
            for node in self.model.graph.node:
                if node.op_type == 'MatMul' \
                    and node.name.find("/MultiHeadDotProductAttention") < 0  \
                    and node.name.find("/attention") < 0  \
                    and node.name.find("/self_attention") < 0:
                    output = node.output[0]
                    for add_node in self.model.graph.node:
                        if add_node.op_type == 'Add' and (add_node.input[0] == output or add_node.input[1] == output):
                            #print(node, add_node)
                            flag_found = True
                            dim = self.inputs[node.input[1]].shape[1]
                            dense_node = onnx.helper.make_node('Dense', inputs=node.input, outputs=[add_node.output[0]], out_dim=dim, name="Dense_"+str(dense_idx))
                            dense_idx += 1
                            #print(dense_node)
                            break
                    if flag_found and dense_node:
                        self.model.graph.node.insert(idx, dense_node)
                        self.model.graph.node.remove(add_node)
                        self.model.graph.node.remove(node)
                        break
                elif node.op_type == 'Gemm':
                    flag_found = True
                    dim = self.inputs[node.input[1]].shape[0]
                    dense_node = onnx.helper.make_node('Dense', inputs=node.input, outputs=[node.output[0]], out_dim=dim, name="Dense_"+str(dense_idx))
                    dense_idx += 1
                    self.model.graph.node.insert(idx, dense_node)
                    self.model.graph.node.remove(node)
                    break
                idx += 1
            flag = flag_found

    def _fusion_layer(self, name):
        stop_node = None
        if type(name) == tuple:
            stop_node = name[1]
            name = name[0]
        target_idx = 1
        REMOVED = []
        while True:
            idx = 0
            flag_found = False
            is_target = False
            matmul_weight_for_dense = None
            for node in self.model.graph.node:
                if node.name.find("/"+name) >= 0 \
                    and (not stop_node or node.name.find(stop_node) == -1): #for not standard fusion pattern

                    if (name == "attention" and node.name.find("LayerNorm") >0): #do not fuse LayerNorm into multihead_attention
                        idx += 1
                        continue
                    else:
                        flag_found = True
                        if not is_target:
                            is_target = True
                            target_input = node.input[:]

                        if name == "embeddings" and node.name.find("/embedding_lookup") >= 0 and len(target_input) < 2:
                            node.input.insert(1, target_input[0])
                            target_input = node.input[:] #tensorflow embedding weights
                        
                        if (name == "attention" or name == "MultiHeadDotProductAttention") and node.op_type == "MatMul":
                            # target_input.append()
                            # print(node.input)
                            # print("\n*****->", node.name)
                            for inp in node.input:
                                if inp in self.inputs.keys():
                                    if type(self.inputs[inp]) == np.ndarray and inp not in target_input:
                                        # print(inp, ": ", self.inputs[inp].shape)
                                        target_input.append(inp) #weights for multihead_attention
                                #     else:
                                #         print(inp, ": ", self.inputs[inp])
                                # else:
                                #     print("->", inp)

                        target_output = node.output
                        if name == "Dense" and node.op_type == "MatMul":
                            matmul_weight_for_dense = node.input[1:]
                        node.name = "#REMOVED"
                        REMOVED.append(node)
                else:
                    if is_target:
                        is_target = False
                        if name == "Dense" and matmul_weight_for_dense != None:
                            target_input.extend(list(matmul_weight_for_dense))
                            node = onnx.helper.make_node(name, inputs=target_input, outputs=target_output, name="Ext"+name+"_"+str(target_idx))
                        node = onnx.helper.make_node(name, inputs=target_input, outputs=target_output, name="Ext"+name+"_"+str(target_idx))

                        self.model.graph.node.insert(idx, node)
                        target_idx += 1
                        break

                idx += 1

            if not flag_found:
                break
        for node in REMOVED:
            self.model.graph.node.remove(node)

    
class ONNXModelKeras(ONNXModel):
    def __init__(self, onnx_model, umodel=None, transformer=False, pass_weights=False):
        super(ONNXModelKeras, self).__init__(onnx_model=onnx_model, umodel=umodel, pass_weights=pass_weights)
        self.transformer=transformer
        for node in onnx_model.graph.node:
            if node.name.find("u_front_keras/") != -1:
                node.name = node.name[node.name.find("keras_model/")+len("u_front_keras/")+1:]
            # print(node.name)
        for initializer in self.model.graph.initializer:
            if ('/bias' in initializer.name or '/BiasAdd/ReadVariableOp' in initializer.name )and 'dense' in initializer.name:
                # self.symbol_table[initializer.name] = self._create_initializer_tensor(ffconfig, ffmodel, initializer)
                pass
            else:
                tensor = ONNXTensor(initializer.name, initializer.dims, 2)
                self.inputs[initializer.name] = tensor
    #TODO fix constant
    def _create_initializer_tensor(self, ffconfig, input):
        if len(input.dims) == 1:
            dims = [ffconfig.batch_size, input.dims[0]]
            # print("dims", dims)
        else:
            assert 0
        tensor = self.umodel.create_tensor(dims, DataType.Float, True, "constant_tensor"+str(ONNXModel.const_tensor_idx))
        ONNXModel.const_tensor_idx += 1
        # print("create constant", input.name)
        return tensor

class UFrontONNX(ONNXModel):
    def __init__(
        self,
        onnx_model,
        batch_size,
        verbose=False,
        seq_length=None,
        transformer=False,
        simplify = False,
        pass_weights = False
    ): 
        self.umodel = Model() # Ufront Rust model
        super(UFrontONNX, self).__init__(onnx_model, self.umodel, simplify, pass_weights)
        # self.input_names = input_names
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.operators = []
        self._metrics = []
        self._loss = LossType.SPARSE_CATEGORICAL_CROSSENTROPY
        self._label_type = DataType.Int32
        self.transformer = transformer
        self.umodel.optimizer = Optimizer(params={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"})

    def __call__(self, inputs):
        return self.apply(inputs)


    def softmax(self, input, name="softmax"):
        softmax_op = self.umodel.softmax(input=input, name=name)
        self.operators.append(softmax_op)
        return softmax_op.get_output(0)

    def dump_ir(self):
        return self.umodel.dump_ir()
    
    def dump_tosa_ir(self):
        return self.umodel.dump_tosa_ir()
    

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

