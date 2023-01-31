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
from collections import OrderedDict
import logging
import onnx
import struct
from ..ufront import (OpType, ActiMode, AggrMode, PoolType, TensorF32, DataType, ParamSyncType, Initializer)

# logging.basicConfig(level=logging.DEBUG)

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
    def __init__(self, filename):
        if type(filename) == str:
            model = onnx.load(filename)
        else:
            model = filename
        # for node in model.graph.node:
        #     print(node)
        self.inputs = {}
        for input in model.graph.input:
            tensor = ONNXTensor(input.name, input.type.tensor_type.shape.dim, 1)
            self.inputs[input.name] = tensor
        self.outputs = {}
        for output in model.graph.output:
            self.outputs[output.name] = output
        self.model = model
        self.symbol_table = {}

    def handleAdd(self, ffmodel, node, node_to_output):
        input0 = node_to_output[node.input[0]]
        input1 = node_to_output[node.input[1]]
        return ffmodel.add(x=input0, y=input1, name=node.name)
        
    def handleSub(self, ffmodel, node, node_to_output):
        print(node)
        input0 = node_to_output[node.input[0]]
        input1 = node_to_output[node.input[1]]
        return ffmodel.subtract(x=input0, y=input1, name=node.name)
        
    def handleMul(self, ffmodel, node, node_to_output):
        input0 = node_to_output[node.input[0]]
        input1 = node_to_output[node.input[1]]
        return ffmodel.multiply(x=input0, y=input1, name=node.name)

    def handleConcat(self, ffmodel, node, node_to_output):
        inputs = [node_to_output[i] for i in node.input]
        attribute = {x.name: x for x in node.attribute}
        return ffmodel.concat(tensors=inputs, axis=attribute['axis'].i, name=node.name)

    def handleSplit(self, ffmodel, node, node_to_output):
        input = node_to_output[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        split = list(attribute['split'].ints)
        if 'axis' in attribute:
            axis = attribute['axis'].i
        else:
            axis = 0
        return ffmodel.split(input=input, sizes=split, axis=axis, name=node.name)

    def handleAveragePool(self, ffmodel, node, node_to_output):
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
            assert 0, "padding is missing"
        return ffmodel.pool2d(input=input, kernel_h=kernel[0], kernel_w=kernel[1], 
                        stride_h=stride[0], stride_w=stride[1], padding_h=padding[0], padding_w=padding[1], 
                        pool_type=PoolType.POOL_AVG, name=node.name)

    def handleGlobalAveragePool(self,ffmodel,node, node_to_output):
        input = node_to_output[node.input[0]]
        return ffmodel.pool2d(input=input, kernel_h=input.dims[2], kernel_w=input.dims[3], 
                            stride_h=1, stride_w=1, padding_h=0, padding_w=0, pool_type=PoolType.POOL_AVG, name=node.name)

    def handleBatchNormalization(self, ffmodel, node, node_to_output):
        input = node_to_output[node.input[0]]
        return ffmodel.batch_norm(input=input)

    def handleConv(self, ffmodel, node, node_to_output):
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
            assert 0, "padding is missing"
        group = attribute["group"].i
        out_channels = self.inputs[node.input[1]].dims[0]
        return ffmodel.conv2d(input=input, out_channels=out_channels, kernel_h=kernel[0], 
                kernel_w=kernel[1], stride_h=stride[0], stride_w=stride[1], 
                padding_h=padding[0], padding_w=padding[1], 
                activation=ActiMode.AC_MODE_NONE, groups=group, name=node.name)

    def handleDropout(self, ffmodel, node, node_to_output):
        input = node_to_output[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        rate = attribute["ratio"].f
        seed = 0
        return ffmodel.dropout(input=input, rate=rate, seed=0, name=node.name)

    def handleFlatten(self, ffmodel, node, node_to_output):
        input = node_to_output[node.input[0]]
        return ffmodel.flat(input=input, name=node.name)

    # def handleGemm(self, ffmodel, node):
    #     input = self.symbol_table[node.input[0]]
    #     dim = self.inputs[node.input[1]].dims[0]
    #     output = ffmodel.dense(input, dim, name=node.name)
    #     self.symbol_table[node.output[0]] = output
    #     logging.debug("ffmodel.dense({}, {}, name={})".format(node.input[0], dim, node.name))
        
    def handleDense(self, ffmodel, node, node_to_output):
        input = node_to_output[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        dim = attribute["out_dim"].i
        return ffmodel.dense(input=input, out_dim=dim, name=node.name)

    def handleMaxPool(self, ffmodel, node, node_to_output):
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
            assert 0, "padding is missing"
        return ffmodel.pool2d(input=input, kernel_h=kernel[0], kernel_w=kernel[1], 
                            stride_h=stride[0], stride_w=stride[1], 
                            padding_h=padding[0], padding_w=padding[1], name=node.name)

    def handleRelu(self, ffmodel, node, node_to_output):
        input = node_to_output[node.input[0]]
        return ffmodel.relu(input=input, name=node.name)

    def handlePad(self, ffmodel, node, node_to_output):
        input = node_to_output[node.input[0]]
        output = input
        return output

    def handleSoftmax(self, ffmodel, node, node_to_output):
        input = node_to_output[node.input[0]]
        return ffmodel.softmax(input=input, name=node.name)

    def handleReshape(self, ffmodel, node, node_to_output):
        input = node_to_output[node.input[0]]
        shape = node_to_output[node.input[1]]
        return ffmodel.reshape(input=input, shape=list(shape.int64_data), name=node.name)
    
    def handleCast(self, ffmodel, node, node_to_output):
        # TODO: add cast
        input = node_to_output[node.input[0]]
        return input
        
    def handleUnsqueeze(self, ffmodel, node, node_to_output):
        # TODO: add unsqueeze
        input = node_to_output[node.input[0]]
        attribute = {x.name: x for x in node.attribute}
        axes = attribute["axes"].ints
        return input

    #TODO fix constant  
    def handleConstant(self, ffmodel, node, node_to_output):
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
            output = ffmodel.create_constant(tensor,dims, value[0], data_type)
            logging.warning("ffmodel.create_constant: {}, {}, {}".format(dims, value[0], data_type))
        else:
            output = value[0]
        return output
        
    def handleRange(self, ffmodel, node, node_to_output):
        # TODO: add range
        start = node_to_output[node.input[0]]
        limit = node_to_output[node.input[1]]
        delta = node_to_output[node.input[2]]
        return start

    def apply(self, ffmodel, input_tensors):
        self._fusion()
        node_to_output = OrderedDict()

        inputs = {}
        for i in range(len(self.model.graph.input)):
            input = self.model.graph.input[i]
            if input.name.find("input") >=0:
            # tensor = ONNXTensor(input.name, input.type.tensor_type.shape.dim, 1)
                inputs[input.name] = input_tensors[i]

        outputs = OrderedDict()

        for output in self.model.graph.output:
            outputs[output.name] = output

        node_to_output.update(inputs)
        input_index = 0
        operators = []
        # for node in self.model.graph.node:
        #     print(node.name)

        for node in self.model.graph.node:
            handler_name = 'handle' + node.op_type
            if hasattr(self, handler_name):
                handler = getattr(self, handler_name)
                operator = handler(ffmodel, node, node_to_output)
                operators.append(operator)
                if node.op_type == "Split":
                    node_output = []
                    for i in range(operator.num_of_outputs()):
                        node_output.append(operator.get_output(i))
                else:
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
        return tensor_outputs, operators
        
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
        
        # for node in self.model.graph.node:
        #     print(node)
        
class ONNXModelKeras(ONNXModel):
    def __init__(self, filename, ffconfig=None, ffmodel=None):
        super(ONNXModelKeras, self).__init__(filename)
        for initializer in self.model.graph.initializer:
            if ('/bias' in initializer.name or '/BiasAdd/ReadVariableOp' in initializer.name )and 'dense' in initializer.name:
                # self.symbol_table[initializer.name] = self._create_initializer_tensor(ffconfig, ffmodel, initializer)
                pass
            else:
                tensor = ONNXTensor(initializer.name, initializer.dims, 2)
                self.inputs[initializer.name] = tensor
        
    # def handleMatMul(self, ffmodel, node):
    #     print("########################################I am in Keras MatMul")
    #     input = self.symbol_table[node.input[0]]
    #     dim = self.inputs[node.input[1]].dims[1]
    #     output = ffmodel.dense(input, dim, use_bias=False, name=node.name)
    #     self.symbol_table[node.output[0]] = output
    #     logging.debug("ffmodel.dense({}, {})".format(node.input[0], dim))
        
    def handleTranspose(self, ffmodel, node):
        input = self.symbol_table[node.input[0]]
        self.symbol_table[node.output[0]] = input
        logging.debug("ffmodel.tranpose({})".format(node.input[0]))
        
    def handleReshape(self, ffmodel, node):
        print("########################################I am in Keras Reshape")
        self.handleFlatten(ffmodel, node)

    #TODO fix constant
    def _create_initializer_tensor(self, ffconfig, ffmodel, input):
        if len(input.dims) == 1:
            dims = [ffconfig.batch_size, input.dims[0]]
            print("dims", dims)
        else:
            assert 0
        tensor = ffmodel.create_constant(dims, 0.0, DataType.FLOAT)
        print("create constant", input.name)
        return tensor
