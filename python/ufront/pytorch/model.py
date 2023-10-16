# Copyright 2023 Enflame Tech, Stanford University (legacy code)
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
from enum import Enum
from typing import List
import typing
import numpy as np
import math
import io

from ..ufront import (OpType, ActiMode, AggrMode, PoolType, Tensor, DataType, ParamSyncType, Initializer)
from ..ufront import Model, PyOperator, Tensor, Optimizer, LossType, MetricsType, WeightType #Rust frontend
from ..utils import list_product, onnx_to_ufront_dtype, numpy_to_ufront_dtype, ufront_to_numpy_dtype, torch_to_ufront_dtype

try:
    import torch
    from torch.fx.immutable_collections import immutable_dict
except:
    print("You need to first install pytorch before using pytorch models!")


IR_DELIMITER = "; "
INOUT_NODE_DELIMITER = ','

class Comparator(Enum):
    EQ = 0
    GEQ = 1


class Node():
    """This base class represents a node in the model computational graph (to
    be used internally for PyTorch to FlexFlow conversion)."""
    def __init__(self, node):
        self.name = node.name
        self.op_type = None
        self._ir_string = None

    def __repr__(self):
        return f"{type(self).__name__}: {self.name}"

    def parse_inoutnodes(self, nodes):
        """Parses the given input or output nodes, and returns a string
        representation."""
        if nodes is None:
            return ""
        assert type(nodes) is list or type(nodes) is tuple or \
            type(nodes) is dict
        return INOUT_NODE_DELIMITER.join([node.name for node in nodes]) + \
            INOUT_NODE_DELIMITER

    def assert_num_args(self, num_args, cmp):
        if cmp == Comparator.EQ:
            assert len(self.innodes) == num_args, \
                f"{self.op_type.as_str()} expects {num_args}" \
                "arguments"
        elif cmp == Comparator.GEQ:
            assert len(self.innodes) >= num_args, \
                f"{self.op_type.as_str()} expects at least " \
                f"{num_args} arguments"

    @property
    def ir_string(self):
        """Returns the string representation of the node."""
        if self._ir_string is None:
            self.parse()
        return self._ir_string

    def parse(self):
        """Parses the node to populate ``self._ir_string`` with a string
        representation."""
        raise NotImplementedError

    class StringData():
        """Wraps the data in the string representation returned by
        ``self.string``."""
        def __init__(self, string):
            self.items = [i.strip() for i in string.strip().split(';')]
            n = len(self.items)
            self.name = self.items[0]
            if n < 4:
                assert n == 2
                self.op_type = OpType.as_enum(self.items[1])
                assert self.op_type == OpType.ATTRIBUTE
            else:
                self.innodes = self.get_inout_nodes(self.items[1])
                self.outnodes = self.get_inout_nodes(self.items[2])
                self.op_type = OpType.as_enum(self.items[3])

        def get_inout_nodes(self, inout_string):
            node_list = inout_string.split(INOUT_NODE_DELIMITER)
            filtered_node_list = []
            for node in node_list:
                node_stripped = node.strip()
                if node_stripped != "":
                    filtered_node_list.append(node_stripped)
            return filtered_node_list

    def __call__(self, umodel, node_to_output):
        assert 0, f"`__call__()` is not implemented for {self.name}"

    @staticmethod
    def string_to_node_class(string):
        data = Node.StringData(string)
        op_type = data.op_type
        if op_type == OpType.CONV2D: return Conv2dNode
        elif op_type == OpType.EMBEDDING: return EmbeddingNode
        elif op_type == OpType.POOL2D: return Pool2dNode
        elif op_type == OpType.LINEAR: return LinearNode
        elif op_type == OpType.SOFTMAX: return SoftmaxMNode
        elif op_type == OpType.CONCAT: return ConcatNode
        elif op_type == OpType.FLAT: return FlattenNode
        elif op_type == OpType.BATCH_NORM: return BatchNorm2dNode
        elif op_type == OpType.RELU: return ReLUMNode
        elif op_type == OpType.SIGMOID: return SigmoidNode
        elif op_type == OpType.TANH: return TanhMNode
        elif op_type == OpType.ELU: return ELUNode
        elif op_type == OpType.DROPOUT: return DropoutMNode
        elif op_type == OpType.BATCH_MATMUL: return BatchMatMulNode
        elif op_type == OpType.SPLIT: return SplitChunkNode
        elif op_type == OpType.SPLIT: return SplitChunkNode
        elif op_type == OpType.RESHAPE: return ReshapeNode
        elif op_type == OpType.TRANSPOSE: return TransposeNode
        elif op_type == OpType.ADD: return AddNode
        elif op_type == OpType.MULTIPLY: return MulNode
        elif op_type == OpType.POW: return PowNode
        elif op_type == OpType.MEAN: return MeanNode
        elif op_type == OpType.RSQRT: return RsqrtNode
        elif op_type == OpType.INPUT: return InputNode
        elif op_type == OpType.OUTPUT: return OutputNode
        elif op_type == OpType.GETITEM: return GetItemNode
        elif op_type == OpType.GETATTR: return GetAttrNode
        elif op_type == OpType.EXPAND: return ExpandNode
        elif op_type == OpType.LAYER_NORM: return LayerNormNode
        elif op_type == OpType.FLOOR_DIVIDE: return ScalarFloorDivNode
        elif op_type == OpType.IDENTITY: return IdentityNode
        elif op_type == OpType.GELU: return GeluMNode
        elif op_type == OpType.PERMUTE: return PermuteNode
        elif op_type == OpType.SCALAR_MULTIPLY: return ScalarMulNode
        elif op_type == OpType.SCALAR_FLOORDIV: return ScalarFloorDivNode
        elif op_type == OpType.SCALAR_ADD: return ScalarAddNode
        elif op_type == OpType.SCALAR_SUB: return ScalarSubNode
        elif op_type == OpType.SCALAR_TRUEDIV: return ScalarTrueDivNode
        elif op_type == OpType.FLOAT: return FloatNode
        elif op_type == OpType.CONTIGUOUS: return ContiguousNode
        elif op_type == OpType.TO: return ToNode
        elif op_type == OpType.UNSQUEEZE: return UnsqueezeNode
        elif op_type == OpType.TYPE_AS: return TypeAsNode
        elif op_type == OpType.VIEW: return ViewNode
        elif op_type == OpType.ATTRIBUTE: return AttributeNode
        elif op_type == OpType.EQ: return EqNode
        assert 0, f"Unsupported op type: {op_type}"

    @staticmethod
    def torch_to_dtype(torch_dtype):
        if torch_dtype in (torch.float32, torch.float, "float32", "float"):
            return DataType.Float
        elif torch_dtype in (torch.float64, torch.double, "float64", "double"):
            return DataType.Double
        elif torch_dtype in (torch.int32, torch.int, "int32", "int"):
            return DataType.Int32
        elif torch_dtype in (torch.int64, torch.long, "int64", "long"):
            return DataType.Int64
        elif torch_dtype in (torch.float16, torch.half, "float16", "half"):
            return DataType.Half
        elif torch_dtype in (torch.bool, "bool", "BOOL", "boolean", "Boolean"):
            return DataType.Bool
        else:
            assert 0, f"Unknown dtype: {torch_dtype}"


class ModuleNode(Node):
    def __init__(self, node, module):
        super().__init__(node)
        self.innodes = node.args
        self.outnodes = node.users
        self.module = module

    @staticmethod
    def construct_node(node, module):
        if type(module) is torch.nn.modules.linear.Linear:
            return LinearNode(node, module)
        elif type(module) is torch.nn.modules.conv.Conv2d:
            return Conv2dNode(node, module)
        elif type(module) is torch.nn.modules.pooling.MaxPool2d:
            return Pool2dNode(node, PoolType.POOL_MAX, module)
        elif type(module) is torch.nn.modules.pooling.AvgPool2d:
            return Pool2dNode(node, PoolType.POOL_AVG, module)
        elif type(module) is torch.nn.modules.pooling.AdaptiveAvgPool2d:
            return AdaptivePool2dNode(node, PoolType.POOL_ADAPTIVE, module)
        elif type(module) is torch.nn.modules.batchnorm.BatchNorm2d:
            return BatchNorm2dNode(node, module)
        elif type(module) is torch.nn.modules.dropout.Dropout:
            return DropoutMNode(node, module)
        elif type(module) is torch.nn.modules.flatten.Flatten:
            return FlattenNode(node, module)
        elif type(module) is torch.nn.modules.activation.ReLU:
            return ReLUMNode(node, module)
        elif type(module) is torch.nn.modules.activation.Sigmoid:
            return SigmoidNode(node, module)
        elif type(module) is torch.nn.modules.activation.Tanh:
            return TanhMNode(node, module)
        elif type(module) is torch.nn.modules.activation.ELU:
            return ELUNode(node, module)
        elif type(module) is torch.nn.modules.activation.Hardswish:
            return HardswishNode(node, module)
        elif type(module) is torch.nn.modules.activation.Hardsigmoid:
            return HardsigmoidNode(node, module)
        elif type(module) is torch.nn.modules.activation.SiLU:
            return SiLUNode(node, module)
        elif type(module) is torch.nn.modules.activation.Softmax:
            return SoftmaxMNode(node, module)
        elif type(module) is torch.nn.modules.normalization.LayerNorm:
            return LayerNormNode(node, module)
        elif type(module) is torch.nn.Identity:
            return IdentityNode(node, module)
        elif type(module) is torch.nn.GELU:
            return GeluMNode(node, module)
        elif isinstance(module, torch.nn.Embedding):
            return EmbeddingNode(node, module)
        elif isinstance(module, torch.nn.MultiheadAttention):
            return MultiheadAttentionNode(node, module)
        else:
            # assert 0, f"Unknown module: {module}"
            return None


class LinearNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.LINEAR
        self.acti_mode = ActiMode.AC_MODE_NONE
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        s.append(str(self.module.out_features))
        s.append(str(ActiMode.AC_MODE_NONE.as_int()))
        if self.module.bias is not None:
            s.append("1")
        else:
            s.append("0")
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        if umodel.weight_type == WeightType.INTERNAL:
            return umodel.dense(
            input=input_tensor,
            # weight=operator.get_output(0),
            out_dim=self.module.out_features,
            activation=self.acti_mode,
            use_bias=(self.module.bias is not None),
            name=self.name,
            )
        else:
            requires_grad = self.module.weight.requires_grad
            weight = self.module.weight.detach().numpy() if requires_grad \
                else self.module.weight.numpy()
            operator = umodel.parameter(np_tensor=weight, dtype=numpy_to_ufront_dtype(weight.dtype), requires_grad=requires_grad, name=self.name + "_weight")
            if self.module.bias != None:
                requires_grad = self.module.bias.requires_grad
                bias = self.module.bias.detach().numpy() if requires_grad \
                    else self.module.bias.numpy()
                bias_op = umodel.parameter(np_tensor=bias, dtype=numpy_to_ufront_dtype(bias.dtype), requires_grad=requires_grad, name=self.name + "_bias")
                return umodel.dense(
                    input=input_tensor,
                    weight=operator.get_output(0),
                    bias=bias_op.get_output(0),
                    out_dim=self.module.out_features,
                    activation=self.acti_mode,
                    name=self.name,
                )
            else:
                return umodel.dense(
                    input=input_tensor,
                    weight=operator.get_output(0),
                    out_dim=self.module.out_features,
                    activation=self.acti_mode,
                    use_bias=(self.module.bias is not None),
                    name=self.name,
                )


class Conv2dNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.CONV2D
        self.acti_mode = ActiMode.AC_MODE_NONE
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        s.append(str(self.module.out_channels))
        s.append(str(self.module.kernel_size[0]))
        s.append(str(self.module.kernel_size[1]))
        s.append(str(self.module.stride[0]))
        s.append(str(self.module.stride[1]))
        s.append(str(self.module.padding[0]))
        s.append(str(self.module.padding[1]))
        s.append(str(self.acti_mode.as_int()))
        s.append(str(self.module.groups))
        if self.module.bias is not None:
            s.append("1")
        else:
            s.append("0")
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        if umodel.weight_type == WeightType.INTERNAL:
            return umodel.conv2d(
                input=input_tensor,
                # weight=operator.get_output(0),
                out_channels=self.module.out_channels,
                kernel=[self.module.kernel_size[0], self.module.kernel_size[1]],
                stride=[self.module.stride[0], self.module.stride[1]],
                pad=[self.module.padding[0], self.module.padding[0], self.module.padding[1], self.module.padding[1]],
                activation=self.acti_mode,
                groups=self.module.groups,
                use_bias=(self.module.bias is not None),
                name=self.name,
            )     
        else:      
            
            requires_grad = self.module.weight.requires_grad
            weight = self.module.weight.detach().numpy() if requires_grad \
                else self.module.weight.numpy()
            
            operator = umodel.parameter(np_tensor=weight, dtype=numpy_to_ufront_dtype(weight.dtype), requires_grad=requires_grad, name=self.name + "_weight")
            
            if self.module.bias != None:
                requires_grad = self.module.weight.requires_grad
                bias = self.module.bias.detach().numpy() if requires_grad \
                    else self.module.bias.numpy()
                
                bias_op = umodel.parameter(np_tensor=bias, dtype=numpy_to_ufront_dtype(bias.dtype), requires_grad=requires_grad, name=self.name + "_weight")
                return umodel.conv2d(
                    input=input_tensor,
                    weight=operator.get_output(0),
                    bias=bias_op.get_output(0),
                    out_channels=self.module.out_channels,
                    kernel=[self.module.kernel_size[0], self.module.kernel_size[1]],
                    stride=[self.module.stride[0], self.module.stride[1]],
                    pad=[self.module.padding[0], self.module.padding[0], self.module.padding[1], self.module.padding[1]],
                    activation=self.acti_mode,
                    groups=self.module.groups,
                    name=self.name,
                )
            else:
                return umodel.conv2d(
                    input=input_tensor,
                    weight=operator.get_output(0),
                    out_channels=self.module.out_channels,
                    kernel=[self.module.kernel_size[0], self.module.kernel_size[1]],
                    stride=[self.module.stride[0], self.module.stride[1]],
                    pad=[self.module.padding[0], self.module.padding[0], self.module.padding[1], self.module.padding[1]],
                    activation=self.acti_mode,
                    groups=self.module.groups,
                    use_bias=(self.module.bias is not None),
                    name=self.name,
                )

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

class Pool2dNode(ModuleNode):
    def __init__(self, node, pool_type, module):
        super().__init__(node, module)
        self.op_type = OpType.POOL2D
        self.pool_type = pool_type
        self.acti_mode = ActiMode.AC_MODE_NONE
        if module == None:
            self.module = objectview(node.kwargs)
            if "padding" not in node.kwargs:
                self.module.padding = 0
            if len(self.innodes) > 1:
                if "kernel_size" not in node.kwargs and type(self.innodes[1])==int:
                    self.module.kernel_size = self.innodes[1]
                self.innodes = (self.innodes[0],)
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        # FIXME MaxPool2d supports ceil_mode
        s.append(str(self.module.kernel_size))
        s.append(str(self.module.stride))
        s.append(str(self.module.padding))
        s.append(str(self.pool_type.as_int()))
        s.append(str(self.acti_mode.as_int()))
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return umodel.pool2d(
            input=input_tensor,
            kernel=[self.module.kernel_size, self.module.kernel_size],
            stride=[self.module.stride, self.module.stride],
            pad=[self.module.padding, self.module.padding],
            pool_type=self.pool_type,
            activation=self.acti_mode,
            name=self.name,
        )


class AdaptivePool2dNode(ModuleNode):
    def __init__(self, node, pool_type, module):
        super().__init__(node, module)
        self.op_type = OpType.POOL2D
        self.pool_type = pool_type
        self.acti_mode = ActiMode.AC_MODE_NONE
        if module == None:
            self.module = objectview(node.kwargs)
            if len(self.innodes) > 1:
                if "output_size" not in node.kwargs and (type(self.innodes[1])==int or type(self.innodes[1])==tuple or type(self.innodes[1])==list):
                    self.module.output_size = self.innodes[1]
                self.innodes = (self.innodes[0],)
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        # FIXME Fix kernel, stride, and padding
        s += ["3", "1", "0"]
        s.append(str(self.pool_type.as_int()))
        s.append(str(self.acti_mode.as_int()))
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        output_size = self.module.output_size if type(self.module.output_size) != int else [self.module.output_size, self.module.output_size]
        return umodel.pool2d(
            input=input_tensor,
            output_size=list(output_size),
            pool_type=self.pool_type,
            activation=self.acti_mode,
            name=self.name,
        )


class BatchNorm2dNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.BATCH_NORM
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())

        s.append(str(self.module.eps))
        s.append(str(self.module.momentum))
        s.append(str(self.module.affine))
        s.append(str(self.module.track_running_stats))

        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]

        if umodel.weight_type == WeightType.INTERNAL:
            return umodel.batch_norm(
                input=input_tensor,
                # weight=weight_op.get_output(0),
                # bias=bias_op.get_output(0),
                eps=float(self.module.eps), momentum=self.module.momentum, affine=self.module.affine,
                track_running_stats=self.module.track_running_stats,
                name=self.name,
            )

        requires_grad = self.module.weight.requires_grad
        weight = self.module.weight.detach().numpy() if requires_grad \
            else self.module.weight.numpy()
        weight = weight.reshape((1, weight.shape[0], 1, 1))
        weight_op = umodel.parameter(np_tensor=weight, dtype=numpy_to_ufront_dtype(weight.dtype), requires_grad=requires_grad, name=self.name + "_weight")

        requires_grad = self.module.bias.requires_grad
        bias = self.module.bias.detach().numpy() if requires_grad \
            else self.module.bias.numpy()
        bias = bias.reshape((1, bias.shape[0], 1, 1))
        
        bias_op = umodel.parameter(np_tensor=bias, dtype=numpy_to_ufront_dtype(bias.dtype), requires_grad=requires_grad, name=self.name + "_weight")
        if self.module.training:
            return umodel.batch_norm(
                input=input_tensor,
                weight=weight_op.get_output(0),
                bias=bias_op.get_output(0),
                eps=float(self.module.eps), momentum=self.module.momentum, affine=self.module.affine,
                track_running_stats=self.module.track_running_stats,
                name=self.name,
            )
        else:
            requires_grad = self.module.running_mean.requires_grad
            running_mean = self.module.running_mean.detach().numpy() if requires_grad \
                else self.module.running_mean.numpy()
            running_mean = running_mean.reshape((1, running_mean.shape[0], 1, 1))
            
            running_mean_op = umodel.parameter(np_tensor=running_mean, dtype=numpy_to_ufront_dtype(running_mean.dtype), requires_grad=requires_grad, name=self.name + "_weight")
            
            requires_grad = self.module.running_var.requires_grad
            running_var = self.module.running_var.detach().numpy() if requires_grad \
                else self.module.running_var.numpy()
            running_var = running_var.reshape((1, running_var.shape[0], 1, 1))
            
            running_var_op = umodel.parameter(np_tensor=running_var, dtype=numpy_to_ufront_dtype(running_var.dtype), requires_grad=requires_grad, name=self.name + "_weight")
            return umodel.batch_norm(
                input=input_tensor,
                weight=weight_op.get_output(0),
                bias=bias_op.get_output(0),
                mean=running_mean_op.get_output(0),
                variance=running_var_op.get_output(0),
                eps=float(self.module.eps), momentum=self.module.momentum, affine=self.module.affine,
                track_running_stats=self.module.track_running_stats,
                name=self.name,
            )


class SoftmaxMNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.SOFTMAX
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return umodel.softmax(
            input=input_tensor, name=self.name,
        )


class DropoutMNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.DROPOUT
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        s.append(str(self.module.p))
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        rate = self.module.p
        return umodel.dropout(
            input=input_tensor, rate=rate, seed=0, training = self.module.training, name=self.name,
        )


class FlattenNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.FLAT

        if module == None:
            self.module = objectview(node.kwargs)
            if len(self.innodes) > 1:
                if "start_dim" not in node.kwargs:
                    if type(self.innodes[1])==int and self.innodes[1] < 4:
                        self.module.start_dim = self.innodes[1]
                    else:
                        self.module.start_dim = 0
                self.innodes = (self.innodes[0],)

            if "end_dim" not in node.kwargs:
                self.module.end_dim = -1

        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        s.append(str(self.module.start_dim))
        s.append(str(self.module.end_dim))

        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return umodel.flat(input=input_tensor, start_dim=self.module.start_dim, end_dim=self.module.end_dim, name=self.name)


class ReLUMNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.RELU
        self.inplace = module.inplace
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        s.append(str(self.inplace))
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return umodel.relu(input=input_tensor, name=self.name, inplace=self.inplace)


class IdentityNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.IDENTITY
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return umodel.identity(
            input=input_tensor,
            name=self.name,
        )


class GeluMNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.GELU
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return umodel.gelu(input=input_tensor, approximate=self.module.approximate!=None, name=self.name)


class LayerNormNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.LAYER_NORM

        if module == None:
            self.module = objectview(node.kwargs)
            if len(self.innodes) > 1:
                if "normalized_shape" not in node.kwargs and (type(self.innodes[1])==int or type(self.innodes[1])==tuple or type(self.innodes[1])==list):
                    self.module.normalized_shape = [self.innodes[1]] if type(self.innodes[1])==int else list(self.innodes[1])
                if "elementwise_affine" not in node.kwargs:
                    self.module.elementwise_affine = True
                self.innodes = (self.innodes[0],)
        self.module.normalized_shape = list(self.module.normalized_shape)
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        s.append(str(self.module.normalized_shape))
        s.append(str(self.module.eps))
        s.append(str(self.module.elementwise_affine))

        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        requires_grad = self.module.weight.requires_grad
        weight = self.module.weight.detach().numpy() if requires_grad \
            else self.module.weight.numpy()
        # weight = weight.reshape((1, weight.shape[0], 1, 1))
        weight_op = umodel.parameter(np_tensor=weight, dtype=numpy_to_ufront_dtype(weight.dtype), requires_grad=requires_grad, name=self.name + "_weight")

        requires_grad = self.module.bias.requires_grad
        bias = self.module.bias.detach().numpy() if requires_grad \
            else self.module.bias.numpy()
        # bias = bias.reshape((1, bias.shape[0], 1, 1))
        bias_op = umodel.parameter(np_tensor=bias, dtype=numpy_to_ufront_dtype(bias.dtype), requires_grad=requires_grad, name=self.name + "_bias")


        return umodel.layer_norm(input=input_tensor, 
                                 weight=weight_op.get_output(0),
                                 bias=bias_op.get_output(0),
                                 normalized_shape=list(self.module.normalized_shape), 
                                  eps=float(self.module.eps), elementwise_affine=self.module.elementwise_affine, name=self.name)


class T5LayerNormNode(Node):
    """
    This coalesces the ``T5LayerNorm`` primitive operation nodes into a single
    node to forward to FlexFlow's layer norm operation.

    NOTE: This forwarding deviates from the ``T5LayerNorm`` implementation,
    which does not subtract the mean or add a bias.
    """
    def __init__(self, to_node, mul_node):
        """
        The symbolic trace of ``T5LayerNorm`` follows the sequence ``to()``,
        ``pow()``, ``mean()``, ``scalar_add()``, ``rsqrt()``, ``multiply()``,
        attribute, and ``multiply()``.

        Args:
            name (str): Name for the node.
            to_node (ToNode): Node giving the first operation in the layer norm
                sequence.
            mul_node (MultiplyNode): Node giving the last operation in the
                layer norm sequence.
        """
        assert isinstance(to_node, ToNode)
        assert isinstance(mul_node, MulNode)
        self._ir_string = None
        # Take the input/output from the first/last nodes in the op sequence
        self.innodes = (to_node.innodes[0],)
        self.outnodes = mul_node.outnodes
        # Adopt the last node's name to respect later dependencies
        self.name = mul_node.name
        self.op_type = OpType.LAYER_NORM

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        # Normalize over the last dimension
        axes = [len(input_tensor.dims) - 1]
        return umodel.layer_norm(
            input=input_tensor,
            axes=axes,
            elementwise_affine=True,
            eps=1e-6,
            name=self.name,
        )


class SigmoidNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.SIGMOID
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return umodel.sigmoid(
            input=input_tensor,
            name=self.name,
        )


class TanhMNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.TANH
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return umodel.tanh(input=input_tensor, name=self.name)


class ELUNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.ELU
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return umodel.elu(input=input_tensor, name=self.name)

class HardswishNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.HARDSWISH
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return umodel.hardswish(input=input_tensor, name=self.name)

class HardsigmoidNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.HARDSIGMOID
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return umodel.hardsigmoid(input=input_tensor, name=self.name)

class SiLUNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.SILU
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return umodel.silu(input=input_tensor, name=self.name)
        
class EmbeddingNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.EMBEDDING
        self.assert_num_args(1, Comparator.GEQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        s.append(str(self.module.num_embeddings))
        s.append(str(self.module.embedding_dim))
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        weight = None
        input_tensor = node_to_output[self.innodes[0].name]
        if self.module != None:
            num_embeddings = self.module.num_embeddings
            embedding_dim = self.module.embedding_dim
            assert type(num_embeddings) is int
            assert type(embedding_dim) is int

            requires_grad = self.module.weight.requires_grad
            weight = self.module.weight.detach().numpy() if requires_grad \
                else self.module.weight.numpy()
        elif len(self.innodes) > 1:
            weight = node_to_output[self.innodes[1].name]
            num_embeddings = weight.shape[0]
            embedding_dim = weight.shape[1]

        if type(weight) == np.ndarray or weight != None:
            if type(weight) != Tensor:
                weight_op = umodel.parameter(np_tensor=weight, dtype=numpy_to_ufront_dtype(weight.dtype), requires_grad=True, name=self.name + "_weight")
            return umodel.embedding(
                input=input_tensor,
                weight=weight_op.get_output(0) if type(weight) != Tensor else weight,
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                name=self.name,
            )
        else:
            # return umodel.embedding(
            #     input=input_tensor,
            #     num_embeddings=num_embeddings,
            #     embedding_dim=embedding_dim,
            #     # aggr=AggrMode.AGGR_MODE_NONE,
            #     name=self.name,
            # )
            assert 0, "Invalid argument for embedding, missing weight or num_embeddings & embedding_dim!"

class MultiheadAttentionNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.innodes = []
        if "query" in node.kwargs:
            self.innodes.append(node.kwargs["query"])
        if "key" in node.kwargs:
            self.innodes.append(node.kwargs["key"])
        if "value" in node.kwargs:
            self.innodes.append(node.kwargs["value"])

        self.innodes = tuple(self.innodes)
        self.op_type = OpType.MULTIHEAD_ATTENTION
        self.assert_num_args(3, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        s.append(str(self.module.embed_dim))
        s.append(str(self.module.num_heads))
        s.append(str(self.module.drop_out))
        s.append(str(self.module.batch_first))
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        q = node_to_output[self.innodes[0].name]
        k = node_to_output[self.innodes[1].name]
        v = node_to_output[self.innodes[2].name]

        embed_dim = self.module.embed_dim
        num_heads = self.module.num_heads
        dropout = self.module.dropout
        batch_first = self.module.batch_first
        if umodel.weight_type == WeightType.INTERNAL:
            return umodel.multihead_attention(
                q=q,
                k=k,
                v=v,
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=float(dropout),
                batch_first=batch_first,
                name=self.name,
            )
        else:
            assert self.module.out_proj.weight != None, "out_proj.weight cannot be None!"
            # assert self.module._qkv_same_embed_dim, "qkv must have same dim!"

            requires_grad_o = self.module.out_proj.weight.requires_grad
            weight_o = self.module.out_proj.weight.detach().numpy() if requires_grad_o \
                    else self.module.out_proj.weight.numpy()
            
            requires_grad_bias_o = self.module.out_proj.bias.requires_grad
            bias_o = self.module.out_proj.bias.detach().numpy() if requires_grad_bias_o \
                    else self.module.out_proj.bias.numpy()
            
            if self.module.in_proj_weight != None:
                requires_grad = self.module.in_proj_weight.requires_grad
                weight_in = self.module.in_proj_weight.detach().numpy() if requires_grad \
                    else self.module.in_proj_weight.numpy()
                
                length = int(weight_in.shape[0] / 3)
                weight_q = weight_in[0:length]
                weight_k = weight_in[length:2*length]
                weight_v = weight_in[2*length:]
                
                requires_grad_q = requires_grad_k = requires_grad_v = requires_grad

                requires_grad_bias = self.module.in_proj_bias.requires_grad
                bias_in = self.module.in_proj_bias.detach().numpy() if requires_grad_bias \
                    else self.module.in_proj_bias.numpy()
                length = int(bias_in.shape[0] / 3)
                
                bias_q = bias_in[0:length]
                bias_k = bias_in[length:2*length]
                bias_v = bias_in[2*length:]
            else:
                requires_grad_q = self.module.q_proj_weight.requires_grad
                weight_q = self.module.q_proj_weight.detach().numpy() if requires_grad_q \
                    else self.module.q_proj_weight.numpy()

                requires_grad_k = self.module.k_proj_weight.requires_grad
                weight_k = self.module.k_proj_weight.detach().numpy() if requires_grad_k \
                    else self.module.k_proj_weight.numpy()

                requires_grad_v = self.module.v_proj_weight.requires_grad
                weight_v = self.module.v_proj_weight.detach().numpy() if requires_grad_v \
                    else self.module.v_proj_weight.numpy()
                
                requires_grad_bias = self.module.bias_q.requires_grad
                bias_q = self.module.bias_q.detach().numpy() if requires_grad_bias \
                    else self.module.bias_q.numpy()
                bias_k = self.module.bias_k.detach().numpy() if requires_grad_bias \
                    else self.module.bias_k.numpy()
                bias_v = self.module.bias_v.detach().numpy() if requires_grad_bias \
                    else self.module.bias_v.numpy()
                            
            operator_q = umodel.parameter(np_tensor=weight_q, dtype=numpy_to_ufront_dtype(weight_q.dtype), requires_grad=requires_grad_q, name=self.name + "_weight_q")
            operator_k = umodel.parameter(np_tensor=weight_k, dtype=numpy_to_ufront_dtype(weight_k.dtype), requires_grad=requires_grad_k, name=self.name + "_weight_k")
            operator_v = umodel.parameter(np_tensor=weight_v, dtype=numpy_to_ufront_dtype(weight_v.dtype), requires_grad=requires_grad_v, name=self.name + "_weight_v")
            operator_bias_q = umodel.parameter(np_tensor=bias_q, dtype=numpy_to_ufront_dtype(bias_q.dtype), requires_grad=requires_grad_bias, name=self.name + "_bias_q")
            operator_bias_k = umodel.parameter(np_tensor=bias_k, dtype=numpy_to_ufront_dtype(bias_k.dtype), requires_grad=requires_grad_bias, name=self.name + "_bias_k")
            operator_bias_v = umodel.parameter(np_tensor=bias_v, dtype=numpy_to_ufront_dtype(bias_v.dtype), requires_grad=requires_grad_bias, name=self.name + "_bias_v")

            operator_o = umodel.parameter(np_tensor=weight_o, dtype=numpy_to_ufront_dtype(weight_o.dtype), requires_grad=requires_grad_o, name=self.name + "_weight_o")
            operator_bias_o = umodel.parameter(np_tensor=bias_o, dtype=numpy_to_ufront_dtype(bias_o.dtype), requires_grad=requires_grad_bias_o, name=self.name + "_bias_o")

            return umodel.multihead_attention(
                q=q,
                k=k,
                v=v,
                weight_q=operator_q.get_output(0),
                weight_k=operator_k.get_output(0),
                weight_v=operator_v.get_output(0),
                bias_q=operator_bias_q.get_output(0),
                bias_k=operator_bias_k.get_output(0),
                bias_v=operator_bias_v.get_output(0),

                weight_o=operator_o.get_output(0),
                bias_o=operator_bias_o.get_output(0),

                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=float(dropout),
                batch_first=batch_first,
                name=self.name,
            )



class FunctionNode(Node):
    tensor_idx = 1
    def __init__(self, node):
        super().__init__(node)
        self.innodes = node.args
        # self.innodes = node.all_input_nodes
        self.outnodes = node.users
        self.function = node.target
        self.kwargs = node.kwargs
        self.op_type = None

    class ScalarPosition(Enum):
        LEFT = 0
        RIGHT = 1

    @staticmethod
    def construct_node(node):
        """
        Args:
            node (torch.fx.node.Node): ``torch.fx`` node from which to
                construct a corresponding :class:`Node`.
        """
        name : str = node.name
        if name.find("add") >= 0:
            if FunctionNode.is_right_scalar_op(node):
                return ScalarAddNode(node, FunctionNode.ScalarPosition.RIGHT)
            elif FunctionNode.is_left_scalar_op(node):
                return ScalarAddNode(node, FunctionNode.ScalarPosition.LEFT)
            elif FunctionNode.is_elemwise_op(node):
                return AddNode(node)
            elif len(node.args) == 2 and type(node.args[0]) == torch.fx.graph.Node and type(node.args[1]) == tuple:
                return AddNodePython(node)
            else:
                assert 0, "Unknown `add()` usage with `innodes`: " \
                    f"{node.innodes}"
        elif name.find("sub") >= 0:
            if FunctionNode.is_right_scalar_op(node):
                return ScalarSubNode(node, FunctionNode.ScalarPosition.RIGHT)
            elif FunctionNode.is_left_scalar_op(node):
                return ScalarSubNode(node, FunctionNode.ScalarPosition.LEFT)
            elif FunctionNode.is_elemwise_op(node):
                return SubNode(node)
            else:
                assert 0, "Unknown `sub()` usage with `innodes`: " \
                    f"{node.innodes}"
        elif name.find("mul") >= 0 and name.find("matmul") < 0:
            if FunctionNode.is_right_scalar_op(node):
                return ScalarMulNode(node, FunctionNode.ScalarPosition.RIGHT)
            elif FunctionNode.is_left_scalar_op(node):
                return ScalarMulNode(node, FunctionNode.ScalarPosition.LEFT)
            elif FunctionNode.is_elemwise_op(node):
                return MulNode(node)
            else:
                return None
                # assert 0, \
                #     f"Unknown `mul()` usage with `innodes`: {node.innodes}"
        elif name.find("floordiv") >= 0 or name.find("floor_divide") >= 0:
            return ScalarFloorDivNode(node)
        elif name.startswith("neg"):
            return NegNode(node)
        elif name.startswith("bool"):
            return BoolNode(node)
        elif name.startswith("invert"):
            return InvertNode(node)
        elif name.startswith("and"):
            return AndNode(node)
        elif name.startswith("detach"):
            return DetachNode(node)
        elif name.startswith("cumsum"):
            return CumsumNode(node)
        elif name.find("arange") >= 0: #name.startswith("arange"):
            return ArangeNode(node)
        elif name.startswith("less") or name.startswith("lt"):
            return LessNode(node)
        elif name.startswith("erf"):
            return ErfNode(node)
        elif name.find("truediv") >= 0: return ScalarTrueDivNode(node)
        elif name.find("cat") == 0: return ConcatNode(node)
        elif name.find("split") >= 0: return SplitChunkNode(node, OpType.SPLIT)
        elif name.find("chunk") >= 0: return SplitChunkNode(node, OpType.CHUNK)
        elif name.find("flatten") >= 0: return FlattenNode(node, None)
        elif name.find("relu") >= 0: return ReLUFNode(node)
        elif name.find("getitem") >= 0: return GetItemNode(node)
        elif name.find("matmul") >= 0: return BatchMatMulNode(node)
        elif name.find("getattr") >= 0: return GetAttrNode(node)
        elif name.find("transpose") >= 0: return TransposeNode(node)
        elif name.find("expand") >= 0: return ExpandNode(node)
        elif name.find("reshape") >= 0: return ReshapeNode(node)
        elif name.find("permute") >= 0: return PermuteNode(node)
        elif name.find("softmax") >= 0: return SoftmaxFNode(node)
        elif name.find("view") >= 0: return ViewNode(node)
        elif name.find("to") == 0: return ToNode(node)
        elif name.find("pow") == 0: return PowNode(node)
        elif name.find("mean") >= 0: return MeanNode(node)
        elif name.find("rsqrt") >= 0: return RsqrtNode(node)
        elif name.find("unsqueeze") >= 0: return UnsqueezeNode(node)
        elif name.find("float") >= 0: return FloatNode(node)
        elif name.find("type_as") >= 0: return TypeAsNode(node)
        elif name.find("dropout") >= 0: return DropoutFNode(node)
        elif name.find("contiguous") >= 0: return ContiguousNode(node)
        elif name.find("tanh") >= 0: return TanhFNode(node)
        elif name.find("gelu") >= 0: return GeluFNode(node)
        elif name.find("eq") == 0: return EqNode(node)
        elif name.find("_assert") >= 0: return AssertNode(node)
        elif name.find("dim") == 0: return GetAttrNode(node)
        elif name.find("dims") >= 0: return GetAttrNode(node)
        elif name.find("shape") >= 0: return GetAttrNode(node)
        elif name.find("size") >= 0: return GetAttrNode(node)
        elif name.find("adaptive_avg_pool2d") >= 0: return AdaptivePool2dNode(node, PoolType.POOL_ADAPTIVE, None)
        elif name.find("avg_pool2d") >= 0: return Pool2dNode(node, PoolType.POOL_AVG, None)
        elif name.find("max_pool2d") >= 0: return Pool2dNode(node, PoolType.POOL_MAX, None)
        elif name.find("conv2d") >= 0: return Conv2dNode(node)
        elif name.find("linear") >= 0: return LinearNode(node)
        elif name.find("layer_norm") >= 0: return LayerNormNode(node, None)
        elif name.find("embedding") >= 0: return EmbeddingNode(node, None)
        elif name.find("batch_norm") >= 0: return BatchNorm2dNode(node)
        else:
            return None
            # assert 0, f"Unknown function or method: {name}"
            #  print(f"Unknown function or method: {name}")


    @staticmethod
    def is_right_scalar_op(node):
        """
        Args:
            node (torch.fx.node.Node): ``torch.fx`` node to check.
        """
        innodes = node.args
        if len(innodes) != 2:
            return False
        return type(innodes[1]) is float or \
            type(innodes[1]) is int

    @staticmethod
    def is_left_scalar_op(node):
        """
        Args:
            node (torch.fx.node.Node): ``torch.fx`` node to check.
        """
        innodes = node.args
        if len(innodes) != 2:
            return False
        return type(innodes[0]) is float or \
            type(innodes[1]) is int

    @staticmethod
    def is_elemwise_op(node):
        """
        Args:
            node (torch.fx.node.Node): ``torch.fx`` node to check.
        """
        innodes = node.args
        if len(innodes) != 2:
            return False
        return type(innodes[0]) is torch.fx.node.Node and \
            type(innodes[1]) is torch.fx.node.Node

    @staticmethod
    def parse_scalar_op(node, node_to_output):
        """Parses the node representing a scalar op, and returns the input
        tensor and scalar."""
        assert hasattr(node, "scalar_pos") and node.scalar_pos is not None
        if node.scalar_pos == FunctionNode.ScalarPosition.RIGHT:
            input_tensor = node_to_output[node.innodes[0].name]
            scalar = node.innodes[1]
        elif node.scalar_pos == FunctionNode.ScalarPosition.LEFT:
            input_tensor = node_to_output[node.innodes[1].name]
            scalar = node.innodes[0]
        else:
            assert 0, f"Unknown scalar position: {node.scalar_pos}"
        # if type(scalar) == int:
        #     scalar = scalar * 1.0
        # assert type(scalar) is float
        return input_tensor, scalar

    @staticmethod
    def get_view_shape(input_tensor, view_shape):
        for dim in view_shape:
            assert type(dim) is int

        # Find the input numel
        input_shape = list(input_tensor.shape)
        numel = 1
        for dim_size in input_shape:
            numel *= dim_size

        # Find the new numel
        infer_dim = -1
        new_numel = 1
        shape = []
        for dim, dim_size in enumerate(view_shape):
            if dim_size == -1:
                if infer_dim >= 0:
                    assert 0, \
                        f"Already inferring dim {infer_dim}; cannot also " \
                        f"infer dim {dim}"
                infer_dim = dim
            elif dim_size >= 0:
                new_numel *= dim_size
            else:
                assert 0, \
                    f"Invalid dim size {dim_size} for dim {dim}"
            shape.append(dim_size)

        # Try to check and infer the new shape
        def check_and_infer(numel, new_numel, infer_dim, shape):
            if (numel == new_numel) or \
                    (infer_dim >= 0 and numel % new_numel == 0):
                if infer_dim >= 0:
                    shape[infer_dim] = numel // new_numel
                return shape
            return None  # invalid

        new_shape = check_and_infer(numel, new_numel, infer_dim, shape)
        if new_shape:
            return new_shape
        assert 0, f"Shape {view_shape} is invalid for input of size {numel}"

    @staticmethod
    def get_unsqueeze_shape(input_tensor, dim):
        assert type(dim) is int
        shape = list(input_tensor.shape)
        shape.insert(dim, 1)
        return shape

    @staticmethod
    def get_broadcast_shape(shape1, shape2):
        """Returns the tensor shape after broadcasting either ``shape1`` to
        ``shape2``or vice versa, or returns ``None`` if neither shape can be
        broadcast to the other."""
        # Ensure that `tensor1` has no more dimensions that `tensor2`
        if len(shape1) > len(shape2):
            shape1, shape2 = shape2, shape1
        bc_shape = list(shape2)
        i = len(shape1) - 1
        j = len(shape2) - 1
        while i >= 0 and j >= 0:
            if shape1[i] == shape2[j]:
                i -= 1
                j -= 1
            elif shape1[i] == 1:
                bc_shape[j] = shape2[j]
                i -= 1
                j -= 1
            elif shape2[j] == 1:
                bc_shape[j] = shape1[i]
                i -= 1
                j -= 1
            else:
                return None  # cannot broadcast
        return bc_shape

    @staticmethod
    def broadcast_tensors(tensor1, tensor2, umodel):
        shape1 = tensor1.dims
        shape2 = tensor2.dims
        bc_shape = FunctionNode.get_broadcast_shape(shape1, shape2)
        x = tensor1
        y = tensor2
        if bc_shape is None:
            assert 0, "Operands cannot be broadcast together: " \
                f"{shape1} {shape2}"
        if list(shape1) != bc_shape:
            np1 = tensor1.get_tensor(umodel, ParamSyncType.PS)
            np1 = np.broadcast_to(np1, bc_shape)
            np1 = np.ascontiguousarray(np1)
            dtype = numpy_to_ufront_dtype(np1.dtype)
            x = umodel.create_tensor(bc_shape, dtype, True, "broadcast_tensor_x_" + str(FunctionNode.tensor_idx))
            FunctionNode.tensor_idx += 1
            x.set_ndarray(np1)
        if list(shape2) != bc_shape:
            np2 = tensor2.get_tensor(umodel, ParamSyncType.PS)
            np2 = np.broadcast_to(np2, bc_shape)
            np2 = np.ascontiguousarray(np2)
            dtype = numpy_to_ufront_dtype(np2.dtype)
            y = umodel.create_tensor(bc_shape, dtype, True, "broadcast_tensor_y_" + str(FunctionNode.tensor_idx))
            FunctionNode.tensor_idx += 1
            y.set_ndarray(np2)
        return x, y


class ScalarAddNode(FunctionNode):
    def __init__(self, node, scalar_pos):
        super().__init__(node)
        self.op_type = OpType.SCALAR_ADD
        self.assert_num_args(2, Comparator.EQ)
        self.scalar_pos = scalar_pos

    def parse(self):
        s = [self.name]
        if self.scalar_pos == FunctionNode.ScalarPosition.RIGHT:
            innodes = (self.innodes[0],)
            scalar = self.innodes[1]
        elif self.scalar_pos == FunctionNode.ScalarPosition.LEFT:
            innodes = (self.innodes[1],)
            scalar = self.innodes[0]
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        s.append(str(scalar))
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor, scalar = \
            FunctionNode.parse_scalar_op(self, node_to_output)
        return umodel.sadd(
            input=input_tensor, scalar=scalar, name=self.name,
        )


class SubNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.SUBTRACT
        self.assert_num_args(2, Comparator.EQ)

    def __call__(self, umodel, node_to_output):
        input_tensor1 = node_to_output[self.innodes[0].name]
        input_tensor2 = node_to_output[self.innodes[1].name]
        return umodel.subtract(x=input_tensor1, y=input_tensor2, name=self.name)
    
class AddNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.ADD
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor1 = node_to_output[self.innodes[0].name]
        input_tensor2 = node_to_output[self.innodes[1].name]
        res = umodel.add(x=input_tensor1, y=input_tensor2, name=self.name)
        return res

class AddNodePython(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.ADD
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor1 = node_to_output[self.innodes[0].name]
        if type(self.innodes[1]) == tuple or type(self.innodes[1]) == list:
            input_tensor2 = list(self.innodes[1])
        else:
            input_tensor2 = node_to_output[self.innodes[1].name]
        return input_tensor1 + input_tensor2

class ScalarSubNode(FunctionNode):
    def __init__(self, node, scalar_pos):
        super().__init__(node)
        self.op_type = OpType.SCALAR_SUB
        self.assert_num_args(2, Comparator.EQ)
        self.scalar_pos = scalar_pos

    def parse(self):
        s = [self.name]
        if self.scalar_pos == FunctionNode.ScalarPosition.RIGHT:
            innodes = (self.innodes[0],)
            scalar = self.innodes[1]
        elif self.scalar_pos == FunctionNode.ScalarPosition.LEFT:
            innodes = (self.innodes[1],)
            scalar = self.innodes[0]
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        s.append(str(scalar))
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        # if self.scalar_pos == None:
        #     return node_to_output[self.innodes[0]] - node_to_output[self.innodes[1]]
        input_tensor, scalar = \
            FunctionNode.parse_scalar_op(self, node_to_output)
        return umodel.ssub(
            input=input_tensor, scalar=scalar, name=self.name,
        )


class ScalarTrueDivNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.SCALAR_TRUEDIV
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        scalar = self.innodes[1]
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        s.append(str(scalar))
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        scalar = self.innodes[1]
        if hasattr(scalar, "name"):
            scalar = node_to_output[scalar.name]
        if type(scalar) == Tensor:
            scalar = float(scalar.ndarray[0])
        assert type(scalar) is float
        return umodel.struediv(
            input=input_tensor, scalar=scalar, name=self.name,
        )


class ConcatNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        if len(self.innodes) < 2 and "dim" in node.kwargs:
            self.innodes = (self.innodes[0], node.kwargs["dim"])
        self.op_type = OpType.CONCAT
        self.assert_num_args(2, Comparator.GEQ)

    def parse(self):
        s = [self.name]
        innodes = self.innodes[0]
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        if len(self.innodes) == 1:
            s.append("1")
        else:
            s.append(str(self.innodes[1]))
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensors = []
        for input_node in self.innodes[0]:
            input_tensors.append(node_to_output[input_node.name])
        axis = 1 if len(self.innodes) == 1 else self.innodes[1]
        assert type(axis) is int
        return umodel.concat(
            tensors=input_tensors, axis=axis, name=self.name,
        )


class SplitChunkNode(FunctionNode):
    def __init__(self, node, op_type):
        super().__init__(node)
        # FIXME May be 3
        self.op_type = op_type # OpType.SPLIT
        self.dim = node.kwargs["dim"]
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        s.append(str(self.innodes[1]))
        s.append(str(self.dim))

        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        sizes = self.innodes[1] #len(self.outnodes)
        axis = self.dim
        assert type(axis) is int
        #TODO split support sizes of int or list
        if self.op_type==OpType.SPLIT:
            return umodel.split(
                input=input_tensor, sizes=sizes, axis=axis, name=self.name,
            ) 
        else:
            return umodel.chunk(
                input=input_tensor, sizes=sizes, axis=axis, name=self.name,
            )


class ReLUFNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.RELU
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return umodel.relu(input=input_tensor, name=self.name)


class GetItemNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.GETITEM
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        if type(self.innodes[1]) is int:
            s.append(str(self.innodes[1]))
        else:
            # Append each slice separately
            for sl in self.innodes[1]:
                s.append(str(sl))
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        if type(input_tensor) is Tensor:
            slices = self.innodes[1]
            if hasattr(slices, "name"):
                slice_tensor = node_to_output[slices.name]
                np_slices = slice_tensor.ndarray.astype(ufront_to_numpy_dtype(slice_tensor.dtype))
                output_shape = np.zeros(shape=input_tensor.shape, dtype=np.float32)[np_slices].shape
            else:
                assert type(slices) is tuple, f"Expected tuple slices but got {type(slices)}"
                # slice_tensor = list(slices)
                isEllipsis  = False
                for slice in slices:
                    if slice is Ellipsis:
                        isEllipsis = True
                        break

                if isEllipsis:
                    slice_tensor = str(slices)
                else:
                    if slices[0] != None:
                        start, step, stop = slices[0].start,  slices[0].step,  slices[0].stop
                        slice_tensor = "[[{}, {}, {}], {}]".format(start if start!= None else "\"None\"", step if step!= None else "\"None\"", stop if stop!= None else "\"None\"", slices[1] if slices[1]!= None else "\"None\"")
                    else:
                        start, step, stop = slices[1].start,  slices[1].step,  slices[1].stop
                        slice_tensor = "[{}, [{}, {}, {}]]".format(slices[0] if slices[0]!= None else "\"None\"", start if start!= None else "\"None\"", step if step!= None else "\"None\"", stop if stop!= None else "\"None\"")
                    
                output_shape = np.zeros(shape=input_tensor.shape, dtype=np.float32)[slices].shape
            
            # print(np.max(slices), " ", np.min(slices), " : ", list(slices))
            if type(slice_tensor) == Tensor:
                return umodel.slice_tensor(x=input_tensor, y=slice_tensor, output_shape=list(output_shape), name=self.name)
            else:
                return umodel.slice_tensor(input=input_tensor, slices=slice_tensor, output_shape=list(output_shape), name=self.name)

        assert type(input_tensor) is list or \
            type(input_tensor) is tuple, \
            f"Expected list or tuple but got {type(input_tensor)}"
        index = self.innodes[1]
        # assert type(index) is int
        return input_tensor[index]

    @staticmethod
    def slice_tensor(umodel, tensor, slices, name):
        """Returns a reshaped tensor based on the given slices."""
        def is_colon(slice_elem):
            """Returns if the slice is equivalent to `:`."""
            return slice_elem == slice(None, None, None)

        def is_unsqueeze(slice_elem):
            """Returns if the slice is equivalent to unsqueezing that
            dimension."""
            return slice_elem is None
        shape = tensor.dims
        # Match dimensions from right to left
        new_shape = []  # append then reverse
        j = len(shape) - 1
        for slice_elem in reversed(slices):
            if is_colon(slice_elem):
                assert j >= 0
                new_shape.append(shape[j])
                j -= 1
            elif is_unsqueeze(slice_elem):
                new_shape.append(1)
            else:
                assert 0, f"Unsupported slice element: {slice_elem}"
        new_shape.reverse()
        return umodel.reshape(
            input=tensor, shape=list(new_shape), name=name,
        )

    @staticmethod
    def strings_to_slices(strings: List[str]):
        # Extract slice elements
        slices = [sl.strip() for sl in strings]

        def string_to_slice_val(s):
            """Converts the string representation of a slice value (i.e. either
            start, stop, or step) to the actual value."""
            if s == "None":
                return None
            try:
                return int(s)
            except ValueError:
                assert 0, f"Invalid slice value: {s}"

        def string_to_slice(s):
            """Converts the string representation of a slice to the actual
            slice object."""
            if s == "None":
                return None
            elif s.startswith("slice"):
                # The slice should contain three elements: start, stop, step
                s = s[5:]
                # Remove left and right parentheses
                assert s[0] == '('
                assert s[-1] == ')'
                s = s[1:-1]
                # Extract slice values
                sls = [v.strip() for v in s.split(',')]
                assert len(sls) == 3
                # Convert the slice elements from string
                sls = [string_to_slice_val(v) for v in sls]
                return slice(*sls)
            else:
                assert 0, f"Invalid slice: {s}"

        # Convert the slices from string
        slices = [string_to_slice(sl) for sl in slices]
        return tuple(slices)


class BatchMatMulNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.BATCH_MATMUL
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor1 = node_to_output[self.innodes[0].name]
        input_tensor2 = node_to_output[self.innodes[1].name]
        return umodel.batch_matmul(
            x=input_tensor1, y=input_tensor2, name=self.name,
        )


class NegNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.SCALAR_MULTIPLY
        self.assert_num_args(1, Comparator.EQ)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        if type(input_tensor) == Tensor:
            return umodel.smultiply(
                input=input_tensor, scalar=-1.0, name=self.name,
            )
        else:
            return input_tensor * -1.0

class BoolNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.BOOL
        self.assert_num_args(1, Comparator.EQ)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        if type(input_tensor) == Tensor:
            return umodel.bool(
                input=input_tensor, name=self.name,
            )
        else:
            return input_tensor > 0

class InvertNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.INVERT
        self.assert_num_args(1, Comparator.EQ)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        if type(input_tensor) == Tensor:
            return umodel.invert(
                input=input_tensor, name=self.name,
            )
        elif type(input_tensor) == np.ndarray:
            return np.invert(input_tensor)
        else:
            return ~input_tensor

class AndNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.AND
        self.assert_num_args(2, Comparator.EQ)

    def __call__(self, umodel, node_to_output):
        input_tensor1 = node_to_output[self.innodes[0].name]
        input_tensor2 = node_to_output[self.innodes[1].name]
        if type(input_tensor1) == Tensor and type(input_tensor2) == Tensor:
            return umodel.And(
                x=input_tensor1, y=input_tensor2, name=self.name,
            )
        else:
            return input_tensor1 & input_tensor2

class LessNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.LESS
        self.assert_num_args(2, Comparator.EQ)

    def __call__(self, umodel, node_to_output):
        input_tensor1 = node_to_output[self.innodes[0].name]
        input_tensor2 = node_to_output[self.innodes[1].name]
        if type(input_tensor1) == Tensor and type(input_tensor2) == Tensor:
            return umodel.less(
                x=input_tensor1, y=input_tensor2, name=self.name,
            )
        else:
            return input_tensor1 < input_tensor2
        
class DetachNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.DETACH
        self.assert_num_args(1, Comparator.EQ)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        if type(input_tensor) == Tensor:
            return umodel.detach(
                input=input_tensor, name=self.name,
            )
        else:
            return input_tensor
        
class CumsumNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.CUMSUM
        self.axis = node.kwargs["dim"]
        self.assert_num_args(1, Comparator.EQ)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        if type(input_tensor) == Tensor:
            return umodel.cumsum(
                input=input_tensor, axis=self.axis, name=self.name,
            )
        else:
            return np.cumsum(input_tensor, self.axis)

class ArangeNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.ARANGE
        self.assert_num_args(1, Comparator.EQ)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        np_tensor = np.arange(start=0, stop=input_tensor, step=1, dtype=np.int64)
        return umodel.parameter(np_tensor=np_tensor, dtype=numpy_to_ufront_dtype(np_tensor.dtype), requires_grad=False, name=self.name)
        # return umodel.arange(start=0, end=input_tensor, step=1, name=self.name)

        
                      
class ScalarMulNode(FunctionNode):
    def __init__(self, node, scalar_pos):
        super().__init__(node)
        self.op_type = OpType.SCALAR_MULTIPLY
        self.assert_num_args(2, Comparator.EQ)
        self.scalar_pos = scalar_pos

    def parse(self):
        s = [self.name]
        if self.scalar_pos == FunctionNode.ScalarPosition.RIGHT:
            innodes = (self.innodes[0],)
            scalar = self.innodes[1]
        elif self.scalar_pos == FunctionNode.ScalarPosition.LEFT:
            innodes = (self.innodes[1],)
            scalar = self.innodes[0]
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        s.append(str(scalar))
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor, scalar = \
            FunctionNode.parse_scalar_op(self, node_to_output)
        if type(input_tensor) == Tensor:
            return umodel.smultiply(
                input=input_tensor, scalar=scalar, name=self.name,
            )
        else:
            return input_tensor * scalar


class MulNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.MULTIPLY
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor1 = node_to_output[self.innodes[0].name]
        input_tensor2 = node_to_output[self.innodes[1].name]
        if type(input_tensor1) == Tensor and type(input_tensor2) == Tensor:
            return umodel.multiply(
                x=input_tensor1, y=input_tensor2, name=self.name
            )
        else:
            return input_tensor1 * input_tensor2


class GetAttrNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.GETATTR
        if len(self.innodes) == 1: #func attribute in tensor, e.g, tensor.dim()
            tmp = list(self.innodes)
            tmp.append(self.function)
            self.innodes = tuple(tmp)

        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        s.append(str(self.innodes[1]))
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        attr = self.innodes[1]
        if attr == "shape" or attr == "size":
            return input_tensor.shape
        if attr == "dims" or attr == "dim":
            return input_tensor.dims
        if attr == "device":
            return "cpu"
        
        if hasattr(self, "function") and self.function == "size":
            return input_tensor.shape[attr]

        if attr in ["float", "bool", "double", "int"]:
            return umodel.cast(input_tensor, to=attr)
        else:
            return getattr(input_tensor, attr)


class TransposeNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.TRANSPOSE
        self.assert_num_args(3, Comparator.EQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        s.append(str(self.innodes[1]))
        s.append(str(self.innodes[2]))
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        dim0 = self.innodes[1]
        dim1 = self.innodes[2]
        assert type(dim0) is int and type(dim1) is int
        perms = list(range(input_tensor.dims if type(input_tensor.dims)==int else len(input_tensor.dims) ))
        perms[dim0], perms[dim1] = perms[dim1], perms[dim0]
        return umodel.transpose(
            input=input_tensor, perms=perms, name=self.name,
        )


class ExpandNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.EXPAND
        self.assert_num_args(1, Comparator.GEQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        args = self.innodes[1:]
        expand_as = type(args[-1]) is not int
        if expand_as:
            tensors = []
            for other in args:
                assert type(other) is torch.fx.node.Node
                tensors.append(other.name)
            s.append(','.join(tensors) + ',')
        else:
            for dim in args:
                s.append(str(dim))
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        output_shape = []
        shapes = self.innodes[1:]
        for i in range(len(shapes)):
            if hasattr(shapes[i], "name"):
                output_shape = node_to_output[shapes[i].name].shape
                break
            elif type(shapes[i]) == str:
                output_shape.append(node_to_output[shapes[i]])
            elif type(shapes[i]) == int:
                output_shape.append(shapes[i])

        return umodel.expand(
            input=input_tensor, sizes=output_shape, name=self.name,
        )


class ScalarFloorDivNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.SCALAR_FLOORDIV
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        scalar = self.innodes[1]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        s.append(str(scalar))
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        scalar = self.innodes[1]
        if type(input_tensor) is float or type(input_tensor) is int:
            return input_tensor // scalar


class ReshapeNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.RESHAPE
        self.assert_num_args(2, Comparator.GEQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        if len(self.innodes) == 2:
            shape = self.innodes[1]
        else:
            shape = self.innodes[1:]
        if type(shape[-1]) is int:
            for dim in shape:
                s.append(str(dim))
        else:
            tensors = []
            for other in shape:
                assert type(other) is torch.fx.node.Node
                tensors.append(other.name)
            s.append(INOUT_NODE_DELIMITER.join(tensors) + INOUT_NODE_DELIMITER)
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        reshape_as = len(self.innodes) == 2 and \
            type(self.innodes[1]) is not int and \
                type(self.innodes[1]) is not str
        if not reshape_as:
            shape = list(self.innodes[1:])
            for i in range(len(shape)):
                if hasattr(shape[i], "name"):
                    shape[i] = node_to_output[shape[i].name]
            shape = tuple(shape)
        else:
            other = self.innodes[1]
            shape = node_to_output[other.name].dims
        for dim in shape:
            assert type(dim) is int
        return umodel.reshape(input=input_tensor, shape=list(shape), name=self.name)


class PermuteNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.PERMUTE
        self.assert_num_args(1, Comparator.GEQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        dims_as_list = isinstance(self.innodes[1], list)
        dims = self.innodes[1] if dims_as_list else self.innodes[1:]
        for dim in dims:
            s.append(str(dim))
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        perm_as_list = isinstance(self.innodes[1], list)
        perms = self.innodes[1] if perm_as_list else self.innodes[1:]
        for dim in perms:
            assert type(dim) is int
        return umodel.transpose(input=input_tensor, perms=list(perms), name=self.name)

class ErfNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.ERF
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return umodel.erf(input=input_tensor, approximate=True, name=self.name)
    
class SoftmaxFNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.SOFTMAX
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return umodel.softmax(input=input_tensor, name=self.name)


class ViewNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.VIEW
        self.assert_num_args(2, Comparator.GEQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        for dim in self.innodes[1:]:
            assert type(dim) is int
            s.append(str(dim))
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        view_shape = list(self.innodes[1:])
        for i in range(len(view_shape)):
            if type(view_shape[i])==torch.fx.node.Node and hasattr(view_shape[i], "name"):
                view_shape[i] = node_to_output[view_shape[i].name]

        shape = FunctionNode.get_view_shape(input_tensor, view_shape)
        # Treat as a special case of `reshape()`
        return umodel.reshape(
            input=input_tensor, shape=list(shape), name=self.name
        )


class ToNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.TO
        self.assert_num_args(1, Comparator.GEQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        if len(self.innodes) == 2 and \
            (isinstance(self.innodes[1], torch.dtype)
             or type(self.innodes[1]) is int
             or type(self.innodes[1]) is float):
            s.append(str(self.innodes[1]))
        elif len(self.innodes) == 1 and "dtype" in self.kwargs:
            s.append(str(self.kwargs["dtype"]))

        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        if hasattr(self, "kwargs") and "dtype" in self.kwargs:
            dtype = self.kwargs["dtype"]
        elif len(self.innodes) > 1 and type(self.innodes[1]) == torch.dtype:
            dtype = self.innodes[1]
        else:
            assert 0, "Invalid dtype to cast!"
        return umodel.cast(input=input_tensor, dtype=torch_to_ufront_dtype(dtype))


class PowNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.POW
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        s.append(str(self.innodes[1]))
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        exponent = self.innodes[1]
        return umodel.pow(
            input=input_tensor, exponent=exponent, name=self.name,
        )


class MeanNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.MEAN
        self.assert_num_args(2, Comparator.GEQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        for dim in self.innodes[1:]:
            s.append(str(dim))
        if "keepdim" in self.kwargs:
            s.append(str(self.kwargs["keepdim"]))
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        if "keepdim" in self.kwargs:
            keepdims = self.kwargs["keepdim"]
        else:
            keepdims = False
        dims = list(self.innodes)[1:]
        if type(dims[0])!=int:
            dims = list(dims[0])
        # Infer the -1 dimension if needed
        for i in range(len(dims)):
            if dims[i] == -1:
                dims[i] = len(input_tensor.dims) - 1
            assert dims[i] >= 0 and dims[i] < input_tensor.dims if type(input_tensor.dims)==int else len(input_tensor.dims)

        return umodel.mean(
            input=input_tensor, dims=dims, keepdims=keepdims, name=self.name,
        )


class RsqrtNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.RSQRT
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return umodel.rsqrt(input=input_tensor, name=self.name)


class UnsqueezeNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.UNSQUEEZE
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        innodes = (self.innodes[0],)
        s.append(self.parse_inoutnodes(innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        s.append(str(self.innodes[1]))
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        dim = self.innodes[1]
        shape = FunctionNode.get_unsqueeze_shape(input_tensor, dim)
        # Treat as a special case of `reshape()`
        return umodel.reshape(
            input=input_tensor, shape=list(shape), name=self.name,
        )


class FloatNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.FLOAT
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        if type(input_tensor) == Tensor:
            return umodel.float(input=input_tensor, name=self.name)
        else:
            assert type(input_tensor) == np.ndarray, "Only accept Tensor or numpy array for converting to float type!"
            return input_tensor.astype(np.float)


class TypeAsNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.TYPE_AS
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return input_tensor


class DropoutFNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.DROPOUT
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        s.append(str(self.kwargs["p"]))
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        rate = self.kwargs["p"]
        if "training" in self.kwargs:
            training = self.kwargs["training"]
        else:
            training = False
        return umodel.dropout(
            input=input_tensor, rate=rate, seed=0, training=training, name=self.name,
        )


class ContiguousNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.CONTIGUOUS
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return input_tensor


class TanhFNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.TANH
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return umodel.tanh(input=input_tensor, name=self.name)

class MaskedFillNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.MASKEDFILL
        self.assert_num_args(3, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        mask_tensor = node_to_output[self.innodes[1].name]
        value = self.innodes[2]
        return umodel.masked_fill(input=input_tensor, mask=mask_tensor, value=value, name=self.name)
    
class RepeatNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.REPEAT
        self.assert_num_args(2, Comparator.GEQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        sizes = self.innodes[1:]
        return umodel.repeat(input=input_tensor, sizes=list(sizes), name=self.name)
    

class AssertNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.ASSERT
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        if hasattr(self.innodes[0], 'name'):
            v = node_to_output[self.innodes[0].name]
        else:
            v = self.innodes[0]
        assert v, self.innodes[1]

    
class EqNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.EQ
        self.assert_num_args(2, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        if hasattr(self.innodes[0], 'name'):
            input_tensor1 = node_to_output[self.innodes[0].name]
        else:
            input_tensor1 = self.innodes[0]

        if hasattr(self.innodes[1], 'name'):
            input_tensor2 = node_to_output[self.innodes[1].name]
        else:
            input_tensor2 = self.innodes[1]

        if type(input_tensor1) == Tensor and type(input_tensor2) == Tensor:
            return input_tensor1 == input_tensor2
        elif type(input_tensor1) == Tensor:
            return umodel.eq(input=input_tensor1, comparator=input_tensor2, name=self.name)
        elif type(input_tensor2) == Tensor:
            return umodel.eq(input=input_tensor2, comparator=input_tensor1, name=self.name)
        else:
            return input_tensor1 == input_tensor2

class GeluFNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.GELU
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        input_tensor = node_to_output[self.innodes[0].name]
        return umodel.gelu(input=input_tensor, name=self.name)


class AttributeNode(Node):
    def __init__(self, node, model):
        super().__init__(node)
        self.innodes = node.args
        self.outnodes = node.users
        self.attr_name = node.target
        self.attr = self.fetch_attr(model)
        self.op_type = OpType.ATTRIBUTE

    def fetch_attr(self, model):
        atoms = self.attr_name.split('.')
        attr_iter = model
        for i, atom in enumerate(atoms):
            if not hasattr(attr_iter, atom):
                assert 0, f"Nonexistent target: {'.'.join(atoms[:i])}"
            attr_iter = getattr(attr_iter, atom)
        return attr_iter

    def parse(self):
        s = [self.name]
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        return self.attr_to_tensor(umodel)

    def attr_to_tensor(self, umodel):
        torch_tensor = self.attr
        requires_grad = torch_tensor.requires_grad
        np_tensor = torch_tensor.detach().numpy() if requires_grad \
            else torch_tensor.numpy()

        return umodel.parameter(np_tensor=np_tensor, dtype=numpy_to_ufront_dtype(np_tensor.dtype), requires_grad=requires_grad, name=self.attr_name)

                    
class CallNode(FunctionNode):
    def __init__(self, node, callback):
        super().__init__(node)
        self.op_type = OpType.CALL
        self.target = node.target
        self.callback = callback
        if hasattr(node.target, "__annotations__"):
            self.target_argnames = node.target.__annotations__
        else: 
            self.target_argnames = {}

        self.funcname = self.name
        if hasattr(node.target, "__name__"):
            self.funcname = node.target.__name__

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output):
        idx = 0
        args = OrderedDict()
        argtypes = OrderedDict()
        for key, vtype in self.target_argnames.items():
            if key == "return": continue
            if idx < len(self.innodes):  
                args[key] = node_to_output[self.innodes[idx].name] if type(self.innodes[idx])==torch.fx.node.Node else self.innodes[idx]
            elif key in self.kwargs:
                args[key] = node_to_output[self.kwargs[key].name] if type(self.kwargs[key])==torch.fx.node.Node else self.kwargs[key]

            idx += 1

        return umodel.call(
            func=self.funcname, args=args, argtypes = self.target_argnames, callback=self.callback, name=self.name
        )

    
class TorchCallNode(CallNode):
    def __init__(self, node, callback):
        super().__init__(node, callback)
        self.args = node.args
        self.funcname = node.target.__name__
                
    def __call__(self, umodel, node_to_output):
        args = OrderedDict()
        argtypes = OrderedDict()
        idx = 1
        for arg in self.args:
            if hasattr(arg, "name"):
                v = node_to_output[arg.name]
                argtypes["arg"+str(idx)] = type(v)
                args["arg"+str(idx)] = v
            else:
                argtypes["arg"+str(idx)] = type(arg)
                args["arg"+str(idx)] = arg
            idx += 1

        for k, v in self.kwargs.items():
            argtypes[k] = type(v)
            args[k] = v
            idx += 1

        return umodel.call(
            func=self.funcname, args=args, argtypes = argtypes, callback=self.callback, name=self.name
        )

    
class MathCallNode(CallNode):
    def __init__(self, node, callback):
        super().__init__(node, callback)
        self.args = node.args
        self.funcname = node.target.__name__
                
    def __call__(self, umodel, node_to_output):
        args = OrderedDict()
        argtypes = OrderedDict()
        idx = 1
        for arg in self.args:
            if hasattr(arg, "name"):
                v = node_to_output[arg.name]
                argtypes["arg"+str(idx)] = type(v)
                args["arg"+str(idx)] = v
            else:
                argtypes["arg"+str(idx)] = type(arg)
                args["arg"+str(idx)] = arg
            idx += 1
        func = getattr(math, self.funcname)
        assert func!=None, "Unable to obtain function " + self.funcname + " from math!" 
        return func(*list(args.values()))
        # return umodel.call(
        #     func=self.funcname, args=args, argtypes = argtypes, callback=self.callback, name=self.name
        # )
    
class InputNode(Node):
    def __init__(self, node):
        super().__init__(node)
        self.innodes = None
        self.outnodes = node.users
        self.op_type = OpType.INPUT

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        # s.append(self.op_type.as_str())
        s.append(self.op_type.as_str())
        
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, input_tensors, input_index):
        return input_tensors[input_index] if input_index < len(input_tensors) else None


class OutputNode(Node):
    def __init__(self, node):
        super().__init__(node)
        self.innodes = node.args[0]
        if type(self.innodes) != tuple and type(self.innodes) != list:
            self.innodes = (self.innodes, ) 
        self.outnodes = None
        self.op_type = OpType.OUTPUT

    def parse(self):
        # TODO: Assumes only one output
        self.assert_num_args(len(self.innodes), Comparator.EQ)
        s = [self.name]
        # if type(self.innodes) is tuple:
        #     s.append(self.parse_inoutnodes(self.innodes))
        #     s.append(self.parse_inoutnodes(self.outnodes))
        # # NOTE: This case targets MT5Model
        # elif type(self.innodes[0]) is immutable_dict and \
        #         "last_hidden_state" in self.innodes[0]:
        #     innodes = (self.innodes[0]["last_hidden_state"],)
        #     s.append(self.parse_inoutnodes(innodes))
        #     s.append(self.parse_inoutnodes(self.outnodes))
        # # NOTE: This case targets MT5ForConditionalGeneration
        # elif type(self.innodes[0]) is immutable_dict and \
        #         "logits" in self.innodes[0]:
        #     innodes = (self.innodes[0]["logits"],)
        #     s.append(self.parse_inoutnodes(innodes))
        #     s.append(self.parse_inoutnodes(self.outnodes))
        # else:
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(self.op_type.as_str())
        self._ir_string = IR_DELIMITER.join(s)

    def __call__(self, umodel, node_to_output, output_tensors):
        for other in self.innodes:
            # Destructively modify `output_tensors`
            if type(other) is immutable_dict:
                assert "last_hidden_state" in other or "logits" in other
                # NOTE: This case targets MT5Model
                if "last_hidden_state" in other:
                    output_tensors[:] += \
                        [node_to_output[other["last_hidden_state"].name]]
                # NOTE: This case targets MT5ForConditionalGeneration
                elif "logits" in other:
                    # NOTE: Manually add a softmax layer since the model is
                    # traced from PyTorch, which includes the softmax in its
                    # `CrossEntropyLoss()` implementation
                    logits = node_to_output[other["logits"].name]
                    softmax_logits = umodel.softmax(
                        input=logits, name=self.name,
                    )
                    output_tensors[:] += [softmax_logits]
            else:
                if other != None:
                    if type(other) == tuple:
                        kv_cache = []
                        for (k, v) in list(other):
                            kv_cache[:] += [(node_to_output[k.name], node_to_output[v.name])]
                        output_tensors[:] += [[kv_cache]]
                    else:
                        output_tensors[:] += [node_to_output[other.name]]
                else:
                    output_tensors.append(other)
                

class UFrontTorch():
    def __init__(
        self,
        model,
        batch_size,
        pass_weights, #pass Pytorch weights to compiler, set to True for pretrained models.
        verbose = False,
        seq_length=None,
    ):
        assert isinstance(model, torch.nn.Module)
        self.model = model
        self.ufront_model = Model() # Ufront Rust model
        self.batch_size = batch_size
        self.pass_weights = pass_weights
        self.seq_length = seq_length
        self.operators = []
        self._metrics = []
        self._loss = LossType.SPARSE_CATEGORICAL_CROSSENTROPY
        self._label_type = DataType.Int32
        self.ufront_model.weight_type = WeightType.EXTERNAL if pass_weights else WeightType.INTERNAL
        self.ufront_model.optimizer = Optimizer(params={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"})
        self.external_functions = {}
        # NOTE: We default `seq_length` to `None` instead of matching
        # the HuggingFace `symbolic_trace()`'s default of `(128, 128)` to
        # decouple the two implementations
    def normal_callback(self, **kwargs):
        # print(kwargs)
        assert(len(kwargs["args"]) > 0)
        for key, v in kwargs["args"].items():
            if type(v) == Tensor:
                dtype = ufront_to_numpy_dtype(v.dtype)
                kwargs["args"][key] = torch.from_numpy(v.ndarray.astype(dtype))

        ret = self.external_functions[kwargs['func']](**kwargs["args"])
        if type(ret) == torch.Tensor:
            # print("Results after calling the external function: ", ret.shape)
            arr = ret.numpy()
            return Tensor(np_tensor=arr, dtype=numpy_to_ufront_dtype(arr.dtype), name=kwargs['func']) # convert to Rust f32 tensor
        else:
            return ret
        
    def torch_callback(self, **kwargs):
        # print(kwargs)
        assert(len(kwargs["args"]) > 0)
        args = []
        for key, v in kwargs["args"].items():
            if type(v) == Tensor:
                dtype = ufront_to_numpy_dtype(v.dtype)
                kwargs["args"][key] = torch.from_numpy(v.ndarray.astype(dtype))
            elif type(v) == int or type(v) == float:
                if kwargs["argtypes"][key] == torch.Tensor:
                    kwargs["args"][key] = torch.Tensor([v])
                else:
                    kwargs["args"][key] = v
            else:
                kwargs["args"][key] = v
            args.append(kwargs["args"][key])
        
        ret = self.external_functions[kwargs['func']](*args)
        if type(ret) == torch.Tensor:
            # print("Results after calling the external function: ", ret.shape)
            nparray = ret.numpy()
            # if kwargs['func'] == "swapaxes":
            # nparray = np.zeros(nparray.shape, dtype=np.float32)
            return Tensor(np_tensor=nparray, dtype=numpy_to_ufront_dtype(nparray.dtype), name=kwargs['func']) # convert to Rust f32 tensor
        else:
            return ret

    def math_callback(self, **kwargs):
        # print(kwargs)
        assert(len(kwargs["args"]) > 0)
        args = []
        for key, v in kwargs["args"].items():
            args.append(kwargs["args"][key])
        
        ret = self.external_functions[kwargs['func']](*args)
        return ret
                
    def _trace_model(self):
        class UFrontTracer(torch.fx.Tracer):
            """
            ``Tracer`` is the class that implements the symbolic tracing functionality
            of ``torch.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is equivalent
            to ``Tracer().trace(m)``.
            This Tracer override the ``is_leaf_module`` function to make symbolic trace
            right in some cases.
            """
            def __init__(self, *args, customed_leaf_module=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.customed_leaf_module = customed_leaf_module

            def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
                """
                A method to specify whether a given ``nn.Module`` is a "leaf" module.
                Leaf modules are the atomic units that appear in
                the IR, referenced by ``call_module`` calls. By default,
                Modules in the PyTorch standard library namespace (torch.nn)
                are leaf modules. All other modules are traced through and
                their constituent ops are recorded, unless specified otherwise
                via this parameter.
                Args:
                    m (Module): The module being queried about
                    module_qualified_name (str): The path to root of this module. For example,
                        if you have a module hierarchy where submodule ``foo`` contains
                        submodule ``bar``, which contains submodule ``baz``, that module will
                        appear with the qualified name ``foo.bar.baz`` here.
                """
                if self.customed_leaf_module and isinstance(m, self.customed_leaf_module):
                    return True
                
                if hasattr(m, '_is_leaf_module') and m._is_leaf_module:
                    return True

                return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)
        tracer = UFrontTracer()
        traced_graph = tracer.trace(self.model)
        # Convert the fx graph to an internal graph representation
        name_to_module = {}
        for name, module in self.model.named_modules():
            name_to_module[name] = module
            # print(name)
        graph = []
        for fx_node in traced_graph.nodes:
            # if fx_node.op == "placeholder":
                # print(fx_node.op, " : ", fx_node.name)
            if fx_node.op == "output":
                node = OutputNode(fx_node)
            elif fx_node.op == "placeholder":
                node = InputNode(fx_node)
            elif fx_node.op == "get_attr":
                node = AttributeNode(fx_node, self.model) 
            elif fx_node.op == "call_function" or fx_node.op == "call_method" or fx_node.op == "call_module":
                if fx_node.op == "call_module":
                    module_name = fx_node.target
                    module = name_to_module[module_name]
                    node = ModuleNode.construct_node(fx_node, module)
                    if node is None:
                        node = FunctionNode.construct_node(fx_node)
                else:
                    node = FunctionNode.construct_node(fx_node)

                if node is None:
                    # if fx_node.name.find("einsum") >=0 or fx_node.name.find("swapaxes") >=0: 
                    if hasattr(fx_node.target, "__module__") and (fx_node.target.__module__ == "torch.functional" or fx_node.target.__module__=="torch"):
                        # print(fx_node.target.__module__)
                        node = TorchCallNode(fx_node, self.torch_callback)
                    elif hasattr(fx_node.target, "__module__") and fx_node.target.__module__ == "math":
                        # print(fx_node.target.__module__)
                        node = MathCallNode(fx_node, self.math_callback)
                    elif fx_node.target == 'masked_fill':
                        node = MaskedFillNode(fx_node)
                    elif fx_node.target == 'repeat':
                        node = RepeatNode(fx_node)
                    else:
                        node = CallNode(fx_node, self.normal_callback)
            else:
                assert 0, f"Unknown operator type: {fx_node.op}"
            if node != None:
                graph.append(node)
                if type(node) == CallNode or type(node) == TorchCallNode or type(node) == MathCallNode:
                    self.external_functions[node.name] = node.target
        return graph
    
    def __call__(self, inputs, verbose=False):
        return self.apply(inputs, verbose=verbose)

    def softmax(self, input, name="softmax"):
        softmax_op = self.ufront_model.softmax(input=input, name=name)
        self.operators.append(softmax_op)
        return softmax_op.get_output(0)

    def dump_ir(self):
        return self.ufront_model.dump_ir()
    
    def dump_tosa_ir(self):
        return self.ufront_model.dump_tosa_ir()

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


    def apply(self, inputs, verbose=False):
        """
        Traces the PyTorch model wrapped by this ``PyTorchModel`` instance,
        and adds operators to ``umodel`` coresponding to the computational
        graph nodes from the trace.

        Args:
            input_tensors (List[Tensor]): Input tensors to the model.
            verbose (bool, optional): If ``True``, then prints the string
                representation of each computational graph node. Default:
                ``False``.

        Returns:
            output_tensors (List[Tensor]): Output tensors of the model.
        """
        graph = self._trace_model()
        output_tensors = []
        node_to_output = OrderedDict()
        input_index = 0
        input_tensors = []
        idx = 1
        for input in inputs:
            if type(input) == torch.Tensor:
                input = input.numpy()
            input1 = np.ones(shape=input.shape, dtype=input.dtype)
            input1[:] = input
            input_tensor = Tensor(np_tensor=input1, dtype=numpy_to_ufront_dtype(input1.dtype), name="input" + str(idx)) # convert to Rust f32 tensor
            input_tensors.append(input_tensor)
            idx += 1

        inputs_nodes = []
        for node in graph:
            if verbose:
                print(f"{node.ir_string}")
            if isinstance(node, InputNode):
                node_output = node(input_tensors, input_index)
                if node_output:
                    inputs_nodes.append(node_output)
                    input_index += 1
            elif isinstance(node, OutputNode):
                node(self.ufront_model, node_to_output, output_tensors)
                node_output = None
            else:
                # if type(node) in [GetItemNode, GetAttrNode, AttributeNode, EqNode, AssertNode, ScalarAddNode, ScalarFloorDivNode, ScalarMulNode, ScalarSubNode, ScalarTrueDivNode]:
                #     print(type(node), ": ", node.name)
                if len(inputs_nodes) > 0:
                    self.ufront_model.input(tensors=inputs_nodes, num_of_inputs=len(inputs_nodes))
                    inputs_nodes = []
                operator = node(self.ufront_model, node_to_output)
                if type(operator) == PyOperator:
                    self.operators.append(operator)
                    if isinstance(node, SplitChunkNode):
                        node_output = []
                        for i in range(operator.num_of_outputs()):
                            node_output.append(operator.get_output(i))
                    elif isinstance(node, MultiheadAttentionNode):
                        node_output = [operator.get_output(0), None] #multiheaded attention return tensor output and weights
                    else:
                        node_output = operator.get_output(0)
                else:
                    node_output = operator

            # Save the node output for later nodes
            if node_output is not None:
                node_to_output[node.name] = node_output

        return output_tensors

