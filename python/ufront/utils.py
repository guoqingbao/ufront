import numpy as np
from functools import reduce
from .ufront import (OpType, ActiMode, AggrMode, PoolType, Tensor, DataType, ParamSyncType, Initializer)
try:
    import onnx
except:
    print("You need to first install onnx package before compiling onnx models!")

try:
    import torch
except:
    print("You need to first install pytorch package before compiling pytorch models!")

def list_product(lst):
    return reduce(lambda x, y: x*y, lst)

def onnx_to_ufront_dtype(datatype):
    if datatype == onnx.TensorProto.FLOAT:
        return DataType.Float
    elif datatype == onnx.TensorProto.DOUBLE:
        return DataType.Double
    elif datatype == onnx.TensorProto.INT32:
        return DataType.Int32
    elif datatype == onnx.TensorProto.INT64:
        return DataType.Int64
    elif datatype == onnx.TensorProto.FLOAT16:
        return DataType.Half
    elif datatype == onnx.TensorProto.BOOL:
        return DataType.Bool
    else:
        assert 0, "Unsupported datatype"

def torch_to_ufront_dtype(torch_dtype):
    if torch_dtype == torch.float32:
        return DataType.Float
    elif torch_dtype == torch.float64:
        return DataType.Double
    elif torch_dtype == torch.int32:
        return DataType.Int32
    elif torch_dtype == torch.int64:
        return DataType.Int64
    elif torch_dtype == torch.half:
        return DataType.Half
    elif torch_dtype == torch.bfloat16:
        return DataType.BHalf
    elif torch_dtype == torch.bool:
        return DataType.Bool
    else:
        assert 0, f"Unknown dtype: {torch_dtype}"

def numpy_to_ufront_dtype(numpy_dtype):
    if numpy_dtype in (np.float32, "float32", "float"):
        return DataType.Float
    elif numpy_dtype in (np.float64, np.double, "float64", "double"):
        return DataType.Double
    elif numpy_dtype in (np.int32, "int32"):
        return DataType.Int32
    elif numpy_dtype in (np.int64, "int64", "long"):
        return DataType.Int64
    elif numpy_dtype in (np.float16, np.half, "float16", "half"):
        return DataType.Half
    elif hasattr(numpy_dtype, "name") and numpy_dtype.name == "bfloat16":
        return DataType.BHalf
    elif numpy_dtype in (bool, "bool", "BOOL", "boolean", "Boolean"):
        return DataType.Bool
    else:
        assert 0, f"Unknown dtype: {numpy_dtype}"

def ufront_to_numpy_dtype(ff_dtype):
    if ff_dtype ==DataType.Float:
        return np.float32
    elif ff_dtype ==DataType.Double:
        return np.float64
    elif ff_dtype == DataType.Int32:
        return np.int32
    elif ff_dtype == DataType.Int64:
        return np.int64
    elif ff_dtype == DataType.Half:
        return np.float16
    elif ff_dtype == DataType.Bool:
        return bool
    else:
        assert 0, f"Unknown dtype: {ff_dtype}"