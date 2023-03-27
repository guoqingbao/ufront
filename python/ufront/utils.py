import numpy as np
from functools import reduce
from .ufront import (OpType, ActiMode, AggrMode, PoolType, TensorF32, DataType, ParamSyncType, Initializer)
try:
    import onnx
except:
    print("You need to first install onnx package before using onnx models!")


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

def numpy_to_ufront_dtype(numpy_dtype):
    if numpy_dtype in (np.float32, np.float, "float32", "float"):
        return DataType.Float
    elif numpy_dtype in (np.float64, np.double, "float64", "double"):
        return DataType.Double
    elif numpy_dtype in (np.int32, np.int, "int32", "int"):
        return DataType.Int32
    elif numpy_dtype in (np.int64, np.long, "int64", "long"):
        return DataType.Int64
    elif numpy_dtype in (np.float16, np.half, "float16", "half"):
        return DataType.Half
    elif numpy_dtype in (np.bool, "bool", "BOOL", "boolean", "Boolean"):
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
        return np.bool
    else:
        assert 0, f"Unknown dtype: {ff_dtype}"