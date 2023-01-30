import ufront
# from pydantic import BaseModel
import numpy as np;
from ufront import OpType

model = ufront.Model()
# from type import OpType
print(OpType.CONV2D)
operator = ufront.PyOperator(OpType.CONV2D, {"input_channel":"1", "output_channel":"16", "kernel_size":"[3,3]"})
model.add_operator(operator)

arr = np.ones((3,3,3), dtype=np.float32)
# tensorf32 = ufront.TensorF32(arr)
operator.add_input_ndarray(arr)


print(operator.raw_ptr)
operator.num_of_inputs()

operator.num_of_outputs()

model.compile()
model.forward()



# tensorf32.set_ndarray(arr)
# print("Obtained dimension: ", tensorf32.get_dims())

# a = tensorf32.get_ndarray()
a = operator.get_input_ndarray(0)
print(a)
print("Retrive tensor from Rust:\n ", a)

print(operator.op_type)

model.remove_operator(operator)

print("No. of operators: ", model.num_of_operators())
