
import ufront
import numpy as np;
from ufront import OpType, PoolType, LossType, MetricsType, Optimizer

model = ufront.Model()
batch_size = 1

input = np.ones((batch_size,3,32,32), dtype=np.float32)

tensor_input1 = ufront.TensorF32(input, name="input1")
tensor_input2 = ufront.TensorF32(input, name="input2")

x = model.conv2d(input=tensor_input1, out_channels=32, kernel=[3, 3], stride=[1, 1], pad=[0, 0], groups=1)
x1 = model.relu(input=x.get_output(0))


x = model.conv2d(input=tensor_input2, out_channels=32, kernel=[3, 3], stride=[1, 1], pad=[0, 0], groups=1)
x2 = model.relu(input=x.get_output(0))

x = model.concat(tensors=[x1.get_output(0), x2.get_output(0)], axis=1)

x = model.split(input=x.get_output(0), sizes = [32, 32], axis=1)

x = model.concat(tensors=[x.get_output(0), x.get_output(1)], axis=1)

x = model.conv2d(input=x.get_output(0), out_channels=64, kernel=[3, 3], stride=[1, 1], pad=[0, 0], groups=1)
x = model.relu(input=x.get_output(0))
x = model.pool2d(input=x.get_output(0), kernel=[2, 2], stride=[2, 2], pad=[0, 0], pool_type=PoolType.POOL_MAX)

x = model.conv2d(input=x.get_output(0), out_channels=64, kernel=[3, 3], stride=[1, 1], pad=[0, 0], groups=1)
x = model.relu(input=x.get_output(0))

x = model.conv2d(input=x.get_output(0), out_channels=64, kernel=[3, 3], stride=[1, 1], pad=[0, 0], groups=1)
x = model.relu(input=x.get_output(0))
x = model.pool2d(input=x.get_output(0), kernel=[2, 2], stride=[2, 2], pad=[0, 0], pool_type=PoolType.POOL_MAX)

x = model.flat(input=x.get_output(0))
x = model.dense(input=x.get_output(0), out_dim=512)

x = model.relu(input=x.get_output(0))
x = model.dense(input=x.get_output(0), out_dim=10)
x = model.softmax(input=x.get_output(0))

# operator = ufront.PyOperator(OpType.CONV2D, {"input_channel":"1", "output_channel":"16", "kernel_size":"[3,3]"})
# model.add_operator(operator)


# tensorf32 = ufront.TensorF32(arr)


model.optimizer = Optimizer(params={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"})

model.compile(loss=LossType.CATEGORICAL_CROSSENTROPY, metrics=[MetricsType.ACCURACY, MetricsType.SPARSE_CATEGORICAL_CROSSENTROPY])
# model.forward()

model_name = "ComplexCNN"
print("No. of operators: ", model.num_of_operators())

print("\r\n\r\nIR for ", model_name)

# for operator in operators:
#   print(operator.ir) #show ir for each operator
modelir= model.dump_ir()
print(modelir)

import pathlib
path = str(pathlib.Path(__file__).parent.resolve()) + "/output_ir/native_" + model_name + ".ir"
f = open(path, "w")
f.write(modelir)
f.close()

