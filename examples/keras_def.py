from tensorflow.keras import backend
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Input, Concatenate
from tensorflow.keras import Model

def SequentialCNN(shape=(3, 32, 32), dtype="float32", num_classes = 10):
    input_tensor1 = Input(shape=shape, dtype=dtype)
    
    output_tensor = Conv2D(filters=32, input_shape=(3,32,32), kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu")(input_tensor1)
    output_tensor = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu")(output_tensor)
    output_tensor = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor)
    output_tensor = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu")(output_tensor)
    output_tensor = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu")(output_tensor)
    output_tensor = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor)
    output_tensor = Flatten()(output_tensor)
    output_tensor = Dense(512, activation="relu")(output_tensor)
    output_tensor = Dense(num_classes)(output_tensor)
    output_tensor = Activation("softmax")(output_tensor)

    return {1: input_tensor1}, output_tensor

def ConcatenatedCNN(shape=(3, 32, 32), dtype="float32", num_classes = 10):
    input_tensor1 = Input(shape=shape, dtype=dtype)
    
    o1 = Conv2D(filters=32, input_shape=(3,32,32), kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu")(input_tensor1)
    o2 = Conv2D(filters=32, input_shape=(3,32,32), kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu")(input_tensor1)
    output_tensor = Concatenate(axis=1)([o1, o2])
    output_tensor = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu")(output_tensor)
    output_tensor = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor)
    output_tensor = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu")(output_tensor)
    output_tensor = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu")(output_tensor)
    output_tensor = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor)
    output_tensor = Flatten()(output_tensor)
    output_tensor = Dense(512, activation="relu")(output_tensor)
    output_tensor = Dense(num_classes)(output_tensor)
    output_tensor = Activation("softmax")(output_tensor)

    return {1: input_tensor1}, output_tensor

def NestedCNN(shape=(3, 32, 32), dtype="float32", num_classes = 10):
    input_tensor1 = Input(shape=shape, dtype=dtype)
    output_tensor1 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu")(input_tensor1)
    output_tensor1 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu")(output_tensor1)
    output_tensor1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor1)
    model1 = Model(input_tensor1, output_tensor1)
    
    input_tensor2 = Input(shape=(32, 14, 14), dtype="float32")
    output_tensor2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu")(input_tensor2)
    output_tensor2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="valid", activation="relu")(output_tensor2)
    output_tensor2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(output_tensor2)
    output_tensor2 = Flatten()(output_tensor2)
    output_tensor2 = Dense(512, activation="relu")(output_tensor2)
    output_tensor2 = Dense(num_classes)(output_tensor2)
    output_tensor2 = Activation("softmax")(output_tensor2)
    model2 = Model(input_tensor2, output_tensor2)
    
    input_tensor3 = Input(shape=(3, 32, 32), dtype="float32")
    output_tensor3 = model1(input_tensor3)
    output_tensor3 = model2(output_tensor3)
    
    return {3: input_tensor3}, output_tensor3