from tensorflow.keras import backend
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import cifar10

from ufront.keras.model import UFrontKeras
from keras_def import SequentialCNN, ConcatenatedCNN, NestedCNN

if __name__ == "__main__":
    backend.set_image_data_format('channels_first')
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    y_train = y_train.astype('int32')
    print("shape: ", x_train.shape)
    
    # inputs, outputs = SequentialCNN(shape=(3, 32, 32), dtype="float32", num_classes=10)
    # inputs, outputs = ConcatenatedCNN(shape=(3, 32, 32), dtype="float32", num_classes=10)
    inputs, outputs = NestedCNN(shape=(3, 32, 32), dtype="float32", num_classes=10)

    model = UFrontKeras(inputs = inputs, outputs = outputs, batch_size = 32)

    model.compile(optimizer={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"},
                         loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])

    # opt = optimizers.SGD(learning_rate=0.01)
    # model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
    
    print(model.summary()) 

    print(model.dump_ir())
    # model.fit(x_train, y_train, epochs=1)

