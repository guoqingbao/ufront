from tensorflow.keras import backend
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import cifar10

from ufront.keras.model import UFrontKeras
from keras_def import SequentialCNN, ConcatenatedCNN, NestedCNN

from tensorflow.keras.applications import ResNet50, ResNet50V2, EfficientNetB0, Xception, MobileNetV2, MobileNetV3Small, DenseNet121, InceptionV3, VGG16
from tensorflow.keras import layers
from tensorflow.keras.models import Model

if __name__ == "__main__":
    backend.set_image_data_format('channels_first')
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    y_train = y_train.astype('int32')
    print("shape: ", x_train.shape)
    
    # inputs, outputs, model_name = SequentialCNN(shape=(3, 32, 32), dtype="float32", num_classes=10)
    # inputs, outputs, model_name = ConcatenatedCNN(shape=(3, 32, 32), dtype="float32", num_classes=10)
    # inputs, outputs, model_name = NestedCNN(shape=(3, 32, 32), dtype="float32", num_classes=10)
    input = layers.Input(shape=(3, 224, 224))

    # base_model = ResNet50(weights=None, include_top=True, input_tensor=input)
    # base_model = ResNet50V2(weights=None, include_top=True, input_tensor=input)

    # base_model = EfficientNetB0(weights=None, include_top=True, input_tensor=input) 

    base_model = Xception(weights=None, include_top=True, input_tensor=input)
    # base_model = MobileNetV2(weights=None, include_top=True, input_tensor=input)
    # base_model = MobileNetV3Small(weights=None, include_top=True, input_tensor=input)

    # base_model = DenseNet121(weights=None, include_top=True, input_tensor=input)
    # base_model = InceptionV3(weights=None, include_top=True, input_tensor=input)
    # base_model = VGG17(weights=None, include_top=True, input_tensor=input)


    model_name = base_model.name

    inputs, outputs = {1:base_model.input}, base_model.output

    # model =  Model(inputs=inputs, outputs=outputs)
    # print(model.summary())

    model = UFrontKeras(inputs = inputs, outputs = outputs, batch_size = 32)

    model.compile(optimizer={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"},
                         loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])

    # opt = optimizers.SGD(learning_rate=0.01)
    # model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
    
    print(model.summary()) 

    print("\r\n\r\nIR for ", model_name)

    # for operator in operators:
    #   print(operator.ir) #show ir for each operator
    modelir= model.dump_ir()
    print(modelir)

    import pathlib
    path = str(pathlib.Path(__file__).parent.resolve()) + "/output_ir/keras_" + model_name + ".ir"
    f = open(path, "w")
    f.write(modelir)
    f.close()

