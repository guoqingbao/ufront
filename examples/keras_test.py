from tensorflow.keras import backend
backend.set_image_data_format('channels_first')
from ufront.keras.model import UFrontKeras
from keras_def import SequentialCNN, ConcatenatedCNN, NestedCNN, ShuffleNet, SqueezeNet_11, ResNet18

from tensorflow.keras.applications import ResNet50, ResNet50V2, EfficientNetB0, Xception, MobileNetV2, MobileNetV3Small, DenseNet121, InceptionV3, VGG16
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from vit_keras import vit

if __name__ == "__main__":
    # backend.set_image_data_format('channels_first')
    # num_classes = 10
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # x_train = x_train.astype('float32') / 255
    # y_train = y_train.astype('int32')
    # print("shape: ", x_train.shape)
    
    # inputs, outputs, model_name = SequentialCNN(shape=(3, 32, 32), dtype="float32", num_classes=10)
    # inputs, outputs, model_name = ConcatenatedCNN(shape=(3, 32, 32), dtype="float32", num_classes=10)
    # inputs, outputs, model_name = NestedCNN(shape=(3, 32, 32), dtype="float32", num_classes=10)
    input = layers.Input(shape=(3, 224, 224))
    # base_model = ResNet18(classes=1000, input_shape=(3, 224, 224))
    # base_model = vit.vit_b16(image_size=224, activation='relu', pretrained=False, include_top=True, pretrained_top=False, channel_first=True)
    # base_model = ResNet50(weights=None, include_top=True, input_tensor=input) # no batch norm
    # base_model = ResNet50V2(weights=None, include_top=True, input_tensor=input) # with batch norm

    # base_model = EfficientNetB0(weights=None, include_top=True, input_tensor=input) 

    # base_model = Xception(weights=None, include_top=True, input_tensor=input)
    # base_model = MobileNetV2(weights=None, include_top=True, input_tensor=input)
    # base_model = MobileNetV3Small(weights=None, include_top=True, input_tensor=input)

    # base_model = DenseNet121(weights=None, include_top=True, input_tensor=input)
    # base_model = InceptionV3(weights=None, include_top=True, input_tensor=input)
    # base_model = VGG16(weights=None, include_top=True, input_tensor=input)

    # base_model = SqueezeNet_11(input_shape=(3, 224, 224), nb_classes=1000, channel_first=True)
    base_model = ShuffleNet(include_top=True, input_shape=(3, 224, 224))
    
    model_name = base_model.name

    inputs, outputs = {1:base_model.input}, base_model.output

    # model =  Model(inputs=inputs, outputs=outputs)
    # print(model.summary())
    transformer = True if model_name.find("Transformer") > 0 or model_name.find("vit") >= 0 else False

    model = UFrontKeras(inputs = inputs, outputs = outputs, batch_size = 1, transformer=transformer)

    if transformer:
      last_op = model.get_output_operator()
      output = model.umodel().softmax(input=last_op.get_output(0), name="softmax_out")

    model.compile(optimizer={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"},
                        loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
    
    # except Exception as e:
    #     print(e)
    # opt = optimizers.SGD(learning_rate=0.01)
    # model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
    
    # print(model.summary()) 

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

