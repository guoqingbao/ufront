from tensorflow.keras import backend
backend.set_image_data_format('channels_first')
from ufront.keras.model import UFrontKeras
from keras_def import SequentialCNN, ConcatenatedCNN, NestedCNN, ShuffleNet, SqueezeNet_11, ResNet18

from tensorflow.keras.applications import ResNet50, ResNet50V2, EfficientNetB0, Xception, MobileNetV2, MobileNetV3Small, DenseNet121, InceptionV3, VGG16
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from vit_keras import vit
import numpy as np
import iree.compiler as ireec
from iree import runtime
import tensorflow as tf

def decode_result(result):
  return tf.keras.applications.resnet50.decode_predictions(result, top=5)[0]
    
def load_read_image():
    content_path = tf.keras.utils.get_file(
    'YellowLabradorLooking_new.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

    image = tf.io.read_file(content_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = image[tf.newaxis, :]
    return np.moveaxis(image.numpy(), -1, 1)/ 255.0

if __name__ == "__main__":
    GPU = True
    backend.set_image_data_format('channels_first')
    # num_classes = 10
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # x_train = x_train.astype('float32') / 255
    # y_train = y_train.astype('int32')
    # print("shape: ", x_train.shape)
    
    # inputs, outputs, model_name = SequentialCNN(shape=(3, 32, 32), dtype="float32", num_classes=10)
    # inputs, outputs, model_name = ConcatenatedCNN(shape=(3, 32, 32), dtype="float32", num_classes=10)
    # inputs, outputs, model_name = NestedCNN(shape=(3, 32, 32), dtype="float32", num_classes=10)
    input = load_read_image()
    # keras_input = layers.Input(shape=input.shape[1:])

    # base_model = ResNet18(classes=1000, input_shape=(3, 224, 224))
    # base_model = vit.vit_b16(image_size=224, activation='relu', pretrained=False, include_top=True, pretrained_top=False, channel_first=True)
    # base_model = ResNet50(weights=None, include_top=True, input_tensor=keras_input) # no batch norm
    # a = np.moveaxis(input, 1, -1)

    base_model = ResNet50V2(weights="imagenet", include_top=True) # with batch norm

    # base_model = EfficientNetB0(weights=None, include_top=True, input_shape=input.shape[1:]) 

    # base_model = Xception(weights=None, include_top=True, input_shape=input.shape[1:])
    # base_model = MobileNetV2(weights=None, include_top=True, input_shape=input.shape[1:])
    # base_model = MobileNetV3Small(weights=None, include_top=True, input_shape=input.shape[1:])

    # base_model = DenseNet121(weights=None, include_top=True, input_shape=input.shape[1:])
    # base_model = InceptionV3(weights=None, include_top=True, input_shape=input.shape[1:])
    # base_model = VGG16(weights=None, include_top=True, input_shape=input.shape[1:])

    # base_model = SqueezeNet_11(input_shape=input.shape[1:], nb_classes=1000, channel_first=True)
    # base_model = ShuffleNet(include_top=True, input_shape=input.shape[1:])
    # a = np.moveaxis(input, 1, -1)
    # print("TF/Keras: ", decode_result(base_model(tf.constant(a)).numpy()))

    model_name = base_model.name

    # inputs, outputs = {1:input}, base_model.output

    # model =  Model(inputs=inputs, outputs=outputs)
    # print(model.summary())
    transformer = True if model_name.find("Transformer") > 0 or model_name.find("vit") >= 0 else False

    model = UFrontKeras(base_model, inputs = [input], batch_size = 1, transformer=transformer, pass_weights=True)

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

    tosa_ir= model.dump_tosa_ir()

    # import pathlib
    # path = str(pathlib.Path(__file__).parent.resolve()) + "/output_ir/keras_" + model_name + ".ir"
    # f = open(path, "w")
    # f.write(modelir)
    # f.close()

    print("Compiling TOSA model...")
    if GPU:
        binary = ireec.compile_str(tosa_ir,
                        target_backends=["cuda"], 
                        input_type=ireec.InputType.TOSA)
        module = runtime.load_vm_flatbuffer(binary, driver="cuda")
    else:
        binary = ireec.compile_str(tosa_ir,
                        target_backends=["llvm-cpu"], 
                        input_type=ireec.InputType.TOSA)
        module = runtime.load_vm_flatbuffer(binary,backend="llvm-cpu") 

    print("UFront: ", decode_result(module.forward(input).to_host()))