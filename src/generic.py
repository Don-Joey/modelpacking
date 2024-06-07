import os
import torch
import shutil
from torch.utils.mobile_optimizer import optimize_for_mobile
from PIL import Image
test_benchs = {
    "resnet50": "ImageClassification",
    "inception_v3": "ImageClassification",
    "mobilenet_v2": "ImageClassification",
    "vgg16": "ImageClassification",
    "vit_b_16": "ImageClassification",
}
def deserialize_bits_to_directory(bits, target_dir):
    """
    Deserialize the bit string to reconstruct the directory structure and files.
    """
    def bits_to_int(bits):
        """Convert a string of bits to an integer."""
        return int(bits, 2)
    i = 0
    while i < len(bits):
        # Extract path length (32 bits = 4 bytes, each byte = 8 bits)
        path_length_bits = bits[i:i+32]
        path_length = bits_to_int(path_length_bits)
        i += 32
        
        # Extract the file path (path_length bits)
        path_bits = bits[i:i + path_length * 8]  # 8 bits per character
        path = ''.join(chr(bits_to_int(path_bits[j:j+8])) for j in range(0, len(path_bits), 8))
        i += path_length * 8
        
        # Extract content length
        content_length_bits = bits[i:i+32]
        content_length = bits_to_int(content_length_bits)
        i += 32
        
        # Extract file content (content_length bytes)
        content_bits = bits[i:i + content_length * 8]  # 8 bits per byte
        content_bytes = bytes(bits_to_int(content_bits[j:j+8]) for j in range(0, len(content_bits), 8))
        i += content_length * 8
        
        # Reconstruct the file
        full_path = os.path.join(target_dir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'wb') as f:
            f.write(content_bytes)
def serialize_directory_to_bits(dir_path):
    """
    Serialize the directory structure and files to a bit string (conceptually).
    Here we use a simple text representation for demonstration.
    """
    serialized = ""
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, dir_path)
            with open(file_path, 'rb') as f:
                content = f.read()
                # Example encoding: [path_length]:[path][content_length]:[content]
                serialized += f"{len(relative_path)}:{relative_path}{len(content)}:".encode() + content
    return serialized

def read_pytorch_model(cover_model,data_format):
    if cover_model == "deeplabv3_resnet50":
        # use deeplabv3_resnet50 instead of resnet101 to reduce the model size
        model = torch.hub.load('pytorch/vision:v0.7.0', 'deeplabv3_resnet50', pretrained=True)
        model.eval()
    
    if data_format == "float32":
        return torch.jit.script(model)

def pytorch_mobile_deploy(cover_model, data_format, secret, only_load):
    # load cover model
    if only_load:
        cover_model = torch.jit.load(cover_model)
    if data_format == 'float32':
        from src.pymobile_bit import pytorch_float32_embedding_secret
        cover_model = pytorch_float32_embedding_secret(cover_model, secret)
    else:
        from src.pymobile_bit import pytorch_quantize_embedding_secret
        cover_model = pytorch_quantize_embedding_secret(cover_model, secret)
    return cover_model

def tflite_deploy(cover_model, data_format, secret, only_load):
    # load cover model
    if only_load:
        cover_model = torch.jit.load(cover_model)
    if data_format == 'float32':
        from src.tflite_bit import tf_float32_embedding_secret
        cover_model = tf_float32_embedding_secret(cover_model, secret)
    else:
        from src.tflite_bit import tf_quantize_embedding_secret
        cover_model = tf_quantize_embedding_secret(cover_model, secret)
    return cover_model

def onnx_deploy(cover_model, data_format, secret, only_load):
    # load cover model
    if only_load:
        cover_model = torch.jit.load(cover_model)
    if data_format == 'float32':
        from src.onnx_bit import onnx_float32_embedding_secret
        cover_model = onnx_float32_embedding_secret(cover_model, secret)
    else:
        from src.onnx_bit import onnx_quantize_embedding_secret
        cover_model = onnx_quantize_embedding_secret(cover_model, secret)
    return cover_model

def coreml_deploy(cover_model, data_format, secret, only_load):
    # load cover model
    if only_load:
        cover_model = torch.jit.load(cover_model)
    if data_format == 'float32':
        from src.coreml_bit import coreml_float32_embedding_secret
        cover_model = coreml_float32_embedding_secret(cover_model, secret)
    else:
        from src.onnx_bit import coreml_quantize_embedding_secret
        cover_model = coreml_quantize_embedding_secret(cover_model, secret)
    return cover_model

def testing_cover_model(cover_model, framework, cover_model_name):
    def test_api(benchname, framework):
        if framework == "pytorch-mobile":
            if benchname=="ImageClassification":
                from cv.imageclassification import py_evaluate_model
                return py_evaluate_model
            elif benchname=="QA":
                return
            elif benchname=="Speech":
                return 
        elif framework == "tflite":
            if benchname=="ImageClassification":
                from cv.imageclassification import tf_evaluate_model
                return tf_evaluate_model
            elif benchname=="QA":
                return
            elif benchname=="Speech":
                return 
        elif framework == "onnx":
            if benchname=="ImageClassification":
                from cv.imageclassification import onnx_evaluate_model
                return onnx_evaluate_model
            elif benchname=="QA":
                return
            elif benchname=="Speech":
                return 
        elif framework == "coreml":
            if benchname=="ImageClassification":
                from cv.imageclassification import coreml_evaluate_model
                return coreml_evaluate_model
            elif benchname=="QA":
                return
            elif benchname=="Speech":
                return 
    return test_api(test_benchs[cover_model_name], framework)(cover_model)

def testing_secret():
    pass

def load_cover_model(cover_model, framework, data_format):
    if cover_model == "resnet50" and  framework=="pytorch-mobile":
        from torchvision.models import resnet50
        cover_model = resnet50(pretrained=True)
        if data_format == "float32":
            cover_model.eval()
            #scripted_cover_model = torch.jit.script(cover_model)
            #cover_model = optimize_for_mobile(scripted_cover_model)
        elif data_format == "quantized":
            #cover_model.to('cpu')
            cover_model.eval()
            backend = "x86"
            cover_model.qconfig = torch.quantization.get_default_qconfig(backend)
            torch.backends.quantized.engine = backend
            model_static_quantized = torch.quantization.prepare(cover_model, inplace=False)
            #activation parameter
            input_fp32 = torch.randn(64, 3, 224, 224)
            model_static_quantized(input_fp32)
            model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
            cover_model = model_static_quantized#torch.jit.script(model_static_quantized)
    if cover_model == "resnet50" and framework=="tflite":
        from keras.applications.resnet50 import ResNet50
        import tensorflow as tf
        import pathlib
        cover_model = ResNet50(weights='imagenet')
        tf.keras.models.save_model(cover_model, "/data/qiyuan/smallmodels/resnet50/")
        if data_format == "float32":
            converter = tf.lite.TFLiteConverter.from_saved_model("/data/qiyuan/smallmodels/resnet50")
            tflite_model = converter.convert()
            tflite_model_file = pathlib.Path("/data/qiyuan/smallmodels/resnet50.tflite")
            tflite_model_file.write_bytes(tflite_model)
            cover_model = '/data/qiyuan/smallmodels/resnet50.tflite'
        elif data_format == "quantized":
            converter = tf.lite.TFLiteConverter.from_saved_model("/data/qiyuan/smallmodels/resnet50")
            tflite_model = converter.convert()
            tflite_model_file = pathlib.Path("/data/qiyuan/smallmodels/resnet50.tflite")
            tflite_model_file.write_bytes(tflite_model)
            interpreter = tf.lite.TFLiteConverter.from_saved_model('/data/qiyuan/smallmodels/resnet50')
            interpreter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_quant_model = interpreter.convert()
            tflite_model_quant_file = pathlib.Path("/data/qiyuan/smallmodels/resnet50_quant.tflite")
            tflite_model_quant_file.write_bytes(tflite_quant_model)
            cover_model = '/data/qiyuan/smallmodels/resnet50_quant.tflite'
    if cover_model == "resnet50" and framework=="onnx":
        from torchvision.models import resnet50
        cover_model = resnet50(weights="IMAGENET1K_V2")
        cover_model.eval()
        if data_format == "float32":
            dummy_input = torch.randn(1, 3, 224, 224)
            onnx_model_path = '/data/qiyuan/smallmodels/resnet50.onnx'
            torch.onnx.export(cover_model,               # model being run
                            dummy_input,         # model input (or a tuple for multiple inputs)
                            onnx_model_path,     # where to save the model (can be a file or file-like object)
                            export_params=True,  # store the trained parameter weights inside the model file
                            opset_version=11,    # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'], # the model's output names
                            dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                            'output' : {0 : 'batch_size'}})
            cover_model = onnx_model_path
        elif data_format == "quantized":
            from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static
            dummy_input = torch.randn(1, 3, 224, 224)
            onnx_model_path = '/data/qiyuan/smallmodels/resnet50.onnx'
            torch.onnx.export(cover_model,               # model being run
                            dummy_input,         # model input (or a tuple for multiple inputs)
                            onnx_model_path,     # where to save the model (can be a file or file-like object)
                            export_params=True,  # store the trained parameter weights inside the model file
                            opset_version=11,    # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'], # the model's output names
                            dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                            'output' : {0 : 'batch_size'}})
            # Define the path to the original ONNX model and the output path for the quantized model
            input_model_path = onnx_model_path
            output_model_path = '/data/qiyuan/smallmodels/resnet50_quan.onnx'

            # Perform dynamic quantization (Note: Other methods like static quantization are also available)
            quantized_model = quantize_dynamic(model_input=input_model_path,
                            model_output=output_model_path,
                            weight_type=QuantType.QUInt8)
            cover_model = output_model_path
    if cover_model == "resnet50" and framework=="coreml":
        from torchvision.models import resnet50
        from shutil import copytree
        cover_model = resnet50(weights="IMAGENET1K_V2")
        cover_model.eval()
        if data_format == "float32":
            import coremltools as ct
            
            # Trace the model with random data.
            example_input = torch.rand(1, 3, 224, 224) 
            traced_model = torch.jit.trace(cover_model, example_input)
            '''
            import urllib
            label_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
            class_labels = urllib.request.urlopen(label_url).read().decode("utf-8").splitlines()
            class_labels = class_labels[1:] # remove the first class which is background
            assert len(class_labels) == 1000
            scale = 1/(0.226*255.0)
            bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]

            image_input = ct.ImageType(name="input_1",
                                    shape=example_input.shape,
                                    scale=scale, bias=bias)

            # Using image_input in the inputs parameter:
            # Convert to Core ML program using the Unified Conversion API.
            cover_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=example_input.shape)],
                classifier_config = ct.ClassifierConfig(class_labels),
                convert_to="neuralnetwork"
            )
            cover_model.save("/data/qiyuan/smallmodels/ResNet50.mlmodel")
            cover_model = "/data/qiyuan/smallmodels/ResNet50.mlmodel"
            '''
            # Convert using the same API. Note that we need to provide "inputs" for pytorch conversion.
            '''
            model_from_torch = ct.convert(
                traced_model,
                convert_to="mlprogram",
				inputs=[ct.TensorType(name="input", shape=example_input.shape)]
            )
            model_from_torch.save("/data/qiyuan/smallmodels/ResNet50.mlpackage")
            cover_model = "/data/qiyuan/smallmodels/ResNet50.mlpackage"
            mlmodel = ct.models.MLModel(cover_model, compute_units=ct.ComputeUnit.CPU_ONLY)
            compiled_model_path = mlmodel.get_compiled_model_path()
            print(compiled_model_path)
            exit()
            '''
            onnx_path = "resnet50.onnx"
            torch.onnx.export(cover_model,               # PyTorch Model
                  example_input,                   # Model input (or a tuple for multiple inputs)
                  onnx_path,           # Where to save the model
                  export_params=True,  # Store the trained parameter weights inside the model file
                  opset_version=11,    # The ONNX version to export the model to
                  do_constant_folding=True,  # Whether to execute constant folding for optimization
                  input_names=['input'],   # The model's input names
                  output_names=['output'], # The model's output names
                  dynamic_axes={'input': {0: 'batch_size'},    # Variable length axes
                                'output': {0: 'batch_size'}})
            from onnx import onnx_pb
            model_file = open("/data/qiyuan/smallmodels/resnet50.onnx", 'rb')
            model_proto = onnx_pb.ModelProto()
            coreml_model = ct.convert(model_proto, source='auto')
            
        elif data_format == "quantized":
            from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static
            dummy_input = torch.randn(1, 3, 224, 224)
            onnx_model_path = '/data/qiyuan/smallmodels/resnet50.onnx'
            torch.onnx.export(cover_model,               # model being run
                            dummy_input,         # model input (or a tuple for multiple inputs)
                            onnx_model_path,     # where to save the model (can be a file or file-like object)
                            export_params=True,  # store the trained parameter weights inside the model file
                            opset_version=11,    # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'], # the model's output names
                            dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                            'output' : {0 : 'batch_size'}})
            # Define the path to the original ONNX model and the output path for the quantized model
            input_model_path = onnx_model_path
            output_model_path = '/data/qiyuan/smallmodels/resnet50_quan.onnx'

            # Perform dynamic quantization (Note: Other methods like static quantization are also available)
            quantized_model = quantize_dynamic(model_input=input_model_path,
                            model_output=output_model_path,
                            weight_type=QuantType.QUInt8)
            cover_model = output_model_path
    if cover_model == "inception_v3" and  framework=="pytorch-mobile":
        from torchvision.models import inception_v3
        cover_model = inception_v3(weights="IMAGENET1K_V1")
        if data_format == "float32":
            cover_model.eval()
            #scripted_cover_model = torch.jit.script(cover_model)
            #cover_model = optimize_for_mobile(scripted_cover_model)
        elif data_format == "quantized":
            #cover_model.to('cpu')
            cover_model.eval()
            backend = "x86"
            cover_model.qconfig = torch.quantization.get_default_qconfig(backend)
            torch.backends.quantized.engine = backend
            model_static_quantized = torch.quantization.prepare(cover_model, inplace=False)
            #activation parameter
            input_fp32 = torch.randn(64, 3, 224, 224)
            model_static_quantized(input_fp32)
            model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
            cover_model = torch.jit.script(model_static_quantized)
    if cover_model == "inception_v3" and framework=="tflite":
        from keras.applications.inception_v3 import InceptionV3
        import tensorflow as tf
        import pathlib
        cover_model = InceptionV3(weights='imagenet')
        tf.keras.models.save_model(cover_model, "/data/qiyuan/smallmodels/inception_v3/")
        if data_format == "float32":
            converter = tf.lite.TFLiteConverter.from_saved_model("/data/qiyuan/smallmodels/inception_v3")
            tflite_model = converter.convert()
            tflite_model_file = pathlib.Path("/data/qiyuan/smallmodels/inception_v3.tflite")
            tflite_model_file.write_bytes(tflite_model)
            cover_model = '/data/qiyuan/smallmodels/inception_v3.tflite'
        elif data_format == "quantized":
            converter = tf.lite.TFLiteConverter.from_saved_model("/data/qiyuan/smallmodels/inception_v3")
            tflite_model = converter.convert()
            tflite_model_file = pathlib.Path("/data/qiyuan/smallmodels/inception_v3.tflite")
            tflite_model_file.write_bytes(tflite_model)
            interpreter = tf.lite.TFLiteConverter.from_saved_model('/data/qiyuan/smallmodels/inception_v3')
            interpreter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_quant_model = interpreter.convert()
            tflite_model_quant_file = pathlib.Path("/data/qiyuan/smallmodels/inception_v3_quant.tflite")
            tflite_model_quant_file.write_bytes(tflite_quant_model)
            cover_model = '/data/qiyuan/smallmodels/inception_v3_quant.tflite'
    if cover_model == "inception_v3" and framework=="onnx":
        from torchvision.models import inception_v3
        cover_model = inception_v3(weights="IMAGENET1K_V1")
        cover_model.eval()
        if data_format == "float32":
            dummy_input = torch.randn(1, 3, 224, 224)
            onnx_model_path = '/data/qiyuan/smallmodels/inception_v3.onnx'
            torch.onnx.export(cover_model,               # model being run
                            dummy_input,         # model input (or a tuple for multiple inputs)
                            onnx_model_path,     # where to save the model (can be a file or file-like object)
                            export_params=True,  # store the trained parameter weights inside the model file
                            opset_version=11,    # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'], # the model's output names
                            dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                            'output' : {0 : 'batch_size'}})
            cover_model = onnx_model_path
        elif data_format == "quantized":
            from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static
            dummy_input = torch.randn(1, 3, 224, 224)
            onnx_model_path = '/data/qiyuan/smallmodels/inception_v3.onnx'
            torch.onnx.export(cover_model,               # model being run
                            dummy_input,         # model input (or a tuple for multiple inputs)
                            onnx_model_path,     # where to save the model (can be a file or file-like object)
                            export_params=True,  # store the trained parameter weights inside the model file
                            opset_version=11,    # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'], # the model's output names
                            dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                            'output' : {0 : 'batch_size'}})
            # Define the path to the original ONNX model and the output path for the quantized model
            input_model_path = onnx_model_path
            output_model_path = '/data/qiyuan/smallmodels/inception_v3_quan.onnx'

            # Perform dynamic quantization (Note: Other methods like static quantization are also available)
            quantized_model = quantize_dynamic(model_input=input_model_path,
                            model_output=output_model_path,
                            weight_type=QuantType.QUInt8)
            cover_model = output_model_path
    if cover_model == "mobilenet_v2" and  framework=="pytorch-mobile":
        from torchvision.models import mobilenet_v2
        cover_model = mobilenet_v2(weights="IMAGENET1K_V2")
        if data_format == "float32":
            cover_model.eval()
            #scripted_cover_model = torch.jit.script(cover_model)
            #cover_model = optimize_for_mobile(scripted_cover_model)
        elif data_format == "quantized":
            #cover_model.to('cpu')
            cover_model.eval()
            backend = "x86"
            cover_model.qconfig = torch.quantization.get_default_qconfig(backend)
            torch.backends.quantized.engine = backend
            model_static_quantized = torch.quantization.prepare(cover_model, inplace=False)
            #activation parameter
            input_fp32 = torch.randn(64, 3, 224, 224)
            model_static_quantized(input_fp32)
            model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
            cover_model = torch.jit.script(model_static_quantized)
    if cover_model == "mobilenet_v2" and framework=="tflite":
        from keras.applications.mobilenet_v2 import MobileNetV2
        import tensorflow as tf
        import pathlib
        cover_model = MobileNetV2(weights='imagenet')
        tf.keras.models.save_model(cover_model, "/data/qiyuan/smallmodels/mobilenet_v2/")
        if data_format == "float32":
            converter = tf.lite.TFLiteConverter.from_saved_model("/data/qiyuan/smallmodels/mobilenet_v2")
            tflite_model = converter.convert()
            tflite_model_file = pathlib.Path("/data/qiyuan/smallmodels/mobilenet_v2.tflite")
            tflite_model_file.write_bytes(tflite_model)
            cover_model = '/data/qiyuan/smallmodels/mobilenet_v2.tflite'
        elif data_format == "quantized":
            converter = tf.lite.TFLiteConverter.from_saved_model("/data/qiyuan/smallmodels/mobilenet_v2")
            tflite_model = converter.convert()
            tflite_model_file = pathlib.Path("/data/qiyuan/smallmodels/mobilenet_v2.tflite")
            tflite_model_file.write_bytes(tflite_model)
            interpreter = tf.lite.TFLiteConverter.from_saved_model('/data/qiyuan/smallmodels/mobilenet_v2')
            interpreter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_quant_model = interpreter.convert()
            tflite_model_quant_file = pathlib.Path("/data/qiyuan/smallmodels/mobilenet_v2_quant.tflite")
            tflite_model_quant_file.write_bytes(tflite_quant_model)
            cover_model = '/data/qiyuan/smallmodels/mobilenet_v2_quant.tflite'
    if cover_model == "mobilenet_v2" and framework=="onnx":
        from torchvision.models import mobilenet_v2
        cover_model = mobilenet_v2(weights="IMAGENET1K_V2")
        cover_model.eval()
        if data_format == "float32":
            dummy_input = torch.randn(1, 3, 224, 224)
            onnx_model_path = '/data/qiyuan/smallmodels/mobilenet_v2.onnx'
            torch.onnx.export(cover_model,               # model being run
                            dummy_input,         # model input (or a tuple for multiple inputs)
                            onnx_model_path,     # where to save the model (can be a file or file-like object)
                            export_params=True,  # store the trained parameter weights inside the model file
                            opset_version=11,    # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'], # the model's output names
                            dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                            'output' : {0 : 'batch_size'}})
            cover_model = onnx_model_path
        elif data_format == "quantized":
            from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static
            dummy_input = torch.randn(1, 3, 224, 224)
            onnx_model_path = '/data/qiyuan/smallmodels/mobilenet_v2.onnx'
            torch.onnx.export(cover_model,               # model being run
                            dummy_input,         # model input (or a tuple for multiple inputs)
                            onnx_model_path,     # where to save the model (can be a file or file-like object)
                            export_params=True,  # store the trained parameter weights inside the model file
                            opset_version=11,    # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'], # the model's output names
                            dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                            'output' : {0 : 'batch_size'}})
            # Define the path to the original ONNX model and the output path for the quantized model
            input_model_path = onnx_model_path
            output_model_path = '/data/qiyuan/smallmodels/mobilenet_v2_quan.onnx'

            # Perform dynamic quantization (Note: Other methods like static quantization are also available)
            quantized_model = quantize_dynamic(model_input=input_model_path,
                            model_output=output_model_path,
                            weight_type=QuantType.QUInt8)
            cover_model = output_model_path
    if cover_model == "vgg16" and  framework=="pytorch-mobile":
        from torchvision.models import vgg16
        cover_model = vgg16(weights="IMAGENET1K_V1")
        if data_format == "float32":
            cover_model.eval()
            #scripted_cover_model = torch.jit.script(cover_model)
            #cover_model = optimize_for_mobile(scripted_cover_model)
        elif data_format == "quantized":
            #cover_model.to('cpu')
            cover_model.eval()
            backend = "x86"
            cover_model.qconfig = torch.quantization.get_default_qconfig(backend)
            torch.backends.quantized.engine = backend
            model_static_quantized = torch.quantization.prepare(cover_model, inplace=False)
            #activation parameter
            input_fp32 = torch.randn(64, 3, 224, 224)
            model_static_quantized(input_fp32)
            model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
            cover_model = torch.jit.script(model_static_quantized)
    if cover_model == "vgg16" and framework=="tflite":
        from keras.applications.vgg16 import VGG16
        import tensorflow as tf
        import pathlib
        cover_model = VGG16(weights='imagenet')
        tf.keras.models.save_model(cover_model, "/data/qiyuan/smallmodels/vgg16/")
        if data_format == "float32":
            converter = tf.lite.TFLiteConverter.from_saved_model("/data/qiyuan/smallmodels/vgg16")
            tflite_model = converter.convert()
            tflite_model_file = pathlib.Path("/data/qiyuan/smallmodels/vgg16.tflite")
            tflite_model_file.write_bytes(tflite_model)
            cover_model = '/data/qiyuan/smallmodels/vgg16.tflite'
        elif data_format == "quantized":
            converter = tf.lite.TFLiteConverter.from_saved_model("/data/qiyuan/smallmodels/vgg16")
            tflite_model = converter.convert()
            tflite_model_file = pathlib.Path("/data/qiyuan/smallmodels/vgg16.tflite")
            tflite_model_file.write_bytes(tflite_model)
            interpreter = tf.lite.TFLiteConverter.from_saved_model('/data/qiyuan/smallmodels/vgg16')
            interpreter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_quant_model = interpreter.convert()
            tflite_model_quant_file = pathlib.Path("/data/qiyuan/smallmodels/vgg16_quant.tflite")
            tflite_model_quant_file.write_bytes(tflite_quant_model)
            cover_model = '/data/qiyuan/smallmodels/vgg16_quant.tflite'
    if cover_model == "vgg16" and framework=="onnx":
        from torchvision.models import vgg16
        cover_model = vgg16(weights="IMAGENET1K_V1")
        cover_model.eval()
        if data_format == "float32":
            dummy_input = torch.randn(1, 3, 224, 224)
            onnx_model_path = '/data/qiyuan/smallmodels/vgg16.onnx'
            torch.onnx.export(cover_model,               # model being run
                            dummy_input,         # model input (or a tuple for multiple inputs)
                            onnx_model_path,     # where to save the model (can be a file or file-like object)
                            export_params=True,  # store the trained parameter weights inside the model file
                            opset_version=11,    # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'], # the model's output names
                            dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                            'output' : {0 : 'batch_size'}})
            cover_model = onnx_model_path
        elif data_format == "quantized":
            from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static
            dummy_input = torch.randn(1, 3, 224, 224)
            onnx_model_path = '/data/qiyuan/smallmodels/vgg16.onnx'
            torch.onnx.export(cover_model,               # model being run
                            dummy_input,         # model input (or a tuple for multiple inputs)
                            onnx_model_path,     # where to save the model (can be a file or file-like object)
                            export_params=True,  # store the trained parameter weights inside the model file
                            opset_version=11,    # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'], # the model's output names
                            dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                            'output' : {0 : 'batch_size'}})
            # Define the path to the original ONNX model and the output path for the quantized model
            input_model_path = onnx_model_path
            output_model_path = '/data/qiyuan/smallmodels/vgg16_quan.onnx'

            # Perform dynamic quantization (Note: Other methods like static quantization are also available)
            quantized_model = quantize_dynamic(model_input=input_model_path,
                            model_output=output_model_path,
                            weight_type=QuantType.QUInt8)
            cover_model = output_model_path
    if cover_model == "vit_b_16" and  framework=="pytorch-mobile":
        from torchvision.models import vit_b_16
        cover_model = vit_b_16(weights="IMAGENET1K_V1")
        if data_format == "float32":
            cover_model.eval()
            #scripted_cover_model = torch.jit.script(cover_model)
            #cover_model = optimize_for_mobile(scripted_cover_model)
        elif data_format == "quantized":
            #cover_model.to('cpu')
            cover_model.eval()
            backend = "x86"
            cover_model.qconfig = torch.quantization.get_default_qconfig(backend)
            torch.backends.quantized.engine = backend
            model_static_quantized = torch.quantization.prepare(cover_model, inplace=False)
            #activation parameter
            #input_fp32 = torch.randn(64, 3, 224, 224)
            #model_static_quantized(input_fp32)
            model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
            cover_model = torch.jit.script(model_static_quantized)
    if cover_model == "vit_b_16" and framework=="tflite":
        from transformers import AutoImageProcessor, TFViTForImageClassification
        import tensorflow as tf
        import pathlib
        cover_model = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        cover_model.save_pretrained("/data/qiyuan/smallmodels/vit_b_16/")
        if data_format == "float32":
            converter = tf.lite.TFLiteConverter.from_saved_model("/data/qiyuan/smallmodels/vit_b_16")
            tflite_model = converter.convert()
            tflite_model_file = pathlib.Path("/data/qiyuan/smallmodels/vit_b_16.tflite")
            tflite_model_file.write_bytes(tflite_model)
            cover_model = '/data/qiyuan/smallmodels/vit_b_16.tflite'
        elif data_format == "quantized":
            converter = tf.lite.TFLiteConverter.from_saved_model("/data/qiyuan/smallmodels/vit_b_16")
            tflite_model = converter.convert()
            tflite_model_file = pathlib.Path("/data/qiyuan/smallmodels/vit_b_16.tflite")
            tflite_model_file.write_bytes(tflite_model)
            interpreter = tf.lite.TFLiteConverter.from_saved_model('/data/qiyuan/smallmodels/vit_b_16')
            interpreter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_quant_model = interpreter.convert()
            tflite_model_quant_file = pathlib.Path("/data/qiyuan/smallmodels/vit_b_16_quant.tflite")
            tflite_model_quant_file.write_bytes(tflite_quant_model)
            cover_model = '/data/qiyuan/smallmodels/vit_b_16_quant.tflite'
    if cover_model == "vit_b_16" and framework=="onnx":
        from torchvision.models import vit_b_16
        cover_model = vit_b_16(weights="IMAGENET1K_V1")
        cover_model.eval()
        if data_format == "float32":
            dummy_input = torch.randn(1, 3, 224, 224)
            onnx_model_path = '/data/qiyuan/smallmodels/vit_b_16.onnx'
            torch.onnx.export(cover_model,               # model being run
                            dummy_input,         # model input (or a tuple for multiple inputs)
                            onnx_model_path,     # where to save the model (can be a file or file-like object)
                            export_params=True,  # store the trained parameter weights inside the model file
                            opset_version=11,    # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'], # the model's output names
                            dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                            'output' : {0 : 'batch_size'}})
            cover_model = onnx_model_path
        elif data_format == "quantized":
            from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static
            dummy_input = torch.randn(1, 3, 224, 224)
            onnx_model_path = '/data/qiyuan/smallmodels/vit_b_16.onnx'
            torch.onnx.export(cover_model,               # model being run
                            dummy_input,         # model input (or a tuple for multiple inputs)
                            onnx_model_path,     # where to save the model (can be a file or file-like object)
                            export_params=True,  # store the trained parameter weights inside the model file
                            opset_version=11,    # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'], # the model's output names
                            dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                            'output' : {0 : 'batch_size'}})
            # Define the path to the original ONNX model and the output path for the quantized model
            input_model_path = onnx_model_path
            output_model_path = '/data/qiyuan/smallmodels/vit_b_16_quan.onnx'

            # Perform dynamic quantization (Note: Other methods like static quantization are also available)
            quantized_model = quantize_dynamic(model_input=input_model_path,
                            model_output=output_model_path,
                            weight_type=QuantType.QUInt8)
            cover_model = output_model_path
    return cover_model


def save_cover_model(test_cover_model, framework, cover_model_name, secret):
    if "py" in framework:
        test_cover_model._save_for_lite_interpreter("/home/zhangqiyuan/bitplane_experiment/cover_model/"+cover_model_name+"-"+secret+".pt") 