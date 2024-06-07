import onnx
from onnx import numpy_helper
import numpy as np
from .bitplane_operation import model_np_32bitplane_edit, model_np_8bitplane_edit, model_np_32bitplane_secret_edit, pack_32bit_onnx_array, pack_8bit_onnx_array
from .bitplane_operation import read_8bit_onnx_array, read_32bit_onnx_array
def onnx_32bit_embedding(onnx_model_name, information=None, fraction=19):
    onnx_model = onnx.load(onnx_model_name)
    onnx.checker.check_model(onnx_model)
    if not information:
        for initializer in onnx_model.graph.initializer:
            # Assuming the parameter is a weight/bias and you want to modify its value
            # Here, we are converting the tensor to a numpy array, modifying it, and converting it back
            # This example sets all values to 0, but you can perform any operation you need
            tensor = numpy_helper.to_array(initializer)
            modified_tensor = np.zeros_like(tensor)
            modified_tensor = model_np_32bitplane_edit(tensor, modified_tensor, fraction=fraction)
            modified_tensor = numpy_helper.from_array(modified_tensor, initializer.name)
            initializer.ClearField('raw_data')
            initializer.raw_data = modified_tensor.raw_data
    else:
        pass
    onnx.save(onnx_model, onnx_model_name[:-len('.onnx')]+"_"+str(fraction)+'.onnx')
    return onnx_model_name[:-len('.onnx')]+"_"+str(fraction)+'.onnx'

def onnx_32bit_secret_embedding(onnx_model_name, secret, information=None, fraction=19):
    onnx_model = onnx.load(onnx_model_name)
    onnx.checker.check_model(onnx_model)
    if not information:
        for initializer in onnx_model.graph.initializer:
            # Assuming the parameter is a weight/bias and you want to modify its value
            # Here, we are converting the tensor to a numpy array, modifying it, and converting it back
            # This example sets all values to 0, but you can perform any operation you need
            tensor = numpy_helper.to_array(initializer)
            modified_tensor = np.zeros_like(tensor)
            modified_tensor, secret = model_np_32bitplane_secret_edit(tensor, modified_tensor, secret, fraction=fraction)
            modified_tensor = numpy_helper.from_array(modified_tensor, initializer.name)
            initializer.ClearField('raw_data')
            initializer.raw_data = modified_tensor.raw_data
    else:
        pass
    onnx.save(onnx_model, onnx_model_name[:-len('.onnx')]+"_"+str(fraction)+'.onnx')
    return onnx_model_name[:-len('.onnx')]+"_"+str(fraction)+'.onnx'



def onnx_quantizedbit_embedding(onnx_model_name, fraction, information=None):
    onnx_model = onnx.load(onnx_model_name)
    onnx.checker.check_model(onnx_model)
    if not information:
        for initializer in onnx_model.graph.initializer:
            tensor = numpy_helper.to_array(initializer)
            if tensor.dtype == "float32":
                modified_tensor = np.zeros_like(tensor)
                if modified_tensor.size == 1:
                    continue
                modified_tensor = model_np_32bitplane_edit(tensor, modified_tensor, fraction=fraction)
                modified_tensor = numpy_helper.from_array(modified_tensor, initializer.name)
                initializer.ClearField('raw_data')
                initializer.raw_data = modified_tensor.raw_data
            elif tensor.dtype == "uint8":
                modified_tensor = np.zeros_like(tensor)
                if modified_tensor.size == 1:
                    continue
                modified_tensor = model_np_8bitplane_edit(tensor, modified_tensor, fraction=fraction)
                modified_tensor = numpy_helper.from_array(modified_tensor, initializer.name)
                initializer.ClearField('raw_data')
                initializer.raw_data = modified_tensor.raw_data
    else:
        pass
    onnx.save(onnx_model, onnx_model_name[:-len('.onnx')]+"_"+str(fraction)+'.onnx')
    return onnx_model_name[:-len('.onnx')]+"_"+str(fraction)+'.onnx'


def onnx_quantizedbit_secret_embedding(onnx_model_name, secret, fraction, information=None):
    onnx_model = onnx.load(onnx_model_name)
    #onnx.checker.check_model(onnx_model)
    if not information:
        for initializer in onnx_model.graph.initializer:
            # Assuming the parameter is a weight/bias and you want to modify its value
            # Here, we are converting the tensor to a numpy array, modifying it, and converting it back
            # This example sets all values to 0, but you can perform any operation you need
            tensor = numpy_helper.to_array(initializer)
            if tensor.dtype == "float32":
                if modified_tensor.size == 1:
                    continue
                modified_tensor = np.zeros_like(tensor)
                modified_tensor,secret = model_np_32bitplane_edit(tensor, modified_tensor, secret, fraction=fraction)
                modified_tensor = numpy_helper.from_array(modified_tensor, initializer.name)
            elif tensor.dtype == "uint8":
                modified_tensor = np.zeros_like(tensor)
                if modified_tensor.size == 1:
                    continue
                modified_tensor,secret = model_np_8bitplane_edit(tensor, modified_tensor, secret, fraction=fraction)
                modified_tensor = numpy_helper.from_array(modified_tensor, initializer.name)
            initializer.ClearField('raw_data')
            initializer.raw_data = modified_tensor.raw_data
    else:
        pass
    onnx.save(onnx_model, onnx_model_name[:-len('.onnx')]+"_"+str(fraction)+'.onnx')
    return onnx_model_name[:-len('.onnx')]+"_"+str(fraction)+'.onnx'

def onnx_float32_embedding_secret(onnx_model_name, secret=None, fraction=19):
    if type(secret) == int:
        fraction=secret
        return onnx_32bit_embedding(onnx_model_name, fraction=fraction)
    else:
        return onnx_32bit_secret_embedding(onnx_model_name, secret=secret, fraction=fraction)



def onnx_quantize_embedding_secret(onnx_model_name, secret=None, fraction=1):
    if type(secret) == int:
        fraction=secret
        return onnx_quantizedbit_embedding(onnx_model_name, fraction=secret)
    else:
        return onnx_quantizedbit_secret_embedding(onnx_model_name, secret=secret, fraction=fraction)


def onnx_float32_pack(onnx_model_name, information=None):
    onnx_model = onnx.load(onnx_model_name)
    onnx.checker.check_model(onnx_model)
    if not information:
        for initializer in onnx_model.graph.initializer:
            # Assuming the parameter is a weight/bias and you want to modify its value
            # Here, we are converting the tensor to a numpy array, modifying it, and converting it back
            # This example sets all values to 0, but you can perform any operation you need
            tensor = numpy_helper.to_array(initializer)
            modified_tensor = np.zeros_like(tensor)
            modified_tensor = pack_32bit_onnx_array(tensor, modified_tensor, fraction=20)
            modified_tensor = numpy_helper.from_array(modified_tensor, initializer.name)
            initializer.ClearField('raw_data')
            initializer.raw_data = modified_tensor.raw_data
    else:
        pass
    onnx.save(onnx_model, onnx_model_name[:-len('.onnx')]+"_packed"+'.onnx')
    return onnx_model_name[:-len('.onnx')]+"_packed.onnx"



def onnx_int8_pack(onnx_model_name, information=None):
    onnx_model = onnx.load(onnx_model_name)
    onnx.checker.check_model(onnx_model)
    if not information:
        for initializer in onnx_model.graph.initializer:
            tensor = numpy_helper.to_array(initializer)
            if tensor.dtype == "uint8":
                modified_tensor = np.zeros_like(tensor)
                if modified_tensor.size == 1:
                    continue
                modified_tensor = pack_8bit_onnx_array(tensor, modified_tensor, fraction=2)
                modified_tensor = numpy_helper.from_array(modified_tensor, initializer.name)
                initializer.ClearField('raw_data')
                initializer.raw_data = modified_tensor.raw_data
    else:
        pass
    onnx.save(onnx_model, onnx_model_name[:-len('.onnx')]+"_packed"+'.onnx')
    return onnx_model_name[:-len('.onnx')]+"_packed.onnx"

def onnx_float32_read(onnx_model_name, information=None):
    onnx_model = onnx.load(onnx_model_name)
    onnx.checker.check_model(onnx_model)
    written_array = []
    if not information:
        for initializer in onnx_model.graph.initializer:
            tensor = numpy_helper.to_array(initializer)
            # Assuming the parameter is a weight/bias and you want to modify its value
            # Here, we are converting the tensor to a numpy array, modifying it, and converting it back
            # This example sets all values to 0, but you can perform any operation you need
            modified_tensor = np.zeros_like(tensor)
            tensor = numpy_helper.to_array(initializer)
            modified_tensor = read_32bit_onnx_array(tensor, modified_tensor, fraction=20)
            written_array.extend(modified_tensor)
    else:
        pass
    print(written_array)
    return written_array


def onnx_int8_read(onnx_model_name, information=None):
    onnx_model = onnx.load(onnx_model_name)
    onnx.checker.check_model(onnx_model)
    written_array = []
    if not information:
        for initializer in onnx_model.graph.initializer:
            tensor = numpy_helper.to_array(initializer)
            modified_tensor = np.zeros_like(tensor)
            if tensor.dtype == "uint8":
                modified_tensor = read_8bit_onnx_array(tensor, modified_tensor, fraction=2)
                written_array.extend(modified_tensor)
            else:
                tensor = numpy_helper.to_array(initializer)
                modified_tensor = read_32bit_onnx_array(tensor, modified_tensor, fraction=20)
                written_array.extend(modified_tensor)
    else:
        pass
    return written_array


def judge_onnx_int8(onnx_model_name, information=None):
    onnx_model = onnx.load(onnx_model_name)
    onnx.checker.check_model(onnx_model)
    if not information:
        for initializer in onnx_model.graph.initializer:
            tensor = numpy_helper.to_array(initializer)
            if tensor.dtype == "uint8":
                return True
    return False