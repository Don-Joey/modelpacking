import copy
import random
import re
import struct
import sys
import os

import tensorflow as tf
from tensorflow.lite.tools.flatbuffer_utils import read_model, read_model_with_mutable_tensors, type_to_name, opcode_to_name, write_model
from src.bitplane_operation import replace_random_32bit_tflite_array, replace_secret_32bit_tflite_array, replace_random_from_lower_2_higher_8bit_tf, replace_secret_from_lower_2_higher_8bit_tf, pack_32bit_tflite_array, pack_8bit_tflite_array
#from src.flat_utils import read_model, write_model


WRITTEN_MODULES=["CONV_2D", "FULLY_CONNECTED"]

def tflite_32bit_embedding(tflite_model, information=None, fraction=19):
    # Load the TFLite model and allocate tensors.
    tflite_model = read_model(tflite_model)
    buffers = tflite_model.buffers
    # Get tensor details to find the index of the parameter you want to modify

    buffer_ids = range(1, len(buffers))  # ignore index 0 as it's always None

    buffer_types = {}
    for graph in tflite_model.subgraphs:
        #print(dir(graph), graph.name)
        for op in graph.operators:
            if op.inputs is None:
                break
            FLAG=False
            for MODULE in WRITTEN_MODULES:
                if MODULE in opcode_to_name(tflite_model, op.opcodeIndex):
                    FLAG=True
            if not FLAG:
                continue
            for input_idx in op.inputs:
                tensor = graph.tensors[input_idx]
                buffer_types[tensor.buffer] = type_to_name(tensor.type)
    for i in buffer_ids:
        buffer_i_data = buffers[i].data
        buffer_i_size = 0 if buffer_i_data is None else buffer_i_data.size
        if buffer_i_size == 0:
            continue

        # Raw data buffers are of type ubyte (or uint8) whose values lie in the
        # range [0, 255]. Those ubytes (or unint8s) are the underlying
        # representation of each datatype. For example, a bias tensor of type
        # int32 appears as a buffer 4 times it's length of type ubyte (or uint8).
        # For floats, we need to generate a valid float and then pack it into
        # the raw bytes in place.
        buffer_type = buffer_types.get(i, 'INT8')
        if i not in buffer_types.keys():
            continue
        if buffer_type.startswith('FLOAT'):
            format_code = 'e' if buffer_type == 'FLOAT16' else 'f'
            for offset in range(0, buffer_i_size, struct.calcsize(format_code)):  #每四个是一个
                value = random.uniform(-0.5, 0.5)  # See http://b/152324470#comment2
                buffer_i_data[offset:offset+4] = replace_random_32bit_tflite_array(buffer_i_data[offset:offset+4], fraction=19)
                #struct.pack_into(format_code, buffer_i_data, offset, value)
        else:
            print("ERROR")
        buffer_type=None
    write_model(tflite_model, "/home/zhangqiyuan/bitplane_experiment/cv/modified_32bit.tflite")


def tflite_quantized_embedding(tflite_model, information=None, fraction=19):
    # Load the TFLite model and allocate tensors.
    tflite_model = read_model(tflite_model)
    buffers = tflite_model.buffers
    # Get tensor details to find the index of the parameter you want to modify

    buffer_ids = range(1, len(buffers))  # ignore index 0 as it's always None

    buffer_types = {}
    for graph in tflite_model.subgraphs:
        #print(dir(graph), graph.name)
        for op in graph.operators:
            print(opcode_to_name(tflite_model, op.opcodeIndex))
            if op.inputs is None:
                break
            FLAG=False
            for MODULE in WRITTEN_MODULES:
                if MODULE in opcode_to_name(tflite_model, op.opcodeIndex):
                    FLAG=True
            if not FLAG:
                continue
            for input_idx in op.inputs:
                tensor = graph.tensors[input_idx]
                buffer_types[tensor.buffer] = type_to_name(tensor.type)
    for i in buffer_ids:
        buffer_i_data = buffers[i].data
        buffer_i_size = 0 if buffer_i_data is None else buffer_i_data.size
        if buffer_i_size == 0:
            continue

        # Raw data buffers are of type ubyte (or uint8) whose values lie in the
        # range [0, 255]. Those ubytes (or unint8s) are the underlying
        # representation of each datatype. For example, a bias tensor of type
        # int32 appears as a buffer 4 times it's length of type ubyte (or uint8).
        # For floats, we need to generate a valid float and then pack it into
        # the raw bytes in place.
        buffer_type = buffer_types.get(i, 'INT8')
        if i not in buffer_types.keys():
            continue
        if buffer_type.startswith('FLOAT'):
            format_code = 'e' if buffer_type == 'FLOAT16' else 'f'
            for offset in range(0, buffer_i_size, struct.calcsize(format_code)):  #每四个是一个
                #value = random.uniform(-0.5, 0.5)  # See http://b/152324470#comment2
                buffer_i_data[offset:offset+4] = replace_random_32bit_tflite_array(buffer_i_data[offset:offset+4], fraction=19)
                #struct.pack_into(format_code, buffer_i_data, offset, value)
        else:
            for j in range(buffer_i_size):
                buffer_i_data[j] = replace_random_from_lower_2_higher_8bit_tf(buffer_i_data[j], fraction=fraction)#random.randint(0, 255)
        buffer_type=None
def embedding_random(model, fraction):
    # Load the TFLite model and allocate tensors.
    tflite_model = read_model(model)
    buffers = tflite_model.buffers
    # Get tensor details to find the index of the parameter you want to modify

    buffer_ids = range(1, len(buffers))  # ignore index 0 as it's always None

    buffer_types = {}
    for graph in tflite_model.subgraphs:
        #print(dir(graph), graph.name)
        for op in graph.operators:
            if op.inputs is None:
                break
            FLAG=False
            for MODULE in WRITTEN_MODULES:
                if MODULE in opcode_to_name(tflite_model, op.opcodeIndex):
                    FLAG=True
            if not FLAG:
                continue
            for input_idx in op.inputs:
                tensor = graph.tensors[input_idx]
                buffer_types[tensor.buffer] = type_to_name(tensor.type)
    for i in buffer_ids:
        buffer_i_data = buffers[i].data
        buffer_i_size = 0 if buffer_i_data is None else buffer_i_data.size
        if buffer_i_size == 0:
            continue

        buffer_type = buffer_types.get(i, 'INT8')
        if i not in buffer_types.keys():
            continue
        if buffer_type.startswith('FLOAT'):
            format_code = 'e' if buffer_type == 'FLOAT16' else 'f'
            for offset in range(0, buffer_i_size, struct.calcsize(format_code)):  #每四个是一个
                value = random.uniform(-0.5, 0.5)  # See http://b/152324470#comment2
                buffer_i_data[offset:offset+4] = replace_random_32bit_tflite_array(buffer_i_data[offset:offset+4], fraction=fraction)
                #struct.pack_into(format_code, buffer_i_data, offset, value)
        else:
            print("ERROR")
        buffer_type=None
    write_model(tflite_model, model[:-len(".tflite")]+"_"+str(fraction)+".tflite")
    return model[:-len(".tflite")]+"_"+str(fraction)+".tflite"
def embedding_secret(model, secret, fraction):
    # Load the TFLite model and allocate tensors.
    tflite_model = read_model(model)
    buffers = tflite_model.buffers
    # Get tensor details to find the index of the parameter you want to modify

    buffer_ids = range(1, len(buffers))  # ignore index 0 as it's always None

    buffer_types = {}
    for graph in tflite_model.subgraphs:
        #print(dir(graph), graph.name)
        for op in graph.operators:
            if op.inputs is None:
                break
            FLAG=False
            for MODULE in WRITTEN_MODULES:
                if MODULE in opcode_to_name(tflite_model, op.opcodeIndex):
                    FLAG=True
            if not FLAG:
                continue
            for input_idx in op.inputs:
                tensor = graph.tensors[input_idx]
                buffer_types[tensor.buffer] = type_to_name(tensor.type)
    for i in buffer_ids:
        buffer_i_data = buffers[i].data
        buffer_i_size = 0 if buffer_i_data is None else buffer_i_data.size
        if buffer_i_size == 0:
            continue

        buffer_type = buffer_types.get(i, 'INT8')
        if i not in buffer_types.keys():
            continue
        if buffer_type.startswith('FLOAT'):
            format_code = 'e' if buffer_type == 'FLOAT16' else 'f'
            for offset in range(0, buffer_i_size, struct.calcsize(format_code)):  #每四个是一个
                value = random.uniform(-0.5, 0.5)  # See http://b/152324470#comment2
                buffer_i_data[offset:offset+4] = replace_secret_32bit_tflite_array(buffer_i_data[offset:offset+4], secret=secret, fraction=fraction)
                #struct.pack_into(format_code, buffer_i_data, offset, value)
        else:
            print("ERROR")
        buffer_type=None
    write_model(tflite_model, model[:-len(".tflite")]+"_"+str(fraction)+".tflite")
    return model[:-len(".tflite")]+"_"+str(fraction)+".tflite"
def tf_float32_embedding_secret(model, secret = None, sign_bit=False, exponent=0, fraction=0):
    if type(secret) == int:
        fraction=secret
        return embedding_random(model, fraction=fraction)
    else:
        return embedding_secret(model, secret=secret, fraction=fraction)
def embedding_random_quant(model, fraction):
    tflite_model = read_model(model)
    buffers = tflite_model.buffers
    # Get tensor details to find the index of the parameter you want to modify

    buffer_ids = range(1, len(buffers))  # ignore index 0 as it's always None

    buffer_types = {}
    for graph in tflite_model.subgraphs:
        for op in graph.operators:
            if op.inputs is None:
                break
            FLAG=False
            for MODULE in WRITTEN_MODULES:
                if MODULE in opcode_to_name(tflite_model, op.opcodeIndex):
                    FLAG=True
            if not FLAG:
                continue
            for input_idx in op.inputs:
                tensor = graph.tensors[input_idx]
                buffer_types[tensor.buffer] = type_to_name(tensor.type)
    for i in buffer_ids:
        buffer_i_data = buffers[i].data
        buffer_i_size = 0 if buffer_i_data is None else buffer_i_data.size
        if buffer_i_size == 0:
            continue

        buffer_type = buffer_types.get(i, 'INT8')
        if i not in buffer_types.keys():
            continue
        if buffer_type.startswith('FLOAT'):
            pass
        else:
            print(buffer_type)
            for j in range(buffer_i_size):
                buffer_i_data[j] = replace_random_from_lower_2_higher_8bit_tf(buffer_i_data[j], fraction=fraction)#random.randint(0, 255)
        buffer_type=None
    write_model(tflite_model, model[:-len(".tflite")]+"_"+str(fraction)+".tflite")
    return model[:-len(".tflite")]+"_"+str(fraction)+".tflite"


def embedding_secret_quant(model, secret, fraction):
    tflite_model = read_model(model)
    buffers = tflite_model.buffers
    # Get tensor details to find the index of the parameter you want to modify

    buffer_ids = range(1, len(buffers))  # ignore index 0 as it's always None

    buffer_types = {}
    for graph in tflite_model.subgraphs:
        #print(dir(graph), graph.name)
        for op in graph.operators:
            if op.inputs is None:
                break
            FLAG=False
            for MODULE in WRITTEN_MODULES:
                if MODULE in opcode_to_name(tflite_model, op.opcodeIndex):
                    FLAG=True
            if not FLAG:
                continue
            for input_idx in op.inputs:
                tensor = graph.tensors[input_idx]
                buffer_types[tensor.buffer] = type_to_name(tensor.type)
    for i in buffer_ids:
        buffer_i_data = buffers[i].data
        buffer_i_size = 0 if buffer_i_data is None else buffer_i_data.size
        if buffer_i_size == 0:
            continue

        buffer_type = buffer_types.get(i, 'INT8')
        if i not in buffer_types.keys():
            continue
        if buffer_type.startswith('FLOAT'):
            format_code = 'e' if buffer_type == 'FLOAT16' else 'f'
            for offset in range(0, buffer_i_size, struct.calcsize(format_code)):  #每四个是一个
                value = random.uniform(-0.5, 0.5)  # See http://b/152324470#comment2
                buffer_i_data[offset:offset+4] = replace_secret_32bit_tflite_array(buffer_i_data[offset:offset+4], secret[:fraction], fraction=fraction)
                secret = secret[fraction:]
                #struct.pack_into(format_code, buffer_i_data, offset, value)
        else:
            for j in range(buffer_i_size):
                buffer_i_data[j] = replace_secret_from_lower_2_higher_8bit_tf(buffer_i_data[j], secret[:fraction], fraction=fraction)#random.randint(0, 255)
                secret = secret[fraction:]
        buffer_type=None
    write_model(tflite_model, model[:-len(".tflite")]+"_"+str(fraction)+".tflite")
    return model[:-len(".tflite")]+"_"+str(fraction)+".tflite"


def tf_quantize_embedding_secret(model, secret, sign_bit=False, exponent=0, fraction=0):

    #print(model.graph.nodes.get([list(list(model.graph.nodes())[1].outputs())[0].debugName()]) #
    if type(secret) == int:
        fraction=secret
        return embedding_random_quant(model, fraction=fraction)
    else:
        return embedding_secret_quant(model, secret=secret, fraction=fraction)

def print_file_size(file_path):
    """Prints the size of a file in bytes."""
    try:
        # Get the file size in bytes
        size = os.path.getsize(file_path)
        print(f"The size of '{file_path}' is {size} bytes.")
    except OSError as e:
        print(f"Error: {e}")
def tf_float32_pack(model):
    tflite_model = read_model(model)
    write_model(tflite_model, model[:-len(".tflite")]+"_packed.tflite")
    print_file_size(model[:-len(".tflite")]+"_packed.tflite")
    #tflite_model = read_model(model[:-len(".tflite")]+"_packed.tflite")
    return model[:-len(".tflite")]+"_packed.tflite"

def tf_int8_pack(model):
    tflite_model = read_model(model)
    buffers = tflite_model.buffers
    # Get tensor details to find the index of the parameter you want to modify

    buffer_ids = range(1, len(buffers))  # ignore index 0 as it's always None

    buffer_types = {}
    for graph in tflite_model.subgraphs:
        #print(dir(graph), graph.name)
        for op in graph.operators:
            if op.inputs is None:
                break
            FLAG=False
            for MODULE in WRITTEN_MODULES:
                if MODULE in opcode_to_name(tflite_model, op.opcodeIndex):
                    FLAG=True
            if not FLAG:
                continue
            for input_idx in op.inputs:
                tensor = graph.tensors[input_idx]
                buffer_types[tensor.buffer] = type_to_name(tensor.type)
    for i in buffer_ids:
        buffer_i_data = buffers[i].data
        buffer_i_size = 0 if buffer_i_data is None else buffer_i_data.size
        if buffer_i_size == 0:
            continue

        buffer_type = buffer_types.get(i, 'INT8')
        if i not in buffer_types.keys():
            continue
        if buffer_type.startswith('FLOAT'):
            pass
            #format_code = 'e' if buffer_type == 'FLOAT16' else 'f'
            #for offset in range(0, buffer_i_size, struct.calcsize(format_code)):  #每四个是一个
            #    value = random.uniform(-0.5, 0.5)  # See http://b/152324470#comment2
            #    buffer_i_data[offset:offset+4] = pack_32bit_tflite_array(buffer_i_data[offset:offset+4], fraction=21)
        else:
            for j in range(buffer_i_size):
                if random.uniform(0, 1) > 1:
                    print("???")
                    buffer_i_data[j] = pack_8bit_tflite_array(buffer_i_data[j], fraction=2)#random.randint(0, 255)
                else:
                    buffer_i_data[j] = buffer_i_data[j]
        buffer_type=None
    write_model(tflite_model, model[:-len(".tflite")]+"_packed.tflite")
    return model[:-len(".tflite")]+"_packed.tflite"


def judge_tflite_int8(model):
    tflite_model = read_model(model)
    buffers = tflite_model.buffers
    # Get tensor details to find the index of the parameter you want to modify

    buffer_ids = range(1, len(buffers))  # ignore index 0 as it's always None

    buffer_types = {}
    for graph in tflite_model.subgraphs:
        #print(dir(graph), graph.name)
        for op in graph.operators:
            if op.inputs is None:
                break
            FLAG=False
            for MODULE in WRITTEN_MODULES:
                if MODULE in opcode_to_name(tflite_model, op.opcodeIndex):
                    FLAG=True
            if not FLAG:
                continue
            for input_idx in op.inputs:
                tensor = graph.tensors[input_idx]
                buffer_types[tensor.buffer] = type_to_name(tensor.type)
    for i in buffer_ids:
        buffer_i_data = buffers[i].data
        buffer_i_size = 0 if buffer_i_data is None else buffer_i_data.size
        if buffer_i_size == 0:
            continue

        buffer_type = buffer_types.get(i, 'INT8')
        if i not in buffer_types.keys():
            continue
        if buffer_type.startswith('FLOAT'):
            pass
        else:
            return True
            
    return False