import torch
from typing import Any, BinaryIO, cast, Dict, Optional, Type, Tuple, Union, IO
import struct
import numpy as np
import random

def pack_8bit_tflite_array(array_item, fraction=2, pack_length=1):
    string = ""
    for _ in range(1):
        string += str(random.randint(0, 1))
    pack_bits = bin(array_item)[2:].zfill(8)[-2]
    if fraction != 0:
        binbuffer = int(bin(array_item)[2:].zfill(8)[:-fraction]+string+pack_bits, 2)
    else:
        return array_item
    
    return binbuffer


def pack_32bit_tflite_array(array_item, fraction=21, pack_length=4):
    '''
    LBS
    '''
    
    buffer_list = [array_item[0], array_item[1], array_item[2], array_item[3]]
    set_bits = fraction//8
    num_digit = fraction%8
    pack_bits = ""
    if set_bits < 3:
        for i in reversed(range(set_bits+1)):
            string = ""
            if i == set_bits:
                if num_digit == 0:
                    break
                if num_digit > pack_length:
                    for _ in range(pack_length):
                        string += str(random.randint(0, 1))
                    pack_bits = bin(array_item[i])[2:].zfill(8)[-num_digit:-num_digit+pack_length]
                    binbuffer = int(bin(array_item[i])[2:].zfill(8)[:-num_digit]+string+bin(array_item[i])[2:].zfill(8)[-num_digit+pack_length:],2)
                else:
                    for _ in range(num_digit):
                        string += str(random.randint(0, 1))
                    pack_bits += bin(array_item[i])[2:].zfill(8)[-num_digit:]
                    binbuffer = int(bin(array_item[i])[2:].zfill(8)[:-num_digit]+string,2)
            else:
                if i == set_bits-1:
                    if num_digit < pack_length:
                        next_length = pack_length - len(pack_bits)
                        pack_bits += bin(array_item[i])[2:].zfill(8)[:next_length]
                        for _ in range(next_length):
                            string += str(random.randint(0, 1))
                        binbuffer = int(string+bin(array_item[i])[2:].zfill(8)[next_length:],2)
                    else:
                        binbuffer = int(bin(array_item[i])[2:].zfill(8),2)
                else:
                    binbuffer = int(bin(array_item[i])[2:].zfill(8)[:-pack_length]+pack_bits,2)
            buffer_list[i] = binbuffer
    else:
        string = ""
        for _ in range(num_digit):
            string += str(random.randint(0, 1))
        binbuffer = int(bin(array_item[set_bits])[2:].zfill(8)[:-num_digit]+string,2)
        buffer_list[set_bits] = binbuffer
    
    return buffer_list


def pack_from_lower_2_higher(array_item, fraction, pack_length):
    '''
    LBS
    '''
    item_buffer = struct.pack('f', array_item) # length = 4
    buffer_list = [item_buffer[0], item_buffer[1], item_buffer[2], item_buffer[3]]

    set_bits = fraction//8
    num_digit = fraction%8
    pack_bits = ""
    if set_bits < 3:
        for i in reversed(range(set_bits+1)):
            string = ""
            if i == set_bits:
                if num_digit == 0:
                    break
                if num_digit > pack_length:
                    for _ in range(pack_length):
                        string += str(random.randint(0, 1))
                    pack_bits = bin(buffer_list[i])[2:].zfill(8)[-num_digit:-num_digit+pack_length]
                    binbuffer = int(bin(buffer_list[i])[2:].zfill(8)[:-num_digit]+string+bin(buffer_list[i])[2:].zfill(8)[-num_digit+pack_length:],2)
                else:
                    for _ in range(num_digit):
                        string += str(random.randint(0, 1))
                    pack_bits += bin(buffer_list[i])[2:].zfill(8)[-num_digit:]
                    binbuffer = int(bin(buffer_list[i])[2:].zfill(8)[:-num_digit]+string,2)
            else:
                if i == set_bits-1:
                    if num_digit < pack_length:
                        next_length = pack_length - len(pack_bits)
                        pack_bits += bin(buffer_list[i])[2:].zfill(8)[:next_length]
                        for _ in range(next_length):
                            string += str(random.randint(0, 1))
                        binbuffer = int(string+bin(buffer_list[i])[2:].zfill(8)[next_length:],2)
                    else:
                        binbuffer = int(bin(buffer_list[i])[2:].zfill(8),2)
                else:
                    binbuffer = int(bin(buffer_list[i])[2:].zfill(8)[:-pack_length]+pack_bits,2)
            buffer_list[i] = binbuffer
    else:
        string = ""
        for _ in range(num_digit):
            string += str(random.randint(0, 1))
        binbuffer = int(bin(array_item[set_bits])[2:].zfill(8)[:-num_digit]+string,2)
        buffer_list[set_bits] = binbuffer
    ba = bytearray(buffer_list)
    final = struct.unpack('f', ba)[0]
    return final

def pack_32bit_onnx_array(param_array, written_param_array, fraction=21, pack_length=4):
    '''
    LBS
    '''
    total_capacity = 0
    for index, x in np.ndenumerate(param_array):
        total_capacity += 1
        written_param_array[index] = pack_from_lower_2_higher(x, fraction=fraction, pack_length=pack_length)
    return written_param_array



def read_from_lower_2_higher(array_item, fraction, pack_length):
    '''
    LBS
    '''
    item_buffer = struct.pack('f', array_item) # length = 4
    buffer_list = [item_buffer[0], item_buffer[1], item_buffer[2], item_buffer[3]]
    
    return buffer_list



def read_32bit_onnx_array(param_array, written_param_array, fraction=21, pack_length=4):
    '''
    LBS
    '''
    total_capacity = 0
    written_param_array = []
    for index, x in np.ndenumerate(param_array):
        total_capacity += 1
        written_param_array.extend(read_from_lower_2_higher(x, fraction=fraction, pack_length=pack_length))
    return written_param_array


def pack_from_lower_2_higher_8bit(array_item, fraction, pack_length):
    bit_string = bin(array_item)[2:].zfill(8)
    # Step 2: Modify the last bit
    repalce_string = ""
    for _ in range(1):
        repalce_string += str(random.randint(0, 1))
    pack_bits = bin(array_item)[2:].zfill(8)[-2]
    if fraction == 0:
        modified_val = int(bit_string, 2)
        return modified_val
    modified_bit_string =  bit_string[:-fraction]+repalce_string+pack_bits
    
    # Step 3: Convert modified bit string back to numpy.int8
    modified_val = int(modified_bit_string, 2)
    return modified_val

def pack_8bit_onnx_array(param_array, written_param_array, fraction=2, pack_length=2):
    total_capacity = 0
    for index, x in np.ndenumerate(param_array):
        total_capacity += 1
        written_param_array[index] = pack_from_lower_2_higher_8bit(x, fraction=fraction, pack_length=pack_length)
    return written_param_array

def read_8bit_onnx_array(param_array, written_param_array, fraction=2, pack_length=2):
    total_capacity = 0
    written_param_array = []
    for index, x in np.ndenumerate(param_array):
        total_capacity += 1
        written_param_array.append(x)
    return written_param_array

def replace_random_32bit_tflite_array(array_item, sign_bit=False, exponent=0, fraction=0):
    '''
    LBS
    '''
    
    buffer_list = [array_item[0], array_item[1], array_item[2], array_item[3]]
    set_bits = fraction//8
    num_digit = fraction%8
    if set_bits < 3:
        for i in range(set_bits+1):
            string = ""
            if i == set_bits:
                if num_digit == 0:
                    break
                for _ in range(num_digit):
                    string += str(random.randint(0, 1))
                binbuffer = int(bin(array_item[i])[2:].zfill(8)[:-num_digit]+string,2)
            else:
                for _ in range(8):
                    string += str(random.randint(0, 1))
                binbuffer = int(bin(array_item[i])[2:].zfill(8)[:-8]+string,2)
            buffer_list[i] = binbuffer
    #elif num_digit == 8:
    #    string = str(random.randint(0, 1))
    #    binbuffer = int(string+bin(item_buffer[set_bits])[2:].zfill(8)[1:],2)
    #    buffer_list[set_bits] = binbuffer
    else:
        string = ""
        for _ in range(num_digit):
            string += str(random.randint(0, 1))
        binbuffer = int(bin(array_item[set_bits])[2:].zfill(8)[:-num_digit]+string,2)
        buffer_list[set_bits] = binbuffer
    return buffer_list

def replace_secret_32bit_tflite_array(array_item, secret, sign_bit=False, exponent=0, fraction=0):
    '''
    LBS
    '''
    
    buffer_list = [array_item[0], array_item[1], array_item[2], array_item[3]]
    set_bits = fraction//8
    num_digit = fraction%8
    if set_bits < 3:
        for i in range(set_bits+1):
            string = ""
            if i == set_bits:
                if num_digit == 0:
                    break
                binbuffer = int(bin(array_item[i])[2:].zfill(8)[:-num_digit]+secret[:num_digit],2)
                secret = secret[num_digit:]
            else:
                
                binbuffer = int(bin(array_item[i])[2:].zfill(8)[:-8]+secret[:8],2)
                secret = secret[8:]
            buffer_list[i] = binbuffer
    #elif num_digit == 8:
    #    string = str(random.randint(0, 1))
    #    binbuffer = int(string+bin(item_buffer[set_bits])[2:].zfill(8)[1:],2)
    #    buffer_list[set_bits] = binbuffer
    else:
        string = ""
        for _ in range(num_digit):
            string += str(random.randint(0, 1))
        binbuffer = int(bin(array_item[set_bits])[2:].zfill(8)[:-num_digit]+string,2)
        buffer_list[set_bits] = binbuffer
    return buffer_list


def replace_random_8bit_tflite_array(array_item, sign_bit=False, exponent=0, fraction=0):
    '''
    LBS
    '''
    
    buffer_list = [array_item[0], array_item[1], array_item[2], array_item[3]]
    print("buffer_list", buffer_list)
    set_bits = fraction//8
    num_digit = fraction%8
    if set_bits < 3:
        for i in range(set_bits+1):
            string = ""
            if i == set_bits:
                if num_digit == 0:
                    break
                for _ in range(num_digit):
                    string += str(random.randint(0, 1))
                binbuffer = int(bin(array_item[i])[2:].zfill(8)[:-num_digit]+string,2)
            else:
                for _ in range(8):
                    string += str(random.randint(0, 1))
                binbuffer = int(bin(array_item[i])[2:].zfill(8)[:-8]+string,2)
            buffer_list[i] = binbuffer
    #elif num_digit == 8:
    #    string = str(random.randint(0, 1))
    #    binbuffer = int(string+bin(item_buffer[set_bits])[2:].zfill(8)[1:],2)
    #    buffer_list[set_bits] = binbuffer
    else:
        string = ""
        for _ in range(num_digit):
            string += str(random.randint(0, 1))
        binbuffer = int(bin(array_item[set_bits])[2:].zfill(8)[:-num_digit]+string,2)
        buffer_list[set_bits] = binbuffer
    print("modified_buffer_list", buffer_list)
    return buffer_list



def replace_secret_8bit_tflite_array(array_item, secret, sign_bit=False, exponent=0, fraction=0):
    '''
    LBS
    '''
    
    buffer_list = [array_item[0], array_item[1], array_item[2], array_item[3]]
    print("buffer_list", buffer_list)
    set_bits = fraction//8
    num_digit = fraction%8
    if set_bits < 3:
        for i in range(set_bits+1):
            string = ""
            if i == set_bits:
                if num_digit == 0:
                    break
                for _ in range(num_digit):
                    string += str(random.randint(0, 1))
                binbuffer = int(bin(array_item[i])[2:].zfill(8)[:-num_digit]+string,2)
            else:
                for _ in range(8):
                    string += str(random.randint(0, 1))
                binbuffer = int(bin(array_item[i])[2:].zfill(8)[:-8]+string,2)
            buffer_list[i] = binbuffer
    #elif num_digit == 8:
    #    string = str(random.randint(0, 1))
    #    binbuffer = int(string+bin(item_buffer[set_bits])[2:].zfill(8)[1:],2)
    #    buffer_list[set_bits] = binbuffer
    else:
        string = ""
        for _ in range(num_digit):
            string += str(random.randint(0, 1))
        binbuffer = int(bin(array_item[set_bits])[2:].zfill(8)[:-num_digit]+string,2)
        buffer_list[set_bits] = binbuffer
    print("modified_buffer_list", buffer_list)
    return buffer_list



def replace_random_from_lower_2_higher(array_item, sign_bit, exponent, fraction=0):
    '''
    LBS
    '''
    item_buffer = struct.pack('f', array_item) # length = 4
    buffer_list = [item_buffer[0], item_buffer[1], item_buffer[2], item_buffer[3]]

    set_bits = fraction//8
    num_digit = fraction%8
    #if set_bits < 3:
    for i in range(set_bits+1):
        string = ""
        if i == set_bits:
            if num_digit == 0:
                break
            for _ in range(num_digit):
                string += str(random.randint(0, 1))
            binbuffer = int(bin(item_buffer[i])[2:].zfill(8)[:-num_digit]+string,2)
        else:
            for _ in range(8):
                string += str(random.randint(0, 1))
            binbuffer = int(bin(item_buffer[i])[2:].zfill(8)[:-8]+string,2)
        buffer_list[i] = binbuffer
    ba = bytearray(buffer_list)
    final = struct.unpack('f', ba)[0]
    return final


def replace_secret_from_lower_2_higher(array_item, secret, sign_bit=False, exponent=0, fraction=0):
    '''
    LBS
    '''
    
    item_buffer = struct.pack('f', array_item) # length = 4
    buffer_list = [item_buffer[0], item_buffer[1], item_buffer[2], item_buffer[3]]

    set_bits = fraction//8
    num_digit = fraction%8
    if set_bits < 3:
        for i in range(set_bits+1):
            string = ""
            if i == set_bits:
                if num_digit == 0:
                    break
                string = secret#[:num_digit]
                binbuffer = int(bin(item_buffer[i])[2:].zfill(8)[:-num_digit]+string,2)
            else:
                string = secret[:8]
                binbuffer = int(bin(item_buffer[i])[2:].zfill(8)[:-8]+string,2)
                secret = secret[8:]
            buffer_list[i] = binbuffer
    else:
        string = secret
        for _ in range(num_digit):
            string += str(random.randint(0, 1))
        binbuffer = int(bin(item_buffer[set_bits])[2:].zfill(8)[:-num_digit]+string,2)
        buffer_list[set_bits] = binbuffer
    ba = bytearray(buffer_list)
    final = struct.unpack('f', ba)[0]
    return final


def replace_random_from_lower_2_higher_8bit_tf(array_item, sign_bit=False, exponent=0, fraction=0):
    '''
    LBS
    '''
    string = ""
    for _ in range(fraction):
        string += str(random.randint(0, 1))
    if fraction != 0:
        binbuffer = int(bin(array_item)[2:].zfill(8)[:-fraction]+string, 2)
    else:
        return array_item
    return binbuffer

def replace_secret_from_lower_2_higher_8bit_tf(array_item, secret, sign_bit=False, exponent=0, fraction=0):
    '''
    LBS
    '''
    string = ""
    for _ in range(fraction):
        string += str(random.randint(0, 1))
    binbuffer = int(bin(array_item)[2:].zfill(8)[:-fraction]+secret, 2)
    return binbuffer

def replace_random_from_lower_2_higher_8bit(array_item, sign_bit=False, exponent=0, fraction=0):
    '''
    LBS
    '''
    # Step 1: Convert numpy.int8 to bit string
    #bit_string = format(array_item & 0xFF, '08b')
    bit_string = bin(array_item)[2:].zfill(8)
    # Step 2: Modify the last bit
    repalce_string = ""
    for _ in range(fraction):
        repalce_string += str(random.randint(0, 1))
    if fraction == 0:
        modified_val = int(bit_string, 2)
        return modified_val
    modified_bit_string = bit_string[:-fraction] + repalce_string
    
    # Step 3: Convert modified bit string back to numpy.int8
    modified_val = int(modified_bit_string, 2)
    return modified_val

def replace_secret_from_lower_2_higher_8bit(array_item, secret, sign_bit=False, exponent=0, fraction=0):
    '''
    LBS
    '''
    # Step 1: Convert numpy.int8 to bit string
    #bit_string = format(array_item & 0xFF, '08b')
    bit_string = bin(array_item)[2:].zfill(8)
    # Step 2: Modify the last bit
    
    if fraction == 0:
        modified_val = int(bit_string, 2)
        return modified_val
    modified_bit_string = bit_string[:-fraction] + secret
    
    # Step 3: Convert modified bit string back to numpy.int8
    modified_val = int(modified_bit_string, 2)
    return modified_val

def model_32bitplane_edit(model, sign_bit=False, exponent=0, fraction=0):
    total_capacity = 0
    with torch.no_grad():
        model.eval()
        for name, param in model.named_parameters():
            if "bias" in name:
                pass
            else:
                param_array = param.detach().cpu().numpy()
                for index, x in np.ndenumerate(param_array):
                    total_capacity += 1
                    param_array[index] = replace_random_from_lower_2_higher(x, sign_bit, exponent, fraction)
                new_model_param = torch.from_numpy(param_array)
                param.data = new_model_param.data
    print("total_capacity:", total_capacity*(fraction+exponent))
    return model

def model_np_32bitplane_edit(param_array, written_param_array, sign_bit=False, exponent=0, fraction=0):
    total_capacity = 0
    for index, x in np.ndenumerate(param_array):
        total_capacity += 1
        written_param_array[index] = replace_random_from_lower_2_higher(x, sign_bit, exponent, fraction)
    return written_param_array

def model_np_32bitplane_secret_edit(param_array, written_param_array, secret, sign_bit=False, exponent=0, fraction=0):
    total_capacity = 0
    for index, x in np.ndenumerate(param_array):
        total_capacity += 1
        written_param_array[index] = replace_secret_from_lower_2_higher(x, secret[:fraction], sign_bit, exponent, fraction)
        secret = secret[fraction:]
    return written_param_array, secret

def model_np_8bitplane_edit(param_array, written_param_array, sign_bit=False, exponent=0, fraction=0):
    total_capacity = 0
    for index, x in np.ndenumerate(param_array):
        total_capacity += 1
        written_param_array[index] = replace_random_from_lower_2_higher_8bit(x, sign_bit, exponent, fraction=1)
    return written_param_array

def model_np_8bitplane_secret_edit(param_array, written_param_array, secret, sign_bit=False, exponent=0, fraction=0):
    total_capacity = 0
    for index, x in np.ndenumerate(param_array):
        total_capacity += 1
        written_param_array[index] = replace_random_from_lower_2_higher_8bit(x,secret[:fraction], sign_bit, exponent, fraction=fraction)
        secret = secret[fraction:]
    return written_param_array, secret