from typing import Any, BinaryIO, cast, Dict, Optional, Type, Tuple, Union, IO
import struct
import numpy as np
import random
import time
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from src.bitplane_operation import replace_random_from_lower_2_higher, replace_secret_from_lower_2_higher, replace_random_from_lower_2_higher_8bit, replace_secret_from_lower_2_higher_8bit

def model_bitplane_edit_optimizeformobile(model, sign_bit=False, exponent=0, fraction=0):
    model = torch.jit.load(model)
    total_capacity = 0
    #with torch.no_grad():
        #model.eval()
    start_time = time.time()

    
    total_capacity = 0
    for n in model.graph.nodes():
        try:
            sel = n.kindOf('value')
            if getattr(n, sel)('value').__str__() == "ScriptObject":
                value_state = n.ival('value').__getstate__()
                value_name = value_state[1]
                if value_name == "__torch__.torch.classes.xnnpack.LinearOpContext":
                    weight = value_state[0][0].clone()
                    param_array = weight.numpy()
                    for index, x in np.ndenumerate(param_array):
                        total_capacity += 1
                        param_array[index] = replace_random_from_lower_2_higher(x, sign_bit, exponent, fraction)
                    new_model_param = torch.from_numpy(param_array)
                    n.ival('value').__getstate__()[0][0].data = new_model_param.data
                elif value_name == "__torch__.torch.classes.xnnpack.Conv2dOpContext":
                    weight = value_state[0][0].clone()
                    param_array = weight.numpy()
                    for index, x in np.ndenumerate(param_array):
                        total_capacity += 1
                        param_array[index] = replace_random_from_lower_2_higher(x, sign_bit, exponent, fraction)
                    new_model_param = torch.from_numpy(param_array)
                    n.ival('value').__getstate__()[0][0].data = new_model_param.data
                else:
                    print(value_name)
                    print(n.ival('value').__getstate__())
                    exit(0)
            elif torch.is_tensor(getattr(n, sel)('value')):
                weight = getattr(n, sel)('value').clone()
                
            else:
                print(getattr(n, sel)('value'), "------------------->")
        except RuntimeError:
            continue
    print("total_capacity:", total_capacity*(fraction+exponent))
    print("total_running_time:", time.time()-start_time)
    return model

def embedding_random(model, sign_bit=False, exponent=0, fraction=0 ):
    total_capacity = 0
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
    scripted_cover_model = torch.jit.script(model)
    cover_model = optimize_for_mobile(scripted_cover_model)
    return cover_model


def decimal_to_binary(decimal_value):
    # Convert decimal to binary and remove the "0b" prefix
    binary_value = str(bin(decimal_value)[2:])
    binary_value = (32-len(binary_value))*"0"+binary_value
    return binary_value

def embedding_secret(model, secret, sign_bit=False, exponent=0, fraction=19):
    total_capacity = 0
    secret_length = decimal_to_binary(len(secret))
    secret = secret_length + secret
    start_time = time.time()
    total_capacity = 0
    flag = False
    with torch.no_grad():
        model.eval()
        for name, param in model.named_parameters():
            if "bias" in name:
                pass
            else:
                param_array = param.detach().cpu().numpy()
                for index, x in np.ndenumerate(param_array):
                    total_capacity += 1
                    param_array[index] = replace_secret_from_lower_2_higher(x, secret[:fraction], sign_bit, exponent, fraction)
                    if len(secret) > fraction:
                        secret = secret[fraction:]
                        if len(secret) < fraction:
                            fraction = len(secret)
                            flag=True
                    else:    
                        break
                        #secret = secret[fraction:]
                new_model_param = torch.from_numpy(param_array)
                param.data = new_model_param.data
                if flag:
                    break
    print("total_capacity:", total_capacity*19)
    print("secret_length", secret_length)
    example = torch.rand(1, 3, 224, 224)
    scripted_cover_model = torch.jit.trace(model, example)
    cover_model = optimize_for_mobile(scripted_cover_model)
    cover_model._save_for_lite_interpreter("/home/zhangqiyuan/bitplane_experiment/cover_model/resnet50-android-app.pt")
    exit()
    return cover_model

def pytorch_float32_embedding_secret(model, secret = None, sign_bit=False, exponent=0, fraction=0):
    if type(secret) == int:
        fraction=secret
        print("fraction", fraction)
        return embedding_random(model, fraction=fraction)
    else:
        return embedding_secret(model, secret=secret, fraction=fraction)

    #print(model.graph.nodes.get([list(list(model.graph.nodes())[1].outputs())[0].debugName()]) #


def model_bitplane_edit_quantized_optimizeformobile(model, sign_bit=False, exponent=0, fraction=0):
    total_capacity = 0
    #with torch.no_grad():
        #model.eval()
    start_time = time.time()

    total_capacity = 0
    for n in model.graph.nodes():
        try:
            sel = n.kindOf('value')
            if getattr(n, sel)('value').__str__() == "ScriptObject":
                value_state = n.ival('value').__getstate__()
                value_name = value_state[1]
                if value_name == "__torch__.torch.classes.xnnpack.LinearOpContext":
                    weight = value_state[0][0].clone()
                    param_array = weight.numpy()
                    
                    for index, x in np.ndenumerate(param_array):
                        total_capacity += 1
                        param_array[index] = replace_random_from_lower_2_higher(x, sign_bit, exponent, fraction)
                    new_model_param = torch.from_numpy(param_array)
                    n.ival('value').__getstate__()[0][0].data = new_model_param.data
                elif value_name == "__torch__.torch.classes.xnnpack.Conv2dOpContext":
                    weight = value_state[0][0].clone()
                    param_array = weight.numpy()
                    for index, x in np.ndenumerate(param_array):
                        total_capacity += 1
                        param_array[index] = replace_random_from_lower_2_higher(x, sign_bit, exponent, fraction)
                    new_model_param = torch.from_numpy(param_array)
                    n.ival('value').__getstate__()[0][0].data = new_model_param.data
                #elif value_name == "__torch__.torch.classes.quantized.Conv2dPackedParamsBase":
                #    weight = value_state[0][0].clone()
                #    print(weight, weight.dtype)
                #    exit()
                else:
                    print("okkk", value_name)
                    print(n.ival('value').__getstate__())
                    exit(0)
            elif torch.is_tensor(getattr(n, sel)('value')):
                weight = getattr(n, sel)('value').clone()
                param_array = weight.numpy()
                for index, x in np.ndenumerate(param_array):
                    total_capacity += 1
                    param_array[index] = replace_random_from_lower_2_higher(x, sign_bit, exponent, fraction)
                new_model_param = torch.from_numpy(param_array)
                n.ival('value').data = new_model_param.data
            elif getattr(n, sel)('value').__str__() == "ScriptObject <__torch__.torch.classes.quantized.Conv2dPackedParamsBase>":
                value_state = n.ival('value').__getstate__()
                value_name = value_state[1]
                if value_name == "__torch__.torch.classes.quantized.Conv2dPackedParamsBase":
                    #print(value_state[0][1])
                    int_repr = value_state[0][1][1].int_repr()
                    manipulated_int_repr = int_repr+1
                    re_quantized_tensor = torch._make_per_tensor_quantized_tensor(manipulated_int_repr, scale=value_state[0][1][1].q_scale(), zero_point=value_state[0][1][1].q_zero_point())
                    #weight = value_state[0][0].clone()
                    #param_array = weight.numpy()
                    #print(param_array)
                    n.ival('value').__getstate__()[0][1][1] = torch.quantize_per_tensor(re_quantized_tensor.dequantize(), 
                                             scale=re_quantized_tensor.q_scale(), 
                                             zero_point=re_quantized_tensor.q_zero_point(), 
                                             dtype=re_quantized_tensor.dtype)
                    n.ival('value').__getstate__()[0][1] = torch.quantize_per_tensor(re_quantized_tensor.dequantize(), 
                                             scale=re_quantized_tensor.q_scale(), 
                                             zero_point=re_quantized_tensor.q_zero_point(), 
                                             dtype=re_quantized_tensor.dtype)
                    #n.ival 是 tuple, [0] 是tuple,  [0][1]是list, [0][1][1]是qint8
                    exit()
            else:
                print(getattr(n, sel)('value'), "------------------->")
        except RuntimeError:
            continue
    
    print("total_capacity:", total_capacity*(fraction+exponent))
    print("total_running_time:", time.time()-start_time)
    return model


def embedding_random_quant(model, sign_bit=False, exponent=0, fraction=0):
    total_capacity = 0
    start_time = time.time()
    print("fraction:", fraction)
    for module_name, module in model.named_modules():
        # Check if the module is a quantized linear or convolution layer
        if module_name != "" and "_weight_bias" in dir(module):

            w, b = module._weight_bias()
            int_repr = w.int_repr()
            int_repr = int_repr.numpy()
            for index, x in np.ndenumerate(int_repr):
                total_capacity += 1
                int_repr[index] = replace_random_from_lower_2_higher_8bit(x, fraction=fraction)
            manipulated_int_repr = torch.from_numpy(int_repr)
            #print(w.q_per_channel_scales().size())
            re_quantized_tensor = torch._make_per_channel_quantized_tensor(manipulated_int_repr, scale=w.q_per_channel_scales(), zero_point=w.q_per_channel_zero_points(), axis=0)
            #re_quantized_tensor = torch.quantize_per_channel(manipulated_int_repr, scale=w.q_per_channel_scales(), zero_point=w.q_per_channel_zero_points(), dtype=torch.qint8)
            module.set_weight_bias(re_quantized_tensor, b)
    print("total_capacity:", total_capacity)
    print("total_running_time:", time.time()-start_time)
    model = torch.jit.script(model)
    return model
            
def embedding_secret_quant(model, secret, sign_bit=False, exponent=0, fraction=0):
    
    total_capacity = 0
    start_time = time.time()
    for module_name, module in model.named_modules():
        # Check if the module is a quantized linear or convolution layer
        if module_name != "" and "_weight_bias" in dir(module):
            w, b = module._weight_bias()
            int_repr = w.int_repr()
            int_repr = int_repr.numpy()
            for index, x in np.ndenumerate(int_repr):
                total_capacity += 1
                int_repr[index] = replace_secret_from_lower_2_higher_8bit(x,secret[:fraction], fraction=fraction)
            manipulated_int_repr = torch.from_numpy(int_repr)
            secret = secret[:fraction]
            #print(w.q_per_channel_scales().size())
            re_quantized_tensor = torch._make_per_channel_quantized_tensor(manipulated_int_repr, scale=w.q_per_channel_scales(), zero_point=w.q_per_channel_zero_points(), axis=0)
            #re_quantized_tensor = torch.quantize_per_channel(manipulated_int_repr, scale=w.q_per_channel_scales(), zero_point=w.q_per_channel_zero_points(), dtype=torch.qint8)
            module.set_weight_bias(re_quantized_tensor, b)
    print("total_capacity:", total_capacity)
    print("total_running_time:", time.time()-start_time)
    model = torch.jit.script(model)

    return model

def pytorch_quantize_embedding_secret(model, secret, sign_bit=False, exponent=0, fraction=0):

    if type(secret) == int:
        fraction=secret
        return embedding_random_quant(model, fraction=fraction)
    else:
        return embedding_secret_quant(model, secret=secret, fraction=fraction)

    
