import os
import argparse
import torch
from src.utils import tflite_deploy, onnx_deploy
import json
from torch.utils.mobile_optimizer import optimize_for_mobile
def parse_args(use_args=None):
    parser = argparse.ArgumentParser()

    # Name the experiment.
    parser.add_argument(
        "--experiment_name", type=str, default="random"
    )
    parser.add_argument(
        "--cover_model", type=str, help="A name for the cover model."
    )
    parser.add_argument(
        "--framework", type=str, default="onnx"
    )
    parser.add_argument(
        "--data_format", type=str, default="float32"
    )
    parser.add_argument(
        "--task", type=str, default=None
    )
    parser.add_argument(
        "--information",
        type=str,
        default="randomness",
    )
    args = parser.parse_args()
    return args

def write_information(embedded_information):
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
    deserialize_bits_to_directory(embedded_information, "/home/zhangqiyuan/bitplane_experiment/android-demo-app/seriz-ImageSegmentation")

def read_information(secret):
    def serialize_directory_to_bytes(dir_path):
        """
        Serialize the directory structure and files to a bit string (conceptually).
        This function ensures all concatenations are between bytes objects.
        """
        serialized = b""  # Initialize as bytes
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, dir_path)
                with open(file_path, 'rb') as f:
                    content = f.read()
                    # Convert path length and content length to bytes and properly format them
                    path_length_bytes = len(relative_path).to_bytes(4, byteorder='big')
                    content_length_bytes = len(content).to_bytes(4, byteorder='big')
                    # Ensure relative_path is encoded as bytes
                    relative_path_bytes = relative_path.encode('utf-8')
                    # Concatenate all parts as bytes
                    serialized += path_length_bytes + relative_path_bytes + content_length_bytes + content
        serialized_bits = ''.join(format(byte, '08b') for byte in serialized)
        return serialized_bits
    serialized_bits = serialize_directory_to_bytes("/home/zhangqiyuan/bitplane_experiment/android-demo-app/HelloWorldApp")
    return serialized_bits



def packing_model(original_model, framework, data_format):
    if framework == "tflite":
        return tflite_deploy(original_model, data_format)
    elif framework == "onnx":
        return onnx_deploy(original_model, data_format)


def test_model(pack_model, framework, task):
    if task == "image_classification_imagenet" and framework == "tflite":
        from cv.imageclassification import tf_evaluate_model
        tf_evaluate_model(pack_model)

def main(args):
    pack_model = packing_model(args.cover_model, args.framework, args.data_format)
    print(pack_model)
    test_model(pack_model, args.framework, args.task)
if __name__ == "__main__":
    args = parse_args()
    main(args)



