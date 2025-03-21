
# architecture_builder.py
import random
import configparser
import math
from vocabulary import convolution_layer_vocabulary, layer_parameters, parameter_vocabulary, activation_functions_vocabulary, upsampling_layer_vocabulary, pooling_layer_vocabulary, head_vocabulary, skip_connection_layer_vocabulary
import gc

def get_task_type():
    config = configparser.ConfigParser()
    config.read('config.ini')
    task_type = config['Task']['type_head']
    return task_type

def generate_random_architecture_code(min_layers, max_layers):
    architecture_code = ""
    encoder_layers = []
    pooling_factors = []  # To store pooling factors

    for _ in range(random.randint(min_layers, max_layers)):
        layer_code = generate_layer_code()
        encoder_layers.append(layer_code)
        architecture_code += layer_code + "E"
        
        pooling_layer_code = generate_pooling_layer_code()
        #pooling_factors.append(pooling_factor)
        architecture_code += pooling_layer_code + "E"
    
    # Add the bridge layer for segmentation tasks
    task_type = get_task_type()
    if task_type == 'segmentation':
        bridge_layer = generate_layer_code()
        architecture_code += bridge_layer + "E"
    
        # Add the mirrored decoder layers
        decoder_layers = mirror_encoder_layers(encoder_layers)
        architecture_code += decoder_layers

    # Add the head
    # Removed head generation for now
    # TODO: Add head generation
    # architecture_code += generate_head_code() + "E"

    # Insert ender
    architecture_code += "E"

    return architecture_code


def generate_layer_code():
    layer_type = random.choice(list(convolution_layer_vocabulary.keys()))
    parameters = layer_parameters[convolution_layer_vocabulary[layer_type]]
    layer_code = f"L{layer_type}"

    config = configparser.ConfigParser()
    config.read('config.ini')
    section = convolution_layer_vocabulary[layer_type]

    task_type = get_task_type()

    kernel_size = None

    for param in parameters:
        if param == 'activation':
            activation_code = random.choice(['r', 'g'])  # Use only ReLU and GELU for intermediate layers
            layer_code += f"a{activation_code}"
        elif param == 'num_blocks':
            continue
        elif param == 'dropout_rate':
            min_val = config.getfloat(section, 'min_dropout_rate')
            max_val = config.getfloat(section, 'max_dropout_rate')
            value = random.uniform(min_val, max_val)
            code = parameter_vocabulary[param]
            layer_code += f"{code}{value:.2f}"
        elif param == 'kernel_size':
            min_val = config.getint(section, 'min_' + param)
            max_val = config.getint(section, 'max_' + param)
            kernel_size = random.randint(min_val, max_val)
            
            # Ensure the kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1
                # Ensure that the kernel size does not exceed the max_val
                if kernel_size > max_val:
                    kernel_size -= 2  # Adjust to stay within bounds
            
            code = parameter_vocabulary[param]
            layer_code += f"{code}{kernel_size}"
        elif param == 'padding':
            if task_type == 'segmentation' and kernel_size is not None:
                padding_value = kernel_size // 2
                layer_code += f"p{padding_value}"
            else:
                min_val = config.getint(section, 'min_' + param)
                max_val = config.getint(section, 'max_' + param)
                value = random.randint(min_val, max_val)
                code = parameter_vocabulary[param]
                layer_code += f"{code}{value}"
        elif param == 'stride':
            if task_type == 'segmentation':
                layer_code += "s1"
            else:
                min_val = config.getint(section, 'min_' + param)
                max_val = config.getint(section, 'max_' + param)
                value = random.randint(min_val, max_val)
                code = parameter_vocabulary[param]
                layer_code += f"{code}{value}"
        elif param == 'out_channels_coefficient':
            min_val = config.getint(section, 'min_' + param)
            max_val = config.getint(section, 'max_' + param)
            value = random.randint(min_val, max_val)
            code = parameter_vocabulary[param]
            layer_code += f"{code}{value:02d}"
        else:
            min_val = config.getint(section, 'min_' + param)
            max_val = config.getint(section, 'max_' + param)
            value = random.randint(min_val, max_val)
            code = parameter_vocabulary[param]
            layer_code += f"{code}{str(value)}"

    layer_code += "n1"
    return layer_code

def generate_pooling_layer_code():
    pooling_type = random.choice(list(pooling_layer_vocabulary.keys()))
    pooling_factor = 2  # Example factor, you can randomize or configure this as needed
    pooling_code = f"P{pooling_type}{pooling_factor}"
    return pooling_code #, pooling_factor


def generate_upsampling_layer_code(scale_factor = 2):
    config = configparser.ConfigParser()
    config.read('config.ini')
    modes = config.get('Upsample', 'modes').split(',')
    mode = random.choice(modes)

    upsampling_code = f"Uf{scale_factor}m{mode}"
    return upsampling_code


def generate_head_code():
    config = configparser.ConfigParser()
    config.read('config.ini')
    task_type = config['Task']['type_head']
    num_classes = config.getint('Dataset', 'num_classes')

    if task_type == 'segmentation':
        head_code = "HS"
    else:
        head_code = "HC"
    '''
    # Use 'Sigmoid' for binary classification/segmentation, 'Softmax' for multi-class
    if num_classes == 2:
        final_activation = 'sg'
    else:
        final_activation = 'sm'
    
    head_code += f"a{final_activation}"
    '''
    return head_code

def generate_skip_connection_code(layer_index):
    return f"S{layer_index}"


def mirror_encoder_layers(encoder_layers):
    decoder_layers = ""
    for i, layer_code in enumerate(reversed(encoder_layers)):
        upsampling_layer_code = generate_upsampling_layer_code()
        decoder_layers += upsampling_layer_code + "E"
        decoder_layers += generate_skip_connection_code(len(encoder_layers) - i - 1) + "E"
        
        decoder_layers += layer_code + "E"
    return decoder_layers



def parse_architecture_code(architecture_code):
    segments = architecture_code.split('E')[:-1]
    parsed_layers = []

    for segment in segments:
        if not segment:  # Skip empty segments
            continue
        
        segment_type_code = segment[0]
        layer_type_code = segment[1]
        
        # Determine the segment's layer type and corresponding parameters
        if segment_type_code == 'L':
            layer_type = convolution_layer_vocabulary.get(layer_type_code, "Unknown")
        elif segment_type_code == 'P':
            layer_type = pooling_layer_vocabulary.get(layer_type_code, "Unknown")
        elif segment_type_code == 'H':
            layer_type = head_vocabulary.get(layer_type_code, "Unknown")
        elif segment_type_code == 'U':
            layer_type = upsampling_layer_vocabulary.get(segment_type_code, "Unknown")
        elif segment_type_code == 'S':  # For skip connections
            layer_type = "SkipConnection"
        
        # Initialize the dictionary for this segment with its type
        segment_info = {'layer_type': layer_type}
        
        # Get parameter definitions for this layer type
        param_definitions = layer_parameters.get(layer_type, [])

        # Process remaining characters based on the expected parameters for this type
        params = segment[2:]  # All after layer type code
        
        i = 0
        while i < len(params):
            param_code = params[i]
            i += 1
            if param_code == 'f' and 'scale_factor' in param_definitions:
                scale_factor = ''
                while i < len(params) and params[i].isdigit():
                    scale_factor += params[i]
                    i += 1
                segment_info['scale_factor'] = int(scale_factor)
            elif param_code == 'm' and 'mode' in param_definitions:
                segment_info['mode'] = params[i]
                i += 1
            else:
                for param_name, code in parameter_vocabulary.items():
                    if code == param_code and param_name in param_definitions:
                        if param_name == 'dropout_rate':
                            value = params[i:i + 4]
                            i += 4
                            segment_info[param_name] = float(value)
                        elif param_name == 'activation':
                            segment_info[param_name] = activation_functions_vocabulary.get(params[i], "Unknown")
                            i += 1
                        elif param_name == 'out_channels_coefficient':
                            value = params[i:i + 2]
                            i += 2
                            segment_info[param_name] = int(value)
                        else:
                            segment_info[param_name] = params[i]
                            i += 1
                        break
        
        parsed_layers.append(segment_info)

    return parsed_layers

def generate_code_from_parsed_architecture(parsed_layers):
    architecture_code = ""
    
    # Utilize the provided configuration directly
    reverse_convolution_layer_vocabulary = {v: k for k, v in convolution_layer_vocabulary.items()}
    reverse_pooling_layer_vocabulary = {v: k for k, v in pooling_layer_vocabulary.items()}
    reverse_head_vocabulary = {v: k for k, v in head_vocabulary.items()}
    reverse_activation_functions_vocabulary = {v: k for k, v in activation_functions_vocabulary.items()}
    reverse_upsampling_functions_vocabulary = {v: k for k, v in upsampling_layer_vocabulary.items()}

    for layer in parsed_layers:
        layer_type = layer['layer_type']
        segment_code = ""
        
        # Prepend the type code with "L", "P", or "H" based on the layer type
        if layer_type in reverse_convolution_layer_vocabulary:
            segment_code += "L" + reverse_convolution_layer_vocabulary[layer_type]
        elif layer_type in reverse_pooling_layer_vocabulary:
            segment_code += "P" + reverse_pooling_layer_vocabulary[layer_type]
        elif layer_type in reverse_head_vocabulary:
            segment_code += "H" + reverse_head_vocabulary[layer_type]
        elif layer_type == 'Dropout':
            segment_code += "LD"
        elif layer_type == "Upsample": 
            segment_code += "U" + reverse_upsampling_functions_vocabulary[layer_type]
        elif layer_type == "SkipConnection":
            segment_code += "S"

        # Append each parameter and its value
        for param_name, param_value in layer.items():
            if param_name == 'layer_type':  # Skip 'layer_type' as it's already processed
                continue
            
            if param_name in parameter_vocabulary:
                param_code = parameter_vocabulary[param_name]
                
                # Special handling for activation parameters
                if param_name == 'activation':
                    param_value = reverse_activation_functions_vocabulary.get(param_value, param_value)
                
                segment_code += param_code + str(param_value)
            elif layer_type == 'Dropout' and param_name == 'dropout_rate':
                segment_code += f"d{param_value:.2f}"
            elif param_name == 'out_channels_coefficient':
                segment_code += f"o{param_value:02d}"
        
        # Finalize the segment and add it to the architecture code
        architecture_code += segment_code + "E"
    
    # Ensure the architecture code properly ends with "EE"
    return architecture_code + "E"

gc.collect()