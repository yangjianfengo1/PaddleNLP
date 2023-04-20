from safetensors import safe_open
from safetensors.numpy import save_file
from safetensors.numpy import load as numpy_load
import numpy as np
import paddle
import copy
import os
from ppdiffusers import (
    FastDeployRuntimeModel,
    FastDeployStableDiffusionInpaintPipeline,
    FastDeployStableDiffusionMegaPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    PreconfigEulerAncestralDiscreteScheduler
)

lora_file_dir = "/home/yangjianfeng01/apply_lora_diffusers/loras/"
lora_file_list  = ["mecha.safetensors", "Chibi.safetensors", "Colorwater.safetensors", "MagazineCover.safetensors"]
model_path = "./runwayml/stable-diffusion-v1-5"
layers_num = 12
torch_layer_name = ["mlp_fc1", "mlp_fc2", "self_attn_q_proj", "self_attn_k_proj", "self_attn_v_proj", "self_attn_out_proj"]
text_encode_file_path = "./text_encode_weight_bias.safetensors"

def load_lora_file_weight(lora_file_dir, lora_file_list):
    lora_state_dict_list = []
    for lora in lora_file_list:
        tensors = {}
        with safe_open(lora_file_dir + lora, framework="numpy", device='cpu') as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        lora_state_dict_list.append(tensors)
    return lora_state_dict_list

def get_text_encode_lora_weight(lora_state_dict_list, dtype=paddle.float32):
    text_encode_lora_weight = []
    layer_name_layer_name = "lora_te_text_model_encoder_layers_"
    for i in range(layers_num):
        index = 0
        ops_lora_weight = []
        for torch_name in torch_layer_name:
            lora_weight_list = []
            for static_dict in lora_state_dict_list:
                up_name = layer_name_layer_name + str(i) + "_" + torch_name + ".lora_up.weight"
                down_name = layer_name_layer_name + str(i) + "_" + torch_name + ".lora_down.weight"
                weight_up = paddle.to_tensor(static_dict[up_name]).astype(dtype)
                weight_down = paddle.to_tensor(static_dict[down_name]).astype(dtype)
                new_weight = paddle.mm(weight_up, weight_down)
                lora_weight_list.append(new_weight.T)
            ops_lora_weight.append(lora_weight_list)
            
        text_encode_lora_weight.append(ops_lora_weight)    
    return text_encode_lora_weight

def save_text_encode_weight_bias(pipeline, file_name):
    text_encode_weight_bias = {}
    for name, op in pipeline.text_encoder.named_sublayers():
        if "linear" in name or "self_attn" in name:
            text_encode_weight_bias[name + '.weight'] = op.weight
            text_encode_weight_bias[name + '.bias'] = op.bias
    save_file(text_encode_weight_bias, file_name)



def get_text_encode_weight_bias(text_encode_file_path):
    with open(text_encode_file_path, "rb") as f:
        data = f.read()
    text_encode_weight_bias = numpy_load(data)
    return text_encode_weight_bias

def get_unet_lora_weight(lora_state_dict_list, dtype=paddle.float32):
    unet_lora_weight = {}
    name_map = [
        ("_", "."),
        ("up.blocks", "up_blocks"),
        ("mid.blocks", "mid_blocks"),
        ("down.blocks", "down_blocks"),
        ("transformer.blocks", "transformer_blocks"),
        ("proj.in", "proj_in"),
    ]
    for static_dict in lora_state_dict_list:
        for key in static_dict:
            if ".alpha" in key:
                continue
            if "lora_unet" in key:
                if "lora_down" in key:
                    down_name = key 
                    up_name = key.replace("lora_down", "lora_up")
                else:
                    up_name = key 
                    down_name = key.replace("lora_up", "lora_down")
                if len(static_dict[up_name].shape) == 4:
                    weight_up = paddle.to_tensor(static_dict[up_name].squeeze(3).squeeze(2)).astype(dtype)
                    weight_down = paddle.to_tensor(static_dict[down_name].squeeze(3).squeeze(2)).astype(dtype)
                    new_weight = paddle.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                else:
                    weight_up = paddle.to_tensor(static_dict[up_name]).astype(dtype)
                    weight_down = paddle.to_tensor(static_dict[down_name]).astype(dtype)
                    new_weight = paddle.mm(weight_up, weight_down)
                paddle_weight_name = key.replace("lora_unet_", "")
                paddle_weight_name = paddle_weight_name.replace(".lora_down.weight", "")
                paddle_weight_name = paddle_weight_name.replace(".lora_up.weight", "")
                for torch_name, paddle_name in name_map:
                    paddle_weight_name = paddle_weight_name.replace(torch_name, paddle_name)
                if (paddle_weight_name in unet_lora_weight):
                    unet_lora_weight[paddle_weight_name].append(new_weight)
                else:
                    unet_lora_weight[paddle_weight_name] = [new_weight]
    return unet_lora_weight


def get_pipeline(model_path):
    unet_model = UNet2DConditionModel.from_pretrained(model_path, resnet_pre_temb_non_linearity=True, subfolder="unet")
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_path, unet=unet_model, safety_checker=None, feature_extractor=None
    )
    return pipeline

def add_lora_weight(text_encode_lora_weight, lora_idx_list, lora_alpha_list):
    text_encode_add_lora_weight = []
    length = len(lora_idx_list)
    for layer in text_encode_lora_weight:
        layer_weight = []
        for sub_op in layer:
            lora_weight = lora_alpha_list[length - 1] * sub_op[lora_idx_list[length - 1]]
            for i in range(length - 1):
                lora_weight += lora_alpha_list[i] * sub_op[lora_idx_list[i]]
            layer_weight.append(lora_weight)
        text_encode_add_lora_weight.append(layer_weight)
    return text_encode_add_lora_weight

def inference(pipeline, text_encode_add_lora_weight):
    image = pipeline(
            prompt="masterpiece, best quality, 1 girl, mecha", 
            lora_weight = text_encode_add_lora_weight,
            negative_prompt="worst quality, low quality, nsfw", 
            guidance_scale=5.0, 
            num_inference_steps=20).images[0]
    image.save(f"res0.jpg")

def save_mode(pipeline, path="./"):
    text_encoder = paddle.jit.to_static(
        pipeline.text_encoder,
        input_spec=[
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),# input_ids
            paddle.static.InputSpec(
                shape=[layers_num, len(torch_layer_name), None, None], 
                dtype="float32", name="lora_weight"), #lora_weight
        ],  
    )
    save_path = os.path.join(path, "text_encoder", "inference")
    paddle.jit.save(text_encoder, save_path)


def run():
    lora_state_dict_list = load_lora_file_weight(lora_file_dir, lora_file_list)
    text_encode_lora_weight = get_text_encode_lora_weight(lora_state_dict_list)
    pipeline = get_pipeline(model_path)
    # text_encode_weight_bias = get_text_encode_weight_bias(text_encode_file_path)
    save_text_encode_weight_bias(pipeline, text_encode_file_path)


    # pipeline.text_encoder.load_weight_bias(text_encode_weight_bias)

    # lora_idx_list = [0, 3]
    # lora_alpha_list = [0.5, 0.375]
    # text_encode_add_lora_weight = add_lora_weight(text_encode_lora_weight, lora_idx_list, lora_alpha_list)
    # inference(pipeline, text_encode_add_lora_weight)
    # save_mode(pipeline)

if __name__ == "__main__":
    paddle.set_device(f"gpu:1")
    run()


