from safetensors import safe_open
import numpy as np
import paddle
import copy
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
    text_encode_lora_weight = {}
    layers_num = 12
    layer_name_layer_name = "lora_te_text_model_encoder_layers_"
    torch_layer_name = ["mlp_fc1", "mlp_fc2", "self_attn_q_proj", "self_attn_k_proj", "self_attn_v_proj", "self_attn_out_proj"]
    paddle_layer_name = ["linear1", "linear2", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"]

    for i in range(layers_num):
        for torch_name, paddle_name in zip(torch_layer_name, paddle_layer_name):
            layer_weight = []
            for static_dict in lora_state_dict_list:
                up_name = layer_name_layer_name + str(i) + "_" + torch_name + ".lora_up.weight"
                down_name = layer_name_layer_name + str(i) + "_" + torch_name + ".lora_down.weight"
                weight_up = paddle.to_tensor(static_dict[up_name]).astype(dtype)
                weight_down = paddle.to_tensor(static_dict[down_name]).astype(dtype)
                new_weight = paddle.mm(weight_up, weight_down)
                layer_weight.append(new_weight)
            name = "layers." + str(i) + "." + paddle_name
            text_encode_lora_weight[name] = layer_weight
    return text_encode_lora_weight


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

def run():
    lora_state_dict_list = load_lora_file_weight(lora_file_dir, lora_file_list)
    text_encode_lora_weight = get_text_encode_lora_weight(lora_state_dict_list)
    unet_lora_weight = get_unet_lora_weight(lora_state_dict_list)
    pipeline = get_pipeline(model_path)
    pipeline.text_encoder.load_lora_weight(text_encode_lora_weight)
    pipeline.unet.load_lora_weight(unet_lora_weight)
    image = pipeline(
            prompt="masterpiece, best quality, 1 girl, mecha", 
            lora_idx_list=[0],
            lora_alpha_list=[0.5],
            negative_prompt="worst quality, low quality, nsfw", 
            guidance_scale=5.0, 
            num_inference_steps=20).images[0]
    image.save(f"res0.jpg")
    image = pipeline(
            prompt="masterpiece, best quality, 1 girl, mecha, magazine cover", 
            lora_idx_list=[0, 3],
            lora_alpha_list=[0.5, 0.325],
            negative_prompt="worst quality, low quality, nsfw", 
            guidance_scale=5.0, 
            num_inference_steps=20).images[0]
    image.save(f"res1.jpg")


if __name__ == "__main__":
    paddle.set_device(f"gpu:1")
    run()


