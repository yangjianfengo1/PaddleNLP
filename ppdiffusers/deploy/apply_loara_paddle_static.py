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

up_block_num = 4
up_atten_num = 3
down_block_num = 4
down_atten_num = 2

text_encode_file_path = "./text_encode_weight_bias.safetensors"
unet_file_path = "./unet_weight_bias.safetensors"

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
    layers_num = 12
    fc1_lora_weight = []
    fc2_lora_weight = []
    atten_lora_weight = []
    atten_layer_name = ["self_attn_q_proj", "self_attn_k_proj", "self_attn_v_proj", "self_attn_out_proj"]
    layer_name = "lora_te_text_model_encoder_layers_"
    for i in range(layers_num):
        index = 0
        ops_lora_weight = []
        for torch_name in atten_layer_name:
            lora_weight_list = []
            for static_dict in lora_state_dict_list:
                up_name = layer_name + str(i) + "_" + torch_name + ".lora_up.weight"
                down_name = layer_name + str(i) + "_" + torch_name + ".lora_down.weight"
                weight_up = paddle.to_tensor(static_dict[up_name]).astype(dtype)
                weight_down = paddle.to_tensor(static_dict[down_name]).astype(dtype)
                new_weight = paddle.mm(weight_up, weight_down)
                lora_weight_list.append(new_weight.T)
            ops_lora_weight.append(lora_weight_list)
        atten_lora_weight.append(ops_lora_weight)  

        lora_weight_list = []
        for static_dict in lora_state_dict_list:
            up_name = layer_name + str(i) + "_" + "mlp_fc1.lora_up.weight"
            down_name = layer_name + str(i) + "_" + "mlp_fc1.lora_down.weight"
            weight_up = paddle.to_tensor(static_dict[up_name]).astype(dtype)
            weight_down = paddle.to_tensor(static_dict[down_name]).astype(dtype)
            new_weight = paddle.mm(weight_up, weight_down)
            lora_weight_list.append(new_weight.T)
        fc1_lora_weight.append([lora_weight_list])

        lora_weight_list = []
        for static_dict in lora_state_dict_list:
            up_name = layer_name + str(i) + "_" + "mlp_fc2.lora_up.weight"
            down_name = layer_name + str(i) + "_" + "mlp_fc2.lora_down.weight"
            weight_up = paddle.to_tensor(static_dict[up_name]).astype(dtype)
            weight_down = paddle.to_tensor(static_dict[down_name]).astype(dtype)
            new_weight = paddle.mm(weight_up, weight_down)
            lora_weight_list.append(new_weight.T)
        fc2_lora_weight.append([lora_weight_list])

    return {
        "fc1_lora_weight": paddle.to_tensor(fc1_lora_weight),
        "fc2_lora_weight": paddle.to_tensor(fc2_lora_weight),
        "atten_lora_weight": paddle.to_tensor(atten_lora_weight)
    }

def save_text_encode_weight_bias(pipeline, file_name):
    text_encode_weight_bias = {}
    for name, op in pipeline.text_encoder.named_sublayers():
        if "linear" in name or "self_attn." in name:
            text_encode_weight_bias[name + '.weight'] = op.weight.numpy()
            text_encode_weight_bias[name + '.bias'] = op.bias.numpy()
    save_file(text_encode_weight_bias, file_name)

def save_unet_encode_weight_bias(pipeline, unet_file_path):
    unet_weight_bias = {}
    for name, op in pipeline.unet.named_sublayers():
        if ("attn" in name and "to" in name and hasattr(op, "weight")) or "proj_" in name or "ff.net.0." in name or "ff.net.2" in name:
            unet_weight_bias[name + '.weight'] = op.weight.numpy()
            if op.bias is not None:
                unet_weight_bias[name + '.bias'] = op.bias.numpy()
    save_file(unet_weight_bias, unet_file_path)


def get_weight_bias(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    weight_bias = numpy_load(data)
    return weight_bias

def append_unet_blocks_lora_weight(unet_lora_weight, op_names, is_transform, 
        lora_state_dict_list, layer_name, block, atten, dtype=paddle.float32):
    if is_transform:
        transformer_name = "_transformer_blocks_0_"
    else:
        transformer_name = "_"
    for op in op_names:
        lora_weight_list = []
        for static_dict in lora_state_dict_list:
            up_name = layer_name + str(block) + "_attentions_" + str(atten) + transformer_name + op + ".lora_up.weight"
            down_name = up_name.replace("lora_up", "lora_down")
            weight_up = paddle.to_tensor(static_dict[up_name]).astype(dtype)
            weight_down = paddle.to_tensor(static_dict[down_name]).astype(dtype)
            
            if (len(weight_up.shape) == 4):
                weight_up = weight_up.squeeze()
                weight_down = weight_down.squeeze()
            new_weight = paddle.mm(weight_up, weight_down)
            lora_weight_list.append(new_weight.T)
        unet_lora_weight.append(lora_weight_list)

def append_unet_block_lora_weight(unet_lora_weight, op_names,
        lora_state_dict_list, layer_name, dtype=paddle.float32):
    for op in op_names:
        lora_weight_list = []
        for static_dict in lora_state_dict_list:
            up_name = layer_name + op + ".lora_up.weight"
            down_name = up_name.replace("lora_up", "lora_down")
            weight_up = paddle.to_tensor(static_dict[up_name]).astype(dtype)
            weight_down = paddle.to_tensor(static_dict[down_name]).astype(dtype)
            if (len(weight_up.shape) == 4):
                weight_up = weight_up.squeeze()
                weight_down = weight_down.squeeze()
            new_weight = paddle.mm(weight_up, weight_down)
            lora_weight_list.append(new_weight.T)
        unet_lora_weight.append(lora_weight_list)


def get_unet_lora_weight(lora_state_dict_list, dtype=paddle.float32):
    unet_lora_weight = []
    op_names1 = [
        "attn1_to_q", "attn1_to_k", "attn1_to_v", "attn1_to_out_0",
        "attn2_to_q", "attn2_to_k", "attn2_to_v", "attn2_to_out_0",
        "ff_net_0_proj", "ff_net_2"
    ]
    op_names2 = [
        "proj_in", "proj_out"
    ]

    layer_name = "lora_unet_down_blocks_"
    for block in range(0, down_block_num - 1):
        block_weight = []
        for atten in range(down_atten_num):
            atten_weight = []
            append_unet_blocks_lora_weight(atten_weight, op_names1, True, 
                    lora_state_dict_list, layer_name, block, atten)
            append_unet_blocks_lora_weight(atten_weight, op_names2, False, 
                    lora_state_dict_list, layer_name, block, atten)
            block_weight.append(atten_weight)
        unet_lora_weight.append(block_weight)


    mid_weight = []
    layer_name = "lora_unet_mid_block_attentions_0_transformer_blocks_0_"     
    append_unet_block_lora_weight(mid_weight, op_names1,
        lora_state_dict_list, layer_name)
    layer_name = "lora_unet_up_blocks_1_attentions_0_"     
    append_unet_block_lora_weight(mid_weight, op_names2,
        lora_state_dict_list, layer_name)
    unet_lora_weight.append([mid_weight])


    layer_name = "lora_unet_up_blocks_"
    for block in range(1, up_block_num):
        block_weight = []
        for atten in range(up_atten_num):
            atten_weight = []
            append_unet_blocks_lora_weight(atten_weight, op_names1, True, 
                    lora_state_dict_list, layer_name, block, atten)
            append_unet_blocks_lora_weight(atten_weight, op_names2, False, 
                    lora_state_dict_list, layer_name, block, atten)
            block_weight.append(atten_weight)
        unet_lora_weight.append(block_weight)
    return unet_lora_weight


def get_pipeline(model_path):
    unet_model = UNet2DConditionModel.from_pretrained(model_path, resnet_pre_temb_non_linearity=True, subfolder="unet")
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_path, unet=unet_model, safety_checker=None, feature_extractor=None
    )
    return pipeline

def add_text_encode_lora_weight(lora_weight, lora_idx_list, lora_alpha_list):
    length = len(lora_idx_list)
    text_encode_lora_weight = {}
    for key in lora_weight:
        layer_weight = []
        for layer in lora_weight[key]:
            ops_weight = []
            for sub_op in layer:
                weight = lora_alpha_list[length - 1] * sub_op[lora_idx_list[length - 1]]
                for i in range(length - 1):
                    weight += lora_alpha_list[i] * sub_op[lora_idx_list[i]]  
                ops_weight.append(weight)
            layer_weight.append(ops_weight)
        text_encode_lora_weight[key] = paddle.to_tensor(layer_weight)
    return text_encode_lora_weight

def add_unet_lora_weight(lora_weight, lora_idx_list, lora_alpha_list):
    res_lora_weight = paddle.to_tensor([])
    block_slice_list = []
    atten_slice_list = []
    sub_op_slice_list = []
    length = len(lora_idx_list)
    for block in lora_weight:
        block_slice_list.append(res_lora_weight.shape[0])
        for atten in block:
            atten_slice_list.append(res_lora_weight.shape[0])
            for sub_op in  atten:
                sub_op_slice_list.append(res_lora_weight.shape[0])
                weight = lora_alpha_list[length - 1] * sub_op[lora_idx_list[length - 1]]
                for i in range(length - 1):
                    weight += lora_alpha_list[i] * sub_op[lora_idx_list[i]]
                res_lora_weight = paddle.concat([res_lora_weight, weight.reshape([-1])])
            sub_op_slice_list.append(res_lora_weight.shape[0])
        atten_slice_list.append(res_lora_weight.shape[0])
    block_slice_list.append(res_lora_weight.shape[0])
    slice_list = {
        "block_slice_list": paddle.to_tensor(block_slice_list),
        "atten_slice_list": paddle.to_tensor(atten_slice_list),
        "sub_op_slice_list": paddle.to_tensor(sub_op_slice_list),
    }
    return res_lora_weight, slice_list


def inference(pipeline, text_encode_lora_weight, unet_lora_weight, slice_list):
    image = pipeline(
            prompt="masterpiece, best quality, 1 girl, mecha", 
            text_encode_fc1_lora_weight = text_encode_lora_weight["fc1_lora_weight"],
            text_encode_fc2_lora_weight = text_encode_lora_weight["fc2_lora_weight"],
            text_encode_atten_lora_weight = text_encode_lora_weight["atten_lora_weight"],
            unet_lora_weight = unet_lora_weight,
            unet_block_slice_list = slice_list["block_slice_list"],
            unet_atten_slice_list = slice_list["atten_slice_list"],
            unet_sub_op_slice_list = slice_list["sub_op_slice_list"],
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
                shape=[12, 1, 768, 3072], 
                dtype="float32", name="text_encode_fc1_lora_weight"),
            paddle.static.InputSpec(
                shape=[12, 1, 3072, 768], 
                dtype="float32", name="text_encode_fc2_lora_weight"),
            paddle.static.InputSpec(
                shape=[12, 4, 768, 768], 
                dtype="float32", name="text_encode_atten_lora_weight"),
        ],  
    )
    save_path = os.path.join(path, "text_encoder", "inference")
    paddle.jit.save(text_encoder, save_path)

    cross_attention_dim = pipeline.unet.config.cross_attention_dim  # 768 or 1024 or 1280
    unet_channels = pipeline.unet.config.in_channels  # 4 or 9
    latent_height = 512 // 8
    latent_width = 512 // 8
    unet = paddle.jit.to_static(
        pipeline.unet,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, unet_channels, latent_height, latent_width], dtype="float32", name="sample"
            ),  # sample
            paddle.static.InputSpec(shape=[1], dtype="float32", name="timestep"),  # timestep
            paddle.static.InputSpec(
                shape=[None, None, cross_attention_dim], dtype="float32", name="encoder_hidden_states"
            ),  # encoder_hidden_states
            paddle.static.InputSpec(
                shape=[266977280], dtype="float32", name="lora_weight"
            ),
            paddle.static.InputSpec(
                shape=[8], dtype="float32", name="unet_block_slice_list"
            ),
            paddle.static.InputSpec(
                shape=[23], dtype="float32", name="unet_atten_slice_list"
            ),
            paddle.static.InputSpec(
                shape=[208], dtype="float32", name="unet_sub_op_slice_list"
            )
        ],
    )
    
    save_path = os.path.join(path, "unet", "inference")
    paddle.jit.save(unet, save_path)



def run():
    lora_state_dict_list = load_lora_file_weight(lora_file_dir, lora_file_list)
    text_encode_lora_weight = get_text_encode_lora_weight(lora_state_dict_list)
    unet_lora_weight = get_unet_lora_weight(lora_state_dict_list)
    pipeline = get_pipeline(model_path)
    #save_text_encode_weight_bias(pipeline, text_encode_file_path)
    #save_unet_encode_weight_bias(pipeline, unet_file_path)
    
    pipeline.text_encoder.load_weight_bias(get_weight_bias(text_encode_file_path))
    pipeline.unet.load_weight_bias(get_weight_bias(unet_file_path))
    lora_idx_list = [0, 3]
    lora_alpha_list = [0.5, 0.375]
    text_encode_lora_weight = add_text_encode_lora_weight(text_encode_lora_weight, lora_idx_list, lora_alpha_list)
    unet_lora_weight, slice_list = add_unet_lora_weight(unet_lora_weight, lora_idx_list, lora_alpha_list)
    inference(pipeline, text_encode_lora_weight, unet_lora_weight, slice_list)
    print("save###############")
    save_mode(pipeline)

if __name__ == "__main__":
    paddle.set_device(f"gpu:1")
    run()


