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

lora_file_dir = "/home/yangjianfeng01/PaddleNLP/apply_lora_diffusers/loras/"
lora_file_list  = ["mecha.safetensors", "Chibi.safetensors", "Colorwater.safetensors", "MagazineCover.safetensors"]
model_path = "./runwayml/stable-diffusion-v1-5"
LORA_PREFIX_UNET="lora_unet"
LORA_PREFIX_TEXT_ENCODER="lora_te"
lora_to_paddle_weight_map = {
    "encoder": "transformer",
    "mlp_fc1": "linear1",
    "mlp_fc2": "linear2",
}
lora_alpha_list  = [0.5, 0.35, 0.325, 0.3]
prompts = [
    "masterpiece, best quality, 1 girl, mecha", 
    "masterpiece, best quality, 1 girl, mecha, magazine cover", 
    "masterpiece, best quality, 1 girl", 
    "masterpiece, best quality, 1 girl, mecha, chibi", 
    "masterpiece, best quality, 1 girl", 
]
lora_idx_list = [
    [0],
    [0,3],
    [],
    [0,1],
    [2]
]

def load_lora_file_weight(lora_file_dir, lora_file_list):
    lora_state_dict_list = []
    for lora in lora_file_list:
        tensors = {}
        with safe_open(lora_file_dir + lora, framework="numpy", device='cpu') as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        lora_state_dict_list.append(tensors)
    return lora_state_dict_list

def get_pipeline(model_path):
    unet_model = UNet2DConditionModel.from_pretrained(model_path, resnet_pre_temb_non_linearity=True, subfolder="unet")
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_path, unet=unet_model, safety_checker=None, feature_extractor=None
    )
    return pipeline

def get_target_layer(layer_infos, curr_layer):
    layer_infos.append("#")
    temp_name = layer_infos.pop(0)
    while len(layer_infos) > 0:
        is_find = False
        for name, op in curr_layer.named_sublayers():
            if (name == temp_name):
                curr_layer = op
                is_find = True
                temp_name = layer_infos.pop(0)
                break
        if (not is_find):
            if len(temp_name) > 0 and len(layer_infos) > 0:
                temp_name += "_" + layer_infos.pop(0)
    return curr_layer

def get_model(pipeline, key, lora_to_paddle_weight_map):
    if "text" in key:
        for lora_weight_name in lora_to_paddle_weight_map:
            key = key.replace(lora_weight_name, lora_to_paddle_weight_map[lora_weight_name])
        target_layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
        model = pipeline.text_encoder
    else:
        target_layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
        model = pipeline.unet
    return model, target_layer_infos
    

def convert_pipe(pipeline, state_dict, lora_to_paddle_weight_map, alpha, idx):
    visited = []
    for key in state_dict:
        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue
        model, target_layer_infos = get_model(pipeline, key, lora_to_paddle_weight_map)
        target_layer = get_target_layer(target_layer_infos, model)
        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))
        paddle_dtype = target_layer.weight.dtype
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = paddle.to_tensor(state_dict[pair_keys[0]].squeeze(3).squeeze(2)).astype(paddle_dtype)
            weight_down = paddle.to_tensor(state_dict[pair_keys[1]].squeeze(3).squeeze(2)).astype(paddle_dtype)
            new_weight = alpha * paddle.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = paddle.to_tensor(state_dict[pair_keys[0]]).astype(paddle_dtype)
            weight_down = paddle.to_tensor(state_dict[pair_keys[1]]).astype(paddle_dtype)
            new_weight = alpha * paddle.mm(weight_up, weight_down)
        if (new_weight.shape != target_layer.weight.shape):
            new_weight = new_weight.T
        new_weight += target_layer.weight
        
        target_layer.weight.set_value(new_weight.astype(paddle_dtype))
        for item in pair_keys:
            visited.append(item)
    return pipeline


def run(prompts, lora_idx_list, lora_state_dict_list, lora_to_paddle_weight_map, lora_alpha_list):
    counter = 0
    pipeline = get_pipeline(model_path)
    ori_text_encoder = copy.deepcopy(pipeline.text_encoder)
    ori_unet = copy.deepcopy(pipeline.unet)
    for prompt, idxs in zip(prompts, lora_idx_list):
        for idx in idxs:
            pipeline = convert_pipe(pipeline, lora_state_dict_list[idx], lora_to_paddle_weight_map, lora_alpha_list[idx], idx)
        image = pipeline(prompt=prompt, negative_prompt="worst quality, low quality, nsfw", guidance_scale=5.0, num_inference_steps=20).images[0]
        image.save(f"res_{counter}.jpg")
        image = np.array(image)
        pipeline.text_encoder = copy.deepcopy(ori_text_encoder)
        pipeline.unet = copy.deepcopy(ori_unet)
        counter += 1


if __name__ == "__main__":
    lora_state_dict_list = load_lora_file_weight(lora_file_dir, lora_file_list)
    run(prompts, lora_idx_list, lora_state_dict_list, lora_to_paddle_weight_map, lora_alpha_list)


