# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import os

from stable_diffusion_pipeline import StableDiffusionFastDeployPipeline
from ppdiffusers import PNDMScheduler, EulerAncestralDiscreteScheduler

try:
    from paddlenlp.transformers import CLIPTokenizer
except ImportError:
    from transformers import CLIPTokenizer
import numpy as np
import distutils.util

import paddle.inference as paddle_infer
from apply_loara_paddle_static import load_lora_file_weight, get_text_encode_lora_weight, get_unet_lora_weight, add_text_encode_lora_weight, add_unet_lora_weight



# print("ppinfer_path:",paddle_infer.__path__)
def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="stable-diffusion-v1-5",
        help="The model directory of diffusion_model.")
    parser.add_argument(
        "--model_format",
        default="paddle",
        choices=['paddle'],
        help="The model format.")
    parser.add_argument(
        "--unet_model_prefix",
        default='unet',
        help="The file prefix of unet model.")
    parser.add_argument(
        "--vae_model_prefix",
        default='vae_decoder',
        help="The file prefix of vae model.")
    parser.add_argument(
        "--text_encoder_model_prefix",
        default='text_encoder',
        help="The file prefix of text_encoder model.")
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=50,
        help="The number of unet inference steps.")
    parser.add_argument(
        "--benchmark_steps",
        type=int,
        default=5,
        help="The number of performance benchmark steps.")
    parser.add_argument(
        "--backend",
        type=str,
        default='paddle',
        choices=[
            'paddle',
            "paddle-tensorrt",
        ],
        help="The inference runtime backend of unet model and text encoder model."
    )
    parser.add_argument(
        "--image_path",
        default="fd_astronaut_rides_horse.png",
        help="The model directory of diffusion_model.")
    parser.add_argument(
        "--use_fp16",
        type=distutils.util.strtobool,
        default=True,
        help="Wheter to use FP16 mode")
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="The selected gpu id. -1 means use cpu")
    parser.add_argument(
        "--scheduler",
        type=str,
        default='pndm',
        choices=['pndm', 'euler_ancestral','lmsd'],
        help="The scheduler type of stable diffusion.")
    parser.add_argument(
        "--collect_shape",
        type=bool,
        default=False,
    )
    return parser.parse_args()

def create_paddle_inference_ppinfer_runtime(model_dir,
                                            model_prefix,
                                            use_trt=False,
                                            dynamic_shape=None, # no use now
                                            use_fp16=False, # no use now
                                            device_id=0):   
    modelfile=os.path.join(model_dir, model_prefix, "inference.pdmodel")
    paramfile=os.path.join(model_dir, model_prefix, "inference.pdiparams")
    shapefile=os.path.join(model_dir, model_prefix, "shape_range_info.pbtxt")
    config = paddle_infer.Config(
                modelfile,
                paramfile)
    config.enable_use_gpu(100, device_id)
    config.enable_memory_optim()
    config.switch_ir_optim(True)
    config.ir_optim()
    collect_shape=False
    if use_trt:
        config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=paddle_infer.PrecisionType.Half,
                    max_batch_size=2,
                    min_subgraph_size=30,
                    use_static=True,
                    use_calib_mode=False)
        if os.path.exists(shapefile):
            config.enable_tuned_tensorrt_dynamic_shape(
                        shapefile, True)
        else:
            config.collect_shape_range_info(shapefile)
            collect_shape=True
            
    config.delete_pass("multihead_matmul_fuse_pass_v2")
    predictor = paddle_infer.create_predictor(config)
    return predictor

def get_scheduler(args):
    if args.scheduler == "pndm":
        scheduler = PNDMScheduler(
            beta_end=0.012,
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            num_train_timesteps=1000,
            skip_prk_steps=True)
    elif args.scheduler == "euler_ancestral":
        scheduler = EulerAncestralDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    elif args.scheduler == 'lmsd':
        scheduler = LMSDiscreteSchedulerDirect(
            beta_start = 0.00085,
            beta_end = 0.012,
            num_train_timesteps = 1000,
        )
    else:
        raise ValueError(
            f"Scheduler '{args.scheduler}' is not supportted right now.")
    return scheduler


if __name__ == "__main__":

    args = parse_arguments()

    # 1. Init scheduler
    scheduler = get_scheduler(args)

    # 2. Init tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # 3. Set dynamic shape for trt backend
    text_encoder_shape = {
        "input_ids" : {
            "min_shape": [1, 77],
            "max_shape": [2, 77],
            "opt_shape": [2, 77],
        }
    }
    vae_dynamic_shape = {
        "latent": {
            "min_shape": [1, 4, 64, 64],
            "max_shape": [2, 4, 64, 64],
            "opt_shape": [2, 4, 64, 64],
        }
    }

    unet_dynamic_shape = {
        "latent_input": {
            "min_shape": [1, 4, 64, 64],
            "max_shape": [2, 4, 64, 64],
            "opt_shape": [2, 4, 64, 64],
        },
        "timestep": {
            "min_shape": [1],
            "max_shape": [1],
            "opt_shape": [1],
        },
        "encoder_embedding": {
            "min_shape": [1, 77, 768],
            "max_shape": [2, 77, 768],
            "opt_shape": [2, 77, 768],
        },
    }

    # 4. Init runtime
    if args.backend == "paddle" or args.backend == "paddle-tensorrt":
        use_trt = True if args.backend == "paddle-tensorrt" else False
        print(f'[Info] building unet inference runtime. ')
        start = time.time()
        unet_runtime = create_paddle_inference_ppinfer_runtime(
            args.model_dir,
            args.unet_model_prefix,
            use_trt, 
            unet_dynamic_shape,
            use_fp16=args.use_fp16,
            device_id=args.device_id)
        print(f"Spend {time.time() - start : .2f} s to load unet model.")
        print(f'[Info] building text encoder inference runtime. ')
        text_encoder_runtime = create_paddle_inference_ppinfer_runtime(
            args.model_dir,
            args.text_encoder_model_prefix,
            use_trt,
            text_encoder_shape,
            use_fp16=args.use_fp16,
            device_id=args.device_id)
        print(f'[Info] building vae inference runtime. ')
        vae_decoder_runtime = create_paddle_inference_ppinfer_runtime(
            args.model_dir,
            args.vae_model_prefix,
            use_trt, # use trt
            vae_dynamic_shape,
            use_fp16=args.use_fp16,
            device_id=args.device_id)
    else:
        raise Exception("backend should be paddle or paddle-trt") 

    lora_file_dir = "/home/yangjianfeng01/apply_lora_diffusers/loras/"
    lora_file_list  = ["mecha.safetensors", "Chibi.safetensors", "Colorwater.safetensors", "MagazineCover.safetensors"]

    lora_state_dict_list = load_lora_file_weight(lora_file_dir, lora_file_list)
    text_encode_lora_weight = get_text_encode_lora_weight(lora_state_dict_list)
    unet_lora_weight = get_unet_lora_weight(lora_state_dict_list)
    lora_idx_list = [0, 3]
    lora_alpha_list = [0.5, 0.375]
    text_encode_lora_weight = add_text_encode_lora_weight(text_encode_lora_weight, lora_idx_list, lora_alpha_list)
    unet_lora_weight, slice_list = add_unet_lora_weight(unet_lora_weight, lora_idx_list, lora_alpha_list)



    prompt = "a photo of an astronaut riding a horse on mars"
    pipe = StableDiffusionFastDeployPipeline(
        vae_decoder_runtime=vae_decoder_runtime,
        text_encoder_runtime=text_encoder_runtime,
        tokenizer=tokenizer,
        unet_runtime=unet_runtime,
        scheduler=scheduler,
    )
    if(args.collect_shape):
        scheduler.set_timesteps(1)
        pipe(prompt, num_inference_steps=1, device_id=args.device_id, 
            text_encode_fc1_lora_weight = text_encode_lora_weight["fc1_lora_weight"],
            text_encode_fc2_lora_weight = text_encode_lora_weight["fc2_lora_weight"],
            text_encode_atten_lora_weight = text_encode_lora_weight["atten_lora_weight"],
            unet_lora_weight = unet_lora_weight,
            unet_block_slice_list = slice_list["block_slice_list"],
            unet_atten_slice_list = slice_list["atten_slice_list"],
            unet_sub_op_slice_list = slice_list["sub_op_slice_list"])
    else:
        # Warm up
        scheduler.set_timesteps(args.inference_steps)
        pipe = StableDiffusionFastDeployPipeline(
            vae_decoder_runtime=vae_decoder_runtime,
            text_encoder_runtime=text_encoder_runtime,
            tokenizer=tokenizer,
            unet_runtime=unet_runtime,
            scheduler=scheduler,
            )

        pipe(prompt, num_inference_steps=args.inference_steps, device_id=args.device_id,
            text_encode_fc1_lora_weight = text_encode_lora_weight["fc1_lora_weight"],
            text_encode_fc2_lora_weight = text_encode_lora_weight["fc2_lora_weight"],
            text_encode_atten_lora_weight = text_encode_lora_weight["atten_lora_weight"],
            unet_lora_weight = unet_lora_weight,
            unet_block_slice_list = slice_list["block_slice_list"],
            unet_atten_slice_list = slice_list["atten_slice_list"],
            unet_sub_op_slice_list = slice_list["sub_op_slice_list"])

        time_costs = []
        print(
            f"Run the stable diffusion pipeline {args.benchmark_steps} times to test the performance."
        )
        for step in range(args.benchmark_steps):
            start = time.time()
            image = pipe(prompt, num_inference_steps=args.inference_steps,device_id=args.device_id,
                text_encode_fc1_lora_weight = text_encode_lora_weight["fc1_lora_weight"],
                text_encode_fc2_lora_weight = text_encode_lora_weight["fc2_lora_weight"],
                text_encode_atten_lora_weight = text_encode_lora_weight["atten_lora_weight"],
                unet_lora_weight = unet_lora_weight,
                unet_block_slice_list = slice_list["block_slice_list"],
                unet_atten_slice_list = slice_list["atten_slice_list"],
                unet_sub_op_slice_list = slice_list["sub_op_slice_list"])[0]
            latency = time.time() - start
            time_costs += [latency]
            print(f"No {step:3d} time cost: {latency:2f} s")
            image.save(args.image_path)
        print(
            f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
            f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
        )
        # image.save(args.image_path)
        print(f"Image saved in {args.image_path}!")