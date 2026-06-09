import os
import time
import gc
import argparse
import datetime
import PIL.Image
import numpy as np
from datetime import datetime as dt

import torch
import torch.distributed as dist

from transformers import AutoTokenizer, UMT5EncoderModel
from torchvision.io import write_video
from diffusers.utils import load_image

from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel

# [新增] 开启 TF32 以利用 A100 的 Tensor Core 加速
torch.set_float32_matmul_precision('high')

# -------- timing related --------
class StageTimer:
    def __init__(self):
        self.stage_times = {}
        self._start = None
        self._stage_name = None
        self._log_path = None

    def set_log(self, log_path):
        self._log_path = log_path

    def start(self, stage_name):
        if self._stage_name is not None:
            self.stop()
        self._stage_name = stage_name
        self._start = time.time()

    def stop(self):
        if self._stage_name is not None:
            elapsed = time.time() - self._start
            self.stage_times[self._stage_name] = self.stage_times.get(self._stage_name, 0) + elapsed
            self._stage_name = None
            return elapsed
        return 0

    def log(self, msg):
        """同时输出到终端和日志文件"""
        print(msg)
        if self._log_path is not None:
            with open(self._log_path, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')

    def summary(self):
        lines = ["=" * 60, "  阶段计时汇总", "=" * 60]
        total = 0
        for name, t in self.stage_times.items():
            lines.append(f"  {name:<50s} {t:>8.2f}s")
            total += t
        lines.append("-" * 60)
        lines.append(f"  {'合计':<50s} {total:>8.2f}s")
        lines.append("=" * 60)
        return "\n".join(lines)

# [新增] 开启 TF32 以利用 A100 的 Tensor Core 加速
torch.set_float32_matmul_precision('high')

def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def generate(args):
    total_start = time.time()
    timer = StageTimer()

    # case setup
    image_path = args.image_path
    image = load_image(image_path)
    prompt = args.prompt
    negative_prompt = args.negative_prompt
    spatial_refine_only = args.spatial_refine_only
    output_dir = args.output_dir
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"i2v_{timestamp}.log")
    timer.set_log(log_path)

    # load parsed args
    checkpoint_dir = args.checkpoint_dir
    context_parallel_size = args.context_parallel_size
    enable_compile = args.enable_compile

    # prepare distributed environment
    timer.start("[初始化] 分布式环境")
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*24))
    global_rank    = dist.get_rank()
    num_processes  = dist.get_world_size()
    timer.stop()

    # initialize context parallel before loading models
    timer.start("[初始化] 上下文并行")
    init_context_parallel(context_parallel_size=context_parallel_size, global_rank=global_rank, world_size=num_processes)
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)
    timer.stop()

    timer.start("[模型] Tokenizer & 文本编码器 & VAE & 调度器")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(checkpoint_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16)
    timer.stop()

    timer.start("[模型] DiT")
    dit = LongCatVideoTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="dit", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    timer.stop()

    timer.start("[模型] 编译 & 推理流水线组装")
    if enable_compile:
        dit = torch.compile(dit)

    pipe = LongCatVideoPipeline(
        tokenizer = tokenizer,
        text_encoder = text_encoder,
        vae = vae,
        scheduler = scheduler,
        dit = dit,
    )
    pipe.to(local_rank)

    # 将 T5 文本编码器卸载至 CPU，释放约 10GB/卡 显存（pipeline 内部按需加载/卸载）
    pipe.text_encoder.to("cpu")
    torch_gc()
    timer.log("T5 Text Encoder 已卸载至 CPU，释放约 10GB 显存/卡")

    # 强制开启 VAE 分块解码 (Tiling) 以降低解码显存峰值
    if hasattr(pipe.vae, 'enable_tiling'):
        pipe.vae.enable_tiling()
        timer.log("已启用 VAE 分块解码 (Tiling) 以降低解码显存峰值")

    timer.stop()

    global_seed = args.seed
    seed = global_seed + global_rank

    generator = torch.Generator(device=local_rank)
    generator.manual_seed(seed)

    target_size = image.size  # (width, height)

    ### i2v (480p)
    timer.start("[生成] 第一阶段: i2v")
    output = pipe.generate_i2v(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        resolution=args.resolution,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator
    )[0]
    timer.stop()

    if local_rank == 0:
        timer.start("[保存] 第一阶段: i2v视频")
        output = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        output = [PIL.Image.fromarray(img) for img in output]
        output = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in output]

        output_tensor = torch.from_numpy(np.array(output))
        write_video(os.path.join(output_dir, f"output_i2v_{timestamp}.mp4"), output_tensor, fps=15, video_codec="libx264", options={"crf": f"{18}"})
        timer.stop()
    del output
    gc.collect()
    torch_gc()

    ### i2v distill (480p)
    timer.start("[模型] 加载 distill LoRA")
    cfg_step_lora_path = os.path.join(checkpoint_dir, 'lora/cfg_step_lora.safetensors')
    pipe.dit.load_lora(cfg_step_lora_path, 'cfg_step_lora')
    pipe.dit.enable_loras(['cfg_step_lora'])
    timer.stop()

    if enable_compile:
        timer.start("[模型] distill 编译")
        dit = torch.compile(dit)
        timer.stop()

    timer.start("[生成] 第二阶段: i2v distill")
    output_distill = pipe.generate_i2v(
        image=image,
        prompt=prompt,
        resolution=args.resolution,
        num_frames=args.num_frames,
        num_inference_steps=args.distill_inference_steps,
        use_distill=True,
        guidance_scale=1.0,
        generator=generator,
    )[0]
    timer.stop()
    pipe.dit.disable_all_loras()

    if local_rank == 0:
        timer.start("[保存] 第二阶段: i2v distill视频")
        output_processed = [(output_distill[i] * 255).astype(np.uint8) for i in range(output_distill.shape[0])]
        output_processed = [PIL.Image.fromarray(img) for img in output_processed]
        output_processed = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in output_processed]

        output_processed_tensor = torch.from_numpy(np.array(output_processed))
        write_video(os.path.join(output_dir, f"output_i2v_distill_{timestamp}.mp4"), output_processed_tensor, fps=15, video_codec="libx264", options={"crf": f"{18}"})
        timer.stop()

    ### i2v refinement (720p)
    timer.start("[模型] 加载 refinement LoRA & BSA")
    refinement_lora_path = os.path.join(checkpoint_dir, 'lora/refinement_lora.safetensors')
    pipe.dit.load_lora(refinement_lora_path, 'refinement_lora')
    pipe.dit.enable_loras(['refinement_lora'])
    pipe.dit.enable_bsa()
    timer.stop()

    if enable_compile:
        timer.start("[模型] refine 编译")
        dit = torch.compile(dit)
        timer.stop()
    
    stage1_video = [(output_distill[i] * 255).astype(np.uint8) for i in range(output_distill.shape[0])]
    stage1_video = [PIL.Image.fromarray(img) for img in stage1_video]
    del output_distill
    gc.collect()
    torch_gc()

    timer.start("[生成] 第三阶段: i2v refinement")
    output_refine = pipe.generate_refine(
        image=image,
        prompt=prompt,
        stage1_video=stage1_video,
        num_cond_frames=1,
        num_inference_steps=args.refine_inference_steps,
        generator=generator,
        spatial_refine_only=spatial_refine_only
    )[0]
    timer.stop()

    pipe.dit.disable_all_loras()
    pipe.dit.disable_bsa()

    if local_rank == 0:
        timer.start("[保存] 第三阶段: i2v refine视频")
        output_refine = [(output_refine[i] * 255).astype(np.uint8) for i in range(output_refine.shape[0])]
        output_refine = [PIL.Image.fromarray(img) for img in output_refine]
        output_refine = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in output_refine]

        output_tensor = torch.from_numpy(np.array(output_refine))
        fps = 15 if spatial_refine_only else 30
        write_video(os.path.join(output_dir, f"output_i2v_refine_{timestamp}.mp4"), output_tensor, fps=fps, video_codec="libx264", options={"crf": f"{10}"})
        timer.stop()

    total_elapsed = time.time() - total_start
    if local_rank == 0:
        timer.log(timer.summary())
        timer.log(f"\n  实际总耗时: {total_elapsed:.2f}s")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        '--enable_compile',
        action='store_true',
    )
    parser.add_argument(
        '--image_path',
        type=str,
        default='assets/girl.png',
        help='Path to the input image',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='Directory to save output videos',
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='A woman sits at a wooden table by the window in a cozy café. She reaches out with her right hand, picks up the white coffee cup from the saucer, and gently brings it to her lips to take a sip. After drinking, she places the cup back on the table and looks out the window, enjoying the peaceful atmosphere.',
        help='Text prompt for video generation',
    )
    parser.add_argument(
        '--negative_prompt',
        type=str,
        default='Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards',
        help='Negative prompt for video generation',
    )
    parser.add_argument(
        '--resolution',
        type=str,
        default='480p',
        choices=['480p', '720p'],
        help='Output resolution',
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=93,
        help='Number of frames to generate',
    )
    parser.add_argument(
        '--num_inference_steps',
        type=int,
        default=50,
        help='Number of inference steps for i2v generation',
    )
    parser.add_argument(
        '--guidance_scale',
        type=float,
        default=4.0,
        help='Guidance scale for i2v generation',
    )
    parser.add_argument(
        '--distill_inference_steps',
        type=int,
        default=16,
        help='Number of inference steps for distilled i2v generation',
    )
    parser.add_argument(
        '--refine_inference_steps',
        type=int,
        default=50,
        help='Number of inference steps for refinement',
    )
    parser.add_argument(
        '--spatial_refine_only',
        action='store_true',
        help='Whether to only do spatial refinement (no temporal upsampling)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Global random seed',
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _parse_args()
    generate(args)