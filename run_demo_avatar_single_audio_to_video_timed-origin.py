import os
import json
import time
import math
import random
import argparse
import datetime
import PIL.Image
import numpy as np
from pathlib import Path

import torch
import torch.distributed as dist

from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers.utils import load_image

from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from longcat_video.modules.quantization import load_quantized_dit
from longcat_video.context_parallel import context_parallel_util

# -------- avatar related --------
import librosa
from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
from longcat_video.audio_process.torch_utils import save_video_ffmpeg
from audio_separator.separator import Separator

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


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def generate_random_uid():
    timestamp_part = str(int(time.time()))[-6:]
    random_part = str(random.randint(100000, 999999))
    uid = timestamp_part + random_part
    return uid

def extract_vocal_from_speech(source_path, target_path, vocal_separator, audio_output_dir_temp):
    outputs = vocal_separator.separate(source_path)
    if len(outputs) <= 0:
        print("Audio separate failed. Using raw audio.")
        return None
        
    default_vocal_path = audio_output_dir_temp / "vocals" / outputs[0]
    default_vocal_path = default_vocal_path.resolve().as_posix()
    cmd = f"mv '{default_vocal_path}' '{target_path}'"
    os.system(cmd)    
    return target_path

def generate(args):
    total_start = time.time()
    timer = StageTimer()

    # load parsed args
    input_json = args.input_json
    checkpoint_dir = args.checkpoint_dir
    context_parallel_size = args.context_parallel_size
    stage_1 = args.stage_1
    num_inference_steps = args.num_inference_steps
    text_guidance_scale = args.text_guidance_scale
    audio_guidance_scale = args.audio_guidance_scale
    resolution = args.resolution
    num_segments = max(1, args.num_segments)
    output_dir = args.output_dir
    model_type = args.model_type
    use_distill = args.use_distill
    use_int8 = args.use_int8

    # 设置日志文件
    os.makedirs(output_dir, exist_ok=True)
    # video_name 和 log_path 在加载输入数据后设置

    if use_distill and model_type == "avatar-v1.5":
        num_inference_steps = 8
        text_guidance_scale = 1.0
        audio_guidance_scale = 1.0

    # set up default inference params
    save_fps = 16
    audio_stride = 2
    if model_type == "avatar-v1.5":
        save_fps = 25
        audio_stride = 1
    num_frames = 93
    num_cond_frames = 13

    if resolution == '480p':
        height, width = 480, 832
    elif resolution == '720p':
        height, width = 768, 1280

    # case setup
    with open(input_json, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    prompt = input_data['prompt']
    negative_prompt = "Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    raw_speech_path = input_data['cond_audio']['person1']

    # 根据图片名、音频名、时间戳生成视频名称
    if stage_1 == 'ai2v' and 'cond_image' in input_data:
        image_name = Path(input_data['cond_image']).stem
    else:
        image_name = "at2v"
    audio_name = Path(raw_speech_path).stem
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = f"{image_name}_{audio_name}_{timestamp}"
    log_path = os.path.join(output_dir, f"{video_name}.log")
    timer.set_log(log_path)
    
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

    # initialize context parallel
    timer.start("[初始化] 上下文并行")
    context_parallel_util.init_context_parallel(context_parallel_size=context_parallel_size, global_rank=global_rank, world_size=num_processes)
    cp_rank = context_parallel_util.get_cp_rank()
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)
    timer.stop()

    # initialize models
    timer.start("[模型] Tokenizer & 文本编码器 & VAE & 调度器")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'), subfolder="tokenizer", torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'), subfolder="text_encoder", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    vae = AutoencoderKLWan.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'), subfolder="vae", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if model_type == "avatar-v1.0":
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'), subfolder="scheduler", torch_dtype=torch.bfloat16)
    elif model_type == "avatar-v1.5":
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Expected 'avatar-v1.0' or 'avatar-v1.5'.")
    timer.stop()
    
    timer.start("[模型] DiT")
    if model_type == "avatar-v1.0":
        dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="avatar_single", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    elif model_type == "avatar-v1.5":
        if use_int8:
            timer.log("[信息] 正在加载 INT8 量化 DiT 模型...")
            dit = load_quantized_dit(checkpoint_dir, subfolder="base_model_int8", cp_split_hw=cp_split_hw)
        else:
            dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="base_model", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        if use_distill:
            distill_checkpoint_path = os.path.join(checkpoint_dir, 'lora', f'dmd_lora.safetensors')
            if os.path.exists(distill_checkpoint_path):
                dit.load_lora(distill_checkpoint_path, "dmd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
                dit.enable_loras(["dmd"])
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Expected 'avatar-v1.0' or 'avatar-v1.5'.")
    timer.stop()
    
    # initialize audio models
    timer.start("[模型] 音频编码器 & 特征提取器")
    if model_type == "avatar-v1.0":
        audio_model_checkpoint_path = os.path.join(checkpoint_dir, 'chinese-wav2vec2-base')
    elif model_type == "avatar-v1.5":
        audio_model_checkpoint_path = os.path.join(checkpoint_dir, 'whisper-large-v3')
    audio_encoder = get_audio_encoder(audio_model_checkpoint_path, model_type).to(local_rank)
    audio_feature_extractor = get_audio_feature_extractor(audio_model_checkpoint_path, model_type)
    timer.stop()

    timer.start("[模型] 人声分离器")
    vocal_separator_path = os.path.join(checkpoint_dir, 'vocal_separator/Kim_Vocal_2.onnx')
    audio_output_dir_temp = f"./audio_temp_file"
    os.makedirs(audio_output_dir_temp, exist_ok=True)
    audio_output_dir_temp = Path(audio_output_dir_temp)
    audio_separator_model_path = os.path.dirname(vocal_separator_path)
    audio_separator_model_name = os.path.basename(vocal_separator_path)
    vocal_separator = Separator(
        output_dir=audio_output_dir_temp / "vocals",
        output_single_stem="vocals",
        model_file_dir=audio_separator_model_path,
    )
    vocal_separator.load_model(audio_separator_model_name)
    timer.stop()

    
    # initialize pipeline
    timer.start("[模型] 推理流水线组装")
    pipe = LongCatVideoAvatarPipeline(
        tokenizer = tokenizer,
        text_encoder = text_encoder,
        vae = vae,
        scheduler = scheduler,
        dit = dit,
        audio_encoder=audio_encoder,
        audio_feature_extractor=audio_feature_extractor,
        model_type=model_type
    )
    pipe.to(local_rank)
    timer.stop()

    # === 开启 VAE 的极致显存优化，专治去噪后的 OOM ===
    if hasattr(pipe.vae, 'enable_slicing'):
        pipe.vae.enable_slicing()
    if hasattr(pipe.vae, 'enable_tiling'):
        pipe.vae.enable_tiling()
    # =======================================================

    global_seed = 42
    seed = global_seed + global_rank

    generator = torch.Generator(device=local_rank)
    generator.manual_seed(seed)

    if cp_rank == 0:
        # extract vocal
        timer.start("[音频] 人声提取")
        temp_vocal_path = extract_vocal_from_speech(raw_speech_path, f"/tmp/temp_speech_{generate_random_uid()}_{global_rank}_vocal.wav", vocal_separator, audio_output_dir_temp)
        timer.stop()
        assert temp_vocal_path is not None and os.path.exists(temp_vocal_path), f"No vocal detected"    

        # audio padding to target length
        timer.start("[音频] 音频填充与嵌入")
        generate_duration = num_frames / save_fps + (num_segments-1)*(num_frames-num_cond_frames) / save_fps
        speech_array, sr = librosa.load(temp_vocal_path, sr=16000)
        source_duraion = len(speech_array) / sr
        added_sample_nums = math.ceil((generate_duration - source_duraion) * sr)
        if added_sample_nums > 0:
            speech_array = np.append(speech_array, [0.]*added_sample_nums)

        # audio embedding
        full_audio_emb = pipe.get_audio_embedding(speech_array, fps=save_fps*audio_stride, device=local_rank, sample_rate=sr, model_type=model_type)
        if torch.isnan(full_audio_emb).any():
            raise ValueError(f"broken audio embedding with nan values")
        timer.stop()

        if context_parallel_util.get_cp_size() > 1:
            timer.start("[音频] CP广播音频嵌入")
            full_audio_emb_shape_list = list(full_audio_emb.size())
            full_audio_emb_tensor_shape_list = torch.tensor(full_audio_emb_shape_list, dtype=torch.int64, device=full_audio_emb.device)
            context_parallel_util.cp_broadcast(full_audio_emb_tensor_shape_list)
            context_parallel_util.cp_broadcast(full_audio_emb)
            timer.stop()
        
        if os.path.exists(temp_vocal_path):
            os.remove(temp_vocal_path)

    elif context_parallel_util.get_cp_size() > 1:
        timer.start("[音频] CP接收音频嵌入")
        full_audio_emb_tensor_shape_list = torch.zeros(3, dtype=torch.int64, device=local_rank)
        context_parallel_util.cp_broadcast(full_audio_emb_tensor_shape_list)
        full_audio_emb_shape_list = full_audio_emb_tensor_shape_list.tolist()
        full_audio_emb = torch.zeros(*full_audio_emb_shape_list, dtype=torch.float32, device=local_rank)
        context_parallel_util.cp_broadcast(full_audio_emb)
        timer.stop()

    # prepare audio embedding for the first clip
    indices = torch.arange(2 * 2 + 1) - 2
    audio_start_idx = 0
    audio_end_idx = audio_start_idx + audio_stride * num_frames

    center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0]-1)
    audio_emb = full_audio_emb[center_indices][None,...].to(local_rank)


    if local_rank == 0:
        timer.log(f"正在生成片段 1/{num_segments}...")

    if stage_1 == 'at2v':
        # ==============================
        #          at2v (480P)
        # ==============================
        timer.start("[生成] 第一阶段: 音频转视频(at2v)")
        output_tuple = pipe.generate_at2v(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            generator=generator,
            output_type='both',
            audio_emb=audio_emb,
            use_distill=use_distill,
        )
        timer.stop()
        output, latent = output_tuple 
        output = output[0] 
        video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        video = [PIL.Image.fromarray(img) for img in video]

        if cp_rank == 0:
            timer.start("[保存] 第一阶段: at2v视频")
            output_tensor = torch.from_numpy(np.array(video))
            save_video_ffmpeg(output_tensor, os.path.join(output_dir, video_name), raw_speech_path, fps=save_fps, quality=5)
            timer.stop()
        del output
        torch_gc()
    
    elif stage_1 == 'ai2v':
        # ==============================
        #          ai2v (480P)
        # ==============================
        image_path = input_data['cond_image']
        image = load_image(image_path)
        timer.start("[生成] 第一阶段: 音频图像转视频(ai2v)")
        output_tuple = pipe.generate_ai2v(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            output_type='both',
            generator=generator,
            audio_emb=audio_emb,
            use_distill=use_distill,
        )
        timer.stop()
        output, latent = output_tuple
        output = output[0]
        video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        video = [PIL.Image.fromarray(img) for img in video]

        if cp_rank == 0:
            timer.start("[保存] 第一阶段: ai2v视频")
            output_tensor = torch.from_numpy(np.array(video))
            save_video_ffmpeg(output_tensor, os.path.join(output_dir, video_name), raw_speech_path, fps=save_fps, quality=5)
            timer.stop()
        del output
        torch_gc()
    else:
        raise NotImplementedError(f"Not supported type of stage_1: {stage_1}")

    if context_parallel_util.get_cp_size() > 1:
        torch.distributed.barrier(group=context_parallel_util.get_cp_group())

    # =========================================
    #         long video generation (480P)
    # =========================================
    # load parsed long video args
    ref_img_index = args.ref_img_index
    mask_frame_range = args.mask_frame_range

    width, height = video[0].size
    current_video = video
    ref_latent = latent[:, :, :1].clone()
    all_generated_frames = video

    # === 在进入第二段生成前，释放所有不需要的显存 ===
    torch.cuda.empty_cache()
    # ====================================================

    for segment_idx in range(1, num_segments):
        if local_rank == 0:
            timer.log(f"正在生成片段 {segment_idx+1}/{num_segments}...")
        
        # prepare audio embedding for the next clip
        audio_start_idx = audio_start_idx + audio_stride * (num_frames - num_cond_frames)
        audio_end_idx   = audio_start_idx + audio_stride * num_frames
        center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0]-1)
        audio_emb = full_audio_emb[center_indices][None,...].to(local_rank)
        
        timer.start(f"[生成] AVC片段 {segment_idx+1}/{num_segments}")
        output_tuple = pipe.generate_avc(
            video=current_video,
            video_latent=latent, 
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            generator=generator,
            output_type='both',
            use_kv_cache=True,
            offload_kv_cache=True,
            enhance_hf=True if not use_distill else False,
            audio_emb=audio_emb,
            ref_latent=ref_latent,
            ref_img_index=ref_img_index,
            mask_frame_range=mask_frame_range,
            use_distill=use_distill,
        )
        timer.stop()
        output, latent = output_tuple

        output = output[0]
        new_video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        new_video = [PIL.Image.fromarray(img) for img in new_video]
        del output

        all_generated_frames.extend(new_video[num_cond_frames:])

        current_video = new_video

        if cp_rank == 0:
            timer.start(f"[保存] AVC片段 {segment_idx+1} 视频")
            output_tensor = torch.from_numpy(np.array(all_generated_frames))
            save_video_ffmpeg(output_tensor, os.path.join(output_dir, f"{video_name}_continue_{segment_idx+1}"), raw_speech_path, fps=save_fps, quality=5)
            timer.stop()
            del output_tensor

    total_elapsed = time.time() - total_start
    if local_rank == 0:
        timer.log(timer.summary())
        timer.log(f"\n  实际总耗时: {total_elapsed:.2f}s")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_json',
        type=str,
        default='assets/avatar/single_example_1.json'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs_avatar_single'
    )
    parser.add_argument(
        '--resolution',
        type=str,
        default='480p',
        choices=['480p', '720p']
    )
    parser.add_argument(
        '--num_segments',
        type=int,
        default=1
    )
    parser.add_argument(
        '--num_inference_steps',
        type=int,
        default=50
    )
    parser.add_argument(
        '--ref_img_index',
        type=int,
        default=10
    )
    parser.add_argument(
        '--mask_frame_range',
        type=int,
        default=3
    )
    parser.add_argument(
        '--text_guidance_scale',
        type=float,
        default=4.0
    )
    parser.add_argument(
        '--audio_guidance_scale',
        type=float,
        default=4.0
    )
    parser.add_argument(
        '--stage_1',
        type=str,
        default='ai2v',
        choices=['ai2v', 'at2v']
    )
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./weights/LongCat-Video-Avatar",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="avatar-v1.0",
    )
    parser.add_argument(
        "--use_distill",
        action='store_true',
    )
    parser.add_argument(
        "--use_int8",
        action='store_true',
        help="Load INT8 quantized DiT model for reduced VRAM usage"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
