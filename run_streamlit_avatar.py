import os
import uuid
import math
import tempfile
import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import streamlit as st
import numpy as np
import librosa
from PIL import Image
from diffusers.utils import load_image

from transformers import AutoTokenizer, UMT5EncoderModel

from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from longcat_video.modules.quantization import load_quantized_dit
from longcat_video.context_parallel import context_parallel_util
from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
from longcat_video.audio_process.torch_utils import save_video_ffmpeg
from audio_separator.separator import Separator


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


# ---------- Page config ----------
st.set_page_config(
    page_title="LongCat-Video Avatar 1.5",
    page_icon="🎭",
    layout="wide"
)


# ---------- Init distributed (single GPU for Streamlit) ----------
def init_single_gpu():
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24))
    context_parallel_util.init_context_parallel(
        context_parallel_size=1, global_rank=0, world_size=1
    )


init_single_gpu()


# ---------- Load model ----------
@st.cache_resource
def load_model(checkpoint_dir, base_checkpoint_dir, use_int8=True, use_distill=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    local_rank = 0

    with st.spinner("Loading model... (may take a few minutes)"):
        cp_split_hw = context_parallel_util.get_optimal_split(1)

        # Load tokenizer / text_encoder / vae from base model
        tokenizer = AutoTokenizer.from_pretrained(
            base_checkpoint_dir, subfolder="tokenizer", torch_dtype=torch_dtype
        )
        text_encoder = UMT5EncoderModel.from_pretrained(
            base_checkpoint_dir, subfolder="text_encoder", torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )
        vae = AutoencoderKLWan.from_pretrained(
            base_checkpoint_dir, subfolder="vae", torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )

        # Load scheduler from Avatar-1.5
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            checkpoint_dir, subfolder="scheduler", torch_dtype=torch_dtype
        )

        # Load DiT
        if use_int8:
            st.info("Loading INT8 quantized DiT model...")
            dit = load_quantized_dit(
                checkpoint_dir, subfolder="base_model_int8", cp_split_hw=cp_split_hw
            )
        else:
            dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(
                checkpoint_dir, subfolder="base_model",
                cp_split_hw=cp_split_hw, torch_dtype=torch_dtype,
                low_cpu_mem_usage=True
            )

        # Load distill LoRA
        if use_distill:
            distill_lora_path = os.path.join(checkpoint_dir, "lora", "dmd_lora.safetensors")
            if os.path.exists(distill_lora_path):
                dit.load_lora(
                    distill_lora_path, "dmd",
                    multiplier=1.0, lora_network_dim=128, lora_network_alpha=64
                )
                dit.enable_loras(["dmd"])
                st.info("Distill LoRA (DMD) loaded")

        # Audio models
        audio_model_path = os.path.join(checkpoint_dir, "whisper-large-v3")
        audio_encoder = get_audio_encoder(audio_model_path, "avatar-v1.5").to(local_rank)
        audio_feature_extractor = get_audio_feature_extractor(audio_model_path, "avatar-v1.5")

        # Vocal separator
        vocal_separator_path = os.path.join(checkpoint_dir, "vocal_separator", "Kim_Vocal_2.onnx")
        audio_temp_dir = Path("./audio_temp_file")
        audio_temp_dir.mkdir(exist_ok=True)
        vocal_separator = Separator(
            output_dir=audio_temp_dir / "vocals",
            output_single_stem="vocals",
            model_file_dir=os.path.dirname(vocal_separator_path),
        )
        vocal_separator.load_model(os.path.basename(vocal_separator_path))

        # Pipeline
        pipe = LongCatVideoAvatarPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            dit=dit,
            audio_encoder=audio_encoder,
            audio_feature_extractor=audio_feature_extractor,
            model_type="avatar-v1.5",
        )
        pipe.to(device)

    return pipe, vocal_separator, audio_temp_dir, device


def extract_vocal(raw_speech_path, vocal_separator, audio_temp_dir):
    """Extract vocal from audio using vocal separator."""
    outputs = vocal_separator.separate(raw_speech_path)
    if len(outputs) <= 0:
        st.warning("Audio separation failed, using raw audio.")
        return None
    default_vocal_path = audio_temp_dir / "vocals" / outputs[0]
    default_vocal_path = default_vocal_path.resolve().as_posix()
    target_path = str(audio_temp_dir / f"vocal_{uuid.uuid4().hex[:8]}.wav")
    os.system(f"mv '{default_vocal_path}' '{target_path}'")
    return target_path


def process_audio(pipe, raw_speech_path, vocal_separator, audio_temp_dir,
                  num_frames, num_segments, device):
    """Extract vocal, compute audio embedding."""
    save_fps = 25
    audio_stride = 1
    num_cond_frames = 13

    # Extract vocal
    with st.spinner("Extracting vocal from audio..."):
        temp_vocal_path = extract_vocal(raw_speech_path, vocal_separator, audio_temp_dir)
        if temp_vocal_path is None:
            temp_vocal_path = raw_speech_path

    # Load and pad audio
    sr = 16000
    generate_duration = (
        num_frames / save_fps
        + (num_segments - 1) * (num_frames - num_cond_frames) / save_fps
    )
    speech_array, _ = librosa.load(temp_vocal_path, sr=sr)
    source_duration = len(speech_array) / sr
    added_samples = math.ceil((generate_duration - source_duration) * sr)
    if added_samples > 0:
        speech_array = np.append(speech_array, [0.0] * added_samples)

    # Audio embedding
    with st.spinner("Computing audio embedding..."):
        full_audio_emb = pipe.get_audio_embedding(
            speech_array, fps=save_fps * audio_stride,
            device=device, sample_rate=sr, model_type="avatar-v1.5"
        )
        if torch.isnan(full_audio_emb).any():
            st.error("Audio embedding contains NaN values!")
            return None

    # Prepare embedding for the first clip
    indices = torch.arange(2 * 2 + 1) - 2
    audio_start_idx = 0
    audio_end_idx = audio_start_idx + audio_stride * num_frames
    center_indices = (
        torch.arange(audio_start_idx, audio_end_idx, audio_stride)
        .unsqueeze(1) + indices.unsqueeze(0)
    )
    center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
    audio_emb = full_audio_emb[center_indices][None, ...].to(device)

    # Cleanup temp file
    if temp_vocal_path != raw_speech_path and os.path.exists(temp_vocal_path):
        os.remove(temp_vocal_path)

    return audio_emb, full_audio_emb, save_fps, audio_stride, num_cond_frames


def main():
    st.title("🎭 LongCat-Video Avatar 1.5")
    st.markdown("Audio-driven character video generation (single-speaker)")

    # ---- Sidebar: model config ----
    st.sidebar.title("⚙️ Model Config")
    checkpoint_dir = st.sidebar.text_input(
        "Avatar-1.5 Dir", "/app/weights/LongCat-Video-Avatar-1.5"
    )
    base_checkpoint_dir = st.sidebar.text_input(
        "Base Model Dir", "/app/weights/LongCat-Video"
    )
    use_int8 = st.sidebar.checkbox(
        "INT8 Quantization", value=True, help="Reduced VRAM usage"
    )
    use_distill = st.sidebar.checkbox(
        "Distill Mode (8 steps)", value=True, help="Faster inference with DMD LoRA"
    )

    # Load model
    try:
        pipe, vocal_separator, audio_temp_dir, device = load_model(
            checkpoint_dir, base_checkpoint_dir,
            use_int8=use_int8, use_distill=use_distill
        )
        st.success(f"Model loaded! Device: {device}")
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return

    # ---- Sidebar: generation params ----
    st.sidebar.title("🎯 Generation Params")
    mode = st.sidebar.selectbox(
        "Mode",
        ["AI2V (Audio+Image→Video)", "AT2V (Audio+Text→Video)"],
        index=0
    )
    resolution = st.sidebar.selectbox("Resolution", ["480p", "720p"], index=0)

    num_inference_steps = st.sidebar.number_input(
        "Inference Steps", min_value=1, max_value=100,
        value=8 if use_distill else 50
    )
    text_guidance_scale = st.sidebar.number_input(
        "Text Guidance Scale", min_value=0.1, max_value=10.0,
        value=1.0 if use_distill else 4.0, step=0.5
    )
    audio_guidance_scale = st.sidebar.number_input(
        "Audio Guidance Scale", min_value=0.1, max_value=10.0,
        value=1.0 if use_distill else 4.0, step=0.5
    )
    seed = st.sidebar.number_input(
        "Random Seed", min_value=0, max_value=2**32 - 1, value=42
    )

    # ---- Main interface ----
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Input")

        prompt = st.text_area(
            "Prompt (describe the character and scene)",
            height=100,
            placeholder=(
                "e.g. A young woman with long black hair is speaking and smiling, "
                "wearing a white blouse, sitting in a bright café"
            ),
        )

        # Audio upload (required)
        audio_file = st.file_uploader(
            "Upload Audio *required*",
            type=["mp3", "wav", "flac", "ogg", "m4a"],
            help="The speech audio that drives lip sync"
        )
        if audio_file:
            st.audio(audio_file)

        # Image upload (for AI2V mode)
        image_file = None
        if mode.startswith("AI2V"):
            image_file = st.file_uploader(
                "Upload Portrait Image *required*",
                type=["png", "jpg", "jpeg"],
                help="Portrait image of the character"
            )
            if image_file:
                st.image(image_file, caption="Uploaded Portrait", use_container_width=True)

        generate_btn = st.button("🎭 Generate", type="primary", use_container_width=True)

    with col2:
        st.subheader("🎥 Output")
        result_placeholder = st.empty()

        if generate_btn:
            if not prompt.strip():
                st.error("Please enter a prompt!")
                return
            if audio_file is None:
                st.error("Please upload an audio file!")
                return
            if mode.startswith("AI2V") and image_file is None:
                st.error("Please upload a portrait image for AI2V mode!")
                return

            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

            num_frames = 93
            save_fps = 25

            # Save uploaded audio to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                tmp_audio.write(audio_file.read())
                raw_speech_path = tmp_audio.name

            # Save uploaded image to temp file (if AI2V)
            input_image = None
            if image_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                    tmp_img.write(image_file.read())
                    input_image = load_image(tmp_img.name)

            # Process audio
            audio_result = process_audio(
                pipe, raw_speech_path, vocal_separator, audio_temp_dir,
                num_frames=num_frames, num_segments=1, device=device
            )
            if audio_result is None:
                return
            audio_emb, _, _, _, _ = audio_result

            # Generate
            negative_prompt = (
                "Close-up, Bright tones, overexposed, static, blurred details, subtitles, "
                "style, works, paintings, images, static, overall gray, worst quality, low quality, "
                "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
                "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
                "still picture, messy background, three legs, many people in the background, "
                "walking backwards"
            )

            try:
                with st.spinner("Generating video... (this may take a few minutes)"):
                    if mode.startswith("AI2V"):
                        output = pipe.generate_ai2v(
                            image=input_image,
                            prompt=prompt,
                            negative_prompt=None if use_distill else negative_prompt,
                            resolution=resolution,
                            num_frames=num_frames,
                            num_inference_steps=num_inference_steps,
                            text_guidance_scale=text_guidance_scale,
                            audio_guidance_scale=audio_guidance_scale,
                            generator=generator,
                            audio_emb=audio_emb,
                            use_distill=use_distill,
                        )[0]
                    else:
                        output = pipe.generate_at2v(
                            prompt=prompt,
                            negative_prompt=None if use_distill else negative_prompt,
                            height=480,
                            width=832,
                            num_frames=num_frames,
                            num_inference_steps=num_inference_steps,
                            text_guidance_scale=text_guidance_scale,
                            audio_guidance_scale=audio_guidance_scale,
                            generator=generator,
                            audio_emb=audio_emb,
                            use_distill=use_distill,
                        )[0]

                # Save result
                output_np = (output[0] * 255).astype(np.uint8)
                output_tensor = torch.from_numpy(output_np).unsqueeze(0)

                with tempfile.NamedTemporaryFile(delete=False, suffix="") as out_dir:
                    out_prefix = out_dir.name

                save_video_ffmpeg(
                    output_tensor, out_prefix, raw_speech_path,
                    fps=save_fps, quality=5
                )
                video_path = out_prefix + ".mp4"

                with result_placeholder.container():
                    st.success("Generation complete!")
                    if os.path.exists(video_path):
                        st.video(video_path)
                        with open(video_path, "rb") as f:
                            st.download_button(
                                label="Download Video",
                                data=f.read(),
                                file_name=f"avatar_{mode.split()[0].lower()}_{seed}.mp4",
                                mime="video/mp4",
                            )
                    else:
                        st.warning("Video file not found, but generation completed.")

            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                torch_gc()

            # Cleanup temp audio file
            if os.path.exists(raw_speech_path):
                os.remove(raw_speech_path)


if __name__ == "__main__":
    main()
