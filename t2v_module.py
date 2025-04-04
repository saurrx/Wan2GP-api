# t2v_module.py
# Refactored version of t2v.py for programmatic use
import threading
import os
import time
import torch
import gc
from pathlib import Path
from datetime import datetime
import json
import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import cache_video
from wan.modules.attention import get_attention_modes, get_supported_attention_modes
from mmgp import offload, safetensors2, profile_type
from huggingface_hub import hf_hub_download, snapshot_download
from wan.utils import prompt_parser
import numpy as np # Needed for Teacache coefficients
import traceback # For better error reporting
import random # For random seed

# MMGP version check can be done once in the main server or kept here too
try:
    target_mmgp_version = "3.3.4"
    from importlib.metadata import version
    mmgp_version = version("mmgp")
    if mmgp_version != target_mmgp_version:
        print(f"[t2v_module] WARNING: Incorrect mmgp version ({mmgp_version}), found. {target_mmgp_version} recommended.")
except ImportError:
    print("[t2v_module] WARNING: Could not check mmgp version.")


# --- Helper Functions (Keep as they are needed by generate_video) ---

def download_models(transformer_filename, text_encoder_filename):
    # --- Download logic remains the same ---
    def computeList(filename):
        pos = filename.rfind("/")
        filename = filename[pos+1:]
        return [filename]
    repoId = "DeepBeepMeep/Wan2.1"
    fileList = ["Wan2.1_VAE_bf16.safetensors"] + computeList(text_encoder_filename) + computeList(transformer_filename)
    targetRoot = "ckpts/"
    os.makedirs(targetRoot, exist_ok=True) # Ensure ckpts dir exists
    for onefile in fileList:
        local_path = os.path.join(targetRoot, onefile)
        if not os.path.isfile(local_path):
            print(f"[t2v_module] Downloading {onefile}...")
            try: hf_hub_download(repo_id=repoId, filename=onefile, local_dir=targetRoot)
            except Exception as e: print(f"[t2v_module] ERROR downloading {onefile}: {e}"); raise
    try: # Ensure tokenizer presence
        from transformers import AutoTokenizer
        print("[t2v_module] Ensuring tokenizer is available..."); AutoTokenizer.from_pretrained("google/umt5-xxl")
    except Exception as e: print(f"[t2v_module] Warning: Could not ensure tokenizer. Error: {e}")
    # --- End download_models ---

def load_t2v_model(model_filename, text_encoder_filename, device_id=0):
    # --- Model loading remains the same ---
    cfg = WAN_CONFIGS['t2v-1.3B'] # Assuming 1.3B
    print(f"[t2v_module] Loading '{model_filename}' (using {cfg.__name__})...")
    wan_model = wan.WanT2V(config=cfg, checkpoint_dir="ckpts", device_id=device_id, rank=0, t5_fsdp=False, dit_fsdp=False, use_usp=False, model_filename=model_filename, text_encoder_filename=text_encoder_filename)
    pipe = {"transformer": wan_model.model, "text_encoder": wan_model.text_encoder.model, "vae": wan_model.vae.model}
    return wan_model, pipe
    # --- End load_t2v_model ---

def get_auto_attention():
    # --- Auto attention logic remains the same ---
    attention_modes_supported = get_supported_attention_modes()
    for attn in ["sage2", "sage", "sdpa"]:
        if attn in attention_modes_supported: return attn
    return "sdpa"
    # --- End get_auto_attention ---

def sanitize_file_name(file_name, rep=""):
    # --- Sanitize filename remains the same ---
    return file_name.replace("/", rep).replace("\\", rep).replace(":", rep).replace("|", rep).replace("?", rep).replace("<", rep).replace(">", rep).replace("\"", rep)
    # --- End sanitize_file_name ---

# --- Main Generation Function (Refactored) ---

# Global cache for model and offloader to avoid reloading on every call
_model_cache = {
    "wan_model": None,
    "offloadobj": None,
    "current_config": {}
}
_model_lock = threading.Lock() # For thread-safe model loading/checking

def generate_video(
    # Core parameters
    input_prompt: str,
    output_dir: str = "outputs",
    resolution: str = "832x480",
    frames: int = 81,
    steps: int = 30,
    seed: int = -1,
    guidance_scale: float = 5.0,
    flow_shift: float = 5.0,
    negative_prompt: str = "poor quality, low quality, bad anatomy, wrong anatomy, extra limbs, missing limbs, floating limbs, disconnected limbs, mutation, mutated, disfigured, deformed, poorly drawn hands, too many fingers, missing fingers, extra fingers, fused fingers, disproportionate body, bad proportions, malformed limbs, extra body parts, missing body parts, floating body parts, disfigured face, deformed face, ugly face, bad face anatomy, weird face, double face, two faces, multiple faces, displaced facial features, disproportionate face, unrealistic eyes, crossed eyes, wandering eyes, misaligned eyes, poorly drawn face, face artifacts, unnatural skin, plastic skin, shiny skin, weird expression, unnatural expression, back view, rear view, side view, obscured front, blocked view, cropped body, off-center subject, unbalanced composition, cluttered background, unwanted objects, overlapping figures, overexposed, underexposed, harsh shadows, blurry, grainy, pixelated, out of focus, dark image, washed-out colors, unnatural lighting, flat lighting, static pose, awkward pose, unnatural stance, stiff movement, jerky motion, lack of fluidity, exaggerated motion, unintended angles, disfigured breasts, obscured pussy, clothed figure, unwanted clothing, misplaced clothing, unrealistic genitals, deformed genitals, missing genitals, censored areas, lack of detail, overly smooth skin, lack of muscle definition, cartoonish style, low resolution, oversaturated, undersaturated, artificial look, choppy animation, frame skipping, inconsistent frame rate, looping errors, shaky camera, erratic camera movement, abrupt cuts, slight distortion, subtle disfigurement, minor anatomical errors, uneven body contours, twisted limbs, unnatural joint bends, warped torso", # Use full default later

    # Performance/Hardware parameters
    profile: int = 4,
    attention: str = "auto",
    vae_tile_size: int = 0,
    compile_model: bool = False,
    quantize_t5: bool = False,

    # Feature parameters
    riflex: bool = False,
    teacache: float = 0.0,
    teacache_start_perc: int = 0,

    # Utility
    verbose: int = 1,

    # Internal / Advanced (Optional)
    device_id: int = 0

):
    """
    Generates a video based on the provided parameters.
    Manages model loading and MMGP setup internally.
    Returns the path to the generated video on success, None on failure.
    """
    global _model_cache, _model_lock

    # --- Parameter Validation and Setup ---
    # Adjust frames to 4N+1
    original_frames = frames
    frames = (frames // 4) * 4 + 1
    if frames != original_frames:
        print(f"[t2v_module] Adjusted frame count from {original_frames} to {frames}")

    # Determine filenames based on params
    transformer_filename = "ckpts/wan2.1_text2video_1.3B_bf16.safetensors" # Hardcoded for now
    text_encoder_filename = "ckpts/models_t5_umt5-xxl-enc-quanto_int8.safetensors" if quantize_t5 else "ckpts/models_t5_umt5-xxl-enc-bf16.safetensors"

    # Determine effective attention mode
    effective_attention = attention
    if effective_attention == "auto":
        effective_attention = get_auto_attention()

    # Config hash to check if model needs reloading/re-profiling
    current_config = {
        "transformer": transformer_filename,
        "text_encoder": text_encoder_filename,
        "profile": profile,
        "compile": compile_model,
        # "quantize_transformer": should_quantize_transformer, # Assuming False
    }

    wan_model = None
    offloadobj = None

    # --- Thread-Safe Model Loading / Checking ---
    with _model_lock:
        if _model_cache["wan_model"] is None or _model_cache["current_config"] != current_config:
            print("[t2v_module] Model config changed or not loaded. Loading/Re-profiling...")
            # Cleanup previous if exists
            if _model_cache["offloadobj"] is not None:
                _model_cache["offloadobj"].release()
                print("[t2v_module] Released previous MMGP profile.")
            if _model_cache["wan_model"] is not None:
                del _model_cache["wan_model"] # Allow GC
                print("[t2v_module] Cleared previous model reference.")
            _model_cache["wan_model"] = None
            _model_cache["offloadobj"] = None
            gc.collect()
            torch.cuda.empty_cache()

            offload.default_verboseLevel = verbose
            download_models(transformer_filename, text_encoder_filename)
            wan_model, pipe = load_t2v_model(transformer_filename, text_encoder_filename, device_id)

            # MMGP Profiling
            print(f"[t2v_module] Applying MMGP Profile {profile}...")
            kwargs = {"extraModelsToQuantize": None}
            should_quantize_transformer = False # Default
            if profile == 2 or profile == 4:
                preload = 0; budgets = {"transformer": 100 if preload == 0 else preload, "text_encoder": 100, "*": 1000}; kwargs["budgets"] = budgets
            elif profile == 3:
                kwargs["budgets"] = {"*": "70%"}

            offloadobj = offload.profile(
                pipe, profile_no=profile, compile=compile_model,
                quantizeTransformer=should_quantize_transformer, loras="transformer", **kwargs
            )

            # Store in cache
            _model_cache["wan_model"] = wan_model
            _model_cache["offloadobj"] = offloadobj
            _model_cache["current_config"] = current_config
            print("[t2v_module] Model loaded and profiled.")
        else:
            # Use cached model
            wan_model = _model_cache["wan_model"]
            offloadobj = _model_cache["offloadobj"]
            print("[t2v_module] Using cached model and profile.")
    # --- End Model Loading ---

    # --- Setup per-generation flags and settings ---
    if not hasattr(wan_model, '_interrupt'): wan_model._interrupt = False
    else: wan_model._interrupt = False # Ensure reset

    # Teacache Setup
    trans = wan_model.model
    trans.enable_teacache = teacache > 0
    if trans.enable_teacache:
        print(f"[t2v_module] Enabling Teacache: Mult={teacache}, Start={teacache_start_perc}%")
        trans.teacache_multiplier = teacache
        trans.rel_l1_thresh = 0
        trans.teacache_start_step = int(teacache_start_perc * steps / 100)
        trans.num_steps = steps
        trans.teacache_skipped_steps = 0
        trans.previous_residual_uncond = None
        trans.previous_residual_cond = None
        trans.coefficients = np.array([ # 1.3B T2V coefficients
            2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01
        ])
    else:
        print("[t2v_module] Teacache disabled.")

    # Attention Mode
    print(f"[t2v_module] Using Attention: {effective_attention}")
    if effective_attention not in get_supported_attention_modes():
        print(f"[t2v_module] Warning: Mode '{effective_attention}' might not be installed/supported.")
    offload.shared_state["_attention"] = effective_attention

    # Resolution
    try: width, height = map(int, resolution.split("x"))
    except ValueError: raise ValueError(f"Invalid resolution format '{resolution}'. Use WxH.")

    # Prompt Template Processing
    processed_prompt, errors = prompt_parser.process_template(input_prompt)
    if errors: raise ValueError(f"Prompt template error: {errors}")

    # VAE Tiling Auto-detection
    effective_vae_tile_size = vae_tile_size
    if effective_vae_tile_size == 0:
        try:
            device_mem = torch.cuda.get_device_properties(device_id).total_memory / (1024**2)
            if device_mem < 8000: effective_vae_tile_size = 128
            elif device_mem < 24000: effective_vae_tile_size = 256
            print(f"[t2v_module] Auto VAE Tiles: {effective_vae_tile_size or 'Disabled'} (VRAM: {device_mem:.0f}MB)")
        except Exception as e:
            print(f"[t2v_module] Warning: VRAM detection failed ({e}), defaulting VAE tiles to 256.")
            effective_vae_tile_size = 256

    # Seed
    current_seed = seed if seed != -1 else random.randint(0, 999999999)

    # RIFLEx
    enable_RIFLEx = riflex or frames > (6 * 16)

    # Output Dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Generation ---
    print(f"[t2v_module] Starting generation: Seed={current_seed}, Steps={steps}, Frames={frames}")
    start_time = time.time()
    video_path = None # Initialize

    try:
        gc.collect(); torch.cuda.empty_cache()

        samples = wan_model.generate(
            input_prompt=processed_prompt,
            frame_num=frames,
            size=(width, height),
            shift=flow_shift,
            sampling_steps=steps,
            guide_scale=guidance_scale,
            n_prompt=negative_prompt,
            seed=current_seed,
            offload_model=False, # Handled by MMGP
            enable_RIFLEx=enable_RIFLEx,
            VAE_tile_size=effective_vae_tile_size,
            joint_pass=True
        )

        if samples is not None: samples = samples.to("cpu")

        # Cleanup GPU BEFORE saving (saving can take time)
        if offload.last_offload_obj is not None:
             print("[t2v_module] Unloading models from VRAM...")
             offload.last_offload_obj.unload_all()
        gc.collect(); torch.cuda.empty_cache()

        if samples is None:
            raise RuntimeError("Video generation ABORTED or FAILED (returned None).")

        # --- Saving ---
        print("[t2v_module] Saving video...")
        time_flag = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")
        safe_prompt = sanitize_file_name(processed_prompt[:50].strip())
        file_name = f"{time_flag}_seed{current_seed}_{safe_prompt}.mp4"
        video_path = os.path.join(output_dir, file_name)
        cache_video(tensor=samples[None], save_file=video_path, fps=16, nrow=1, normalize=True, value_range=(-1, 1))

        # --- Metadata ---
        print("[t2v_module] Saving metadata...")
        metadata = {
             "type": "Wan2.1GP - text2video (API)", "prompts": processed_prompt, "resolution": resolution,
             "video_length": frames, "num_inference_steps": steps, "seed": current_seed,
             "guidance_scale": guidance_scale, "flow_shift": flow_shift, "negative_prompt": negative_prompt,
             "riflex_enabled": enable_RIFLEx, "vae_tile_size": effective_vae_tile_size, "model": "1.3B",
             "profile": profile, "attention": effective_attention, "compile": compile_model,
             "teacache_enabled": trans.enable_teacache,
             "teacache_multiplier": teacache if trans.enable_teacache else 0,
             "teacache_start_perc": teacache_start_perc if trans.enable_teacache else 0,
        }
        json_path = video_path.replace('.mp4', '.json')
        try:
            with open(json_path, 'w') as f: json.dump(metadata, f, indent=4)
        except Exception as e: print(f"[t2v_module] Warning: Failed to save metadata: {e}")

        end_time = time.time()
        print(f"[t2v_module] Generation successful in {end_time - start_time:.2f}s.")
        print(f"[t2v_module] Output: {video_path}")
        return video_path # Return path on success

    except Exception as e:
        print(f"\n[t2v_module] --- ERROR DURING GENERATION ---")
        traceback.print_exc()
        print(f"[t2v_module] Error: {str(e)}")
        # No need for separate finally cleanup if using with _model_lock properly
        raise e # Re-raise the exception for the worker thread to catch

# Note: No `if __name__ == "__main__":` block here, this is now a module.
