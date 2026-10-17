"""Gradio UI for UniVid-Intrinsic inference.

Run with:  .venv\\Scripts\\python.exe app.py [--models models.yaml]
Model files come from models.yaml (copy models.example.yaml and edit the paths for this
machine). The model stays loaded between runs and is only rebuilt when a model setting changes.
"""
import argparse
import gc
import json
import os
import sys
from datetime import datetime

# The pipelines print emoji; Windows' default cp1252 console encoding would crash on them
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import gradio as gr
import imageio
import torch
import yaml

from src.pipelines.univid_intrinsic import UniVidIntrinsic
from src.video_io import load_video_clip, save_video
from src.models.wan_video_dit_intrinsic import set_attention_mode
from src.pipelines.long_video import chunk_plan, run_long_r2ain

try:
    import sageattention  # noqa: F401
    SAGE_AVAILABLE = True
except ImportError:
    SAGE_AVAILABLE = False

RESULTS_DIR = "results/gradio"
FPS = 15
HEIGHT, WIDTH, NUM_FRAMES = 480, 640, 21  # trained resolution and window length
RESOLUTION_INFO = ("0 = trained 640×480 · 1 = input video size · any other number = longest side in pixels, "
                   "aspect ratio from the input video. Snapped to multiples of 16; the input is always stretched. "
                   "Higher resolutions are much slower and use more VRAM.")

MODALITIES = ["rgb", "albedo", "irradiance", "normal"]
LETTER_TO_MODALITY = {"R": "rgb", "A": "albedo", "I": "irradiance", "N": "normal"}
MODES = [
    "t2RAIN", "R2AIN", "A2RIN", "I2RAN", "N2RAI",
    "RA2IN", "RI2AN", "RN2AI", "AI2RN", "AN2RI",
    "IN2RA", "AIN2R", "RIN2A", "RAN2I", "RAI2N",
]
DEFAULT_NEGATIVE = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

# UniVid-Intrinsic adapter layout (must match the released checkpoint)
MODEL_PARAMS = {
    "lora_base_model": "dit",
    "lora_target_modules": "self_attn.q,self_attn.k,self_attn.v,self_attn.o,ffn.0,ffn.2",
    "lora_rank": 32,
    "lora_modalities": ["rgb", "albedo", "irradiance", "normal"],
}
DEFAULT_PROMPT = "一只小刺猬，穿着白色小围裙，头上戴着厨师帽，正站在小凳子上，双手举着一个小平底锅，锅里冒着热气，表情专注而自豪，位于一个现代化的迷你厨房中，不锈钢台面反射着明亮的光线，各种小厨具整齐地排列着, 镜头从右向左移动。"

MODEL_LISTS = ["diffusion_models", "unividx_models", "distill_loras", "svi_loras"]
MODEL_FILES = ["text_encoder", "tokenizer", "vae"]

_model = None
_model_key = None


def condition_modalities(mode):
    """Modalities supplied as input videos, e.g. 'RA2IN' -> ['rgb', 'albedo']."""
    cond = mode.split("2")[0]
    return [] if cond == "t" else [LETTER_TO_MODALITY[c] for c in cond]


def describe_mode(mode):
    conds = condition_modalities(mode)
    gens = [m for m in MODALITIES if m not in conds]
    src = "text" if not conds else " + ".join(conds)
    return f"{mode}: {src} → {', '.join(gens)}"


def load_models_config(path):
    """Reads models.yaml. Returns (config with existing files only, list of warnings)."""
    if not os.path.exists(path):
        sys.exit(f"{path} not found. Copy models.example.yaml to {path} and set the paths for this machine.")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    norm = lambda p: str(p).strip().replace("\\", "/")
    config, warnings = {}, []
    for key in MODEL_LISTS:
        entries = raw.get(key) or []
        entries = [entries] if isinstance(entries, str) else entries
        config[key] = []
        for p in map(norm, entries):
            (config[key].append(p) if os.path.exists(p) else warnings.append(f"{key}: not found: {p}"))
    for key in MODEL_FILES:
        p = norm(raw[key]) if raw.get(key) else None
        if not p or not os.path.exists(p):
            sys.exit(f"{path}: '{key}' is missing or not found ({p}).")
        config[key] = p
    for key in ("diffusion_models", "unividx_models"):
        if not config[key]:
            sys.exit(f"{path}: no existing file listed under '{key}'.")
    config["vram_limit"] = raw.get("vram_limit", 16)
    return config, warnings


def choices(paths, allow_none=False):
    """Dropdown (label, value) pairs labelled by file name; the first path is the default."""
    items = [(os.path.basename(p), p) for p in paths]
    return ([("None", "None")] if allow_none else []) + items


def get_model(dit_path, unividx_path, lora_path, lora_strength, extra_path, extra_strength, vram_limit):
    global _model, _model_key
    key = (dit_path, unividx_path, lora_path, float(lora_strength) if lora_path else None,
           extra_path, float(extra_strength) if extra_path else None, vram_limit)
    if _model is not None and _model_key == key:
        return _model
    if _model is not None:
        # Free the old 14B model before building a new one, or RAM usage doubles
        _model = None
        gc.collect()
        torch.cuda.empty_cache()
    params = dict(MODEL_PARAMS)
    params.update(model_paths=json.dumps([MODELS["text_encoder"], MODELS["vae"]]), tokenizer_path=MODELS["tokenizer"],
                  resume_from_checkpoint=unividx_path,
                  dit_path=dit_path, distill_lora_path=lora_path, distill_lora_strength=float(lora_strength),
                  extra_lora_path=extra_path, extra_lora_strength=float(extra_strength), vram_limit=vram_limit)
    model = UniVidIntrinsic(**params)
    model.eval()
    _model, _model_key = model, key
    return model


def load_model_from_ui(dit_path, unividx_path, lora_name, lora_strength, extra_lora, extra_strength, vram_limit, use_sage, progress):
    # Attention backend is a runtime switch, so it never triggers a reload
    set_attention_mode("sage" if use_sage and SAGE_AVAILABLE else "scaled_dot_product")
    lora_path = None if lora_name in (None, "None") else lora_name
    extra_path = None if extra_lora in (None, "None") else extra_lora
    vram_limit = None if not vram_limit else float(vram_limit)
    progress(0, desc="Loading model (only when a model setting changed)...")
    return get_model(dit_path, unividx_path, lora_path, lora_strength, extra_path, extra_strength, vram_limit), vram_limit


def video_size(path):
    """(height, width) of a video file."""
    reader = imageio.get_reader(path)
    width, height = reader.get_meta_data()["size"]
    reader.close()
    return height, width


def target_size(resolution, video_path=None):
    """(height, width) for the resolution setting; multiples of 16, as the Wan VAE and patching need."""
    resolution = int(resolution or 0)
    if resolution == 0:
        return HEIGHT, WIDTH
    src_h, src_w = video_size(video_path) if video_path else (HEIGHT, WIDTH)
    if resolution != 1:
        scale = resolution / max(src_h, src_w)
        src_h, src_w = src_h * scale, src_w * scale
    snap = lambda x: max(16, int(round(x / 16)) * 16)
    return snap(src_h), snap(src_w)


def describe_size(height, width):
    """Resolution note with a rough cost relative to the trained 640×480."""
    area = height * width / (HEIGHT * WIDTH)
    # Attention cost grows with the square of the token count, the rest linearly
    note = "" if abs(area - 1) < 0.05 else f" (~{0.5 * area + 0.5 * area ** 2:.1f}× the sampling time of 640×480)"
    return f"{width}×{height}{note}"


def sampling_params(prompt, negative_prompt, seed, steps, cfg_scale, height=HEIGHT, width=WIDTH):
    return {
        "negative_prompt": negative_prompt,
        "seed": int(seed),
        "num_inference_steps": int(steps),
        # An empty prompt disables CFG, as in the original script
        "cfg_scale": float(cfg_scale) if prompt.strip() else 1.0,
        "cfg_merge": False,
        "height": height, "width": width, "num_frames": NUM_FRAMES,
        "denoising_strength": 1.0,
        "tiled": True, "tile_size": [30, 52], "tile_stride": [15, 26],
        "is_inference": True,
        "prompt": [prompt] * 4,
    }


def to_unit(video):
    """[1,3,T,H,W] or [3,T,H,W] in [-1,1] -> [3,T,H,W] in [0,1] on CPU."""
    if video.ndim == 5:
        video = video[0]
    return (video.detach().float().cpu() * 0.5 + 0.5).clamp(0, 1)


def write_grid(videos, path, fps=FPS):
    """2x2 grid: rgb | albedo / irradiance | normal."""
    top = torch.cat([videos["rgb"], videos["albedo"]], dim=3)
    bottom = torch.cat([videos["irradiance"], videos["normal"]], dim=3)
    grid = torch.cat([top, bottom], dim=2)  # 3,T,2H,2W
    frames = (grid.permute(1, 2, 3, 0) * 255).to(torch.uint8).numpy()
    imageio.mimwrite(path, frames, fps=fps, codec="libx264")


def save_settings(out_dir, settings):
    with open(os.path.join(out_dir, "settings.json"), "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


def update_mode(mode):
    conds = condition_modalities(mode)
    return [gr.update(visible=m in conds) for m in MODALITIES] + [describe_mode(mode)]


def generate(mode, rgb_in, albedo_in, irradiance_in, normal_in, prompt, negative_prompt,
             seed, steps, cfg_scale, resolution, dit_path, unividx_path, lora_name, lora_strength, extra_lora, extra_strength, vram_limit, use_sage,
             progress=gr.Progress(track_tqdm=True)):
    inputs = {"rgb": rgb_in, "albedo": albedo_in, "irradiance": irradiance_in, "normal": normal_in}
    conds = condition_modalities(mode)
    missing = [m for m in conds if not inputs[m]]
    if missing:
        raise gr.Error(f"{mode} needs input video(s): {', '.join(missing)}")
    if mode == "t2RAIN" and not prompt.strip():
        raise gr.Error("t2RAIN generates from text only, so it needs a prompt.")

    model, vram_limit = load_model_from_ui(dit_path, unividx_path, lora_name, lora_strength, extra_lora, extra_strength, vram_limit, use_sage, progress)
    device = torch.device(model.pipe.device)

    # Size follows the first supplied input video (text-only mode uses the trained aspect ratio)
    height, width = target_size(resolution, next((inputs[m] for m in conds), None))
    params = sampling_params(prompt, negative_prompt, seed, steps, cfg_scale, height, width)
    params["training_mode"] = mode
    cond_videos = {}
    for m in MODALITIES:
        video = None
        if m in conds:
            video = load_video_clip(inputs[m], num_frames=NUM_FRAMES, height=height, width=width,
                                    device=device, dtype=torch.bfloat16)
            cond_videos[m] = video
        params[f"inference_{m}"] = video

    with torch.no_grad():
        video_dict = model.pipe(**params)

    out_dir = os.path.join(RESULTS_DIR, f"{datetime.now():%Y%m%d_%H%M%S}_{mode}")
    os.makedirs(out_dir, exist_ok=True)
    videos, outputs = {}, {}
    for m in MODALITIES:
        if m in conds:
            videos[m] = to_unit(cond_videos[m])
            outputs[m] = None
        else:
            videos[m] = to_unit(video_dict["normal_unit" if m == "normal" else m])
            path = os.path.join(out_dir, f"{m}_gen.mp4")
            save_video(videos[m], path, fps=FPS)
            outputs[m] = path
    grid_path = os.path.join(out_dir, "grid.mp4")
    write_grid(videos, grid_path)

    save_settings(out_dir, {
        "mode": mode, "prompt": prompt, "negative_prompt": negative_prompt, "seed": int(seed),
        "steps": int(steps), "cfg_scale": params["cfg_scale"], "resolution_setting": int(resolution or 0),
        "width": width, "height": height, "base_dit": dit_path, "unividx": unividx_path, "distill_lora": lora_name,
        "distill_lora_strength": float(lora_strength), "extra_lora": extra_lora, "extra_lora_strength": float(extra_strength),
        "vram_limit": vram_limit, "sage_attention": bool(use_sage and SAGE_AVAILABLE),
        "inputs": {m: inputs[m] for m in conds},
    })
    return [grid_path] + [outputs[m] for m in MODALITIES] + [f"Saved {width}×{height} to `{out_dir}`"]


def describe_long(video, overlap, max_frames, resolution):
    if not video:
        return "Upload an RGB video."
    try:
        reader = imageio.get_reader(video)
        meta = reader.get_meta_data()
        num = reader.count_frames()
        reader.close()
    except Exception as e:
        return f"Could not read video: {e}"
    if max_frames:
        num = min(num, int(max_frames))
    chunks = len(chunk_plan(num, int(overlap)))
    height, width = target_size(resolution, video)
    return (f"{num} frames at {meta.get('fps', '?')} fps → {chunks} chunk(s) of {NUM_FRAMES} frames, "
            f"{NUM_FRAMES - 1 - 4 * int(overlap)} new frames per chunk after the first. "
            f"Output: {describe_size(height, width)}.")


def generate_long(video, prompt, negative_prompt, seed, steps, cfg_scale, resolution, overlap, max_frames, release_steps,
                  dit_path, unividx_path, lora_name, lora_strength, extra_lora, extra_strength, vram_limit, use_sage,
                  progress=gr.Progress(track_tqdm=True)):
    if not video:
        raise gr.Error("Upload an RGB video.")
    model, vram_limit = load_model_from_ui(dit_path, unividx_path, lora_name, lora_strength, extra_lora, extra_strength, vram_limit, use_sage, progress)
    height, width = target_size(resolution, video)
    params = sampling_params(prompt, negative_prompt, seed, steps, cfg_scale, height, width)
    outputs, fps, rgb = run_long_r2ain(model.pipe, video, params, overlap=int(overlap),
                                       max_frames=int(max_frames or 0), release_steps=int(release_steps), progress=progress)

    out_dir = os.path.join(RESULTS_DIR, f"{datetime.now():%Y%m%d_%H%M%S}_R2AIN_long")
    os.makedirs(out_dir, exist_ok=True)
    videos = {"rgb": to_unit(rgb)}
    paths = {}
    for m, video_out in outputs.items():
        videos[m] = to_unit(video_out)
        paths[m] = os.path.join(out_dir, f"{m}_gen.mp4")
        save_video(videos[m], paths[m], fps=fps)
    grid_path = os.path.join(out_dir, "grid.mp4")
    write_grid(videos, grid_path, fps=fps)

    num_frames = videos["rgb"].shape[1]
    save_settings(out_dir, {
        "mode": "R2AIN (long)", "input": video, "frames": num_frames, "fps": fps,
        "overlap_latents": int(overlap), "free_final_steps": int(release_steps), "chunks": len(chunk_plan(num_frames, int(overlap))),
        "prompt": prompt, "negative_prompt": negative_prompt, "seed": int(seed), "steps": int(steps),
        "resolution_setting": int(resolution or 0), "width": width, "height": height,
        "cfg_scale": params["cfg_scale"], "base_dit": dit_path, "unividx": unividx_path, "distill_lora": lora_name,
        "distill_lora_strength": float(lora_strength), "extra_lora": extra_lora, "extra_lora_strength": float(extra_strength),
        "vram_limit": vram_limit, "sage_attention": bool(use_sage and SAGE_AVAILABLE),
    })
    return [grid_path, paths["albedo"], paths["irradiance"], paths["normal"],
            f"Saved {num_frames} frames at {width}×{height} to `{out_dir}`"]


def sampling_controls(default_prompt):
    prompt = gr.Textbox(default_prompt, label="Prompt", lines=3,
                        info="Chinese prompts are recommended. Leave empty in video-conditioned modes to disable CFG.")
    with gr.Accordion("Negative prompt", open=False):
        negative_prompt = gr.Textbox(DEFAULT_NEGATIVE, show_label=False, lines=3)
    with gr.Row():
        seed = gr.Number(1, label="Seed", precision=0)
        steps = gr.Slider(1, 60, value=4, step=1, label="Steps")
        cfg_scale = gr.Slider(1.0, 10.0, value=1.0, step=0.5, label="CFG scale")
    resolution = gr.Number(0, label="Resolution", precision=0, minimum=0, info=RESOLUTION_INFO)
    return prompt, negative_prompt, seed, steps, cfg_scale, resolution


def build_ui(warnings):
    dits, univids = choices(MODELS["diffusion_models"]), choices(MODELS["unividx_models"])
    distills, svis = choices(MODELS["distill_loras"], allow_none=True), choices(MODELS["svi_loras"], allow_none=True)
    with gr.Blocks(title="UniVid-Intrinsic") as demo:
        gr.Markdown("## UniVid-Intrinsic\nRGB / albedo / irradiance / normal videos (trained at "
                    f"{WIDTH}×{HEIGHT}). The first run loads the model, which takes a few minutes.")
        if warnings:
            gr.Markdown("**Listed in models.yaml but not found on this machine (hidden):**\n" + "\n".join(f"- `{w}`" for w in warnings))
        with gr.Accordion("Model (changing these reloads the model)", open=True):
            with gr.Row():
                dit_path = gr.Dropdown(dits, value=dits[0][1], label="Base DiT",
                                       info="UniVidX was trained on Wan2.1 T2V 14B; Wan2.2 low-noise experts have the same architecture.")
                unividx_path = gr.Dropdown(univids, value=univids[0][1], label="UniVidX checkpoint")
            with gr.Row():
                lora_name = gr.Dropdown(distills, value=distills[1][1] if len(distills) > 1 else "None",
                                        label="Distill LoRA", info="Use 4 steps and CFG 1 with a distill LoRA; 50 steps and CFG 5 without.")
                lora_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="LoRA strength")
            with gr.Row():
                extra_lora = gr.Dropdown(svis, value=svis[1][1] if len(svis) > 1 else "None", label="SVI LoRA",
                                         info="Merged on top of the distill LoRA; improves long-video seams.")
                extra_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="SVI LoRA strength")
                vram_limit = gr.Number(MODELS["vram_limit"], label="VRAM limit (GB)",
                                       info="GPU budget for streaming weights from RAM. 0 = load everything onto the GPU.")
            use_sage = gr.Checkbox(SAGE_AVAILABLE, label="SageAttention", interactive=SAGE_AVAILABLE,
                                   info="About 2x faster attention (~25% faster sampling). Switches per run, no reload."
                                   if SAGE_AVAILABLE else "Not installed in this environment.")
        model_inputs = [dit_path, unividx_path, lora_name, lora_strength, extra_lora, extra_strength, vram_limit, use_sage]

        with gr.Tab("Single window"):
            with gr.Row():
                with gr.Column(scale=1):
                    mode = gr.Dropdown(MODES, value="t2RAIN", label="Mode")
                    mode_info = gr.Markdown(describe_mode("t2RAIN") + f" ({NUM_FRAMES} frames ≈ {NUM_FRAMES / FPS:.1f} s at 15 fps)")
                    with gr.Row():
                        rgb_in = gr.Video(label="RGB input", visible=False)
                        albedo_in = gr.Video(label="Albedo input", visible=False)
                    with gr.Row():
                        irradiance_in = gr.Video(label="Irradiance input", visible=False)
                        normal_in = gr.Video(label="Normal input", visible=False)
                    prompt, negative_prompt, seed, steps, cfg_scale, resolution = sampling_controls(DEFAULT_PROMPT)
                    run = gr.Button("Generate", variant="primary")
                    status = gr.Markdown()
                with gr.Column(scale=1):
                    grid = gr.Video(label="All modalities (RGB | albedo / irradiance | normal)", autoplay=True, loop=True)
                    with gr.Row():
                        rgb_out = gr.Video(label="RGB", loop=True)
                        albedo_out = gr.Video(label="Albedo", loop=True)
                    with gr.Row():
                        irradiance_out = gr.Video(label="Irradiance", loop=True)
                        normal_out = gr.Video(label="Normal", loop=True)

        with gr.Tab("Long video (R2AIN)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("Decomposes an RGB video of any length into albedo, irradiance and normals, "
                                f"in overlapping {NUM_FRAMES}-frame chunks that reuse the previous chunk's latents.")
                    long_video = gr.Video(label="RGB input")
                    with gr.Row():
                        overlap = gr.Radio([1, 2], value=2, label="Overlap (latent frames)",
                                           info="2 = smoother seams, 12 new frames per chunk; 1 = faster, 16 new frames per chunk.")
                        max_frames = gr.Number(0, label="Max frames", precision=0, info="0 = whole video.")
                    release_steps = gr.Slider(0, 3, value=1, step=1, label="Free final steps",
                                              info="Final steps where the overlap is not held fixed, so seams can blend. 0 = fully fixed.")
                    long_info = gr.Markdown("Upload an RGB video.")
                    long_prompt, long_negative, long_seed, long_steps, long_cfg, long_resolution = sampling_controls("")
                    long_run = gr.Button("Generate", variant="primary")
                    long_status = gr.Markdown()
                with gr.Column(scale=1):
                    long_grid = gr.Video(label="All modalities (RGB | albedo / irradiance | normal)", autoplay=True, loop=True)
                    long_albedo = gr.Video(label="Albedo", loop=True)
                    with gr.Row():
                        long_irradiance = gr.Video(label="Irradiance", loop=True)
                        long_normal = gr.Video(label="Normal", loop=True)

        mode.change(update_mode, mode, [rgb_in, albedo_in, irradiance_in, normal_in, mode_info])
        run.click(
            generate,
            [mode, rgb_in, albedo_in, irradiance_in, normal_in, prompt, negative_prompt,
             seed, steps, cfg_scale, resolution] + model_inputs,
            [grid, rgb_out, albedo_out, irradiance_out, normal_out, status],
        )
        for trigger in (long_video.change, overlap.change, max_frames.change, long_resolution.change):
            trigger(describe_long, [long_video, overlap, max_frames, long_resolution], long_info)
        long_run.click(
            generate_long,
            [long_video, long_prompt, long_negative, long_seed, long_steps, long_cfg, long_resolution, overlap, max_frames, release_steps] + model_inputs,
            [long_grid, long_albedo, long_irradiance, long_normal, long_status],
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="models.yaml", help="Model file list for this machine (see models.example.yaml)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    MODELS, missing = load_models_config(args.models)
    for w in missing:
        print(f"[models.yaml] {w}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    build_ui(missing).queue(default_concurrency_limit=1).launch(
        server_port=args.port, show_error=True, allowed_paths=[os.path.abspath("results"), os.path.abspath("assets")])
