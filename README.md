# UniVidX: 24 GB, distilled and long-video fork

This is a community fork of **[UniVidX](https://github.com/houyuanchen111/UniVidX)** by Houyuan Chen, Hong Li, Xianghao Kong, Tianrui Zhu, Yuwei Guo, Weiqing Xiao, Chongjie Ye, Lvmin Zhang, Hao Zhao and Anyi Rao ([paper](https://arxiv.org/pdf/2605.00658), [project page](https://houyuanchen111.github.io/UniVidX.github.io/), [weights](https://huggingface.co/houyuanchen/UniVidX)). The model, the method and the trained weights are entirely theirs; please cite their paper (see [Citation](#-citation) below) if you use this.

This fork only adds **inference tooling** for **UniVid-Intrinsic** (RGB ↔ albedo / irradiance / normal), so you can run it on a single consumer GPU, much faster, and on videos longer than 21 frames. It was developed and tested on Windows 11 with an RTX A5000 (24 GB) and 128 GB of RAM.

## What this fork adds

- **Runs on 24 GB GPUs.** The 14B DiT and the text encoder stay in system RAM and their layers are streamed to the GPU as needed (`vram_limit`). The original code needs the full model in VRAM.
- **Single-file checkpoints.** Load one-file DiTs, such as the ComfyUI repackaged models, instead of the six official shards. The Wan2.1 T2V 14B and Wan2.2 **low-noise** experts (T2V or I2V) can be used as the base model.
- **Step-distill LoRAs** (lightx2v, Wan2.2 Lightning, FastWan): 4 steps with no CFG instead of 50 steps with CFG. That's about **2 minutes per clip instead of about 55**.
- **SVI LoRA stacking.** A second LoRA merged on top of the distill LoRA. Stable Video Infinity improves consistency between long-video chunks.
- **Long-video R2AIN (sliding window).** Decompose an RGB video of any length into albedo, irradiance and normals in overlapping 21-frame chunks that reuse the previous chunk's latents (details below).
- **Optional SageAttention.** About 2× faster attention, about 1.34× faster overall, with no visible quality change.
- **Gradio app** (`app.py`): every mode, long video, model and LoRA selection, and a resolution override, with outputs saved as separate videos, a 2×2 grid and a `settings.json`.
- **Resolution override:** the trained 640×480, the input video's size, or any longest side, keeping the aspect ratio.
- **`models.yaml`:** one file listing your model paths, so the same code runs on machines with different folder layouts.
- **Compatibility fixes** for current PyTorch and Windows: video I/O through imageio (torchvision ≥ 0.22 dropped it) and UTF-8 config reading.

The original CLI scripts and configs still work as upstream intended.

## Results (RTX A5000, 640×480, 21-frame clip)

| Setup | Time per clip |
|---|---|
| Original: 50 steps, CFG 5 | ~55 min |
| Distill LoRA: 4 steps, CFG 1 | ~2 min 40 s |
| Distill LoRA + SageAttention | ~2 min |

Long video: about 2 minutes per chunk (12 new frames each). The first model load takes about 5 minutes, and the app keeps the model loaded between runs.

Seam smoothness for long video, measured as the jump between chunks relative to the model's normal frame-to-frame flicker (lower is better, 1.0 = invisible):

| Base model + LoRAs | Seam jump |
|---|---|
| Wan2.1 T2V + lightx2v T2V | 1.95× |
| Wan2.1 T2V + Lightning I2V | 2.08× |
| Wan2.2 I2V low-noise + Lightning I2V | 3.09× |
| Wan2.2 T2V low-noise + Lightning I2V | 1.35× |
| **Wan2.2 T2V low-noise + Lightning I2V + SVI v2 PRO** | **1.28×** (default) |

## Setup

Python 3.10 and a CUDA GPU. Example with [uv](https://docs.astral.sh/uv/) (plain `pip` in a venv works too):

```bash
git clone https://github.com/apyekitosh/UniVidX.git
cd UniVidX
uv venv .venv --python 3.10
uv pip install --python .venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -r requirement.txt
```

**Optional: SageAttention.** On Windows it isn't on PyPI. Install the wheel from [woct0rdho/SageAttention](https://github.com/woct0rdho/SageAttention/releases) that matches your torch and CUDA versions, plus `triton-windows`. For torch 2.11 + CUDA 12.8:

```bash
uv pip install --python .venv/Scripts/python.exe "triton-windows<3.8" "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/sageattention-2.2.0%2Bcu128torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl"
```

**System RAM:** loading briefly needs a lot of RAM, because the DiT is built in full precision before converting to bf16. This was tested with 128 GB; 64 GB is probably the practical minimum, but that's untested.

### Models

| File | Where to get it |
|---|---|
| Base DiT (recommended): `wan2.2_t2v_low_noise_14B_fp16.safetensors` | [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models) |
| or Wan2.1: `wan2.1_t2v_14B_bf16.safetensors` | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models) |
| UniVidX: `univid_intrinsic.safetensors` | [houyuanchen/UniVidX](https://huggingface.co/houyuanchen/UniVidX) |
| Text encoder: `models_t5_umt5-xxl-enc-bf16.pth` + tokenizer folder `google/umt5-xxl` | [Wan-AI/Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) (ComfyUI's fp8 text encoder won't load) |
| VAE: `wan_2.1_vae.safetensors` (or the official `Wan2.1_VAE.pth`) | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/vae) |
| Distill LoRA: `Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors` | [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy/tree/main/LoRAs/Wan22-Lightning/old) (Wan2.2-Lightning by [lightx2v](https://huggingface.co/lightx2v/Wan2.2-Lightning)) |
| SVI LoRA: `SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors` | [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy/tree/main/LoRAs/Stable-Video-Infinity/v2.0) (Stable Video Infinity by [vita-video-gen](https://huggingface.co/vita-video-gen/svi-model)) |

The text encoder and tokenizer can be fetched in one go:

```bash
.venv/Scripts/hf.exe download Wan-AI/Wan2.1-T2V-14B models_t5_umt5-xxl-enc-bf16.pth google/umt5-xxl/special_tokens_map.json google/umt5-xxl/spiece.model google/umt5-xxl/tokenizer.json google/umt5-xxl/tokenizer_config.json --local-dir ./models/Wan-AI/Wan2.1-T2V-14B
```

Then copy `models.example.yaml` to `models.yaml` and point it at your files. `models.yaml` is git-ignored, so each machine keeps its own.

```yaml
diffusion_models:          # lists become dropdowns; the first entry is the default
  - D:\models\wan2.2_t2v_low_noise_14B_fp16.safetensors
unividx_models:
  - D:\models\univid_intrinsic.safetensors
distill_loras:
  - D:\loras\Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors
svi_loras:
  - D:\loras\SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors
text_encoder: D:\models\models_t5_umt5-xxl-enc-bf16.pth
tokenizer: D:\models\google\umt5-xxl
vae: D:\models\wan_2.1_vae.safetensors
vram_limit: 16             # GB of VRAM for streamed weights; 16 suits a 24 GB card
```

## Using the app

```bash
.venv/Scripts/python.exe app.py
```

Open http://localhost:7860. The first generation loads the model, which takes a few minutes; later runs reuse it.

**Model** (changing any of these reloads the model):
- **Base DiT, UniVidX checkpoint, distill LoRA, SVI LoRA** come from `models.yaml`. LoRA strength 1.0 is the intended value.
- **VRAM limit (GB):** budget for weights kept on the GPU; the rest is streamed from RAM. Lower it if you run out of memory, raise it for a little more speed. 0 loads everything onto the GPU, which needs more than 40 GB.
- **SageAttention:** on when installed. It switches per run and never triggers a reload.

**Single window:** one 21-frame clip (about 1.4 s at 15 fps) in any of the 15 modes. The mode name says inputs → outputs: `R2AIN` = RGB in, albedo + irradiance + normal out; `t2RAIN` = text only. The matching input video slots appear when you pick a mode.

**Long video (R2AIN):** an RGB video of any length.
- **Overlap (latent frames):** 2 gives smoother seams, 12 new frames per chunk; 1 is faster, 16 new frames per chunk.
- **Free final steps:** final denoising steps in which the overlap is not held fixed, so the model can blend it with the new frames. 1 is a good default; 0 holds it fixed for every step and gives more visible seams.
- **Max frames:** 0 processes the whole video.

**Sampling (both tabs):**
- **Steps / CFG:** 4 and 1.0 with a distill LoRA; about 50 and 5.0 without one.
- **Prompt:** optional for video-in modes (leave it empty, which also disables CFG). Chinese prompts are recommended by the authors.
- **Resolution:** 0 = trained 640×480; 1 = input video size; any other number = longest side in pixels, keeping the input's aspect ratio. Sizes are snapped to multiples of 16 and the input is stretched, never cropped. Larger sizes get much slower, and the app shows an estimate.

## How the long-video mode works

UniVidX denoises the four modalities as four latent videos side by side. Inputs (for example RGB in `R2AIN`) are held clean, and the outputs attend to them through cross-modal attention. For a long video:

1. The video is split into 21-frame windows (6 latent frames: one single-frame latent + five four-frame latents).
2. Each new window's RGB slot is the real footage for that window.
3. For albedo, irradiance and normal, the head of each new window is the **previous window's last 1–2 generated latents, reused directly** (no decode/re-encode round trip, which would cause color drift), preceded by one re-encoded **anchor frame**, so the first latent is the single-frame latent the VAE expects.
4. During sampling the head is re-noised to the current noise level after every step, so the model continues from it. The last step is optionally left free so seams can blend.
5. When decoding, the anchor frame is dropped and the overlapping frames are crossfaded with the previous window.

## Notes and limitations

- **Only UniVid-Intrinsic is covered.** UniVid-Alpha works through the original CLI but not in the app, and the new options (distill, SVI, Sage) were only added to the intrinsic pipeline.
- **Tested on Windows only.**
- **Long video was evaluated on a short sample clip.** Very long or fast-moving footage may show more drift.
- **Text-to-video (`t2RAIN`) runs, but long video is only for R2AIN.**
- Using Wan2.2 bases and I2V LoRAs with UniVidX is an off-label combination. It worked well in testing, but the authors trained UniVidX on Wan2.1 T2V 14B.

## Changes from upstream

Per the Apache-2.0 license, these files were modified or added in this fork:

- **Added:** `app.py`, `models.example.yaml`, `src/pipelines/long_video.py`, `src/video_io.py`
- **Modified:**
  - `src/pipelines/univid_intrinsic.py`: single-file DiT loading, I2V input-layer slicing, CPU offload, LoRA merging, configurable tokenizer path, long-video hooks, attention mode
  - `src/pipelines/univid_alpha.py`: single-file DiT loading, CPU offload, LoRA merging, configurable tokenizer path
  - `src/models/wan_video_dit_intrinsic.py`: SageAttention mode
  - `src/lora/__init__.py`: Kohya/ComfyUI/peft-style Wan LoRA merging
  - `scripts/inference_univid_*.py`, `scripts/train.py`, `src/trainers/unividx_trainer.py`: imageio video I/O and UTF-8 config reading
  - `requirement.txt`, `.gitignore`

Everything else is unchanged from upstream and remains under the original [Apache-2.0 license](LICENSE).

---

*The original UniVidX README follows.*

## ___***UniVidX: A Unified Multimodal Framework for Versatile Video Generation via Diffusion Priors***___

<div align="center">
    <a href='https://github.com/houyuanchen111' target='_blank'>Houyuan Chen</a><sup>1</sup>&emsp;
    <a href='https://github.com/Luh1124' target='_blank'>Hong Li</a><sup>2</sup>&emsp;       
    <a href='https://refkxh.github.io/' target='_blank'>Xianghao Kong</a><sup>1</sup>&emsp;   
    <a href='https://github.com/trzhu11' target='_blank'>Tianrui Zhu</a><sup>3</sup>&emsp;
    <a href='https://guoyww.github.io/' target='_blank'>Yuwei Guo</a><sup>4</sup>&emsp;
    <a href='https://github.com/zhuxing0' target='_blank'>Weiqing Xiao</a><sup>3</sup><br>
    <a href='https://hugoycj.github.io/' target='_blank'>Chongjie Ye</a><sup>5</sup>&emsp;
    <a href='https://lllyasviel.github.io/lvmin_zhang/' target='_blank'>Lvmin Zhang</a><sup>6</sup>&emsp;
    <a href='https://sites.google.com/view/fromandto' target='_blank'>Hao Zhao</a><sup>7</sup>&emsp;
    <a href='https://anyirao.com/' target='_blank'>Anyi Rao</a><sup>1,*</sup>
</div>

<div>
<div align="center">
    <sup>1</sup>MMLab, HKUST, &emsp;
    <sup>2</sup>BUAA&emsp;
    <sup>3</sup>NJU&emsp;
    <sup>4</sup>CUHK&emsp;
    <sup>5</sup>FNii, CUHKSZ&emsp;
    <sup>6</sup>Stanford&emsp;
    <sup>7</sup>AIR,THU<br>
</div>
</div>


<p align="center">
  <a href="https://arxiv.org/pdf/2605.00658">
    <img src="https://img.shields.io/badge/arXiv-PDF-b31b1b" alt="Paper">
  </a>
  <a href="https://houyuanchen111.github.io/UniVidX.github.io/">
    <img src="https://raw.githubusercontent.com/prs-eth/Marigold/main/doc/badges/badge-website.svg" alt="Website">
  </a>
  <a href="https://huggingface.co/houyuanchen/UniVidX">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face%20-Model-green" alt="Hugging Face Model">
  </a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0">
    <img src="https://img.shields.io/badge/License-Apache--2.0-929292" alt="License">
  </a>
</p>


<div align="center">
  <img src="assets/teaser.png" width="800px" alt="UniVidX Teaser">
  <br>
</div>


## 📖 Overview

We introduce ***UniVidX***, a unified multimodal video diffusion framework that transcends the boundaries of task-specific models. 
By incorporating Stochastic Condition Masking (SCM), Decoupled Gated LoRA (DGL), and Cross-Modal Self-Attention (CMSA), a single model can achieve versatile video generation and perception. Whether applied to Intrinsic tasks (**UniVid-Intrinsic**) or Alpha channel processing (**UniVid-Alpha**), our approach achieves outstanding performance with remarkable data efficiency (<1k training videos).

---

## 🚀 News
- **[2026/05/04]** Initial release of **UniVidX**.
---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/houyuanchen111/UniVidX.git
cd UniVidX

# Create environment
conda create -n unividx python=3.10
conda activate unividx

# Install dependencies
pip install -r requirements.txt
```

## 🤖 Model Zoo

You can download the weights of backbone Wan2.1-T2V-14B from either **ModelScope** or **Hugging Face**.

**Option 1: ModelScope**
```bash
pip install modelscope
mkdir -p ./checkpoints/Wan-AI
modelscope download Wan-AI/Wan2.1-T2V-14B --local_dir ./checkpoints/Wan-AI/Wan2.1-T2V-14B
```

**Option 2: Hugging Face**
```bash
pip install "huggingface_hub[cli]"
mkdir -p ./models/Wan-AI
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./models/Wan-AI/Wan2.1-T2V-14B
```

Then, download checkpoints of **UniVid-Intrinsic** and **UniVid-Alpha** manually from Hugging Face or let the scripts auto-download them.

| Model Name | Link |
| :--- | :--- |
| UniVid-Intrinsic | [🤗 Download](https://huggingface.co/houyuanchen/UniVidX) |
| UniVid-Alpha | [🤗 Download](https://huggingface.co/houyuanchen/UniVidX) |

---

## 💻 Inference
We use YAML files to centrally manage inference parameters. Below are the configuration templates for **UniVid-Intrinsic** and **UniVid-Alpha**.

#### UniVid-Intrinsic

```yaml
# configs/univid_intrinsic_inference.yaml

experiment_name: "univid_intrinsic_inference"   # Output folder name
mode: "t2RAIN"                                   # Task Mode (One of the 15 supported tasks)

# --- Conditional Inputs ---
# Configure paths based on your chosen 'mode'. Set unused inputs to null.
inference_rgb_path: null
inference_albedo_path: null
inference_irradiance_path: null
inference_normal_path: null

# --- Text Prompt ---
# We recommend using Chinese prompts.
prompt: "一只小刺猬，穿着白色小围裙，头上戴着厨师帽，正站在小凳子上，双手举着一个小平底锅，锅里冒着热气，表情专注而自豪，位于一个现代化的迷你厨房中，不锈钢台面反射着明亮的光线，各种小厨具整齐地排列着, 镜头从右向左移动。"

# --- Model Settings ---
model:
  name: 'UniVidIntrinsic' 
  params:
    # Path to Wan2.1 Backbone
    model_paths: '["models/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth", "models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth"]'
    resume_from_checkpoint: "checkpoints/univid_intrinsic.safetensors"
    
    # LoRA Configuration
    lora_base_model: "dit"
    lora_target_modules: "self_attn.q,self_attn.k,self_attn.v,self_attn.o,ffn.0,ffn.2"
    lora_rank: 32
    lora_modalities: ["rgb", "albedo", "irradiance", "normal"]
```

#### UniVid-Alpha

```yaml
# configs/univid_alpha_inference.yaml

experiment_name: "univid_alpha_inference"   # Output folder name
mode: "R2PFB"                                # Task Mode (One of the 15 supported tasks)


# --- Conditional Inputs ---
# Configure paths based on your chosen 'mode'. Set unused inputs to null.
inference_rgb_path: "./assets/R2PFB/bl.mp4"
inference_pha_path: null
inference_fgr_path: null
inference_bgr_path: null

# --- Text Prompt ---
prompt: ""

# --- Model Settings ---
model:
  name: 'UniVidAlpha' 
  params:
    # Path to Wan2.1 Backbone
    model_paths: '["models/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth", "models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth"]'
    resume_from_checkpoint: "checkpoints/univid_alpha.safetensors"
    
    # LoRA Configuration
    lora_base_model: "dit"
    lora_target_modules: "self_attn.q,self_attn.k,self_attn.v,self_attn.o,ffn.0,ffn.2"
    lora_rank: 32
    lora_modalities: ["com", "pha", "fgr", "bgr"]
```

Once your YAML configuration is ready, run the corresponding inference script:
```bash
# univid_alpha_inference
python scripts/inference_univid_alpha.py --config configs/univid_alpha_inference.yaml

# univid_intrinsic_inference
python scripts/inference_univid_intrinsic.py --config configs/univid_intrinsic_inference.yaml
```

Below are the 15 tasks (modes) supported by **UniVid-Intrinsic** and **UniVid-Alpha**, along with their corresponding inputs and outputs:

| Task Category | UniVid-Intrinsic | UniVid-Alpha |
| :--- | :--- | :--- |
| **Text $\to$ X** | `t2RAIN` | `t2RPFB` |
| **X $\to$ X** | `R2AIN`, `RA2IN`, `RI2AN`, `RN2AI`, `RIN2A`, `RAN2I`, `RAI2N`, `AIN2R` | `R2PFB`, `RP2FB`, `RF2PB`, `RB2PF`, `FB2RP`, `PFB2R`, `RFB2P`, `RPB2F`, `RPF2B` |
| **Text & X $\to$ X** | `A2RIN`, `I2RAN`, `N2RAI`, `AI2RN`, `AN2RI`, `IN2RA` | `P2RFB`, `F2RPB`, `B2RPF`, `PF2RB`, `PB2RF` |


Different tasks can be combined to enable interesting applications. For example:
### 1. `t2RAIN` $\to$ `IN2RA` (Prompt-driven Video Editing)
<table>
  <tr>
    <td style="border: none; vertical-align: top;">
      <table>
        <tr>
          <td><img src="assets/video_editing/rgb_gen.gif" width="100%"></td>
          <td><img src="assets/video_editing/albedo_gen.gif" width="100%"></td>
        </tr>
        <tr>
          <td><img src="assets/video_editing/irradiance_gen.gif" width="100%"></td>
          <td><img src="assets/video_editing/normal_gen.gif" width="100%"></td>
        </tr>
        <tr>
          <td colspan="2" style="text-align: center; font-weight: bold;">
             "一个开放式的现代极简厨房..."<br>t2RAIN
          </td>
        </tr>
      </table>
    </td>
    <td style="border: none; vertical-align: middle; font-size: 50px; padding: 0 20px;">
      &rarr;
    </td>
    <td style="border: none; vertical-align: top;">
      <table>
        <tr>
          <td><img src="assets/video_editing/rgb_edit.gif" width="100%"></td>
        </tr>
        <tr>
          <td><img src="assets/video_editing/albedo_edit.gif" width="100%"></td>
        </tr>
        <tr>
          <td style="text-align: center; font-weight: bold;">
            "橙子放在不锈钢台上..."<br>IN2RA
          </td>
        </tr>
      </table>
    </td>
  </tr>
</table>

### 2. `R2PFB` $\to$ `PB2RF` (Prompt-driven Video Inpainting)
<table>
  <tr>
    <td style="border: none; vertical-align: top;">
      <table>
        <tr>
          <td><img src="assets/video_inpainting/input_bl.gif" width="100%"></td>
          <td><img src="assets/video_inpainting/alpha.gif" width="100%"></td>
        </tr>
        <tr>
          <td><img src="assets/video_inpainting/fg.gif" width="100%"></td>
          <td><img src="assets/video_inpainting/bg.gif" width="100%"></td>
        </tr>
        <tr>
          <td colspan="2" style="text-align: center; font-weight: bold;">
             ""<br>R2PFB
          </td>
        </tr>
      </table>
    </td>
    <td style="border: none; vertical-align: middle; font-size: 50px; padding: 0 20px;">
      &rarr;
    </td>
    <td style="border: none; vertical-align: top;">
      <table>
        <tr>
          <td><img src="assets/video_inpainting/inpaint_bl.gif" width="100%"></td>
        </tr>
        <tr>
          <td><img src="assets/video_inpainting/inpaint_fg.gif" width="100%"></td>
        </tr>
        <tr>
          <td style="text-align: center; font-weight: bold;">
            "带墨镜，穿粉色西装..."<br>PB2RF
          </td>
        </tr>
      </table>
    </td>
  </tr>
</table>

More applications await your exploration.

---

## 🏋️ Training
We utilize `accelerate` for distributed training. The training configurations are provided in `configs/univid_intrinsic_train.yaml` and `configs/univid_alpha_train.yaml`.

Run the following commands to start training:

```bash
# 1. Train UniVid-Intrinsic
accelerate launch --config_file "configs/accelerate_config.yaml" \
    "scripts/train.py" \
    --config "configs/univid_intrinsic_train.yaml"

# 2. Train UniVid-Alpha
accelerate launch --config_file "configs/accelerate_config.yaml" \
    "scripts/train.py" \
    --config "configs/univid_alpha_train.yaml"
```


## 📊 Citation

If you find this work useful, please cite:

```bibtex
@article{chen2026unividx,
  title     = {UniVidX: A Unified Multimodal Framework for Versatile Video Generation via Diffusion Priors},
  author    = {Chen, Houyuan and Li, Hong and Kong, Xianghao and Zhu, Tianrui and Xu, Shaocong and Xiao, Weiqing and Guo, Yuwei and Ye, Chongjie and Zhang, Lvmin and Zhao, Hao and Rao, Anyi},
  journal   = {ACM Transactions on Graphics},
  volume    = {45},
  number    = {4},
  articleno = {51},
  year      = {2026},
  month     = jul,
  doi       = {10.1145/3811304},
  url       = {https://doi.org/10.1145/3811304}
}
```

## 📝 Acknowledgements

Code is built on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio). Thanks all the authors for their excellent contributions!

## 📜 License

This project is released under the [Apache License 2.0](LICENSE).