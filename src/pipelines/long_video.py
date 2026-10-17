"""Chunked R2AIN for RGB videos longer than one 21-frame window.

Each window is 21 frames = 6 latent frames (1 single-frame latent + 5 four-frame latents).
Chunk c >= 1 reuses the previous chunk's generated latents instead of re-encoding pixels,
which avoids the VAE round trip that causes color drift:

  head latents = [encode(anchor frame)] + [last `overlap` latents of the previous chunk]

The anchor is the decoded frame just before the reused latents, encoded on its own so latent 0
is the single-frame latent the VAE expects. The head of each generated slot (albedo, irradiance,
normal) is held fixed during sampling by re-noising it to the current noise level after every
step. The RGB slot is simply the real source frames of the window. When decoding, the anchor
frame is dropped and the overlap frames are crossfaded with the previous chunk.
"""
import math

import imageio
import numpy as np
import torch
import torch.nn.functional as F

WINDOW_FRAMES = 21
WINDOW_LATENTS = 6
GENERATED = ["albedo", "irradiance", "normal"]  # slots 1, 2, 3 in R2AIN
OUTPUT_KEYS = {"albedo": "albedo", "irradiance": "irradiance", "normal": "normal_unit"}


def load_video(path, height, width, max_frames=0):
    """All frames of an mp4 as a uint8 tensor [T,3,H,W] resized to (height, width), plus its fps."""
    reader = imageio.get_reader(path)
    fps = reader.get_meta_data().get("fps", 15)
    frames = []
    for frame in reader:
        frame = torch.from_numpy(np.asarray(frame)[..., :3]).permute(2, 0, 1)[None].float()
        if frame.shape[-2:] != (height, width):
            frame = F.interpolate(frame, size=(height, width), mode="bilinear", align_corners=False)
        frames.append(frame[0].round().clamp(0, 255).to(torch.uint8))
        if max_frames and len(frames) >= max_frames:
            break
    reader.close()
    if not frames:
        raise ValueError(f"Empty video: {path}")
    return torch.stack(frames), fps


def chunk_plan(num_frames, overlap):
    """Window start frames covering num_frames; each window after the first advances by 20 - 4*overlap."""
    step = WINDOW_FRAMES - 1 - 4 * overlap
    if num_frames <= WINDOW_FRAMES:
        return [0]
    return [0] + [i * step for i in range(1, math.ceil((num_frames - WINDOW_FRAMES) / step) + 1)]


def window(frames, start):
    """21 frames from start as [1,3,T,H,W] in [-1,1], padding past the end with the last frame."""
    clip = frames[start:start + WINDOW_FRAMES]
    if clip.shape[0] < WINDOW_FRAMES:
        clip = torch.cat([clip, clip[-1:].repeat(WINDOW_FRAMES - clip.shape[0], 1, 1, 1)])
    return (clip.float() / 127.5 - 1.0).permute(1, 0, 2, 3)[None]


def run_long_r2ain(pipe, video_path, base_params, overlap=2, max_frames=0, release_steps=1, progress=None):
    """Returns ({modality: [3,T,H,W] float16 in [-1,1] on CPU}, fps, rgb frames as [3,T,H,W] in [-1,1])."""
    assert overlap in (1, 2), "overlap is 1 or 2 latent frames"
    height, width = base_params["height"], base_params["width"]
    frames, fps = load_video(video_path, height, width, max_frames)
    total = frames.shape[0]
    starts = chunk_plan(total, overlap)
    num_head = 1 + overlap
    anchor_idx = WINDOW_FRAMES - 1 - 4 * overlap  # local index of the frame just before the reused latents
    blend = 4 * overlap  # overlap frames to crossfade

    outputs = {m: None for m in GENERATED}
    prev_latents, prev_decoded = None, None
    for c, start in enumerate(starts):
        if progress is not None:
            progress(c / len(starts), desc=f"Chunk {c + 1}/{len(starts)} (frames {start}-{min(start + WINDOW_FRAMES, total) - 1})")
        params = dict(base_params)
        params.update(training_mode="R2AIN", return_latents=True, seed=base_params["seed"] + c,
                      inference_rgb=window(frames, start).to(device=pipe.device, dtype=pipe.torch_dtype))
        if prev_latents is not None:
            # Anchor: the previous chunk's decoded frame right before the reused latents, encoded alone
            pipe.load_models_to_device(["vae"])
            anchors = [prev_decoded[m][:, anchor_idx:anchor_idx + 1].clamp(-1, 1).to(pipe.torch_dtype) for m in GENERATED]
            anchor_latents = pipe.vae.encode(anchors, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            tail = prev_latents[[1, 2, 3], :, WINDOW_LATENTS - overlap:]
            params.update(fixed_head_latents=torch.cat([anchor_latents, tail], dim=2), fixed_head_slots=[1, 2, 3],
                          head_release_steps=release_steps)
            assert params["fixed_head_latents"].shape[2] == num_head

        with torch.no_grad():
            video_dict = pipe(**params)
        prev_latents = video_dict["latents"]
        prev_decoded = {m: video_dict[OUTPUT_KEYS[m]].detach() for m in GENERATED}

        for m in GENERATED:
            chunk = prev_decoded[m].float().cpu()  # [3,21,H,W]
            if outputs[m] is None:
                outputs[m] = chunk.half()
                continue
            # Drop the anchor, crossfade the overlap with the previous chunk's tail, append the rest
            new_overlap = chunk[:, 1:1 + blend]
            old_overlap = outputs[m][:, -blend:].float()
            w = torch.arange(1, blend + 1, dtype=torch.float32).view(1, -1, 1, 1) / (blend + 1)
            mixed = (1 - w) * old_overlap + w * new_overlap
            outputs[m] = torch.cat([outputs[m][:, :-blend], mixed.half(), chunk[:, 1 + blend:].half()], dim=1)

    for m in GENERATED:
        outputs[m] = outputs[m][:, :total]
        if m == "normal":
            # Re-normalize after blending so normals stay unit length
            n = outputs[m].float()
            outputs[m] = (n / (n.norm(dim=0, keepdim=True) + 1e-6)).half()
    rgb = (frames.float() / 127.5 - 1.0).permute(1, 0, 2, 3)
    return outputs, fps, rgb
