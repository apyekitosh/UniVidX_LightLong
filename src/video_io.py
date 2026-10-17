"""Video read/write helpers for the app (imageio-ffmpeg; torchvision>=0.22 dropped its video I/O)."""
import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F


def load_video_clip(path, *, num_frames, height, width, device, dtype):
    """First num_frames of an mp4, stretched to (height, width), as [1,3,T,H,W] in [-1,1].

    Shorter videos are padded by repeating the last frame.
    """
    video = torch.from_numpy(np.stack(imageio.mimread(path, memtest=False)))
    if video.numel() == 0:
        raise ValueError(f"Empty video: {path}")
    if video.ndim != 4 or video.shape[-1] < 3:
        raise ValueError(f"Unexpected video shape {tuple(video.shape)} for {path}; expected [T,H,W,3]")

    video = video[..., :3].to(torch.float32) / 255.0
    video = video.permute(0, 3, 1, 2)  # T,H,W,3 -> T,3,H,W
    if video.shape[-2:] != (height, width):
        video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)

    if video.shape[0] >= num_frames:
        video = video[:num_frames]
    else:
        video = torch.cat([video, video[-1:].repeat(num_frames - video.shape[0], 1, 1, 1)], dim=0)

    video = video.permute(1, 0, 2, 3).contiguous() * 2.0 - 1.0  # 3,T,H,W in [-1,1]
    return video.unsqueeze(0).to(device=device, dtype=dtype)


def save_video(tensor, path, fps=15):
    """Writes a [3,T,H,W] tensor in [0,1] as an h264 mp4."""
    assert tensor.ndim == 4, "tensor must be [c,t,h,w]"
    frames = (tensor.detach().cpu().permute(1, 2, 3, 0).clamp(0, 1) * 255).to(torch.uint8).numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, codec="libx264")
