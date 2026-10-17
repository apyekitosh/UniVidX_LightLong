import torch
from safetensors.torch import load_file


def merge_wan_lora(model: torch.nn.Module, path: str, strength: float = 1.0):
    """Merge a Kohya/ComfyUI-style Wan LoRA (lora_down/lora_up [+ alpha], diff, diff_b, diff_m) into the model weights in place."""
    lora = load_file(path)
    params = dict(model.named_parameters())
    merged, skipped = 0, []
    # Layers whose weight delta doesn't fit (e.g. I2V patch_embedding with 36 input channels):
    # their bias delta is skipped too, so a layer is never half-patched
    mismatched = {
        name.removeprefix("diffusion_model.")[:-len(".diff")]
        for name, value in lora.items()
        if name.endswith(".diff") and name.removeprefix("diffusion_model.")[:-len(".diff")] + ".weight" in params
        and params[name.removeprefix("diffusion_model.")[:-len(".diff")] + ".weight"].shape != value.shape
    }
    # peft-style naming (lora_A = down, lora_B = up) -> Kohya-style
    lora = {k.replace(".lora_A.", ".lora_down.").replace(".lora_B.", ".lora_up."): v for k, v in lora.items()}
    with torch.no_grad():
        for key, value in lora.items():
            name = key.removeprefix("diffusion_model.")
            if name.endswith(".lora_down.weight"):
                base = name[:-len(".lora_down.weight")]
                target = base + ".weight"
                up = lora[key.replace(".lora_down.", ".lora_up.")]
                delta = up.float().flatten(1) @ value.float().flatten(1)
                alpha = lora.get(key.replace(".lora_down.weight", ".alpha"))
                if alpha is not None:
                    delta *= alpha.item() / value.shape[0]
            elif name.endswith(".lora_up.weight") or name.endswith(".alpha"):
                continue
            elif name.endswith(".diff_b"):
                base = name[:-len(".diff_b")]
                target = base + ".bias"
                delta = value.float()
                if base in mismatched:
                    skipped.append(name)
                    continue
            elif name.endswith(".diff_m"):
                target = name[:-len(".diff_m")]
                delta = value.float()
            elif name.endswith(".diff"):
                base = name[:-len(".diff")]
                target = base if base in params else base + ".weight"
                delta = value.float()
            else:
                skipped.append(name)
                continue
            if target not in params or params[target].numel() != delta.numel():
                skipped.append(name)
                continue
            param = params[target]
            param.copy_((param.float() + strength * delta.view(param.shape).to(param.device)).to(param.dtype))
            merged += 1
    print(f"Merged LoRA {path} (strength {strength}): {merged} tensors, {len(skipped)} skipped")
    if skipped:
        print(f"  skipped (no matching layer): {skipped[:5]}{' ...' if len(skipped) > 5 else ''}")



class GeneralLoRALoader:
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        self.device = device
        self.torch_dtype = torch_dtype
    
    
    def get_name_dict(self, lora_state_dict):
        lora_name_dict = {}
        for key in lora_state_dict:
            if ".lora_B." not in key:
                continue
            keys = key.split(".")
            if len(keys) > keys.index("lora_B") + 2:
                keys.pop(keys.index("lora_B") + 1)
            keys.pop(keys.index("lora_B"))
            if keys[0] == "diffusion_model":
                keys.pop(0)
            keys.pop(-1)
            target_name = ".".join(keys)
            lora_name_dict[target_name] = (key, key.replace(".lora_B.", ".lora_A."))
        return lora_name_dict


    def load(self, model: torch.nn.Module, state_dict_lora, alpha=1.0):
        updated_num = 0
        lora_name_dict = self.get_name_dict(state_dict_lora)
        for name, module in model.named_modules():
            if name in lora_name_dict:
                weight_up = state_dict_lora[lora_name_dict[name][0]].to(device=self.device, dtype=self.torch_dtype)
                weight_down = state_dict_lora[lora_name_dict[name][1]].to(device=self.device, dtype=self.torch_dtype)
                if len(weight_up.shape) == 4:
                    weight_up = weight_up.squeeze(3).squeeze(2)
                    weight_down = weight_down.squeeze(3).squeeze(2)
                    weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                else:
                    weight_lora = alpha * torch.mm(weight_up, weight_down)
                state_dict = module.state_dict()
                state_dict["weight"] = state_dict["weight"].to(device=self.device, dtype=self.torch_dtype) + weight_lora
                module.load_state_dict(state_dict)
                updated_num += 1
        print(f"{updated_num} tensors are updated by LoRA.")
