"""Why a result matches: shared attributes and an occlusion heatmap (scripts/exp_explain_heatmap.py).

Attributes compare the photo's predicted attributes with the item's precomputed ones. They are a
readable summary of what the two share, not the model's reasoning.

The heatmap is the model's reasoning: grey out one window of the photo at a time and measure how
much its similarity to the matched item drops. Regions whose removal hurts the match most are the
ones it relies on. Occlusion is slower than gradient or attention maps (one forward pass per
window, ~121 per map) but faithful by construction and model-agnostic.
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image

from silhouette_vision.enrich import LABELS, TRUSTED_ON_FARFETCH

COMPARED = ["article_type", "colour", "pattern", "sleeve_length", "neck"]
# Adjacent shades are the colour classifier's main error (Grey vs Black, Blue vs Navy Blue), so
# shades of one family count as a match, labelled "similar".
COLOUR_FAMILY = {"Navy Blue": "Blue", "Turquoise Blue": "Blue", "Teal": "Blue", "Steel": "Grey",
                 "Charcoal": "Grey", "Grey Melange": "Grey", "Silver": "Grey", "Off White": "White",
                 "Cream": "White", "Maroon": "Red", "Rust": "Red", "Magenta": "Pink", "Peach": "Pink",
                 "Lavender": "Purple", "Khaki": "Beige", "Tan": "Brown", "Bronze": "Brown",
                 "Copper": "Brown", "Mustard": "Yellow", "Gold": "Yellow", "Olive": "Green"}
SAME = {("No Sleeves", "Sleeveless")}


def _relation(name: str, a: str, b: str) -> str:
    if a == b or (a, b) in SAME or (b, a) in SAME:
        return "same"
    if name == "colour" and COLOUR_FAMILY.get(a, a) == COLOUR_FAMILY.get(b, b):
        return "similar"
    return "different"


def compare_attributes(query_attrs: list[dict], item_pred: pd.Series, source: str,
                       min_confidence: float = 0.5) -> list[dict]:
    """Confident attributes of the photo and the item side by side, with same/similar/different."""
    query = {a["attribute"]: a for a in query_attrs if a["confidence"] >= min_confidence}
    out = []
    for name in COMPARED:
        if name not in query or (source == "farfetch" and name not in TRUSTED_ON_FARFETCH):
            continue
        value = item_pred.get(f"pred_{name}")
        if not isinstance(value, str) or item_pred.get(f"conf_{name}", 0) < min_confidence:
            continue
        out.append({"label": LABELS[name], "photo": query[name]["value"], "item": value,
                    "relation": _relation(name, query[name]["value"], value)})
    return out


def occlusion_map(encoder, image: Image.Image, target: np.ndarray, grid: int = 12,
                  window: int = 2, batch: int = 64) -> np.ndarray:
    """(grid, grid) map of how much the similarity to `target` drops when each region is hidden.

    Windows of `window` x `window` cells slide one cell at a time; each cell's value is the mean
    drop over the windows covering it. Occluded pixels are set to 0 in normalised space (mid-grey
    for Marqo). Assumes a square-resize transform, so the map aligns with the original photo.
    """
    x = encoder.preprocess(image)
    size = x.shape[-1]
    edges = np.linspace(0, size, grid + 1).round().astype(int)
    positions = [(r, c) for r in range(grid - window + 1) for c in range(grid - window + 1)]
    variants = []
    for r, c in positions:
        v = x.clone()
        v[:, edges[r]:edges[r + window], edges[c]:edges[c + window]] = 0
        variants.append(v)
    target = target.astype(np.float32)
    base = float((encoder.embed_pixels(x[None]) @ target)[0])
    sims = np.concatenate([encoder.embed_pixels(torch.stack(variants[i:i + batch])) @ target
                           for i in range(0, len(variants), batch)])
    heat, count = np.zeros((grid, grid)), np.zeros((grid, grid))
    for (r, c), s in zip(positions, sims, strict=True):
        heat[r:r + window, c:c + window] += base - s
        count[r:r + window, c:c + window] += 1
    return heat / count


def patch_attribution(encoder, image: Image.Image, target: np.ndarray) -> np.ndarray:
    """(14, 14) map of each image patch's contribution to the similarity with `target`.

    Marqo's image embedding comes from attention pooling (a learned query attends over the 196
    patch tokens): before the pooling MLP it is exactly a sum of per-patch terms
    c_i = W_proj (a_i * v_i). Each patch is scored as c_i . d(similarity)/d(pooled), i.e. its
    share of the pooled vector weighted by how the similarity responds to it. One forward and one
    backward pass (~50x cheaper than occlusion); validated against occlusion on LookBench.
    """
    trunk = encoder.model.visual.trunk
    pool = trunk.attn_pool
    x = encoder.preprocess(image)[None].to(encoder.device)
    t = torch.as_tensor(target, dtype=torch.float32, device=encoder.device)
    with torch.no_grad():
        tokens = trunk.forward_features(x)[:, trunk.num_prefix_tokens:]
    if pool.pos_embed is not None:
        tokens = tokens + pool.pos_embed.unsqueeze(0)
    B, N, C = tokens.shape
    H, D = pool.num_heads, pool.head_dim
    with torch.enable_grad():
        q = pool.q(pool.latent.expand(B, -1, -1)).reshape(B, 1, H, D).transpose(1, 2)
        k, v = pool.kv(tokens).reshape(B, N, 2, H, D).permute(2, 0, 3, 1, 4).unbind(0)
        attn = ((pool.q_norm(q) * pool.scale) @ pool.k_norm(k).transpose(-2, -1)).softmax(-1)  # B,H,1,N
        per_patch = (attn.transpose(-2, -1) * v).permute(0, 2, 1, 3).reshape(B, N, C)  # a_i * v_i
        contrib = per_patch @ pool.proj.weight.T  # c_i, (B, N, C)
        pooled = (contrib.sum(1) + pool.proj.bias).detach().requires_grad_(True)
        z = pooled + pool.mlp(pool.norm(pooled))
        z = encoder.model.visual.head(trunk.head(trunk.fc_norm(z)))
        sim = torch.nn.functional.normalize(z, dim=-1) @ t
        (grad,) = torch.autograd.grad(sim.sum(), pooled)
    scores = (contrib.detach() * grad[:, None, :]).sum(-1)[0]
    side = round(N ** 0.5)
    return scores.reshape(side, side).float().cpu().numpy()


def input_gradient_map(encoder, image: Image.Image, target: np.ndarray) -> np.ndarray:
    """(14, 14) gradient x input on the patch embeddings (the network's input tokens).

    Unlike patch_attribution, credit flows back through all attention layers to the image patches
    themselves, so background "register" tokens that merely store global information don't light
    up. One forward + backward pass through the whole network.
    """
    trunk = encoder.model.visual.trunk
    x = encoder.preprocess(image)[None].to(encoder.device)
    t = torch.as_tensor(target, dtype=torch.float32, device=encoder.device)
    store = {}

    def keep(_module, _inputs, output):
        output.retain_grad()
        store["tokens"] = output

    handle = trunk.patch_embed.register_forward_hook(keep)
    try:
        with torch.enable_grad():
            z = encoder.model.encode_image(x)
            sim = torch.nn.functional.normalize(z, dim=-1)[0] @ t
            sim.backward()
    finally:
        handle.remove()
    tokens = store["tokens"]  # (1, N, C)
    scores = (tokens.grad * tokens.detach()).sum(-1)[0]
    side = round(scores.numel() ** 0.5)
    return scores.reshape(side, side).float().cpu().numpy()


def overlay(image: Image.Image, heat: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """The photo with regions that support the match in warm colours; the rest dimmed."""
    h = np.clip(heat, 0, None)
    h = h / h.max() if h.max() > 0 else h
    h = np.asarray(Image.fromarray((h * 255).astype(np.uint8)).resize(image.size, Image.BICUBIC),
                   dtype=np.float32) / 255
    # black -> red -> yellow
    colour = np.stack([np.clip(2 * h, 0, 1), np.clip(2 * h - 1, 0, 1), np.zeros_like(h)], -1)
    img = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    weight = alpha * h[..., None]
    out = img * (1 - weight) * (0.5 + 0.5 * h[..., None]) + colour * weight
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
