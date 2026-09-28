"""Embedding models behind one interface.

Every encoder returns L2-normalized float32 numpy arrays, so cosine similarity is a dot product.
"""

from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(Protocol):
    name: str
    dim: int
    has_text: bool
    transform: object  # picklable image -> tensor transform, safe to send to DataLoader workers

    def preprocess(self, image: Image.Image) -> torch.Tensor: ...
    def embed_pixels(self, pixels: torch.Tensor) -> np.ndarray: ...
    def embed_texts(self, texts: list[str]) -> np.ndarray: ...


class _Base:
    device: str

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        return self.embed_pixels(torch.stack([self.preprocess(im) for im in images]))


class MarqoFashionSigLIP(_Base):
    """Marqo-FashionSigLIP (ViT-B/16 SigLIP): shared image-text space, 768-d."""

    name = "marqo_fashion_siglip"
    dim = 768
    has_text = True
    hub_id = "hf-hub:Marqo/marqo-fashionSigLIP"

    def __init__(self, device: str | None = None):
        import open_clip

        self.device = device or default_device()
        model, _, self.transform = open_clip.create_model_and_transforms(self.hub_id)
        self.model = model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(self.hub_id)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)

    @torch.inference_mode()
    def embed_pixels(self, pixels: torch.Tensor) -> np.ndarray:
        emb = self.model.encode_image(pixels.to(self.device), normalize=True)
        return emb.float().cpu().numpy()

    @torch.inference_mode()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(texts).to(self.device)
        return self.model.encode_text(tokens, normalize=True).float().cpu().numpy()


class GRLite(_Base):
    """GR-Lite (DINOv3 ViT-L/16 fine-tuned for fashion retrieval): image-only, 1024-d.

    The model's own trust_remote_code loader fails on transformers 5, and its reference code
    omits DINOv3's RoPE. The weights load one-to-one into transformers' DINOv3ViTModel, so we
    use that class; `rope=False` reproduces the reference code (identity rotation).
    """

    dim = 1024
    has_text = False
    hub_id = "srpone/gr-lite"
    image_size = 336

    def __init__(self, device: str | None = None, rope: bool = True):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from torchvision import transforms as T
        from transformers import DINOv3ViTConfig, DINOv3ViTModel

        self.device = device or default_device()
        self.rope = rope
        self.name = "gr_lite" if rope else "gr_lite_norope"
        config = DINOv3ViTConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_register_tokens=4,
            image_size=self.image_size,
            patch_size=16,
            key_bias=False,
            layer_norm_eps=1e-6,
        )
        model = DINOv3ViTModel(config)
        state = load_file(hf_hub_download(self.hub_id, "model.safetensors"))
        state = {f"model.{k}" if k.startswith("layer.") else k: v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        if not rope:
            rope_module = model.rope_embeddings
            original = rope_module.forward

            def identity_rope(pixel_values):
                cos, sin = original(pixel_values)
                return torch.ones_like(cos), torch.zeros_like(sin)

            rope_module.forward = identity_rope
        self.model = model.to(self.device).eval()
        self.transform = T.Compose(
            [
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)

    @torch.inference_mode()
    def embed_pixels(self, pixels: torch.Tensor) -> np.ndarray:
        out = self.model(pixel_values=pixels.to(self.device))
        return F.normalize(out.pooler_output, dim=-1).float().cpu().numpy()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError("GR-Lite is image-only; use MarqoFashionSigLIP for text.")


ENCODERS = {
    "marqo_fashion_siglip": MarqoFashionSigLIP,
    "gr_lite": GRLite,
    "gr_lite_norope": lambda device=None: GRLite(device, rope=False),
}


def load_encoder(name: str, device: str | None = None) -> Encoder:
    return ENCODERS[name](device=device)
