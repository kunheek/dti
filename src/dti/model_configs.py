"""Model-specific configurations."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SDXLConfig:
    """SDXL model configuration."""

    pretrained_model_name_or_path: str
    revision: Optional[str] = None
    variant: Optional[str] = None

    def load_models(self):
        """Load all SDXL models."""
        from transformers import (
            CLIPTokenizer,
            CLIPTextModel,
            CLIPTextModelWithProjection,
        )
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

        tokenizer_1 = CLIPTokenizer.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="tokenizer_2"
        )
        text_encoder_1 = CLIPTextModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.revision,
        )
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            revision=self.revision,
        )
        vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.revision,
            variant=self.variant,
        )
        unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.revision,
            variant=self.variant,
        )
        noise_scheduler = DDPMScheduler.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="scheduler"
        )

        return {
            "tokenizer_1": tokenizer_1,
            "tokenizer_2": tokenizer_2,
            "text_encoder_1": text_encoder_1,
            "text_encoder_2": text_encoder_2,
            "vae": vae,
            "unet": unet,
            "noise_scheduler": noise_scheduler,
        }


@dataclass
class SanaConfig:
    """Sana model configuration."""

    pretrained_model_name_or_path: str
    revision: Optional[str] = None
    variant: Optional[str] = None

    def load_models(self):
        """Load all SDXL models."""
        from transformers import (
            AutoTokenizer,
            Gemma2Model,
        )
        from diffusers import (
            AutoencoderDC,
            FlowMatchEulerDiscreteScheduler,
            SanaTransformer2DModel,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        text_encoder = Gemma2Model.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.revision,
        )
        vae = AutoencoderDC.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.revision,
            variant=self.variant,
        )
        transformer = SanaTransformer2DModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="transformer",
            revision=self.revision,
            variant=self.variant,
        )
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="scheduler"
        )

        return {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "vae": vae,
            "transformer": transformer,
            "scheduler": noise_scheduler,
        }
