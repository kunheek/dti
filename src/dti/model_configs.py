"""Model-specific configurations."""

from dataclasses import dataclass


@dataclass
class SDXLConfig:
    """SDXL model configuration."""

    pretrained_model_name_or_path: str
    revision: str | None = None
    variant: str | None = None

    def load_models(self):
        """Load all SDXL models."""
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import (
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPTokenizer,
        )

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
    revision: str | None = None
    variant: str | None = None

    def load_models(self):
        """Load all SDXL models."""
        from diffusers import (
            AutoencoderDC,
            FlowMatchEulerDiscreteScheduler,
            SanaTransformer2DModel,
        )
        from transformers import (
            AutoTokenizer,
            Gemma2Model,
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


@dataclass
class Flux2KleinConfig:
    """FLUX.2-klein model configuration."""

    pretrained_model_name_or_path: str
    revision: str | None = None
    variant: str | None = None

    def load_models(self):
        """Load FLUX.2-klein components."""
        try:
            from diffusers import (
                AutoencoderKLFlux2,
                FlowMatchEulerDiscreteScheduler,
                Flux2Transformer2DModel,
            )
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise ImportError(
                "FLUX.2-klein support requires newer `diffusers`/`transformers` "
                "versions. Please upgrade dependencies and retry."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=self.revision,
        )
        text_encoder = AutoModelForCausalLM.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.revision,
        )
        vae = AutoencoderKLFlux2.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.revision,
            variant=self.variant,
        )
        transformer = Flux2Transformer2DModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="transformer",
            revision=self.revision,
            variant=self.variant,
        )
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="scheduler",
        )

        return {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "vae": vae,
            "transformer": transformer,
            "scheduler": noise_scheduler,
        }


@dataclass
class WanConfig:
    """Wan text-to-video model configuration."""

    pretrained_model_name_or_path: str
    revision: str | None = None
    variant: str | None = None

    def load_models(self):
        """Load Wan model components."""
        from transformers import T5TokenizerFast, UMT5EncoderModel
        from diffusers import (
            AutoencoderKLWan,
            FlowMatchEulerDiscreteScheduler,
            WanTransformer3DModel,
        )

        # Wan2.1 diffusers repo provides a T5 tokenizer and UMT5 encoder.
        # Use explicit classes instead of AutoTokenizer to avoid AutoConfig
        # fallbacks that can trigger "Unrecognized model" on some environments.
        tokenizer = T5TokenizerFast.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=self.revision,
        )
        try:
            text_encoder = UMT5EncoderModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder",
                revision=self.revision,
            )
        except Exception as exc:
            raise ImportError(
                "Failed to load Wan text encoder (UMT5). "
                "Please upgrade `transformers` to a recent version "
                "(e.g., >=4.48) and retry."
            ) from exc
        vae = AutoencoderKLWan.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.revision,
            variant=self.variant,
        )
        transformer = WanTransformer3DModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="transformer",
            revision=self.revision,
            variant=self.variant,
        )
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="scheduler",
        )

        return {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "vae": vae,
            "transformer": transformer,
            "scheduler": noise_scheduler,
        }
