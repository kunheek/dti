from packaging import version

import PIL.Image

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

DIFFUSERS_MODELS = {
    "sd1.5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd2.1_base": "stabilityai/stable-diffusion-2-1-base",
    "sd2.1": "stabilityai/stable-diffusion-2-1",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd3.5_medium": "stabilityai/stable-diffusion-3.5-medium",
    "sd3.5_large": "stabilityai/stable-diffusion-3.5-large",
    "sd3.5_large_turbo": "stabilityai/stable-diffusion-3.5-large-turbo",
    "flux1dev": "black-forest-labs/FLUX.1-dev",
    "flux1schnell": "black-forest-labs/FLUX.1-schnell",
    "sana1.5_1.6b": "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
    "sana1.5_4.8b": "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
}

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]
