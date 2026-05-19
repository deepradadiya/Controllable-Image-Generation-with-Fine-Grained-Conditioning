"""
ControlNet Inference Pipeline
==============================

Full inference pipeline that takes a text prompt and condition image (depth, pose,
or edge map) and produces a generated image using DDIM sampling with classifier-free
guidance (CFG).

Data Flow During Inference:
    1. Encode text prompt → text_embedding (B, 77, 768)
    2. Encode empty string → unconditional_embedding (for CFG)
    3. Initialize random latent noise (B, 4, 64, 64)
    4. For each DDIM timestep:
       a. ControlNet: condition_image → multi-scale features
       b. UNet(latent + text_emb + ControlNet features) → conditional noise pred
       c. UNet(latent + empty_emb, no ControlNet) → unconditional noise pred
       d. CFG: noise_pred = uncond + guidance_scale * (cond - uncond)
       e. Scheduler step → denoised latent
    5. VAE decode final latent → pixel image (512x512 RGB)
"""

from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from model.controlnet import ControlNet


class ControlNetPipeline:
    """
    Full inference pipeline: text_prompt + condition_image → generated_image.

    Supports all 3 condition types (depth, pose, edge) via the condition_type argument.
    Uses classifier-free guidance (CFG) for high-quality generation.
    """

    VALID_CONDITION_TYPES = {"depth", "pose", "edge"}

    def __init__(
        self,
        controlnet: ControlNet,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        scheduler: DDIMScheduler,
    ):
        """
        Initialize the inference pipeline with all required model components.

        Args:
            controlnet: Trained ControlNet adapter that produces conditioning features.
            unet: Frozen SD1.5 UNet for noise prediction.
            vae: Frozen SD1.5 VAE for latent↔pixel conversion.
            text_encoder: Frozen CLIP text encoder for prompt encoding.
            tokenizer: CLIP tokenizer for text tokenization.
            scheduler: DDIM scheduler for the denoising loop.
        """
        self.controlnet = controlnet
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler

        # Determine device from UNet parameters
        self.device = next(unet.parameters()).device
        self.dtype = next(unet.parameters()).dtype

    @torch.no_grad()
    def __call__(
        self,
        text_prompt: str,
        condition_image: Image.Image,
        condition_type: str = "depth",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate an image conditioned on text and spatial control.

        Classifier-Free Guidance (CFG):
        ────────────────────────────────
        CFG improves generation quality by computing TWO noise predictions at each
        denoising step and interpolating between them:

          1. Conditional prediction: UNet receives the text embedding AND ControlNet
             features from the condition image. This predicts noise "guided" by both
             the text prompt and the spatial condition.

          2. Unconditional prediction: UNet receives an empty text embedding and NO
             ControlNet features. This predicts noise with no guidance at all.

          3. Final prediction = unconditional + guidance_scale * (conditional - unconditional)

        The guidance_scale controls how strongly the model follows the prompt/condition:
          - guidance_scale = 1.0 → no guidance (just conditional prediction)
          - guidance_scale = 7.5 → default, good balance of quality and diversity
          - guidance_scale > 10  → very strong adherence, may reduce diversity
          - guidance_scale = 20  → maximum adherence, can cause artifacts

        Higher guidance_scale = stronger adherence to prompt and condition image.

        Args:
            text_prompt: Text description of the desired image (max 77 tokens).
            condition_image: PIL Image of the spatial condition (depth/pose/edge).
                            Will be resized to 512x512 internally.
            condition_type: One of {"depth", "pose", "edge"}.
            guidance_scale: CFG scale factor (default 7.5, range 1.0-20.0).
            num_inference_steps: Number of DDIM denoising steps (default 20).
            seed: Optional random seed for reproducibility.

        Returns:
            PIL Image of size 512x512 in RGB mode.

        Raises:
            ValueError: If condition_type is not one of {"depth", "pose", "edge"}.
        """
        # ─────────────────────────────────────────────────────────────────────
        # Validate condition_type
        # ─────────────────────────────────────────────────────────────────────
        if condition_type not in self.VALID_CONDITION_TYPES:
            raise ValueError(
                f"Invalid condition_type '{condition_type}'. "
                f"Supported types are: {sorted(self.VALID_CONDITION_TYPES)}"
            )

        # ─────────────────────────────────────────────────────────────────────
        # 1. Prepare condition image tensor
        #    Resize to 512x512, normalize to [0, 1], shape (1, 3, 512, 512)
        # ─────────────────────────────────────────────────────────────────────
        condition_image_resized = condition_image.convert("RGB").resize(
            (512, 512), Image.LANCZOS
        )
        condition_tensor = torch.from_numpy(
            np.array(condition_image_resized).astype(np.float32) / 255.0
        )
        # (H, W, C) → (C, H, W) → (1, C, H, W)
        condition_tensor = condition_tensor.permute(2, 0, 1).unsqueeze(0)
        condition_tensor = condition_tensor.to(device=self.device, dtype=self.dtype)

        # ─────────────────────────────────────────────────────────────────────
        # 2. Encode text prompt with CLIP tokenizer + text_encoder (max 77 tokens)
        #    Also encode an empty string for the unconditional prediction in CFG.
        # ─────────────────────────────────────────────────────────────────────
        # Conditional text embedding (from the user's prompt)
        text_inputs = self.tokenizer(
            text_prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        text_embedding = self.text_encoder(text_input_ids)[0]  # (1, 77, 768)

        # Unconditional text embedding (empty string — for CFG)
        uncond_inputs = self.tokenizer(
            "",
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        uncond_input_ids = uncond_inputs.input_ids.to(self.device)
        uncond_embedding = self.text_encoder(uncond_input_ids)[0]  # (1, 77, 768)

        # ─────────────────────────────────────────────────────────────────────
        # 3. Set scheduler timesteps and initialize random latent noise
        #    Latent shape: (1, 4, 64, 64) — matches VAE latent space for 512x512
        # ─────────────────────────────────────────────────────────────────────
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Initialize random latent noise with optional seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        latents = torch.randn(
            (1, 4, 64, 64),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        # Scale initial noise by the scheduler's init_noise_sigma
        latents = latents * self.scheduler.init_noise_sigma

        # ─────────────────────────────────────────────────────────────────────
        # 4. DDIM Denoising Loop with Classifier-Free Guidance
        #
        #    At each timestep t:
        #      a. Run ControlNet with condition_image → get skip connection features
        #      b. Run UNet with text_embedding + ControlNet features → cond_pred
        #      c. Run UNet with empty_embedding (no ControlNet) → uncond_pred
        #      d. Apply CFG formula: pred = uncond + scale * (cond - uncond)
        #      e. Scheduler step to denoise the latent
        # ─────────────────────────────────────────────────────────────────────
        for t in timesteps:
            # Scale model input (required by some schedulers)
            latent_model_input = self.scheduler.scale_model_input(latents, t)

            # (a) Run ControlNet to get multi-scale conditioning features
            #     These features encode the spatial structure from the condition image
            controlnet_output = self.controlnet(
                noisy_latent=latent_model_input,
                timestep=t,
                text_embedding=text_embedding,
                condition_image=condition_tensor,
            )
            down_block_res_samples = controlnet_output["down_block_res_samples"]
            mid_block_res_sample = controlnet_output["mid_block_res_sample"]

            # (b) Conditional noise prediction: UNet with text + ControlNet features
            #     The ControlNet features are injected as additional skip connections
            cond_noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embedding,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            # (c) Unconditional noise prediction: UNet with empty text, NO ControlNet
            #     This gives us the "unguided" prediction for CFG interpolation
            uncond_noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=uncond_embedding,
            ).sample

            # (d) Apply Classifier-Free Guidance formula:
            #     noise_pred = unconditional + guidance_scale * (conditional - unconditional)
            #
            #     This amplifies the difference between guided and unguided predictions,
            #     pushing the generation toward the text prompt and condition image.
            noise_pred = uncond_noise_pred + guidance_scale * (
                cond_noise_pred - uncond_noise_pred
            )

            # (e) DDIM scheduler step: denoise the latent by one step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # ─────────────────────────────────────────────────────────────────────
        # 5. Decode final latent with VAE decoder to produce pixel image
        #    The VAE latent space uses a scaling factor of 0.18215 (from SD1.5).
        #    We divide by this factor before decoding to get proper pixel values.
        # ─────────────────────────────────────────────────────────────────────
        # Scale latents back from the VAE's latent space
        latents = latents / 0.18215

        # Decode latent → pixel image
        decoded = self.vae.decode(latents).sample  # (1, 3, 512, 512)

        # ─────────────────────────────────────────────────────────────────────
        # 6. Convert to PIL Image
        #    VAE output is in range [-1, 1]. We denormalize to [0, 255],
        #    clamp to valid range, convert to uint8, and create a PIL Image.
        # ─────────────────────────────────────────────────────────────────────
        # Denormalize from [-1, 1] to [0, 1]
        image = (decoded / 2 + 0.5).clamp(0, 1)

        # Convert to numpy: (1, 3, H, W) → (H, W, 3), scale to [0, 255]
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image[0] * 255).round().astype(np.uint8)

        # Create PIL Image in RGB mode
        pil_image = Image.fromarray(image, mode="RGB")

        return pil_image

    def save_with_overlay(
        self,
        generated: Image.Image,
        condition: Image.Image,
        condition_type: str,
        output_path: Optional[str] = None,
        output_dir: str = ".",
        index: Optional[int] = None,
    ) -> str:
        """
        Save a side-by-side composite image: condition | generated, with label.

        Creates a visual comparison PNG with the condition image on the left and
        the generated image on the right, both scaled to equal height. A text label
        indicating the condition type is rendered below the images without overlapping
        image content.

        If condition dimensions differ from generated, the condition image is resized
        to match the generated image height while preserving its aspect ratio.

        The output filename includes the condition type and a timestamp or sequential
        index to distinguish between multiple inference runs.

        Args:
            generated: The generated PIL Image (e.g., 512x512 RGB).
            condition: The condition PIL Image (depth/pose/edge map).
            condition_type: One of {"depth", "pose", "edge"} for labeling.
            output_path: Optional explicit file path. If provided, saves directly
                to this path. If None, generates a filename with condition_type
                and timestamp/index in output_dir.
            output_dir: Directory to save the composite when output_path is None.
            index: Optional sequential index for the filename. If None, uses
                a timestamp instead.

        Returns:
            The file path where the composite image was saved.
        """
        import os

        # Convert condition to RGB if needed
        condition_rgb = condition.convert("RGB")

        # Get generated image dimensions
        gen_width, gen_height = generated.size
        cond_width, cond_height = condition_rgb.size

        # Resize condition to match generated height, preserving aspect ratio (Req 11.4)
        if cond_height != gen_height:
            aspect_ratio = cond_width / cond_height
            new_cond_width = int(gen_height * aspect_ratio)
            cond_resized = condition_rgb.resize(
                (new_cond_width, gen_height), Image.LANCZOS
            )
        else:
            cond_resized = condition_rgb
            new_cond_width = cond_width

        # Label area below images (Req 11.2)
        label_height = 30

        # Composite resolution: condition_width + generated_width by shared height (Req 11.3)
        composite_width = new_cond_width + gen_width
        composite_height = gen_height + label_height

        # Create composite canvas
        composite = Image.new("RGB", (composite_width, composite_height), (255, 255, 255))
        composite.paste(cond_resized, (0, 0))
        composite.paste(generated, (new_cond_width, 0))

        # Add condition type label below images without overlapping (Req 11.2)
        try:
            from PIL import ImageDraw, ImageFont

            draw = ImageDraw.Draw(composite)
            # Try to use a default font; fall back to default if unavailable
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
                )
            except (OSError, IOError):
                font = ImageFont.load_default()

            label_text = f"Condition: {condition_type}"
            draw.text((10, gen_height + 5), label_text, fill=(0, 0, 0), font=font)

            gen_label = "Generated"
            draw.text(
                (new_cond_width + 10, gen_height + 5),
                gen_label,
                fill=(0, 0, 0),
                font=font,
            )
        except ImportError:
            # If PIL drawing is not available, save without labels
            pass

        # Determine output path with condition_type and timestamp/index (Req 11.5)
        if output_path is None:
            if index is not None:
                filename = f"{condition_type}_{index:04d}.png"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{condition_type}_{timestamp}.png"
            output_path = os.path.join(output_dir, filename)

        # Save as lossless PNG (Req 11.3)
        composite.save(output_path, format="PNG")

        return output_path
