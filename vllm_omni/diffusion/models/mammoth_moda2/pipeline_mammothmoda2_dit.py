from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.transformers_utils.config import get_config

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.transformers_utils.configs.mammoth_moda2 import Mammothmoda2Config

from .mammothmoda2_dit_model import SimpleQFormerImageRefiner, Transformer2DModel
from .rope_real import RotaryPosEmbedReal
from .schedulers import FlowMatchEulerDiscreteScheduler

logger = init_logger(__name__)


def get_mammoth_moda2_post_process_func(od_config: OmniDiffusionConfig):
    del od_config

    image_processor = VaeImageProcessor(vae_scale_factor=8)

    def post_process_func(
        images: torch.Tensor,
    ) -> list[Image.Image]:
        return image_processor.postprocess(images, output_type="pil")

    return post_process_func


class MammothModa2DiTPipeline(nn.Module, CFGParallelMixin, SupportsComponentDiscovery):
    """
    MammothModa2 DiT + VAE generation stage (non-autoregressive).

    This stage expects "image condition token hidden states" from the upstream AR stage,
    and outputs image tensors via diffusion transformer + VAE decode.

    """

    _dit_modules: ClassVar[list[str]] = ["gen_transformer"]
    _encoder_modules: ClassVar[list[str]] = ["gen_image_condition_refiner"]
    _vae_modules: ClassVar[list[str]] = ["gen_vae"]

    have_multimodal_outputs = True

    # Load only gen_* weights; ignore llm_model.* to prevent loading the entire LLM backbone in the DiT stage.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "llm_model.": None,
            "gen_tokenizer.": None,
        }
    )

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        del prefix

        self.od_config = od_config
        self.parallel_config = od_config.parallel_config

        hf_config = get_config(
            od_config.model,
            trust_remote_code=od_config.trust_remote_code,
            revision=od_config.revision,
        )
        if not isinstance(hf_config, Mammothmoda2Config):
            raise TypeError(f"Expected Mammothmoda2Config, got {type(hf_config)}")

        self.config = hf_config

        # --- Build DiT / VAE modules (names must match checkpoint keys) ---
        if self.config.gen_vae_config is None or self.config.gen_dit_config is None:
            raise ValueError("Mammothmoda2Config.gen_vae_config / gen_dit_config must not be None")

        self.gen_vae = AutoencoderKL.from_config(self.config.gen_vae_config)
        self.gen_transformer = Transformer2DModel.from_config(self.config.gen_dit_config)

        # llm_config is a Mammothmoda2Qwen2_5_VLConfig which has nested text_config
        llm_hidden_size = 0
        text_config = self.config.get_text_config()
        if text_config is None:
            logger.warning("No text config; failed to infer llm_hidden_size.")
        elif not hasattr(text_config, "hidden_size"):
            logger.warning("Text config exists, but has no hidden_size attribute; failed to infer llm_hidden_size.")
        else:
            llm_hidden_size = int(text_config.hidden_size or 0)
        if llm_hidden_size <= 0:
            raise ValueError(
                "Failed to infer llm hidden_size from Mammothmoda2Config.llm_config.text_config.hidden_size"
            )
        self._reinit_caption_embedder(llm_hidden_size)

        # Optional image condition Q-Former. Preview stores it as a standalone
        # module; Dev stores it under the DiT timestep/caption embedder.
        llm_model_type = getattr(self.config.llm_config, "model_type", "")
        refiner_config = self.config.gen_image_condition_refiner_config
        if refiner_config is not None and llm_model_type == "mammothmoda2_qwen3_vl":
            dit_hidden_size = int(self.gen_transformer.hidden_size)
            self.gen_transformer.time_caption_embed.image_embedder = SimpleQFormerImageRefiner(
                hidden_size=llm_hidden_size,
                output_hidden_size=dit_hidden_size,
                num_heads=max(1, dit_hidden_size // 128),
                **refiner_config,
            )
            self.gen_image_condition_refiner = None
        elif refiner_config is not None:
            self.gen_image_condition_refiner = SimpleQFormerImageRefiner(
                hidden_size=llm_hidden_size,
                **refiner_config,
            )
        else:
            self.gen_image_condition_refiner = None

        # Precompute rotary freqs for diffusion transformer
        # IMPORTANT: follow upstream mammothmoda: use top-level `config.gen_axes_*`
        # (the checkpoint's `gen_dit_config.axes_lens` can be as small as 1024,
        # which is insufficient for vLLM dummy-run/cudagraph warmup).
        self.gen_freqs_cis = RotaryPosEmbedReal.get_freqs_real(
            tuple(self.config.gen_axes_dim_rope),
            tuple(self.config.gen_axes_lens),
            theta=10000,
        )

        # vLLM PP interface compatibility
        self.make_empty_intermediate_tensors = lambda: None

        self._llm_hidden_size = llm_hidden_size

    def _reinit_caption_embedder(self, in_features: int) -> None:
        # Align with upstream Mammothmoda2Model's `reinit_caption_embedder`:
        # Use Qwen2RMSNorm(in_features) + Linear(in_features -> out_features).
        out_features = int(getattr(self.gen_transformer, "hidden_size", 0) or self.gen_transformer.config.hidden_size)
        self.gen_transformer.time_caption_embed.caption_embedder = nn.Sequential(
            Qwen2RMSNorm(in_features, eps=1e-5),
            nn.Linear(in_features, out_features, bias=True),
        )

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, object]]:
        if num_reqs <= 0:
            raise ValueError(f"num_reqs must be positive, got {num_reqs}")
        if num_reqs > 1:
            raise NotImplementedError(
                f"get_dummy_runtime_additional_information does not support num_reqs > 1, got {num_reqs}"
            )
        text_prompt_embeds = torch.zeros((1, self._llm_hidden_size), dtype=torch.float32)
        image_prompt_embeds = torch.zeros((1, self._llm_hidden_size), dtype=torch.float32)
        negative_prompt_embeds = torch.zeros((0, self._llm_hidden_size), dtype=torch.float32)
        info = {
            "text_prompt_embeds": text_prompt_embeds,
            "image_prompt_embeds": image_prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_attention_mask": [],
            "image_height": [512],
            "image_width": [512],
            "text_guidance_scale": [1.0],
            "cfg_range": [0.0, 1.0],
            "num_inference_steps": [1],
        }
        return [info for _ in range(num_reqs)]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # DiT stage does not consume token embeddings; return a dummy tensor.
        try:
            dtype = next(self.parameters()).dtype
        except StopIteration:
            dtype = torch.float32
        return torch.zeros(
            (input_ids.numel(), self._llm_hidden_size),
            device=input_ids.device,
            dtype=dtype,
        )

    def _split_ar_conditions(
        self,
        *,
        full_hidden_states: torch.Tensor,
        full_token_ids: list[int],
        answer_start_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split AR-stage hidden states into text / image condition embeds.

        The token ids that distinguish question (text) tokens, generated visual
        tokens, and multi-modal placeholder tokens are read from the model config
        (``gen_vocab_start_index`` and the vision placeholder token ids), so the
        caller no longer needs to pass them. Mirrors the masking the bespoke
        MammothModa2 example performed via ar2dit.
        """
        gen_vocab_start_index = int(self.config.llm_config.gen_vocab_start_index)
        visual_ids = [
            int(self.config.image_token_id),
            int(self.config.video_token_id),
            int(self.config.vision_start_token_id),
            int(self.config.vision_end_token_id),
        ]

        device = full_hidden_states.device
        token_ids = torch.tensor(full_token_ids, dtype=torch.long, device=device)
        positions = torch.arange(token_ids.shape[0], device=device)
        questions_mask = positions < answer_start_index
        answers_mask = ~questions_mask
        gen_token_mask = token_ids >= gen_vocab_start_index
        visual_token_mask = torch.isin(token_ids, torch.tensor(visual_ids, dtype=torch.long, device=device))
        text_mask = questions_mask & ~(visual_token_mask | gen_token_mask)
        image_mask = answers_mask & gen_token_mask

        text_cond = full_hidden_states[text_mask].to(dtype=torch.float32).contiguous()
        image_cond = full_hidden_states[image_mask].to(dtype=torch.float32).contiguous()
        return text_cond, image_cond

    def predict_noise(self, **kwargs: Any) -> torch.Tensor:
        """Run one MammothModa2 CFG branch."""
        return self.gen_transformer(**kwargs)

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if req.num_reqs != 1:
            raise ValueError("MammothModa2 only supports one request at a time")
        first_prompt = req.prompts[0]
        if not isinstance(first_prompt, dict):
            raise TypeError("MammothModa2 requires a dictionary prompt")

        info = first_prompt.get("extra", {})
        sampling_params = req.sampling_params
        extra_args = sampling_params.extra_args or {}

        # Sampling knobs are declared in vllm_omni/model_extras/mammothmodal2_preview.py
        # and routed via extra_body -> sampling_params.extra_args. Standard
        # diffusion knobs use their dedicated sampling-parameter fields.
        text_guidance_scale = float(extra_args.get("text_guidance_scale", 9.0))
        cfg_range_val = extra_args.get("cfg_range", [0.0, 1.0])
        cfg_range = float(cfg_range_val[0]), float(cfg_range_val[1])
        num_inference_steps = int(
            sampling_params.num_inference_steps
            or extra_args.get("num_inference_steps")
            or 50
        )

        negative_cond = info.get("negative_prompt_embeds")
        negative_attention_mask = info.get("negative_prompt_attention_mask")
        image_hw = (
            first_prompt.get("height") or sampling_params.height or 1024,
            first_prompt.get("width") or sampling_params.width or 1024,
        )

        # Split the AR hidden states into text / image conditions. The token ids that
        # drive the split are sourced from the model config (see _split_ar_conditions),
        # formerly supplied by the bespoke example via additional_information. Legacy
        # fallback: ar2dit may have already produced the split conditions.
        if "text_prompt_embeds" in info:
            text_cond = info["text_prompt_embeds"]
            image_cond = info["image_prompt_embeds"]
        else:
            text_cond, image_cond = self._split_ar_conditions(
                full_hidden_states=info["full_hidden_states"],
                full_token_ids=info["full_token_ids"],
                answer_start_index=int(info["answer_start_index"]),
            )

        # Move to model device/dtype.
        model_device = next(self.parameters()).device
        if self.gen_image_condition_refiner is not None:
            target_dtype = next(self.gen_image_condition_refiner.parameters()).dtype
        else:
            target_dtype = next(self.gen_transformer.parameters()).dtype

        def _ensure_2d(x: torch.Tensor, name: str) -> torch.Tensor:
            if x.ndim == 3 and x.shape[0] == 1:
                x = x[0]
            if x.ndim != 2:
                raise ValueError(f"Expected {name} to be 2D [T,H], got shape={tuple(x.shape)}")
            return x

        text_cond = _ensure_2d(text_cond, "text_prompt_embeds")
        image_cond = _ensure_2d(image_cond, "image_prompt_embeds")
        if image_cond.shape[0] == 0:
            answer_token_ids = info.get("full_token_ids", [])[int(info.get("answer_start_index", 0)) :]
            raise ValueError(
                "MammothModa2 AR stage produced no visual-token hidden states; "
                "the DiT stage requires at least one generated visual token. "
                f"Generated token ids: {answer_token_ids[:32]}"
            )
        text_cond = text_cond.to(device=model_device, dtype=target_dtype, non_blocking=True).contiguous()
        image_cond = image_cond.to(device=model_device, dtype=target_dtype, non_blocking=True).contiguous()

        text_embeds = text_cond.unsqueeze(0)  # [1, T_text, H]
        text_attention_mask = torch.ones(
            (1, text_embeds.shape[1]),
            dtype=torch.bool,
            device=text_embeds.device,
        )

        image_embeds = image_cond.unsqueeze(0)  # [1, T_img, H]
        image_attention_mask = torch.ones(
            (1, image_embeds.shape[1]),
            dtype=torch.bool,
            device=image_embeds.device,
        )

        # Apply optional refiner ONLY on image condition tokens.
        if self.gen_image_condition_refiner is not None and image_embeds.shape[1] > 0:
            image_embeds = self.gen_image_condition_refiner(image_embeds, ~image_attention_mask.bool())
            image_attention_mask = torch.ones(
                image_embeds.shape[:2],
                dtype=torch.bool,
                device=image_embeds.device,
            )

        nested_image_embedder = getattr(self.gen_transformer.time_caption_embed, "image_embedder", None)
        if nested_image_embedder is None:
            prompt_embeds = torch.cat([text_embeds, image_embeds], dim=1)
            prompt_attention_mask = torch.cat([text_attention_mask, image_attention_mask], dim=1)
            ar_image_embeds = None
            ar_image_attention_mask = None
        else:
            prompt_embeds = text_embeds
            prompt_attention_mask = text_attention_mask
            ar_image_embeds = image_embeds
            ar_image_attention_mask = image_attention_mask

        # Prepare negative prompt (for CFG). If none provided, fall back to unconditional.
        negative_prompt_embeds = None
        negative_prompt_attention_mask = None
        if text_guidance_scale > 1.0:
            if negative_cond is not None:
                negative_cond = _ensure_2d(negative_cond, "negative_prompt_embeds")
                negative_prompt_embeds = (
                    negative_cond.to(device=model_device, dtype=target_dtype, non_blocking=True)
                    .contiguous()
                    .unsqueeze(0)
                )
                if isinstance(negative_attention_mask, torch.Tensor):
                    neg_mask = negative_attention_mask
                elif isinstance(negative_attention_mask, list):
                    neg_mask = torch.tensor(negative_attention_mask, dtype=torch.bool)
                else:
                    neg_mask = None
                if neg_mask is None:
                    negative_prompt_attention_mask = torch.ones(
                        (1, negative_prompt_embeds.shape[1]),
                        dtype=torch.bool,
                        device=negative_prompt_embeds.device,
                    )
                else:
                    neg_mask = neg_mask.to(device=negative_prompt_embeds.device, dtype=torch.bool)
                    if neg_mask.ndim == 1:
                        neg_mask = neg_mask.unsqueeze(0)
                    negative_prompt_attention_mask = neg_mask
            else:
                hidden_size = int(prompt_embeds.shape[-1])
                negative_prompt_embeds = torch.zeros(
                    (1, 0, hidden_size),
                    dtype=target_dtype,
                    device=prompt_embeds.device,
                )
                negative_prompt_attention_mask = torch.zeros(
                    (1, 0),
                    dtype=torch.bool,
                    device=prompt_embeds.device,
                )

        # Output image size (px), passed from stage input processor.
        height, width = image_hw
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid image size: {height}x{width}")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"Image size must be multiples of 16, got {height}x{width}")
        vae_scale_factor = 16

        latent_channels = int(self.gen_transformer.config.in_channels)
        shape = (1, latent_channels, 2 * height // vae_scale_factor, 2 * width // vae_scale_factor)
        latents = randn_tensor(shape, device=prompt_embeds.device, dtype=prompt_embeds.dtype)

        scheduler = FlowMatchEulerDiscreteScheduler()

        scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            device=prompt_embeds.device,
            num_tokens=latents.shape[-2] * latents.shape[-1],
        )

        # Run diffusion loop (CFG supported when text_guidance_scale > 1.0)
        total_steps = max(1, len(scheduler.timesteps))
        for i, t in enumerate(scheduler.timesteps):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            positive_kwargs = {
                "hidden_states": latents,
                "timestep": timestep,
                "text_hidden_states": prompt_embeds,
                "text_attention_mask": prompt_attention_mask,
                "ref_image_hidden_states": None,
                "ar_image_hidden_states": ar_image_embeds,
                "ar_image_attention_mask": ar_image_attention_mask,
                "freqs_cis": self.gen_freqs_cis,
            }
            guidance_scale = text_guidance_scale if cfg_range[0] <= i / total_steps <= cfg_range[1] else 1.0
            do_true_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

            negative_kwargs = None
            if do_true_cfg:
                negative_kwargs = {
                    "hidden_states": latents,
                    "timestep": timestep,
                    "text_hidden_states": negative_prompt_embeds,
                    "text_attention_mask": negative_prompt_attention_mask,
                    "ref_image_hidden_states": None,
                    "freqs_cis": self.gen_freqs_cis,
                }

            model_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg=do_true_cfg,
                true_cfg_scale=guidance_scale,
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                cfg_normalize=False,
            )
            latents = scheduler.step(model_pred, t, latents, return_dict=False)[0]
            latents = latents.to(dtype=prompt_embeds.dtype)

        # VAE decode
        if self.gen_vae.config.scaling_factor is not None:
            latents = latents / self.gen_vae.config.scaling_factor
        if self.gen_vae.config.shift_factor is not None:
            latents = latents + self.gen_vae.config.shift_factor
        image = self.gen_vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(output=image)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:  # noqa: ARG002
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
