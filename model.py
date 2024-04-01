from typing import Tuple

import bitsandbytes as bnb
import lightning.pytorch as pl
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
)

from modeling_mc_llava import LlavaConfig, LlavaForCausalLM, LlavaMultiModalProjector
from util import find_all_linear_names
from wsd_scheduler import WSDParameters, wsd_scheduler


class MCLLaVAModel(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        scheduler_params: WSDParameters,
        freeze_vision: bool,
        freeze_text: bool,
        use_lora: bool,
        bits: int,
    ):
        super().__init__()
        quant_args = {}
        if bits in [4, 8]:
            quant_args.update(
                dict(
                    load_in_4bit=bits == 4,
                    load_in_8bit=bits == 8,
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=bits == 4,
                        load_in_8bit=bits == 8,
                        llm_int8_skip_modules=["multi_modal_projector", "vision_model"],
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4",
                    ),
                )
            )
        self.model = LlavaForCausalLM.from_pretrained(
            "visheratin/MC-LLaVA-3b",
            **quant_args,
        )
        self.reset_model()
        self.model.train()
        self.model.language_model.gradient_checkpointing_enable()
        self.model.vision_model.vision_tower.gradient_checkpointing_enable()
        if bits in [4, 8]:
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=True
            )
        if use_lora:
            lora_config = LoraConfig(
                r=64,
                lora_alpha=32,
                target_modules=find_all_linear_names(self.model.language_model),
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model.language_model = get_peft_model(self.model.language_model, lora_config)
        if freeze_vision:
            for param in self.model.vision_model.vision_tower.parameters():
                param.requires_grad = False
        if freeze_text:
            for param in self.model.language_model.parameters():
                param.requires_grad = False
        self.learning_rate = lr
        self.scheduler_params = scheduler_params

    def configure_optimizers(self):
        optimizer = bnb.optim.PagedAdamW8bit(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
        )
        scheduler = wsd_scheduler(
            optimizer,
            self.scheduler_params,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def training_step(
        self,
        batch: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        batch_idx: int,
    ):
        input_ids, attention_mask, labels, pixel_values, coords = batch
        original_shape = pixel_values.shape
        pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])
        coords = coords.view(-1, *coords.shape[2:])
        image_features = self.model.vision_model(pixel_values, coords)
        image_features = self.model.multi_modal_projector(image_features)
        image_features = image_features.view(*original_shape[:2], -1)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            image_features=image_features,
            return_dict=True,
        )
        loss = outputs["loss"]
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            batch_size=input_ids.shape[0],
            sync_dist=True,
        )
        self.log(
            "train/epoch",
            self.global_step / self.trainer.max_steps,
            prog_bar=False,
            batch_size=input_ids.shape[0],
            sync_dist=True,
        )
        if self.lr_schedulers() is not None:
            lr = self.lr_schedulers().get_last_lr()[0]
            self.log(
                "train/learning_rate",
                lr,
                prog_bar=False,
                batch_size=input_ids.shape[0],
                sync_dist=True,
            )
        return {"loss": loss}

    def reset_model(self):
        config = LlavaConfig.from_pretrained("visheratin/MC-LLaVA-3b")
        # self.model.language_model = AutoModelForCausalLM.from_pretrained(
        #     "vince62s/phi-2-psy", trust_remote_code=True
        # )
        # self.model.vision_model.vision_tower = SiglipVisionModel.from_pretrained(
        #     "google/siglip-so400m-patch14-384"
        # )
        self.model.vision_model.coord_embed = torch.nn.Sequential(
            torch.nn.Linear(4, config.vision_embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(config.vision_embed_dim, config.vision_embed_dim),
        )
        self.model.multi_modal_projector = LlavaMultiModalProjector(config)
