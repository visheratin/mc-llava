from typing import Tuple

import bitsandbytes as bnb
import lightning.pytorch as pl
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, get_cosine_schedule_with_warmup

from modeling_mc_llava import LlavaForCausalLM
from util import find_all_linear_names


class MCLLaVAModel(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        total_steps: int,
        warmup_steps: int,
        freeze_vision: bool,
        freeze_text: bool,
        use_lora: bool,
        bits: int,
    ):
        super().__init__()
        # self.automatic_optimization = False
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
        self.model.train()
        self.model.language_model.gradient_checkpointing_enable()
        if bits in [4, 8]:
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=True
            )
        if use_lora:
            lora_config = LoraConfig(
                r=64,
                lora_alpha=16,
                target_modules=find_all_linear_names(self.model),
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            for param in self.model.vision_model.parameters():
                param.requires_grad = True
            for param in self.model.multi_modal_projector.parameters():
                param.requires_grad = True
        self.freeze_vision = freeze_vision
        if freeze_vision:
            for param in self.model.vision_model.vision_tower.parameters():
                param.requires_grad = False
        self.freeze_text = freeze_text
        if freeze_text:
            for param in self.model.language_model.parameters():
                param.requires_grad = False
        self.learning_rate = lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def configure_optimizers(self):
        optimizer = bnb.optim.PagedAdamW8bit(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.warmup_steps, self.total_steps
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
        input_ids, attention_mask, labels, images, coords = batch
        image_features = []
        for i in range(images.size(0)):
            image_features.append(self.model.vision_model([images[i]], [coords[i]]))
        image_features = torch.cat(image_features)
        image_features = self.model.multi_modal_projector(image_features)
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
