from dataclasses import dataclass, field
from typing import List, Union

import torch


@dataclass
class Conversation:
    messages: List[str]
    roles: List[str] = field(default_factory=lambda: ["user", "assistant"])

    def encode(self, role: str, message: Union[str, None], tokenizer):
        if message is None:
            text = f"""<|im_start|>{role}
"""
        else:
            text = f"""<|im_start|>{role}
{message}<|im_end|>"""
        input_ids = tokenizer(text, return_tensors="pt").input_ids.squeeze(0)
        return text, input_ids

    def get_prompt(self, tokenizer, ignore_index=-100):
        prompt = ""
        prompt_ids = torch.tensor([]).long()
        prompt_labels = torch.tensor([]).long()
        for i, message in enumerate(self.messages):
            role = self.roles[i % 2]
            text, input_ids = self.encode(
                role=role, message=message, tokenizer=tokenizer
            )
            _, empty_ids = self.encode(role=role, message=None, tokenizer=tokenizer)
            prompt += text
            prompt_ids = torch.cat([prompt_ids, input_ids])
            if role == self.roles[0]:
                labels = torch.full_like(input_ids, ignore_index)
            else:
                labels = input_ids.clone()
                labels[: empty_ids.shape[0]] = ignore_index
            prompt_labels = torch.cat([prompt_labels, labels])
        return prompt, prompt_ids, prompt_labels
