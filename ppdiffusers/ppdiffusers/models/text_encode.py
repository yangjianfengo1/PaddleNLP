from paddlenlp.transformers.clip.modeling import CLIPTextTransformer, CLIPTextModel
from paddlenlp.transformers.clip.configuration import CLIPConfig, CLIPTextConfig
from paddlenlp.transformers.model_outputs import BaseModelOutputWithPooling, ModelOutput
import paddle
import paddle.nn.functional as F
from paddle import nn
from typing import Any, Optional, Tuple, Union


class LoraCLIPTextTransformer(CLIPTextTransformer):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)

    def load_lora_weight(self, weight: dict):
        self.lora_weight = weight

    def update_lora_weight(self, lora_idx_list: list, lora_alpha_list: list):
        for name, op in self.transformer.named_sublayers():
            if name in self.lora_weight:
                new_weight = op.weight
                for idx, alppha in zip(lora_idx_list, lora_alpha_list):
                    new_weight += alppha * self.lora_weight[name][idx].T
                op.weight.set_value(new_weight)      

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        lora_idx_list: list = None,
        lora_alpha_list: list = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        self.update_lora_weight(lora_idx_list, lora_alpha_list)
        return super().forward(
            input_ids,
            attention_mask,
            position_ids,
            output_attentions,
            output_hidden_states,
            return_dict,
            lora_idx_list,
            lora_alpha_list)
        
class LoraCLIPTextModel(CLIPTextModel):
    def __init__(self, config: CLIPTextConfig):
        print("##### LoraCLIPTextModel")
        super().__init__(config)
        self.text_model = LoraCLIPTextTransformer(config)