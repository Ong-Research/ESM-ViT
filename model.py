import torch
from torch import nn
from transformers import EsmPreTrainedModel, EsmModel
from transformers.modeling_outputs import SequenceClassifierOutput
from vit_pytorch import ViT
from typing import Optional, Union, Tuple
from safetensors.torch import load_model

class EsmClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*3, config.hidden_size*3)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size*3, config.num_labels)

    def forward(self, ESMFeatures, AtchleyFeatures, **kwargs):
        x = ESMFeatures[:, 0, :]
        combined_features = torch.cat((x, AtchleyFeatures), dim=1)
        x = self.dropout(combined_features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class EsmMMVit(EsmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        maxl = 36
        self.vit_model = ViT(
            image_size=maxl,
            dim=128,
            patch_size=9,
            num_classes=config.hidden_size,
            channels=5,
            heads=16,
            depth=6,
            mlp_dim=256,
            dropout=0.2,
            emb_dropout=0.2,
            pool='cls'
        ).to(device)
        
        self.num_labels = config.num_labels
        self.config = config

        self.esm1 = EsmModel(config, add_pooling_layer=False)
        self.esm2 = EsmModel(config, add_pooling_layer=False)
        
        self.classifier = EsmClassificationHead(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        is_local_path = 'facebook' not in pretrained_model_name_or_path

        if is_local_path:
            model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            load_model(model, f'{pretrained_model_name_or_path}/model.safetensors')
            return model
        else:
            config = kwargs.pop('config', None)
            
            if config is None:
                config = EsmModel.config_class.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            
            model = cls(config)
            
            pretrained_model = EsmModel.from_pretrained(pretrained_model_name_or_path, config=config)
            
            keys_to_remove = ["pooler.dense.weight", "pooler.dense.bias"]
            sd = pretrained_model.state_dict()
            for key in keys_to_remove:
                if key in sd:
                    del sd[key]
            
            model.esm1.load_state_dict(sd)
            model.esm2.load_state_dict(sd)

            return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids2: Optional[torch.LongTensor] = None,
        attention_mask2: Optional[torch.Tensor] = None,
        atchley_factors_seq: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs1 = self.esm1(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs2 = self.esm2(
            input_ids2,
            attention_mask=attention_mask2,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = torch.cat((outputs1[0], outputs2[0]), dim=2)
        vit_features = self.vit_model(atchley_factors_seq)

        logits = self.classifier(sequence_output, vit_features)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs1[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs1.hidden_states,
            attentions=outputs1.attentions,
        )