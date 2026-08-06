import torch.nn as nn
import torch
from torch.nn import BCELoss


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.manual_dense = nn.Linear(config.feature_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj_new = nn.Linear(config.hidden_size + config.hidden_size, 1)

    def forward(self, features, manual_features=None, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])  [bs,hidden_size]
        y = manual_features.float()  # [bs, feature_size]
        y = self.manual_dense(y)
        y = torch.tanh(y)

        x = torch.cat((x, y), dim=-1)
        x = self.dropout(x)
        x = self.out_proj_new(x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args

    def forward(self, inputs_ids, attn_masks, manual_features=None,
                labels=None, output_attentions=None,
                return_attribution_data=False):
        outputs = self.encoder(
            input_ids=inputs_ids,
            attention_mask=attn_masks,
            output_attentions=output_attentions,
            return_dict=True,
        )

        attention_tensors = outputs.attentions if output_attentions else None
        last_layer_attn_weights = None
        if attention_tensors:
            last_layer_attn_weights = attention_tensors[-1][:, :, 0].detach()

        logits = self.classifier(outputs[0], manual_features)

        prob = torch.sigmoid(logits)
        loss = None
        if labels is not None:
            loss_fct = BCELoss()
            loss = loss_fct(prob, labels.unsqueeze(1).float())

        if return_attribution_data:
            return {
                "probability": prob,
                "logit": logits,
                "attentions": attention_tensors,
                "loss": loss,
            }

        if labels is not None:
            return loss, prob, last_layer_attn_weights
        return prob

