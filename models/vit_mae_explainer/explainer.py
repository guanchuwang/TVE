import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers.utils import ModelOutput
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers import ViTMAEPreTrainedModel
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import ipdb
from typing import Optional, Set, Tuple, Union


@dataclass
class ExplanationOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None

@dataclass
class HeatmapOutput(ModelOutput):

    heatmap: torch.FloatTensor = None
    heatmap_pooling: torch.FloatTensor = None
    heatmap_visual: torch.FloatTensor = None


class ExplainerHeadLayer(nn.Module):
    def __init__(self, seq_len_in, hidden_size_in, seq_len_out, hidden_size_out, layernorm=False, act=None, shortcut=False):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size_in, hidden_size_out)
        self.dense2 = nn.Linear(seq_len_in, seq_len_out)
        self.act1 = ACT2FN[act] if act is not None else nn.Identity(hidden_size_out)
        self.act2 = ACT2FN[act] if act is not None else nn.Identity(seq_len_out)
        self.layernorm = nn.LayerNorm(hidden_size_in, eps=1e-12) if layernorm else nn.Identity(hidden_size_in)
        self.shortcut = shortcut

    def forward(self, x): # x: [Batchsize, seq_len_in, hidden_size_in]

        x_in = x
        x = self.layernorm(x)
        x = self.dense1(x) # x: [Batchsize, seq_len_in, hidden_size_out]
        x = self.act1(x)
        x = x.transpose(1, 2) # x: [Batchsize, hidden_size_out, seq_len_in]
        x = self.dense2(x) # x: [Batchsize, hidden_size_out, seq_len_out]
        x = self.act2(x)
        x = x.transpose(1, 2) # x: [Batchsize, seq_len_out, hidden_size_out]
        if self.shortcut:
            x += x_in

        return x


class ExplainerHead(nn.Module):
    def __init__(self, layer_num, seq_len_in, hidden_size_in, seq_len_out, hidden_size_out):
        super().__init__()

        self.net = nn.Sequential()
        for layer_idx in range(layer_num):
            if layer_idx == 0:
                self.net.add_module("Input Layer",
                    ExplainerHeadLayer(seq_len_in, hidden_size_in, seq_len_out, hidden_size_out, layernorm=True, act=None, shortcut=False),
                )
            elif layer_idx == layer_num - 1:
                self.net.add_module("Output Layer",
                    ExplainerHeadLayer(seq_len_out, hidden_size_out, seq_len_out, hidden_size_out, layernorm=True, act=None, shortcut=False),
                )
            else:
                self.net.add_module("Intermedia Layer {}".format(layer_idx),
                    ExplainerHeadLayer(seq_len_out, hidden_size_out, seq_len_out, hidden_size_out, layernorm=True, act="gelu", shortcut=True),
                )

    def forward(self, x): # x: [Batchsize, seq_len_in, hidden_size_in]
        return self.net(x) # x: [Batchsize, seq_len_out, hidden_size_out]


class GenericExplainer(ViTMAEPreTrainedModel):

    def __init__(self, config, backbone_pos=None, backbone_neg=None, target_encoder=None):
        super().__init__(config)
        self.config = config
        self.backbone_pos = backbone_pos
        self.explainer_head_pos = ExplainerHead(
            layer_num=config.explainerhead_layernum,
            seq_len_in=(backbone_pos.config.image_size//backbone_pos.config.patch_size)**2 + 1, # 197
            hidden_size_in=backbone_pos.config.decoder_hidden_size, # 768
            seq_len_out=config.heatmap_size**2 + 1, # 197
            hidden_size_out=self.config.target_hidden_size # 512/1024/2048
        )

        self.backbone_neg = backbone_neg
        self.explainer_head_neg = ExplainerHead(
            layer_num=config.explainerhead_layernum,
            seq_len_in=(backbone_neg.config.image_size // backbone_neg.config.patch_size) ** 2 + 1,  # 197
            hidden_size_in=backbone_neg.config.decoder_hidden_size,  # 768
            seq_len_out=config.heatmap_size ** 2 + 1,  # 197
            hidden_size_out=self.config.target_hidden_size  # 512/1024/2048
        )

        self.target_encoder = target_encoder
        self.heatmap_size = config.heatmap_size
        self.image_size = config.image_size
        self.perturb_radius = config.perturb_radius
        self.patch_size = backbone_pos.config.patch_size

        # if getattr(config, 'head_init', None):
        #     ipdb.set_trace()
        #     self.explainer_head_init(config, backbone_pos.config, backbone_neg.config)

        # if config.lora:
        #     peft_config = LoraConfig(
        #         target_modules=["query", "value"], # ["q", "v"],
        #         inference_mode=False,
        #         r=config.lora_dim,
        #         lora_alpha=32,
        #         lora_dropout=0.1
        #     )
        #     self.backbone = get_peft_model(self.backbone, peft_config)
        #     self.backbone.print_trainable_parameters()

        self.freeze_model_parameters()
        self.print_model_parameters()

    def explainer_head_init(self, config, backbone_pos_config, backbone_neg_config):
        del self.explainer_head_pos, self.explainer_head_neg
        self.explainer_head_pos = ExplainerHead(
            layer_num=config.explainerhead_layernum,
            seq_len_in=(backbone_pos_config.image_size // backbone_pos_config.patch_size) ** 2 + 1,  # 197
            hidden_size_in=backbone_pos_config.decoder_hidden_size,  # 768
            seq_len_out=config.heatmap_size ** 2 + 1,  # 197
            hidden_size_out=self.config.target_hidden_size  # 512/1024/2048
        )

        self.explainer_head_neg = ExplainerHead(
            layer_num=config.explainerhead_layernum,
            seq_len_in=(backbone_neg_config.image_size // backbone_neg_config.patch_size) ** 2 + 1,  # 197
            hidden_size_in=backbone_neg_config.decoder_hidden_size,  # 768
            seq_len_out=config.heatmap_size ** 2 + 1,  # 197
            hidden_size_out=self.config.target_hidden_size  # 512/1024/2048
        )

    def print_model_parameters(self):
        for n, p in self.named_parameters():
            if p.requires_grad:
                print(f"{n} is trainable...")

    def freeze_model_parameters(self):

        if self.target_encoder is not None:
            for n, p in self.target_encoder.named_parameters():
                p.requires_grad = False

    def forward(
            self,
            pixel_values=None,
            noise=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        generic_explanation_pos = self.backbone_pos(
            pixel_values,
            noise,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
        ).logits

        explanation_pos_hat = self.explainer_head_pos(generic_explanation_pos)

        generic_explanation_neg = self.backbone_neg(
            pixel_values,
            noise,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
        ).logits

        explanation_neg_hat = self.explainer_head_neg(generic_explanation_neg)

        if self.target_encoder is not None:

            with torch.no_grad():
                explanation_index = self.perturb_input(pixel_values, perturb_radius=self.perturb_radius, device=explanation_pos_hat.device)
                perturb_output_pos, perturb_output_neg = self.attribution()

            loss_pos = self.forward_loss(explanation_pos_hat, perturb_output_pos, explanation_index)
            loss_neg = self.forward_loss(explanation_neg_hat, perturb_output_neg, explanation_index)

            loss = loss_pos + loss_neg

        return ExplanationOutput(
            loss=loss if self.target_encoder is not None else None,
            logits=explanation_pos_hat,
            hidden_states=generic_explanation_pos,
        )

    def perturb_input(self, input, perturb_radius=2, device=torch.device("cpu")):

        self.batch_size = input.shape[0]
        perturb_center_xaxis = torch.randint(0, self.heatmap_size, size=(self.batch_size,), device=device)
        perturb_center_yaxis = torch.randint(0, self.heatmap_size, size=(self.batch_size,), device=device)

        explanation_index = perturb_center_yaxis * self.heatmap_size + perturb_center_xaxis

        left_boundary = perturb_center_xaxis - perturb_radius
        up_boundary = perturb_center_yaxis + perturb_radius
        right_boundary = perturb_center_xaxis + perturb_radius
        down_boundary = perturb_center_yaxis - perturb_radius

        left_boundary[left_boundary <= 0] = 0
        up_boundary[up_boundary >= self.heatmap_size] = self.heatmap_size
        right_boundary[right_boundary >= self.heatmap_size] = self.heatmap_size
        down_boundary[down_boundary <= 0] = 0

        grid_x, grid_y = torch.meshgrid(torch.arange(0, self.heatmap_size, device=device),
                                        torch.arange(0, self.heatmap_size, device=device))
        grid_x = grid_x.reshape((1, self.heatmap_size, self.heatmap_size)).repeat((self.batch_size, 1, 1))
        grid_y = grid_y.reshape((1, self.heatmap_size, self.heatmap_size)).repeat((self.batch_size, 1, 1))

        # self.half_mask = ~((grid_x >= left_boundary.reshape(self.half_sample_num, self.batch_size, 1, 1)) & (grid_x <= right_boundary.reshape(self.half_sample_num, self.batch_size, 1, 1)) &
        #                 (grid_y >= down_boundary.reshape(self.half_sample_num, self.batch_size, 1, 1)) & (grid_y <= up_boundary.reshape(self.half_sample_num, self.batch_size, 1, 1)))

        perturb_mask = ((grid_x >= left_boundary.reshape(self.batch_size, 1, 1)) & (
                    grid_x <= right_boundary.reshape(self.batch_size, 1, 1)) &
                                 (grid_y >= down_boundary.reshape(self.batch_size, 1, 1)) & (
                                             grid_y <= up_boundary.reshape(self.batch_size, 1, 1)))

        perturb_mask = perturb_mask.type(torch.float).unsqueeze(dim=1) # Shape batch_size, 1, heatmap_size, heatmap_size
        perturb_mask = torch.cat([perturb_mask, 1 - perturb_mask], dim=0) #

        del left_boundary, up_boundary, right_boundary, down_boundary, grid_x, grid_y

        # import matplotlib.pyplot as plt
        # plt.imshow(self.perturb_mask_pos[0].cpu())
        # plt.savefig("mask.png")
        # plt.close()
        # print(self.explanation_index[0])

        # print(origin_input.reshape(self.sample_num, self.batch_size, -1).sum(dim=2))
        # print(self.mask.shape)

        # mask = torch.cat([self.perturb_mask_pos, 1 - self.perturb_mask_pos], dim=0).unsqueeze(dim=1)  # .reshape(-1, self.height, self.width)
        # self.perturb_image = torch.cat([input, input], dim=0)

        perturb_mask = torch.nn.functional.interpolate(perturb_mask, size=self.image_size, mode='nearest') # Shape batch_size, 1, image_size, image_size
        self.perturb_image = torch.cat([input, input], dim=0)
        self.perturb_image = self.perturb_image * perturb_mask # .repeat(1, input.shape[1], 1, 1)  # Shape batch_size, 1, image_size, image_size

        del perturb_mask

        return explanation_index

    def attribution(self):

        # For ResNet
        # perturb_output = self.target_encoder(self.perturb_image).pooler_output  # 2 * batch_size x embedding_dim
        # For ViT
        # perturb_output = self.target_encoder(self.perturb_image).last_hidden_state[:, 0, :]  # 2 * batch_size x embedding_dim
        # group_sv = perturb_output[:self.batch_size] # - perturb_output[self.batch_size:] # batch_size x embedding_dim

        perturb_output_pos = self.target_encoder.config.get_embedding(self.target_encoder(self.perturb_image[:self.batch_size]))
        perturb_output_neg = self.target_encoder.config.get_embedding(self.target_encoder(self.perturb_image[self.batch_size:]))

        del self.perturb_image

        perturb_output_pos = perturb_output_pos.reshape(self.batch_size, self.config.target_hidden_size)
        perturb_output_neg = perturb_output_neg.reshape(self.batch_size, self.config.target_hidden_size)

        return perturb_output_pos, perturb_output_neg

    def forward_loss(self, pred, target, explanation_index):

        pred = pred.reshape(self.batch_size, -1, self.config.target_hidden_size)
        # explanation_index_expand = explanation_index.unsqueeze(dim=-1).expand(self.batch_size, self.mini_patch_num_square, self.config.target_hidden_size)
        # pred = torch.gather(pred, dim=1, index=explanation_index_expand).transpose(1,2)

        explanation_index_expand = explanation_index.unsqueeze(dim=-1).expand(self.batch_size,
                                                                              self.config.target_hidden_size)
        pred = torch.gather(pred.transpose(1, 2), dim=2, index=explanation_index_expand.unsqueeze(dim=-1)).squeeze(dim=-1)

        loss = (pred - target) ** 2
        loss = loss.mean()

        return loss

    @torch.no_grad()
    def generate_attr(self,
            pixel_values=None,
            noise=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            target_model=None,
            lambda_pos=1,
            lambda_neg=1,
            softmax=True,
    ):

        # print(pixel_values)
        # self.backbone.eval()
        # self.explainer_head.eval()

        generic_explanation_pos = self.backbone_pos(
            pixel_values,
            noise,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
        ).logits

        explainer_output_pos = self.explainer_head_pos(generic_explanation_pos)[:, :-1, :]

        generic_explanation_neg = self.backbone_neg(
            pixel_values,
            noise,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
        ).logits

        explainer_output_neg = self.explainer_head_neg(generic_explanation_neg)[:, :-1, :]

        # generic_explanation = vit_mae_output.logits
        # explainer_output = self.explainer_head(generic_explanation)[:, :-1, :]

        self.class_index_pred = target_model(pixel_values).logits.argmax(dim=-1) # .item()
        # pixel_values[0,0,0,0:100]

        if type(target_model.classifier)==torch.nn.Linear or len(target_model.classifier) == 1:
            target_model_head = target_model.classifier
        else:
            target_model_head = target_model.classifier[-1]

        # target_model_head = target_model.classifier
        # target_model_head.bias = None

        head_output_pos = target_model_head(explainer_output_pos)
        head_output_neg = target_model_head(explainer_output_neg)

        # if softmax: # TODO: vasulization should not use softmax
        #     head_output_pos = head_output_pos.softmax(dim=-1)
        #     head_output_neg = head_output_neg.softmax(dim=-1)

        # head_output = head_output_pos
        # head_output = - head_output_neg

        head_output_pos = self.final_attribution(head_output_pos, "embedding")
        head_output_neg = self.final_attribution(head_output_neg, "embedding")

        # head_output_pos = self.final_attribution(head_output_pos, "log_softmax")
        # head_output_neg = self.final_attribution(head_output_neg, "log_softmax")

        explanation = lambda_pos * head_output_pos - lambda_neg * head_output_neg
        explanation = explanation.reshape((-1, self.config.heatmap_size, self.config.heatmap_size))  #

        explanation_min = explanation.min(dim=-1, keepdim=True).values.min(dim=-1, keepdim=True).values
        explanation_max = explanation.max(dim=-1, keepdim=True).values.max(dim=-1, keepdim=True).values
        explanation_normalized = (explanation - explanation_min) / (explanation_max - explanation_min)

        explanation_pooling = torch.nn.functional.avg_pool2d(explanation.unsqueeze(dim=1),
                                                             kernel_size=2 * self.perturb_radius + 1,
                                                             stride=1,
                                                             padding=self.perturb_radius,
                                                             count_include_pad=False,
                                                             ).squeeze(dim=1)
        explanation_pooling_min = explanation_pooling.min(dim=-1, keepdim=True).values.min(dim=-1, keepdim=True).values
        explanation_pooling_max = explanation_pooling.max(dim=-1, keepdim=True).values.max(dim=-1, keepdim=True).values
        explanation_pooling = (explanation_pooling - explanation_pooling_min) / (
                    explanation_pooling_max - explanation_pooling_min)

        # ipdb.set_trace()

        # print(explanation_pooling)
        return HeatmapOutput(
            heatmap=explanation_normalized,
            heatmap_pooling=explanation_pooling,  # This one is better. Theoretically, this one is better.
            heatmap_visual=explanation_pooling.transpose(1, 2)  # transpose for visualization
        )

    @torch.no_grad()
    def final_attribution(self, head_output, mode):

        # ipdb.set_trace()

        if mode == "embedding":
            return head_output.gather(index=self.class_index_pred.view(-1, 1, 1).repeat((1, head_output.shape[1], 1)), dim=2)

        elif mode == "softmax":
            head_output = head_output.softmax(dim=-1)
            return head_output.gather(index=self.class_index_pred.view(-1, 1, 1).repeat((1, head_output.shape[1], 1)), dim=2)

        elif mode == "log_softmax":
            head_output = head_output.log_softmax(dim=-1)
            return head_output.gather(index=self.class_index_pred.view(-1, 1, 1).repeat((1, head_output.shape[1], 1)), dim=2)

        elif mode == "entropy":
            head_output = head_output.softmax(dim=-1)
            head_output = entropy(head_output)
            return head_output

        else:
            return head_output.gather(index=self.class_index_pred.view(-1, 1, 1).repeat((1, head_output.shape[1], 1)), dim=2)



    @torch.no_grad()
    def generate_single_attr(self,
                      pixel_values=None,
                      noise=None,
                      head_mask=None,
                      output_attentions=None,
                      output_hidden_states=None,
                      return_dict=None,
                      target_model_classifier=None,
                      class_idx=None,
                      lambda_pos=1,
                      softmax=True,
                      ):

        generic_explanation_pos = self.backbone_pos(
            pixel_values,
            noise,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
        ).logits

        explainer_output_pos = self.explainer_head_pos(generic_explanation_pos)[:, :-1, :]

        generic_explanation_neg = self.backbone_neg(
            pixel_values,
            noise,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
        ).logits

        explainer_output_neg = self.explainer_head_neg(generic_explanation_neg)[:, :-1, :]

        if not (type(target_model_classifier)==torch.nn.Linear or len(target_model_classifier) == 1):
            target_model_classifier = target_model_classifier[-1]

        # ipdb.set_trace()

        head_output_pos = target_model_classifier(explainer_output_pos)
        head_output_neg = target_model_classifier(explainer_output_neg)

        head_output = lambda_pos * head_output_pos - head_output_neg
        explanation = head_output.gather(index=class_idx.view(-1, 1, 1).repeat((1, head_output.shape[1], 1)),dim=2)
        explanation = explanation.reshape((-1, self.config.heatmap_size, self.config.heatmap_size))  #

        return explanation

        # [:, :, class_index_pred]
        # classifier_weight = target_model_head.weight[class_index_pred].unsqueeze(dim=-1)

        # if self.target_encoder is not None:
        #
        #     with torch.no_grad():
        #         explanation_index = self.perturb_input(pixel_values, perturb_radius=self.perturb_radius, device=explanation.device)
        #         explanation_target = self.attribution()
        #
        #     loss = self.forward_loss(explainer_output, explanation_target, explanation_index)
        #
        #     print(loss)

        # explanation = torch.matmul(explainer_output.unsqueeze(dim=-1).transpose(2, 3), classifier_weight)
        # print(class_index_pred)


def entropy(prob):
    # return ((prob + 1e-6).log() * prob) + ((1-prob + 1e-6).log() * (1-prob))
    return ((prob + 1e-6).log() * prob).sum(dim=-1)

