# from transformers import AutoImageProcessor, ViTMAEForPreTraining, ViTMAEConfig
import torch
from transformers import ViTImageProcessor, ViTConfig, ViTForImageClassification
from transformers import ViTConfig, ViTModel, DeiTConfig, DeiTModel, SwinConfig, SwinModel
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor, CenterCrop, Resize

from models import GenericExplainer, GenericExplainerConfig
from models import ViTMAEForPreTraining, ViTMAEConfig

import os, json
from utils import explanation_imshow
import ipdb

explainer_checkpoint_dir = "Anonymous-researcher/leta"

## Target Model
target_encoder_config_name = "google/vit-base-patch16-224"
target_encoder_config = ViTConfig.from_pretrained(target_encoder_config_name)
target_model = ViTForImageClassification.from_pretrained(
        target_encoder_config_name,
        config=target_encoder_config,
    )

if "resnet" in target_encoder_config_name:
    def get_embedding(target_model_Output):
        return target_model_Output.pooler_output
    target_encoder_config.get_embedding = get_embedding

elif "vit-base" in target_encoder_config_name or "vit-large" in target_encoder_config_name:
    def get_embedding(target_model_Output):
        return target_model_Output.last_hidden_state[:, 0, :]
    target_encoder_config.get_embedding = get_embedding

elif "swin-base" in target_encoder_config_name or "swin-large" in target_encoder_config_name:
    TargetModelType = SwinModel
    def get_embedding(target_model_Output):
        return target_model_Output.pooler_output
    target_encoder_config.get_embedding = get_embedding

elif "deit-base" in target_encoder_config_name:
    TargetModelType = DeiTModel
    def get_embedding(target_model_Output):
        return target_model_Output.last_hidden_state[:, 0, :]
    target_encoder_config.get_embedding = get_embedding

else:
    raise NotImplementedError("Not support other types of models.")


## Generic Explainer
config_kwargs = {
        "heatmap_patch_num_per_patch": 1,
        "target_hidden_size": target_encoder_config.hidden_size,
        "mask_ratio": 0,
        # "do_normalize": True,
    }
image_processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base", **config_kwargs)
explainer_config = GenericExplainerConfig.from_pretrained(explainer_checkpoint_dir)
backbone_config = ViTMAEConfig.from_dict(explainer_config.backbone_config)
backbone_pos = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base", config=backbone_config)
backbone_neg = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base", config=backbone_config)
generic_explainer = GenericExplainer.from_pretrained(explainer_checkpoint_dir,
                                                     config=explainer_config,
                                                     backbone_pos=backbone_pos,
                                                     backbone_neg=backbone_neg,
                                                     target_encoder=None)

device = torch.device("cuda:0")
generic_explainer.to(device)
target_model.to(device)
generic_explainer.eval()
transforms = Compose(
        [
            Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            # CenterCrop(backbone.config.image_size),
            Resize((backbone_pos.config.image_size, backbone_pos.config.image_size)),
            # ToTensor(),
            # Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
        ]
    )


with open("./imagenet1000_clsidx_to_labels.json", 'r') as json_file:
    class_dict = json.load(json_file)
    for key, value in class_dict.items():
        class_dict[key] = value[:value.find(",")]

for image_fname in os.listdir("output/image"):

    print(image_fname)

    if not (image_fname[-4:] == ".png"):
        continue

    image_image_fname = image_fname
    image = Image.open(os.path.join("output/image", image_image_fname))

    inputs = transforms(image)
    inputs = image_processor(images=inputs, return_tensors="pt")

    with torch.no_grad():

        inputs_pixel_values = inputs["pixel_values"].to(device)
        class_index_pred = target_model(inputs_pixel_values).logits.argmax(dim=-1)  # .item()
        class_str_buf = [class_dict[str(int(class_idx))] for class_idx in class_index_pred]

        explanation = generic_explainer.generate_attr(inputs_pixel_values, target_model=target_model,
                                                      lambda_pos=1,
                                                      lambda_neg=0)

        image_fname_buf = [os.path.join("output/explanation", "heatmap-{}.png".format(image_fname))]
        explanation_imshow(inputs_pixel_values.cpu(), explanation.heatmap_visual.cpu(), image_fname_buf, labels=class_str_buf)


