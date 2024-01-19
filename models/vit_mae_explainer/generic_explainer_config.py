from dataclasses import dataclass
from transformers.configuration_utils import PretrainedConfig
import json

@dataclass
class GenericExplainerConfig(PretrainedConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def load(self, json_file):

        with open(json_file) as f:
            config = json.load(f)

        for key, value in config.items():
            setattr(self, key, value)

        return self

    def backbone_config(self, backbone_config):
        self.backbone_config = backbone_config.__dict__
        # for key, value in backbone_config.items():
        #     setattr(self, "backbone_"+key, value)

    def target_encoder_config(self, target_encoder_config):
        self.target_encoder_config = target_encoder_config.__dict__
        # for key, value in target_encoder_config.items():
        #     setattr(self, "target_encoder_"+key, value)

