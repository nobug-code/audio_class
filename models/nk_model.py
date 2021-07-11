import torch.nn as nn
import torchvision.models as models

class nkModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.model = models.mobilenet_v3_small()
        self.model.fc = nn.Linear(1000, args.class_number)

    def forward(self, inputs):
        pred = self.model(inputs)
        pred = self.model.fc(pred)

        return pred
