import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class TripletModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(TripletModel, self).__init__()

        self.resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        features = self.resnet50(x) 
        features = features.view(features.size(0), -1) 
        embedding = self.fc(features)
        return F.normalize(embedding, p=2, dim=1)