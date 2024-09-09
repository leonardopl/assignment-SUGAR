import torch
import torchvision
from config import DEVICE, LR, MAX_EPOCHS, VAL_AMP

def create_model():
    model = torchvision.models.efficientnet.efficientnet_v2_l(weights="DEFAULT").to(DEVICE)
    # Freeze parameters of the first layer
    for i, param in enumerate(model.parameters()):
        if i == 0:
            param.requires_grad = False
    model.classifier[1] = torch.nn.Linear(1280, 1).to(DEVICE)  # Adjust the output layer to our task
    return model

def create_loss_function():
    return torch.nn.MSELoss()

def create_optimizer(model):
    return torch.optim.Adam(model.parameters(), LR, weight_decay=1e-5)

def create_lr_scheduler(optimizer, train_loader):
    return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=MAX_EPOCHS)

def inference(model, input):
    def _compute(input):
        return model(input)

    if VAL_AMP:
        with torch.autocast(DEVICE):
            return _compute(input)
    else:
        return _compute(input)
