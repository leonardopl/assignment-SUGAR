import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNet18CameraCalibration(nn.Module):
    def __init__(self, num_params=9, image_size=(224, 224)):
        super(ResNet18CameraCalibration, self).__init__()
        
        self.image_size = image_size
        
        # Load pre-trained ResNet18 model
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Replace the last fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_params)

    def forward(self, x):
        params = self.resnet(x)
        
        # Apply constraints to the output
        fx_fy = F.softplus(params[:, 0:2])  # Positive focal length
        cx = self.image_size[1] * torch.sigmoid(params[:, 2]).unsqueeze(1)  # Principal point x
        cy = self.image_size[0] * torch.sigmoid(params[:, 3]).unsqueeze(1)  # Principal point y
        distortion_coeff = torch.tanh(params[:, 4:])  # Distortion coefficients
        
        return torch.cat([fx_fy, cx, cy, distortion_coeff], dim=1)
    

class CameraCalibrationLoss(nn.Module):
    def __init__(self, image_size = (224, 224), lambda_intrinsics=1.0,
                 lambda_distortion=1.0):
        
        """ Camera calibration loss function
        Args:
            image_size (tuple): Image size (height, width)
            lambda_intrinsics (float): Weight for intrinsics loss
            lambda_distortion (float): Weight for distortion loss            
        """
                
        super(CameraCalibrationLoss, self).__init__()
        self.image_size = image_size
        self.lambda_intrinsics = lambda_intrinsics
        self.lambda_distortion = lambda_distortion
        

    def forward(self, pred, target):
        """ Compute camera calibration loss
        Args:
            pred (torch.Tensor): Predicted camera parameters
            target (torch.Tensor): Target camera parameters
        Returns:
            torch.Tensor: Loss value
        """
        
        # Unpack predictions and targets
        pred_intrinsics = pred[:, :4]
        pred_distortion = pred[:, 4:]

        target_intrinsics = target[:, :4]
        target_distortion = target[:, 4:]

        # Intrinsics loss
        intrinsics_loss = F.mse_loss(pred_intrinsics, target_intrinsics)

        # Distortion loss
        distortion_loss = F.mse_loss(pred_distortion, target_distortion)
        
        total_loss = self.lambda_intrinsics * intrinsics_loss + self.lambda_distortion * distortion_loss

        return total_loss

def train_calibration_network(model, train_loader, image_size, num_epochs=50, device='cuda'):
    model = model.to(device)
    # criterion = nn.MSELoss()
    criterion = CameraCalibrationLoss(image_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, true_params in train_loader:
            images, true_params = images.to(device), true_params.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, true_params)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

def predict_camera_params(model, image, device='cuda'):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        params = model(image)
    return params.squeeze().cpu().numpy()