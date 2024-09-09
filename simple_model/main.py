import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from camera_parameters import CameraParameters
from camera_calibration_network import ResNet18CameraCalibration, train_calibration_network, predict_camera_params

class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=(224, 224)):        
        self.images = torch.randn(num_samples, 3, *image_size)  # Simulated image sequences
        # Simulated camera parameters (intrinsics and distortion)
        self.params = torch.tensor([
            [1000, 1000, image_size[1]/2, image_size[0]/2, -0.1, 0.01, 0.001, -0.001, 0]
        ]).repeat(num_samples, 1)
        self.params += torch.randn_like(self.params) * 0.1  # Add some noise

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.params[idx]

def main():  
    # Create synthetic dataset
    dataset = SyntheticDataset()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize and train the model
    model = ResNet18CameraCalibration()
    train_calibration_network(model, train_loader, (224, 224))

    # Test the model
    test_image = torch.FloatTensor(np.random.rand(3, 224, 224))
    predicted_params = predict_camera_params(model, test_image)

    # Convert predicted parameters to CameraParameters object
    camera_params = CameraParameters.from_vector(predicted_params)

    print("Predicted Camera Parameters:")
    print(f"Intrinsics: fx={camera_params.fx:.2f}, fy={camera_params.fy:.2f}, cx={camera_params.cx:.2f}, cy={camera_params.cy:.2f}")
    print(f"Distortion: k1={camera_params.k1:.2f}, k2={camera_params.k2:.2f}, p1={camera_params.p1:.2f}, p2={camera_params.p2:.2f}, k3={camera_params.k3:.2f}")

if __name__ == "__main__":
    main()
