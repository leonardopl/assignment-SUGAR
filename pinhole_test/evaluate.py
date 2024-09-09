import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import sklearn.metrics as skm
from config import DEVICE, ROOT_DIR, VAL_INTERVAL
from model import inference

def plot_results(epoch_loss_values, epoch_val_loss_values, lr_values, mse_values):
    plt.figure("train", (18, 4))

    plt.subplot(1, 3, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    plt.xlabel("epoch")
    plt.plot(x, epoch_loss_values, label="Train Loss", color="red")
    plt.plot(x, epoch_val_loss_values, label="Validation Loss", color="blue")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.title("Val Mean MSE")
    x = [VAL_INTERVAL * (i + 1) for i in range(len(mse_values))]
    plt.xlabel("epoch")
    plt.plot(x, mse_values, color="green")

    plt.subplot(1, 3, 3)
    plt.title("Learning Rate")
    x = [i + 1 for i in range(len(lr_values))]
    plt.xlabel("epoch")
    plt.plot(x, lr_values, color="magenta")

    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, "training_results.png"))
    plt.close()

def evaluate(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for test_data in tqdm(test_loader):
            test_inputs, test_labels = test_data[0].to(DEVICE), test_data[1].to(DEVICE)
            test_outputs = inference(model, test_inputs).view(-1)
                    
            y_true.extend(test_labels.cpu().numpy().tolist())
            y_pred.extend(test_outputs.cpu().numpy().tolist())
            
    mse = skm.mean_squared_error(y_true, y_pred)
    
    return mse, y_true, y_pred