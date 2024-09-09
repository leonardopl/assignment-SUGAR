import os
import torch
from config import *
from data import get_data_loaders
from model import create_model, create_loss_function, create_optimizer, create_lr_scheduler
from train import train
from evaluate import evaluate, plot_results

def main():
    # Create output directory
    if not os.path.exists(ROOT_DIR):
        os.makedirs(ROOT_DIR)
    
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders()

    # Create model, loss function, optimizer, and learning rate scheduler
    model = create_model()
    loss_function = create_loss_function()
    optimizer = create_optimizer(model)
    lr_scheduler = create_lr_scheduler(optimizer, train_loader)

    # Train the model
    epoch_loss_values, epoch_val_loss_values, lr_values, mse_values = train(
        model, train_loader, val_loader, loss_function, optimizer, lr_scheduler
    )

    # Plot training results
    plot_results(epoch_loss_values, epoch_val_loss_values, lr_values, mse_values)

    # Evaluate the model
    print(f"\nEvaluating the model on the test set...")
    
    model.load_state_dict(torch.load(os.path.join(ROOT_DIR, "best_metric_model.pth"), weights_only=True))
    mse, _, _ = evaluate(model, test_loader)
    
    print(f"\nTEST Mean Squared Error: {mse:.4f}")

if __name__ == "__main__":
    main()
