import os
import time
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.regression import MeanSquaredError
from config import *
from model import inference

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.mode = mode

    def early_stop(self, validation_metric):
        if self.mode == 'min':
            improved = (validation_metric + self.min_delta) < self.best_metric
        else:
            improved = (validation_metric - self.min_delta) > self.best_metric

        if improved:
            self.best_metric = validation_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train(model, train_loader, val_loader, loss_function, optimizer, lr_scheduler):
    tb_writer = SummaryWriter(ROOT_DIR)
    early_stopper = EarlyStopper(patience=PATIENCE, min_delta=1e-4, mode='min')
    mse_metric = MeanSquaredError(num_outputs=1).to(DEVICE)
    scaler = torch.GradScaler(DEVICE)

    best_mse = float('inf')
    best_mse_epoch = -1

    epoch_loss_values = []
    epoch_val_loss_values = []
    lr_values = []
    mse_values = []

    total_start = time.time()
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} STARTING TRAINING")

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        print(f"\n{'-' * 30}\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} EPOCH {epoch + 1}/{MAX_EPOCHS}")
        
        # Training
        model.train()
        epoch_loss = train_epoch(model, train_loader, loss_function, optimizer, lr_scheduler, scaler)
        
        epoch_loss_values.append(epoch_loss)
        lr_values.append(lr_scheduler.get_last_lr()[0])
        
        print(f"\nepoch {epoch + 1}, average loss: {epoch_loss:.4f}, lr: {lr_values[-1]:.6f}")
        tb_writer.add_scalar("epoch_train_loss", epoch_loss, epoch + 1)
        tb_writer.add_scalar("learning_rate", lr_values[-1], epoch + 1)
        
        # Validation
        if (epoch + 1) % VAL_INTERVAL == 0:
            val_loss, mse = validate(model, val_loader, loss_function, mse_metric)
            
            epoch_val_loss_values.append(val_loss)
            mse_values.append(mse)
            
            tb_writer.add_scalar("epoch_val_mse", mse, epoch + 1)
            tb_writer.add_scalar("epoch_val_loss", val_loss, epoch + 1)
            
            if mse < best_mse:
                best_mse = mse
                best_mse_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(ROOT_DIR, "best_metric_model.pth"))
                print("\n=> SAVED NEW BEST MODEL!")
            
            print(f"\nCURRENT epoch: {epoch + 1}")
            print(f"CURRENT MSE: {mse:.4f}")
            print(f"BEST MSE: {best_mse:.4f} at epoch: {best_mse_epoch}")
            
            if early_stopper.early_stop(mse):
                print("\n=> EARLY STOPPING")
                break
        
        print(f"\ntime consuming of epoch {epoch + 1} is: {str(datetime.timedelta(seconds=int((time.time() - epoch_start))))}")
    
    total_time = time.time() - total_start
    print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} TRAINING FINISHED")
    print(f"training completed, best_mse: {best_mse:.4f} at epoch: {best_mse_epoch}, total time: {str(datetime.timedelta(seconds=int(total_time)))}")

    return epoch_loss_values, epoch_val_loss_values, lr_values, mse_values

def train_epoch(model, train_loader, loss_function, optimizer, lr_scheduler, scaler):
    epoch_loss = 0
    for step, batch_data in enumerate(train_loader, 1):
        step_start = time.time()
        inputs, labels = batch_data[0].to(DEVICE).float(), batch_data[1].to(DEVICE).float()
        
        optimizer.zero_grad()
        with torch.autocast(DEVICE):
            outputs = model(inputs).view(-1)
            loss = loss_function(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()
        skip_lr_sched = (scale > scaler.get_scale())
        
        epoch_loss += loss.item()
        if not skip_lr_sched:
            lr_scheduler.step()
        
        print(f"step {step}/{len(train_loader)}, train_loss: {loss.item():.4f}, step time: {(time.time() - step_start):.4f}")
    
    return epoch_loss / len(train_loader)

def validate(model, val_loader, loss_function, mse_metric):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for idx, val_data in enumerate(val_loader, 1):
            val_inputs, val_labels = val_data[0].to(DEVICE).float(), val_data[1].to(DEVICE).float()
            val_outputs = inference(model, val_inputs).view(-1)
            val_loss += loss_function(val_outputs, val_labels).item()
            mse_metric.update(val_outputs, val_labels)
            print(f"infer {idx}/{len(val_loader)}")
        
        mse = mse_metric.compute().item()
        mse_metric.reset()
    
    return val_loss / len(val_loader), mse
