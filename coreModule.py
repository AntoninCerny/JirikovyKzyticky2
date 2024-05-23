
import torch
from tqdm.auto import tqdm
import torchvision
from cjm_pytorch_utils.core import move_data_to_device
from torch.amp import autocast

from contextlib import contextmanager
import math
from pathlib import Path
import json

import torch.nn.utils as utils




def train_loop(model, 
               train_dataloader, 
               valid_dataloader, 
               device, 
               epochs, 
               checkpoint_path,
               optimizer = None,  
               lr_scheduler = None, 
               use_scaler=False):

    """
    Main training loop
    input all the good stuff and get shitty model
    """
     

    #START inspired Christian Mills tutorial 
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' and use_scaler else None
    #in the beginning i get worst posibble loss - even bad model gets saved on first iter
    best_loss = float('inf') 
    # Loop over the epochs tqdm does cool counter in terminal 
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Run a training epoch and get the training loss
        train_loss = run_epoch(model, train_dataloader, optimizer, lr_scheduler, device, scaler, epoch, is_training=True)
        # Run an evaluation epoch and get the validation loss
        # TODO create separate function for evaluation with better metric
        with torch.no_grad():
            valid_loss = run_epoch(model, valid_dataloader, None, None, device, scaler, epoch, is_training=False)
                
        # model checkpoint - save .pth file and .json metadata  
        # TODO saving to the timestamp folder because i have overwritten really good model
        if valid_loss < best_loss:
            best_loss = valid_loss
            #creates the .pth weight file
            torch.save(model.state_dict(), checkpoint_path)

            # Save metadata about the training process
            training_metadata = {
                'epoch': epoch,
                'train_loss': train_loss,
                'valid_loss': valid_loss, 
                'model_architecture': "KeypointRCNN"
            }
            with open(Path('save/training_metadata.json'), 'w') as f:
                json.dump(training_metadata, f)
        

    # empty the cache
    # needed because otherwise I had to restart PC after every training
    if device != 'cpu':
        getattr(torch, device).empty_cache()
    
#STOP inspired Christian Mills tutorial

#START inspired Christian Mills tutorial but completely redone 
#acodring to Pytorch documentation
#https://christianjmills.com/posts/pytorch-train-keypoint-rcnn-tutorial/


def run_epoch(model, dataloader, optimizer, lr_scheduler, device, scaler, epoch_id, is_training):
    """
    one epoch, doesnt matter whether training or eval  
    """
   
    # training mode
    model.train()

    current_epoch_loss = 0
    #optimizer and learning rate scheduler inspired by example torch vision
    if is_training:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=50, #TODO finetune - 100 is fine
        gamma=0.9 # 0.9 is fine
        )
    #progress bar in terminal
    progress_bar = tqdm(total=len(dataloader), desc="Train" if is_training else "Eval")

    for batch_id, (images, targets) in enumerate(dataloader):
        MyImage = images[0]
        MyTarget = targets[0]

        #TODO make it move whole list to device - my GPU doesnt have big enough mmry for
        #batchsize>1 (this will do for me)

        #MyImage2 = images[1]
        #MyTarget2 = targets[1]
        #send model and image to device to ensure they are on the same device
        #MyImage2 = MyImage2.to(device)
        MyImage = MyImage.to(device)
        model.to(device)

        MyImage = [MyImage]
        MyTarget = [MyTarget]

        if is_training :
            #send also the target data to device
            #model returns losses - metric to how good the model works
            losses = model(MyImage,move_data_to_device(targets, device))
        else:
            #same thing but without gradiant - actually not trainin
            #TODO create better metric maybe COCO?
            with torch.no_grad():
                losses = model(MyImage,move_data_to_device(MyTarget, device))
         # Compute the loss
        loss = sum([loss for loss in losses.values()])  # Sum up the losses
         # If in training mode
        if is_training:
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                old_scaler = scaler.get_scale()
                scaler.update()
                new_scaler = scaler.get_scale()
                if new_scaler >= old_scaler:
                    lr_scheduler.step()
            else:
                # Gradient clipping because i kept crashing
                clip_value = 1  #TODO fine tune
                loss.backward()
                utils.clip_grad_value_(model.parameters(), clip_value)
                optimizer.step()
                lr_scheduler.step()
                
            optimizer.zero_grad()
        
        loss_item = loss.item()
        current_epoch_loss += loss_item
        # Update progress bar
        progress_bar.set_postfix(loss=loss_item, 
                                 avg_loss=current_epoch_loss/(batch_id+1), 
                                 lr=lr_scheduler.get_last_lr()[0] if is_training else ""
                                 )
        progress_bar.update()
        
        # If loss is NaN or infinity, stop training
        if is_training:
            stop_training_message = f"Loss is NaN or infinite at epoch {epoch_id}, batch {batch_id}. Stopping training."
            assert not math.isnan(loss_item) and math.isfinite(loss_item), stop_training_message
        
    progress_bar.close()
    return current_epoch_loss / (batch_id + 1)
#STOP inspired Christian Mills tutorial