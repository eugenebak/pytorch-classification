import gc
import os
import yaml
import random
import wandb
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn

# Custom
import losses
from test import evaluation
from dataloader import get_dataloader
from utils import get_classifier, set_manual_seed, print_loss, colors, dict2namespace

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train(model, config):
    ### wandb setting ###
    os.environ['WANDB_API_KEY'] = "0ee23525f6f4ddbbab74086ddc0b2294c7793e80"
    wandb.init(project=config.PROJECT, entity="psj", name=config.EXP, tags=[config.DATASET])
    wandb.config.update(config)
    
    ### Optimizers ###
    optimizer = torch.optim.SGD(model.parameters(), lr=config.TRAIN.LR, 
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WEIGHT_DECAY)
    
    ### Load checkpoint ###
    if config.TRAIN.RESUME_EPOCH is not None:
        if config.TRAIN.RESUME_EPOCH == -1:
            resume_name = "final"
        else:
            resume_name = f"epoch{config.TRAIN.RESUME_EPOCH}"
        ckpt_path = os.path.join(config.RESULT_PATH, 
                                 f"{config.CLASSIFIER}_{resume_name}_{config.DATASET}.pth")
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for g in optimizer.param_groups:
            g["lr"] = checkpoint["lr"]
            current_lr = g["lr"]
        logging.info("[*] Load the model at step %d" % checkpoint["iteration"])
        start_iteration = checkpoint["iteration"]
    else:
        start_iteration = 0
        current_lr = config.TRAIN.LR
        
        
    ### Dataloader ###
    train_loader = get_dataloader(dataset_name=config.DATASET, which="train", 
                                  subsample=config.TRAIN_SUBSAMPLE, 
                                  batch_size=config.TRAIN.BATCH_SIZE, config=config)
    test_loader  = get_dataloader(dataset_name=config.DATASET, which="val", 
                                  subsample=config.TRAIN.TEST_SUBSAMPLE,
                                  batch_size=config.TRAIN.EVAL_BATCH_SIZE, config=config)
    
    loss_fn = nn.CrossEntropyLoss()
    losses_dict = {}
    train_loader_iterator = iter(train_loader)
    total_iteration = config.TRAIN.EPOCHS * len(train_loader)
    
    ### Training start ###
    for idx, iteration in enumerate(range(start_iteration, total_iteration)):
        try:
            (x, y) = next(train_loader_iterator)
        except StopIteration:
            np.random.seed()  # Ensure randomness
            # Some cleanup
            train_loader_iterator = None
            torch.cuda.empty_cache()
            gc.collect()
            train_loader_iterator = iter(train_loader)
            (x, y) = next(train_loader_iterator)
        
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        model.train()
        outputs = model(x)
        losses_dict["CE"] = loss_fn(outputs, y)
        total_loss = losses_dict["CE"]
        total_loss.backward()
        optimizer.step()
        
        ### Calculate train accuracy ###
        with torch.no_grad():
            _, y_pred = torch.max(outputs.data, 1)
            correct = (y_pred == y).sum().item()
            losses_dict["train_acc"] = (correct/x.size(0)) * 100
        
        wandb.log({"iteration": iteration})
        ### Print the training process ###
        if iteration % config.TRAIN.PRINT_FREQ == 0:
            logging.info("")
            logging.info("[Epoch %d/%d] [Iteration %d/%d]" % 
                         (iteration // (len(train_loader)),
                          config.TRAIN.EPOCHS, 
                          iteration, 
                          total_iteration))
            print_loss(losses_dict)
            wandb.log(losses_dict.copy())

        ### Learning rate decay
        if iteration % len(train_loader) == 0 and \
            (iteration // (len(train_loader)) in config.TRAIN.LR_DECAY_EPOCHS):
            for g in optimizer.param_groups:
                g["lr"] *= config.TRAIN.LR_DECAY_RATIO
                current_lr = g["lr"]
        wandb.log({"lr": current_lr})
        
        ### Save the model per 10 epoch ###
        if iteration % len(train_loader) == 0 and \
            ((iteration // (len(train_loader))) % config.TRAIN.SAVE_EPOCH == 0):
            ckpt_name = f"{config.CLASSIFIER}_epoch{iteration // len(train_loader)}_{config.DATASET}.pth"
            ckpt_path = os.path.join(config.RESULT_PATH, ckpt_name)
            torch.save({"model"     :model.state_dict(),
                        "optimizer" :optimizer.state_dict(),
                        "iteration" :iteration,
                        "lr"        :current_lr,
                        },
                       ckpt_path)
        
        ### Evaluation every specified iteration ###
        if iteration % config.TRAIN.EVAL_FREQ == 0:
            model.eval()
            acc = evaluation(model, test_loader, config)
            acc_dict = {"test_acc"  : acc,}
            logging.info("")
            logging.info(colors.YELLOW + f"[Evaluation on test subset of {config.DATASET}]" + colors.WHITE)
            for name, acc in acc_dict.items():
                logging.info(f"{name:>18s} | {acc:05.2f}%")
            wandb.log(acc_dict)
            
            torch.cuda.empty_cache()
            
        torch.cuda.empty_cache()
    
    logging.info("[*] Saving the final model...")
    ckpt_path = os.path.join(config.RESULT_PATH, f"{config.CLASSIFIER}_final_{config.DATASET}.pth")
    torch.save({"model"     :model.state_dict(),
                "optimizer" :optimizer.state_dict(),
                "iteration" :iteration,
                "lr"        :current_lr,
                },
               ckpt_path)
    logging.info("[*] Training finished.")
    torch.cuda.empty_cache()
    gc.collect()

def main():
    parser = argparse.ArgumentParser()

    # Training parameter
    parser.add_argument('--config', help='config file path')
    args = parser.parse_args()
    
    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        dict_ = yaml.safe_load(f)
    config = dict2namespace(dict_)
    
    ### Basic configuration ###
    os.environ["CUDA_DEVICE_ORDER"] ="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % config.GPU
    device = 'cuda:%d' % config.GPU if torch.cuda.is_available() else 'cpu'
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    
    ### Set manual seed ###
    set_manual_seed(config.TRAIN.SEED)
    
    ### Result path ###
    if not os.path.exists(config.RESULT_PATH):
        os.makedirs(config.RESULT_PATH)
    config.RESULT_PATH = os.path.join(config.RESULT_PATH, config.PROJECT, config.EXP)
    if not os.path.exists(config.RESULT_PATH):
        os.makedirs(config.RESULT_PATH)
    
    model = get_classifier(config).to(device)
    
    """ Training """
    train(model, config)
    
    """ Test """
    logging.info("[!] Evaluation ...")
    test_loader = get_dataloader(dataset_name= config.DATASET, which="val", 
                                 subsample   = config.TEST.TEST_SUBSAMPLE, 
                                 batch_size  = config.TEST.BATCH_SIZE,
                                 config      = config,)
    acc = evaluation(model, test_loader, config)
    
    logging.info("")
    logging.info(colors.YELLOW + f"[Evaluation results on testset of {config.DATASET}]" + colors.WHITE)
    print_accuracies = {
        "Classifier": config.CLASSIFIER,
        "Accuracy"  : acc,
                        }
    
    for name, acc in print_accuracies.items():
        if type(acc) == float:
            logging.info(f"{name:>20s} | {acc:05.2f}%")
        else:
            logging.info(f"{name:>20s} | {acc}")
    
if __name__ == "__main__":
    main()