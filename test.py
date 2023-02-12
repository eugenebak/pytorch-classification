import os
import time
import torch
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import argparse
import yaml
# Data
from dataloader import get_dataloader
from utils import get_classifier, set_manual_seed, colors, dict2namespace

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def evaluation(model, dataloader, config):
    total_test = 0
    correct = 0
    model.eval()
    for idx, (x, y) in enumerate(tqdm(dataloader)):
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        _, y_pred = torch.max(outputs.data, 1)
        correct += (y_pred == y).sum().item()
        total_test += x.size(0)
    acc = (correct / total_test) * 100
    return acc

def main():
    ### Experiment configuration ###
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file path')
    args = parser.parse_args()
    print(args)
    
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
    set_manual_seed(config.TEST.SEED)
    
    ### Result path ###
    if not os.path.exists(config.RESULT_PATH):
        os.makedirs(config.RESULT_PATH)
    result_path = os.path.join(config.RESULT_PATH, config.PROJECT, config.EXP)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    log_path = os.path.join(result_path, "logs")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    ckpt_path = os.path.join(result_path, f"{config.CLASSIFIER}_final_{config.DATASET}.pth") if config.TEST.CKPT_PATH is None else config.TEST.CKPT_PATH
    
    model = get_classifier(config).to(device)
    
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["model"])
    logging.info("Load the model at step %d" % checkpoint["iteration"])
    
    logging.info("")
    logging.info("[!] Evaluation ...")
    test_loader = get_dataloader(dataset_name=config.DATASET, which="val", 
                                    subsample=config.TEST.TEST_SUBSAMPLE,
                                    batch_size=config.TEST.BATCH_SIZE, config=config)
    
    acc = evaluation(model, test_loader, config)
    
    logging.info("")
    logging.info(colors.YELLOW + f"[Evaluation results on testset of {config.DATASET}]" + colors.WHITE)
    print_accuracies = {
                        "Classifier": config.CLASSIFIER,
                        "Accuracy"  : acc,
                        }
        
    with open(os.path.join(log_path, f"{config.DATASET}_{config.CLASSIFIER}.txt"), "a") as f:
        f.write(f"\n\n Date: {time.strftime('%y%m%d_%H%M%S')}")
        f.write(f"\nWeight: {ckpt_path}")
        f.write(f"\nSeed: {config.TEST.SEED}")
        for name, acc in print_accuracies.items():
            if type(acc) == float:
                logging.info(f"{name:>20s} | {acc:05.2f}%")
                f.write(f"\n{name:>20s} | {acc:05.2f}%")
            else:
                logging.info(f"{name:>20s} | {acc}")
                f.write(f"\n{name:>20s} | {acc}")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()