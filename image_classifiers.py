from __future__ import print_function
from posixpath import split
import numpy as np
import os
import sys

import torch
import torchvision.models as models
from torchvision import transforms, datasets
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm

from CustomImageDataset import CustomImageDataset
from utils import ensure_dir, store_pert, store_adv, run_black_models, run_black_models_single
import argparse
import glob

import argparse
import random

from PIL import Image
import time

use_cuda = True
# use_gpus = 0
print("CUDA Available: ", torch.cuda.is_available())

def MSE(x1, x2): return ((x1-x2)**2).mean()
l1_dist = lambda a, b :  np.linalg.norm((a - b).ravel(), ord = 1)
l2_dist = lambda a, b :  np.linalg.norm((a - b).ravel(), ord = 2)
linf_dist = lambda a, b :  np.linalg.norm((a - b).ravel(), ord = np.inf)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.225])
softmax = torch.nn.Softmax(dim=1)
loss_ce = torch.nn.CrossEntropyLoss()
loss_l1 = torch.nn.L1Loss()
loss_l2 = torch.nn.MSELoss()
sigmoid = torch.nn.Sigmoid()   
loss_l0 = lambda x: torch.norm(x, p=0)

alexnet = nn.DataParallel(models.alexnet(pretrained=True).cuda().eval())
densenet = nn.DataParallel(models.densenet161(pretrained=True).cuda().eval())
inception = nn.DataParallel(models.inception_v3(pretrained=True).cuda().eval())
googlenet = nn.DataParallel(models.googlenet(pretrained=True).cuda().eval())
mnasnet0_5 = nn.DataParallel(models.mnasnet0_5(pretrained=True).cuda().eval())

model_name_all = ["alexnet", "densenet", "googlenet", "inception", "mnasnet0_5"]
model_list_all = [alexnet, densenet, googlenet, inception, mnasnet0_5]
models_dict_all = {model_name_all[i]: model_list_all[i] for i in range(len(model_name_all))}

# annotations_file = "./dataset/neurips17_dataset/images_0.csv"
annotations_file = "./dataset/neurips17_dataset/test_images.csv"
img_root ='./dataset/neurips17_dataset/images/'
root = './dataset/neurips17_dataset/'

#create dataset
data_transforms = transforms.Compose([
    transforms.ToTensor()
])

normed_data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def attack(models, device, test_loader, alpha, eps, save_dir="test/"):
    print("curr_eps", eps)
    # Accuracy counter
    cnt = 0
    missed = 0
    target_success = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, gt_label, target, image_name in test_loader:
        # Send the data and label to the device
        data, gt_label, target = data.to(device), gt_label.to(device) - 1, target.to(device) - 1

        orig_data = data
        pert = torch.zeros_like(data)

        decay_factor = 1.0
        g = 0
        max_ite = 10
        # alpha = eps / max_ite

        for i in tqdm(range(max_ite)):
            pert.requires_grad = True
            data = (orig_data + pert).clamp(0, 1)
            loss_total = 0
            for model in models:
                if(use_gpus):
                    model = nn.DataParallel(model)
                output = model(normalize(data))
                loss_total += loss_ce(output, target)
            torch.autograd.backward(loss_total, inputs=pert,retain_graph=False, create_graph=False)
            with torch.no_grad():
                g = decay_factor * g + pert.grad / torch.norm(pert.grad, p=1)
                pert = pert - alpha * torch.sign(g)
                pert = pert.clamp(min=-eps, max=eps)

        # print(save_dir)
        store_adv((orig_data + pert).clamp(0, 1), image_name, save_dir)

        # Re-classify the perturbed image
        output = model(normalize(orig_data + pert))
        # Check for success
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        if final_pred.item() == target.item():
            target_success += 1
        if final_pred.item() != gt_label.item():
            missed += 1

        print("final", final_pred.item(), "target", target.item(), "gt", gt_label.item())
    final_missed = missed / float(len(test_loader))
    final_target_success = target_success / float(len(test_loader))
    print("Epsilon: {}\tfinal_missed = {} / {} = {}".format(alpha, final_missed, len(test_loader), final_missed))
    print("Epsilon: {}\tfinal_target_success = {} / {} = {}".format(alpha, target_success, len(test_loader),
                                                                    final_target_success))
    print("Epsilon: {}\tmodel accuracy = {} / {} = {}".format(alpha, len(test_loader) - cnt, len(test_loader), cnt))

# Collect a random sample of n+1 distinct models
def getRandomModels(sz, eps, t):
    pm_dir_targeted = f"./batch_results/eps_{eps}_pm_{sz}/task{t}_wb_results_targeted.txt"
    pm_dir_untargeted = f"./batch_results/eps_{eps}_pm_{sz}/task{t}_wb_results_untargeted.txt"

    # print("directory where random model names are stored:", pm_dir)
    ensure_dir(pm_dir_targeted)
    ensure_dir(pm_dir_untargeted)

    randomIndexes = random.sample(range(0,5), sz)
    randomModelNames = []

    for index in randomIndexes:
        randomModelNames.append(model_name_all[index])

    pm_models_dict = {}

    f = open(pm_dir_targeted, "w")
    g = open(pm_dir_untargeted, "w")

    for modelName in randomModelNames:
        pm_models_dict[modelName] = models_dict_all[modelName]
        f.write(modelName + " ")
        g.write(modelName + " ")

    f.close()
    g.close()

    return pm_models_dict
    
def batch_attack(sz, t, K, eps, save_dir_root, device):
    # Start with the benign images
    img_dir = img_root  # img_dir is the path where the input images are found

    # Iterate through t batches
    for i in range(t):
        # print("Task:", i)
        # print("input directory:", img_dir)
        pm_models = getRandomModels(sz, eps, i)

        # Conduct K-1 white-box attacks on wb_models. This is the Meta-train step
        for j in range(K):
            save_dir = save_dir_root + f"task_{i}/K_{j}/"   # The directory where the output adversarial images should be saved

            # The test_loader should contain the images from the previous tasks output
            ImageNet1000 = CustomImageDataset(annotations_file, img_dir, transform=data_transforms)
            test_loader = torch.utils.data.DataLoader(ImageNet1000, batch_size=1, shuffle=True)

            # attack(pm_models.values(), device, test_loader, eps, False, save_dir)
            attack(pm_models.values(), device, test_loader, 1, eps,save_dir)

            # Update the next task's input directory to the output of the current task
            img_dir = root + save_dir
            # break 

# Evaluates the performance of a single white-box model when inputting an adversarial image
def eval_adv_images(model, task, save_dir_root, k, device):
    adv_img_dir = root + save_dir_root + f"task_{task}/K_{k}"

    ImageNet1000_adv = CustomImageDataset(annotations_file, adv_img_dir, transform=normed_data_transforms)
    test_loader_adv = torch.utils.data.DataLoader(ImageNet1000_adv, batch_size=1, shuffle=True)

    cnt, to_target = run_black_models_single(model, test_loader_adv, device)

    adv_accuracy = (len(test_loader_adv) - cnt) / len(test_loader_adv)
    adv_to_target = to_target / len(test_loader_adv)

    return adv_accuracy, adv_to_target

def divide_models(num_tasks, eps, sz):
    pm_models_dict = []
    ho_models_dict = []

    for i in range(num_tasks):
        results_dir = f"./batch_results/eps_{eps}_pm_{sz}/task{i}_wb_results_targeted.txt"
        f = open(results_dir, "r")

        lines = f.readlines()
        pm_models_string = lines[0].rstrip()
        pm_models = pm_models_string.split(" ")

        tmp_models_dict = {}
        for model in models_dict_all.keys():
            tmp_models_dict[model] = models_dict_all[model]

        pm_dict = {}
        for model in pm_models:
            pm_dict[model] = tmp_models_dict[model]
            del tmp_models_dict[model]
        pm_models_dict.append(pm_dict)
        ho_models_dict.append(tmp_models_dict)

    return pm_models_dict, ho_models_dict

def write_wb_results(sz, num_tasks, eps, save_dir_root, k, device):
    pm_models_dict, ho_models_dict = divide_models(num_tasks, eps, sz)

    for i in range(num_tasks):
        results_dir_targeted = f"./batch_results/eps_{eps}_pm_{sz}/task{i}_wb_results_targeted.txt"
        f = open(results_dir_targeted, "a")
        results_dir_untargeted = f"./batch_results/eps_{eps}_pm_{sz}/task{i}_wb_results_untargeted.txt"
        g = open(results_dir_untargeted, "a")
        print("Task {}:".format(i))

        # The file already has all the pm_models written to it, so write the ho models
        for model in ho_models_dict[i].keys():
            f.write(model + " ")
            g.write(model + " ")
        f.write("\n")
        g.write("\n")

        print("\tpm_models:")
        # Write the results of the models in the pm
        for model in pm_models_dict[i].keys():
            # print("\tpm model:", model)

            pm_model = pm_models_dict[i][model] 
            
            print("\t\tAttacking ", model)
            adv_accuracy, adv_to_target = eval_adv_images(pm_model, i, save_dir_root, k-1, device)
            
            f.write(str(adv_to_target) + " ")
            g.write(str(adv_accuracy) + " ")

        print("\tholdout model:")
        for model in ho_models_dict[i].keys():
            ho_model = ho_models_dict[i][model]

            print("\t\tAttacking ", model)
            adv_accuracy, adv_to_target = eval_adv_images(ho_model, i, save_dir_root, k-1, device)

            f.write(str(adv_to_target) + " ")
            g.write(str(adv_accuracy) + " ")
        f.write("\n")
        g.write("\n")

        f.close()
        g.close()

    return pm_models_dict, ho_models_dict

def main():
    parser = argparse.ArgumentParser(description='Find n + 1 random unique models')
    parser.add_argument("--eps", nargs="?", default=0.05, help="perturbation level (linf): 0.01, 0.03, 0.05, 0.08, 0.1, 0.2")
    parser.add_argument("--sz", nargs="?", default=2, help="the number of random models drawn from the ensemble")
    parser.add_argument("--t", nargs="?", default=1, help="the number of tasks for the attack")
    parser.add_argument("--K", nargs="?", default=5, help="the number of iterations for the attack")
    parser.add_argument("--root", nargs="?", default="batch_result", help="root of all the samples to be evaluated")
    parser.add_argument("--mg" , dest = "gpus", action='store_true', help="Enables use of multiple GPUS")
    parser.add_argument("--sg" , dest = "gpus", action='store_false', help="Enables use of multiple GPUS")
    parser.set_defaults(gpus=False)
    args = parser.parse_args()

    eps = float(args.eps)
    sz = int(args.sz)
    t = int(args.t)
    K = int(args.K)
    save_dir_root = args.root 
    global  use_gpus
    use_gpus= args.gpus

    print(use_gpus)
    device = ""

    if(use_gpus == True):
        # This line will use multiple GPUS
        print(f"Using {torch.cuda.device_count()} GPUs")
        device = torch.device("cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu")
    elif(use_gpus == False):
        # This line will use a single GPU
        print(f"Using a single GPU")
        device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    save_dir_root = save_dir_root + f"_{eps}_pm_{sz}/"
    # print(save_dir_root)

    start = time.perf_counter()
    batch_attack(sz, t, K, eps, save_dir_root, device)
    end = time.perf_counter()
    write_wb_results(sz, t, eps, save_dir_root, K, device)

    if(use_gpus == True):
        # This line will use multiple GPUS
        print(f"Using {torch.cuda.device_count()} GPUs")
    elif(use_gpus == False):
        # This line will use a single GPU
        print(f"Using a single GPU")
    print("Running time: ", end-start)

if __name__ == "__main__":
    main()