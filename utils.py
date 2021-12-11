from __future__ import print_function
import numpy as np
import os
from PIL import Image
import glob

root = './dataset/neurips17_dataset/'

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def store_adv(perturbed_data, img_name, adv_dir="adv/"):
    root_adv_dir = root + adv_dir
    if not os.path.exists(root_adv_dir):
        os.makedirs(root_adv_dir)
    #     print(root_adv_dir, img_name)
    img_path = root_adv_dir + img_name[0] + ".png"
    adv_np = perturbed_data.squeeze().detach().cpu().numpy().transpose((1, 2, 0))
    #     print(adv_np.shape)
    #     print(type(adv_np))
    im = Image.fromarray((adv_np * 255).astype(np.uint8))
    im.save(img_path)


def store_pert(pert, img_name, pert_dir="pert/"):
    root_pert_dir = root + pert_dir
    if not os.path.exists(root_pert_dir):
        os.makedirs(root_pert_dir)
    pert = pert.squeeze().detach().cpu().numpy().transpose((1, 2, 0))

    np.save(root_pert_dir + img_name[0] + ".npy", pert)

def store_dummy(img_name,adv_dir="adv/"):
    root_adv_dir = root +adv_dir
    if not os.path.exists(root_adv_dir):
        os.makedirs(root_adv_dir)
    #     print(root_adv_dir, img_name)
    img_path = root_adv_dir + img_name[0] + ".txt"

    fo = open(img_path, "a+")
    fo.write(img_name[0])
    fo.close()

def create_adv(img_root, pert_root, adv_dir="adv_root/", multi=True):
    if multi:
        pert_dirs = glob.glob((pert_root + '*'))
        for dirt in pert_dirs:
            print(dirt)
            print(pert_files)
            pert_files = glob.glob((dirt + '/*'))

            if not os.path.exists(adv_dir):
                os.makedirs(adv_dir)

            for pert_file in pert_files:
                img_file = img_root + pert_file.split("/")[-1]
                img_file = img_file.replace("npy", "png")

                img = Image.open(img_file)
                adv = np.add(np.load(pert_file), np.asarray(img) / 255).clip(0, 1)

                adv_file = pert_file.replace("pert", "adv")
                adv_file = adv_file.replace("npy", "png")
                ensure_dir(adv_file)
                adv = Image.fromarray((adv * 255).astype(np.uint8))
                adv.save(adv_file)

    else:
        pert_files = glob.glob((pert_root + '*'))

        if not os.path.exists(adv_dir):
            os.makedirs(adv_dir)

        for pert_file in pert_files:
            img_file = img_root + pert_file.split("/")[-1]
            img_file = img_file.replace("npy", "png")


            img = Image.open(img_file)
            adv = np.add(np.load(pert_file), np.asarray(img) / 255).clip(0, 1)

            adv_file = img_file.replace("images", "adv")
            adv = Image.fromarray((adv * 255).astype(np.uint8))
            ensure_dir(adv_file)
            adv.save(adv_file)


def run_black_models(model, test_adv_loader, target_dir, device):
    cnt = 0
    to_target = 0
    target_index = int(target_dir.split("/")[-1][-1])
    for data, gt_label, target_list, img_name in test_adv_loader:
        # Send the data and label to the device
        data, gt_label = data.to(device), gt_label.to(device)-1

        target = target_list[target_index].to(device)-1
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # If the initial prediction is wrong, dont bother attacking, just move on
#         print(init_pred.item(),target.item())
        if init_pred.item() != gt_label.item():
            if init_pred.item() == target.item():
                to_target += 1
        else:
            cnt += 1
    return cnt, to_target

def run_black_models_single(model, test_adv_loader, device):
    cnt = 0
    to_target = 0
    for data, gt_label, target, img_name\
            in test_adv_loader:
        # Send the data and label to the device
        data, gt_label, target = data.to(device), gt_label.to(device)-1, target.to(device)-1

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # If the initial prediction is wrong, dont bother attacking, just move on
#         print(init_pred.item(),target.item())
        if init_pred.item() != gt_label.item():
            if init_pred.item() == target.item():
                to_target += 1
        else:
            cnt += 1
    return cnt, to_target

# This is mostly copied from [this answer on Stack Overflow](https://stackoverflow.com/a/49452109/3330552)
# (Thanks [Ryan Tuck](https://github.com/ryantuck)!)
# except that I've included the dependencies and set this up to be called from the command line.
#
# Example call from bash to split a single CSV into multiple 100 line CSVs:
#     python3 split_csv /path/to/my/file.csv /path/to/split_files my_split_file 100
#
# Warning: This doesn't have any validation! This will overwrite existing files if you're not careful.

import csv
import os
import sys

def split_csv(source_filepath, dest_path, result_filename_prefix, row_limit):
    """
    Split a source CSV into multiple CSVs of equal numbers of records,
    except the last file.
    The initial file's header row will be included as a header row in each split
    file.
    Split files follow a zero-index sequential naming convention like so:
        `{result_filename_prefix}_0.csv`
    :param source_filepath {str}:
        File name (including full path) for the file to be split.
    :param dest_path {str}:
        Full path to the directory where the split files should be saved.
    :param result_filename_prefix {str}:
        File name to be used for the generated files.
        Example: If `my_split_file` is provided as the prefix, then a resulting
                 file might be named: `my_split_file_0.csv'
    :param row_limit {int}:
        Number of rows per file (header row is excluded from the row count).
    :return {NoneType}:
    """
    if row_limit <= 0:
        raise Exception('row_limit must be > 0')

    with open(source_filepath, 'r') as source:
        reader = csv.reader(source)
        headers = next(reader)

        file_number = 0
        records_exist = True

        while records_exist:

            i = 0
            target_filename = f'{result_filename_prefix}_{file_number}.csv'
            target_filepath = os.path.join(dest_path, target_filename)

            with open(target_filepath, 'w') as target:
                writer = csv.writer(target)

                while i < row_limit:
                    if i == 0:
                        writer.writerow(headers)

                    try:
                        writer.writerow(next(reader))
                        i += 1
                    except:
                        records_exist = False
                        break

            if i == 0:
                # we only wrote the header, so delete that file
                os.remove(target_filepath)

            file_number += 1