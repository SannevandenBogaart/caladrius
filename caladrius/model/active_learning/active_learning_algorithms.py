import os
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from tqdm import tqdm
from shutil import move
from sklearn.model_selection import train_test_split
import sys

def greedy_k_center(labeled, unlabeled, amount):
    """
    Adapted from: https://github.com/dsgissin/DiscriminativeActiveLearning/blob/master/query_methods.py
    """

    greedy_indices = []
    # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
    min_dist = np.min(distance_matrix(labeled.iloc[0, :].values.reshape((1, labeled.shape[1])), unlabeled), axis=0)
    min_dist = min_dist.reshape((1, min_dist.shape[0]))
    for j in range(1, labeled.shape[0], 20):
        if j + 100 < labeled.shape[0]:
            dist = distance_matrix(labeled[j:j+20, :], unlabeled)
        else:
            dist = distance_matrix(labeled.iloc[j:, :], unlabeled)
        min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
        min_dist = np.min(min_dist, axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))

    # iteratively insert the farthest index and recalculate the minimum distances:
    farthest = np.argmax(min_dist)
    greedy_indices.append(farthest)
    for i in range(amount-1):
        if i % 20 == 0:
            print("At Point " + str(i))
        dist = distance_matrix(unlabeled.iloc[greedy_indices[-1], :].values.reshape((1, unlabeled.shape[1])), unlabeled)
        min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
        min_dist = np.min(min_dist, axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
    indices, max = np.array(greedy_indices, dtype=int), np.max(min_dist)
    print(indices, max)
    return indices, max

def get_images(indices,relu):
    images = []
    for i in indices:
        # 0 = 1 for relu and 1 = 0 for softmax difference approach
        images.append(relu.iloc[i][0])
    return images

# Def to obtain random images for random algorithm
def random_images(relu, amount):
    import random

    random_list = []
    while len(random_list) != amount:
        r = random.randint(0,len(relu))
        if r not in random_list: random_list.append(r)

    images = []
    for i in random_list:
        images.append(relu.iloc[i][1])
    return images

# function to create folder when given images, input and output path
def create_folder(images, input_path, output_path):
    import cv2

    input_folder_before = os.path.join(input_path, 'train/before')
    before_folder_kcenter = os.path.join(output_path, 'temp/before')
    os.makedirs(before_folder_kcenter, exist_ok=True)

    input_folder_after = os.path.join(input_path, 'train/after')
    after_folder_kcenter = os.path.join(output_path, 'temp/after')
    os.makedirs(after_folder_kcenter, exist_ok=True)

    k = 0
    while(k < len(images)):
        path = os.path.join(input_folder_before,images[k])
        img = cv2.imread(path)
        os.chdir(before_folder_kcenter)
        filename = images[k]
        cv2.imwrite(filename, img)
        k += 1

    l = 0
    while(l < len(images)):
        path = os.path.join(input_folder_after, images[l])
        img = cv2.imread(path)
        os.chdir(after_folder_kcenter)
        filename = images[l]
        cv2.imwrite(filename, img)
        l += 1
    return output_path

def damage_quantifier(category):
    """
    Assign value based on damage category.
    Args:
        category (str):damage category

    Returns (float): value of damage
    """

    damage_dict = {
        "no-damage": 0,
        "minor-damage": 1,
        "major-damage": 2,
        "destroyed": 3,
    }
    return damage_dict[category]

def create_label_file(input_path, output_path, images):
    # open label file
    building_file = os.path.join(input_path, 'building_information.csv')
    building_info = pd.read_csv(building_file, na_filter=True,usecols=range(1, 10))

    # clean images from .png in name
    images = [str(i).replace(".png","") for i in images]

    filepath_labels = os.path.join(output_path, "labels.txt")
    with open(filepath_labels, "w+") as labels_file:
        i = 0
        while (i < len(building_info)):
            if building_info.iloc[i][0] in images:
                labels_file.write(
                    "{0}.png {1:.4f}\n".format(
                        building_info.iloc[i][0], damage_quantifier(building_info.iloc[i][6])
                    )
                )
            i += 1
    return filepath_labels

def splitDatapoints(
    filepath_labels,
    path_output,
    path_temp_data,
    train_split,
):
    """
    Split the dataset in train, validation and test set and move all the images to its corresponding folder.
    Args:
        filepath_labels (str): path where labels.txt is saved, which contains the image names of all buildings and their damage score.
    """

    with open(filepath_labels) as file:
        datapoints = file.readlines()

    # make sure training,validation and testing set are random partitions of the data
    datapoints_df = pd.DataFrame(datapoints)
    tmpDatapoints_df = pd.DataFrame(columns=['Name','Damage'])
    tmpDatapoints_df[["Name","Damage"]]=datapoints_df[0].str.split(" ",expand=True)
    tmpDatapoints_df["Damage"] = tmpDatapoints_df["Damage"].replace("\n","",regex=True)
    train_set, val_set = train_test_split(tmpDatapoints_df, test_size=(1-train_split), stratify=tmpDatapoints_df["Damage"], random_state=24)

    train_index = train_set.index.tolist()
    val_index = val_set.index.tolist()

    sets = {
        "train": [datapoints[i] for i in train_index],
        "validation": [datapoints[i] for i in val_index],
    }

    for set in sets:
        # make directory for train, validation and test set
        split_filepath = os.path.join(path_output, set)
        os.makedirs(split_filepath, exist_ok=True)

        split_labels_file = os.path.join(split_filepath, "labels.txt")

        split_before_directory = os.path.join(split_filepath, "before")
        os.makedirs(split_before_directory, exist_ok=True)

        split_after_directory = os.path.join(split_filepath, "after")
        os.makedirs(split_after_directory, exist_ok=True)

        with open(split_labels_file, "w+") as split_file:
            for datapoint in tqdm(sets[set]):
                datapoint_name = datapoint.split(" ")[0]

                before_src = os.path.join(path_temp_data, "before", datapoint_name)
                after_src = os.path.join(path_temp_data, "after", datapoint_name)

                before_dst = os.path.join(split_before_directory, datapoint_name)
                after_dst = os.path.join(split_after_directory, datapoint_name)

                # move the files from the temp folder to the final folder
                move(before_src, before_dst)
                move(after_src, after_dst)

                split_file.write(datapoint)

    return sets

# function choosing the images when softmax approach is used
def prob(file_name_prob, method_prob, amount):
    df = pd.read_csv(file_name_prob, header=None)
    df['max'] = df.max(axis=1)
    df['min'] = df.min(axis=1)
    df['diff'] = df['max'] - df['min']

    # create empty list which will be filled with indices indicating the images chosen by algorithm
    indices =[]

    # depending on probability method used, choices: min softmax approach (min), max softmax approach (max) or
    # minimum difference softmax approach (min_diff) and maximum softmax difference approach (max_diff)
    if method_prob == 'min':
        smallest = df['min'].nsmallest(amount)
        smallest = smallest.to_frame()
        indices = smallest.index.values.tolist()
    if method_prob == 'max':
        biggest = df['max'].nlargest(amount)
        biggest = biggest.to_frame()
        indices = biggest.index.values.tolist()
    if method_prob == 'max_diff':
        max_diff = df['diff'].nlargest(amount)
        max_diff = max_diff.to_frame()
        indices = max_diff.index.values.tolist()
    if method_prob == 'min_diff':
        min_diff = df['diff'].nsmallest(amount)
        min_diff = min_diff.to_frame()
        indices = min_diff.index.values.tolist()
    return indices


def main():
    # input path to all images
    input_path = "./xview2/wind_all/all_wind_processed/"
    # output path to store subset
    output_path = "./xview2/stratify/wind_all_prob_5000"

    # paths to files specifying the value of the relu and softmax layers
    file_name_relu = "./correct_format/matthew_all_correct.csv"
    file_name_prob = "./softmax/all_wind_prob.csv"

    # open and read the relu/softmax file as pandas dataframe
    relu = pd.read_csv(file_name_relu)
    softmax = pd.read_csv(file_name_prob, header = None)

    # specify number of images to include in training/validation set
    amount = 5000

    # specify active learning method to use. Either prob, random or kcenter
    method = "prob"

    # depending on which AL method chosen, obtain the corresponding images
    if method == "random":
        images = random_images(relu, amount)
    if method == "kcenter":
        labeled = pd.read_csv(file_name_relu, nrows= 1, usecols=range(2, 514))
        unlabeled = pd.read_csv(file_name_relu, skiprows=range(1, 2), nrows=len(relu) + 1, usecols=range(2, 514))
        indices, max = greedy_k_center(labeled=labeled, unlabeled=unlabeled, amount=amount)
        images = get_images(indices, relu)
    if method == "prob":
        indices = prob(file_name_prob = file_name_prob, amount = amount, method_prob = "min_diff")
        images = get_images(indices, relu = softmax)

    # after choosing images, create train and validation folder out of it
    create_folder(images, input_path = input_path, output_path = output_path)
    filepath_labels = create_label_file(input_path= input_path, output_path= output_path, images = images)
    splitDatapoints(filepath_labels= filepath_labels, path_output= output_path, path_temp_data = os.path.join(output_path, 'temp'), train_split=0.9)

if __name__ == "__main__":
    main()