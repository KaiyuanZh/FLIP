import datetime
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import image_train
import main
import utils.csv_record as csv_record


def train(helper, start_epoch, local_model, target_model, is_poison, agent_name_keys):
    epochs_submit_update_dict = {}
    num_samples_dict = {}
    if (
        helper.params["type"] == config.TYPE_CIFAR
        or helper.params["type"] == config.TYPE_MNIST
        or helper.params["type"] == config.TYPE_FASHION_MNIST
    ):
        epochs_submit_update_dict, num_samples_dict = image_train.ImageTrain(
            helper, start_epoch, local_model, target_model, is_poison, agent_name_keys
        )
    return epochs_submit_update_dict, num_samples_dict
