import argparse
import copy
import csv
import datetime
import json
import logging
import math
import os
import random
import test
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.autograd import Variable
from torchvision import transforms

import config
import train
import utils.csv_record as csv_record
from image_helper import ImageHelper
from utils.utils import dict_html

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

logger = logging.getLogger("logger")

torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)

if __name__ == "__main__":
    """
    1. load parameters
    2. load data
    3. create models
    4. train
    5. test
    csv_record is used to record the results
    """
    np.random.seed(1)
    time_start_load_everything = time.time()
    parser = argparse.ArgumentParser(description="PPDL")
    parser.add_argument("--params", dest="params")
    args = parser.parse_args()
    with open(f"./{args.params}", "r") as f:
        params_loaded = yaml.load(f)
    current_time = datetime.datetime.now().strftime("%b.%d_%H.%M.%S")
    if params_loaded["type"] == config.TYPE_CIFAR:
        helper = ImageHelper(
            current_time=current_time,
            params=params_loaded,
            name=params_loaded.get("name", "cifar"),
        )
        helper.load_data()
    elif params_loaded["type"] == config.TYPE_MNIST:
        helper = ImageHelper(
            current_time=current_time,
            params=params_loaded,
            name=params_loaded.get("name", "mnist"),
        )
        helper.load_data()
    elif params_loaded["type"] == config.TYPE_FASHION_MNIST:
        helper = ImageHelper(
            current_time=current_time,
            params=params_loaded,
            name=params_loaded.get("name", "fashionmnist"),
        )
        helper.load_data()
    else:
        helper = None

    logger.info(f"load data done")
    helper.create_model()
    logger.info(f"create model done")
    if helper.params["is_poison"]:
        logger.info(
            f"Poisoned following participants: {(helper.params['adversary_list'])}"
        )

    best_loss = float("inf")

    weight_accumulator = helper.init_weight_accumulator(helper.target_model)

    # save parameters:
    with open(f"{helper.folder_path}/params.yaml", "w") as f:
        yaml.dump(helper.params, f)

    submit_update_dict = None
    num_no_progress = 0

    for epoch in range(
        helper.start_epoch,
        helper.params["epochs"] + 1,
        helper.params["aggr_epoch_interval"],
    ):
        start_time = time.time()
        t = time.time()

        agent_name_keys = helper.participants_list
        adversarial_name_keys = []
        if helper.params["is_random_namelist"]:
            if helper.params["is_random_adversary"]:
                agent_name_keys = random.sample(
                    helper.participants_list, helper.params["no_models"]
                )
                for _name_keys in agent_name_keys:
                    if _name_keys in helper.params["adversary_list"]:
                        adversarial_name_keys.append(_name_keys)
            else:
                ongoing_epochs = list(
                    range(epoch, epoch + helper.params["aggr_epoch_interval"])
                )
                for idx in range(0, len(helper.params["adversary_list"])):
                    for ongoing_epoch in ongoing_epochs:
                        if ongoing_epoch in helper.params[str(idx) + "_poison_epochs"]:
                            if (
                                helper.params["adversary_list"][idx]
                                not in adversarial_name_keys
                            ):
                                adversarial_name_keys.append(
                                    helper.params["adversary_list"][idx]
                                )

                nonattacker = []
                for adv in helper.params["adversary_list"]:
                    if adv not in adversarial_name_keys:
                        nonattacker.append(copy.deepcopy(adv))
                benign_num = helper.params["no_models"] - len(adversarial_name_keys)
                random_agent_name_keys = random.sample(
                    helper.benign_namelist + nonattacker, benign_num
                )
                agent_name_keys = adversarial_name_keys + random_agent_name_keys
        else:
            if helper.params["is_random_adversary"] == False:
                adversarial_name_keys = copy.deepcopy(helper.params["adversary_list"])
        logger.info(f"Server Epoch:{epoch} choose agents : {agent_name_keys}.")
        epochs_submit_update_dict, num_samples_dict = train.train(
            helper=helper,
            start_epoch=epoch,
            local_model=helper.local_model,
            target_model=helper.target_model,
            is_poison=helper.params["is_poison"],
            agent_name_keys=agent_name_keys,
        )
        logger.info(f"time spent on training: {time.time() - t}")

        logger.info("=" * 50)
        logger.info(f"epochs_submit_update_dict: {epochs_submit_update_dict.keys()}")
        logger.info(f"num_samples_dict: {num_samples_dict.keys()}")
        logger.info(f"agent_name_keys: {agent_name_keys}")

        each_is_updated = True

        tmp_target_model = copy.deepcopy(
            helper.target_model
        )  # save the target model temporarily
        tmp_weight_accumulator = copy.deepcopy(weight_accumulator)
        tmp_epochs_submit_update_dict = dict()
        tmp_epochs_submit_update_dict = copy.deepcopy(epochs_submit_update_dict)
        tmp_num_samples_dict = copy.deepcopy(num_samples_dict)

        for agent_key in num_samples_dict.keys():
            # each agent name keys pass to the accumulate weight
            each_agent_name_keys = []
            each_agent_name_keys.append(agent_key)

            # reinitialize the target model, to make sure next time the local aggregate with the initial target
            helper.target_model = copy.deepcopy(tmp_target_model)
            weight_accumulator = copy.deepcopy(tmp_weight_accumulator)

            each_weight_accumulator, each_updates = helper.accumulate_weight(
                weight_accumulator,
                epochs_submit_update_dict,
                each_agent_name_keys,
                num_samples_dict,
            )
            if helper.params["aggregation_methods"] == config.AGGR_MEAN:
                # Average the models
                each_is_updated = helper.average_shrink_models(
                    weight_accumulator=each_weight_accumulator,
                    target_model=helper.target_model,
                    epoch_interval=helper.params["aggr_epoch_interval"],
                )
            helper.each_save_model(
                epoch=epoch, each_agent_name_keys=each_agent_name_keys
            )

        logger.info("end test the returned updates")
        logger.info("=" * 50)

        # reinitialize the params, to prevent the value been changed before
        helper.target_model = copy.deepcopy(tmp_target_model)
        weight_accumulator = copy.deepcopy(tmp_weight_accumulator)
        epochs_submit_update_dict = copy.deepcopy(tmp_epochs_submit_update_dict)
        num_samples_dict = copy.deepcopy(tmp_num_samples_dict)

        weight_accumulator, updates = helper.accumulate_weight(
            weight_accumulator,
            epochs_submit_update_dict,
            agent_name_keys,
            num_samples_dict,
        )
        is_updated = True
        if helper.params["aggregation_methods"] == config.AGGR_MEAN:
            # Average the models
            is_updated = helper.average_shrink_models(
                weight_accumulator=weight_accumulator,
                target_model=helper.target_model,
                epoch_interval=helper.params["aggr_epoch_interval"],
            )
            num_oracle_calls = 1

        # clear the weight_accumulator
        weight_accumulator = helper.init_weight_accumulator(helper.target_model)

        temp_global_epoch = epoch + helper.params["aggr_epoch_interval"] - 1

        (
            epoch_loss,
            epoch_acc,
            epoch_was_corret,
            epoch_corret,
            epoch_total,
        ) = test.Mytest(
            helper=helper,
            epoch=temp_global_epoch,
            model=helper.target_model,
            is_poison=False,
            agent_name_key="global",
        )
        csv_record.test_result.append(
            [
                "global",
                temp_global_epoch,
                epoch_loss,
                epoch_acc,
                epoch_was_corret,
                epoch_corret,
                epoch_total,
            ]
        )
        if len(csv_record.scale_temp_one_row) > 0:
            csv_record.scale_temp_one_row.append(round(epoch_acc, 4))

        if helper.params["is_poison"]:
            (
                epoch_loss,
                epoch_acc_p,
                epoch_was_corret,
                epoch_corret,
                epoch_total,
            ) = test.Mytest_poison(
                helper=helper,
                epoch=temp_global_epoch,
                model=helper.target_model,
                is_poison=True,
                agent_name_key="global",
            )

            csv_record.posiontest_result.append(
                [
                    "global",
                    temp_global_epoch,
                    epoch_loss,
                    epoch_acc_p,
                    epoch_was_corret,
                    epoch_corret,
                    epoch_total,
                ]
            )

            # test on local triggers
            csv_record.poisontriggertest_result.append(
                [
                    "global",
                    "combine",
                    "",
                    temp_global_epoch,
                    epoch_loss,
                    epoch_acc_p,
                    epoch_was_corret,
                    epoch_corret,
                    epoch_total,
                ]
            )

        helper.save_model(epoch=epoch, val_loss=epoch_loss)
        logger.info(f"Done in {time.time() - start_time} sec.")
        csv_record.save_result_csv(
            epoch, helper.params["is_poison"], helper.folder_path
        )

    logger.info(
        f"All training and testing above done in {time.time() - time_start_load_everything} sec."
    )

    logger.info("Saving all the graphs.")
    logger.info(f"This run has a label: {helper.params['current_time']}.")
