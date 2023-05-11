import csv
import json
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.cuda.amp
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms.functional
from torch.utils.data import DataLoader
from torchvision import transforms

warnings.filterwarnings("ignore")

from collections import Counter

import config
import main

SEED = [1024, 557540351, 157301989]
SEED = SEED[0]
np.random.seed(SEED)
torch.manual_seed(SEED)

from collections import defaultdict

global local_mat
local_mat = defaultdict(lambda: defaultdict(dict))

from torch.utils.data import DataLoader, Dataset, TensorDataset


class Trigger:
    def __init__(
        self,
        model,
        batch_size=128,
        steps=100,
        img_rows=28,
        img_cols=28,
        img_channels=1,
        num_classes=10,
        attack_succ_threshold=0.9,
        regularization="l1",
        init_cost=1e-3,
    ):
        self.model = model
        self.batch_size = batch_size
        self.steps = steps
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.attack_succ_threshold = attack_succ_threshold
        self.regularization = regularization
        self.init_cost = init_cost

        self.device = config.device
        self.epsilon = 1e-7
        self.patience = 10
        self.cost_multiplier_up = 1.5
        self.cost_multiplier_down = 1.5 ** 1.5

        self.mask_size = [self.img_rows, self.img_cols]
        self.pattern_size = [self.img_channels, self.img_rows, self.img_cols]

    def generate(
        self,
        pair,
        x_train,
        y_train,
        attack_size=100,
        steps=100,
        init_cost=1e-3,
        learning_rate=0.1,
        init_m=None,
        init_p=None,
    ):
        self.model.eval()
        self.steps = steps
        source, target = pair
        cost = init_cost
        cost_up_counter = 0
        cost_down_counter = 0

        mask_best = torch.zeros(self.pattern_size).to(self.device)
        pattern_best = torch.zeros(self.pattern_size).to(self.device)
        reg_best = float("inf")

        if init_m is None:
            init_mask = np.random.random(self.mask_size)
        else:
            init_mask = init_m
        if init_p is None:
            init_pattern = np.random.random(self.pattern_size)
        else:
            init_pattern = init_p
        init_mask = np.clip(init_mask, 0.0, 1.0)
        init_mask = np.arctanh((init_mask - 0.5) * (2 - self.epsilon))
        init_pattern = np.clip(init_pattern, 0.0, 1.0)
        init_pattern = np.arctanh((init_pattern - 0.5) * (2 - self.epsilon))

        self.mask_tensor = torch.Tensor(init_mask).to(self.device)
        self.pattern_tensor = torch.Tensor(init_pattern).to(self.device)
        self.mask_tensor.requires_grad = True
        self.pattern_tensor.requires_grad = True

        if source is not None:
            indices = np.where(y_train == source)[0]
            if indices.shape[0] > attack_size:
                indices = np.random.choice(indices, attack_size, replace=False)
            else:
                attack_size = indices.shape[0]

            if attack_size < self.batch_size:
                self.batch_size = attack_size

            x_set = x_train[indices]
            y_set = torch.full((x_set.shape[0],), target)
        else:
            x_set, y_set = x_train, y_train
            source = self.num_classes
            self.batch_size = attack_size
            loss_start = np.zeros(x_set.shape[0])
            loss_end = np.zeros(x_set.shape[0])

        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        optimizer = torch.optim.Adam(
            [self.mask_tensor, self.pattern_tensor], lr=learning_rate, betas=(0.5, 0.9)
        )

        index_base = np.arange(x_set.shape[0])
        for step in range(self.steps):
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            index_base = index_base[indices]
            x_set = x_set[indices]
            y_set = y_set[indices]

            x_set = x_set.to(self.device)
            y_set = y_set.to(self.device)

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            if self.batch_size != 0:
                for idx in range(x_set.shape[0] // self.batch_size):
                    x_batch = x_set[idx * self.batch_size : (idx + 1) * self.batch_size]
                    y_batch = y_set[idx * self.batch_size : (idx + 1) * self.batch_size]

                    self.mask = (
                        torch.tanh(self.mask_tensor) / (2 - self.epsilon) + 0.5
                    ).repeat(self.img_channels, 1, 1)
                    self.pattern = (
                        torch.tanh(self.pattern_tensor) / (2 - self.epsilon) + 0.5
                    )

                    x_adv = (1 - self.mask) * x_batch + self.mask * self.pattern

                    optimizer.zero_grad()

                    output = self.model(x_adv)

                    pred = output.argmax(dim=1, keepdim=True)
                    acc = pred.eq(y_batch.view_as(pred)).sum().item() / x_batch.shape[0]

                    loss_ce = criterion(output, y_batch)
                    loss_reg = torch.sum(torch.abs(self.mask)) / self.img_channels
                    loss = loss_ce.mean() + loss_reg * cost

                    loss.backward()
                    optimizer.step()

                    loss_ce_list.extend(loss_ce.detach().cpu().numpy())
                    loss_reg_list.append(loss_reg.detach().cpu().numpy())
                    loss_list.append(loss.detach().cpu().numpy())
                    acc_list.append(acc)

                if (
                    source == self.num_classes
                    and step == 0
                    and loss_ce.shape[0] == attack_size
                ):
                    loss_start[index_base] = loss_ce.detach().cpu().numpy()

                avg_loss_ce = np.mean(loss_ce_list)
                avg_loss_reg = np.mean(loss_reg_list)
                avg_loss = np.mean(loss_list)
                avg_acc = np.mean(acc_list)

                if avg_acc >= self.attack_succ_threshold and avg_loss_reg < reg_best:
                    mask_best = self.mask
                    pattern_best = self.pattern
                    reg_best = avg_loss_reg

                    epsilon = 0.01
                    init_mask = mask_best[0, ...]
                    init_mask = init_mask + torch.distributions.Uniform(
                        low=-epsilon, high=epsilon
                    ).sample(init_mask.shape).to(self.device)
                    init_mask = torch.clip(init_mask, 0.0, 1.0)
                    init_mask = torch.arctanh((init_mask - 0.5) * (2 - self.epsilon))
                    init_pattern = pattern_best + torch.distributions.Uniform(
                        low=-epsilon, high=epsilon
                    ).sample(init_pattern.shape).to(self.device)
                    init_pattern = torch.clip(init_pattern, 0.0, 1.0)
                    init_pattern = torch.arctanh(
                        (init_pattern - 0.5) * (2 - self.epsilon)
                    )

                    with torch.no_grad():
                        self.mask_tensor.copy_(init_mask)
                        self.pattern_tensor.copy_(init_pattern)

                    if source == self.num_classes and loss_ce.shape[0] == attack_size:
                        loss_end[index_base] = loss_ce.detach().cpu().numpy()

                if avg_acc >= self.attack_succ_threshold:
                    cost_up_counter += 1
                    cost_down_counter = 0
                else:
                    cost_up_counter = 0
                    cost_down_counter += 1

                if cost_up_counter >= self.patience:
                    cost_up_counter = 0
                    if cost == 0:
                        cost = self.init_cost
                    else:
                        cost *= self.cost_multiplier_up
                elif cost_down_counter >= self.patience:
                    cost_down_counter = 0
                    cost /= self.cost_multiplier_down

                if step % 10 == 0:
                    main.logger.info(
                        "step: %3d, attack: %.3f, loss: %f, ce: %f, reg: %f, reg_best: %f"
                        % (step, avg_acc, avg_loss, avg_loss_ce, avg_loss_reg, reg_best)
                    )
            else:
                pass

        if source == self.num_classes and loss_ce.shape[0] == attack_size:
            indices = np.where(loss_start == 0)[0]
            loss_start[indices] = 1
            loss_monitor = (loss_start - loss_end) / loss_start
            loss_monitor[indices] = 0
        else:
            loss_monitor = np.zeros(x_set.shape[0])

        if (
            len(loss_monitor) > 0
        ):  # when the reg_best is inf, set loss_monitor/speed as 0
            indices = np.where(loss_monitor == 1)[0]
            loss_monitor[indices] = 0

        return mask_best, pattern_best, loss_monitor


class TriggerCombo:
    def __init__(
        self,
        model,
        batch_size=128,
        steps=100,
        img_rows=28,
        img_cols=28,
        img_channels=1,
        num_classes=10,
        attack_succ_threshold=0.9,
        regularization="l1",
        init_cost=1e-3,
    ):
        self.model = model
        self.batch_size = batch_size
        self.steps = steps
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.attack_succ_threshold = attack_succ_threshold
        self.regularization = regularization
        self.init_cost = [init_cost] * 2

        self.device = config.device
        self.epsilon = 1e-7
        self.patience = 10
        self.cost_multiplier_up = 1.5
        self.cost_multiplier_down = 1.5 ** 1.5

        self.mask_size = [2, 1, self.img_rows, self.img_cols]
        self.pattern_size = [2, self.img_channels, self.img_rows, self.img_cols]

    def generate(
        self,
        pair,
        x_set,
        y_set,
        m_set,
        attack_size=50,
        steps=100,
        init_cost=1e-3,
        init_m=None,
        init_p=None,
    ):
        self.model.eval()
        self.batch_size = attack_size
        self.steps = steps
        source, target = pair

        cost = [init_cost] * 2
        cost_up_counter = [0] * 2
        cost_down_counter = [0] * 2

        mask_best = torch.zeros(self.pattern_size).to(self.device)
        pattern_best = torch.zeros(self.pattern_size).to(self.device)
        reg_best = [float("inf")] * 2

        if init_m is None:
            init_mask = np.random.random(self.mask_size)
        else:
            init_mask = init_m
        if init_p is None:
            init_pattern = np.random.random(self.pattern_size)
        else:
            init_pattern = init_p
        init_mask = np.clip(init_mask, 0.0, 1.0)
        init_mask = np.arctanh((init_mask - 0.5) * (2 - self.epsilon))
        init_pattern = np.clip(init_pattern, 0.0, 1.0)
        init_pattern = np.arctanh((init_pattern - 0.5) * (2 - self.epsilon))

        self.mask_tensor = torch.Tensor(init_mask).to(self.device)
        self.pattern_tensor = torch.Tensor(init_pattern).to(self.device)
        self.mask_tensor.requires_grad = True
        self.pattern_tensor.requires_grad = True

        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        optimizer = torch.optim.Adam(
            [self.mask_tensor, self.pattern_tensor], lr=0.1, betas=(0.5, 0.9)
        )

        for step in range(self.steps):
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            x_set = x_set[indices]
            y_set = y_set[indices]
            m_set = m_set[indices]
            x_set = x_set.to(self.device)
            y_set = y_set.to(self.device)
            m_set = m_set.to(self.device)

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(x_set.shape[0] // self.batch_size):
                x_batch = x_set[idx * self.batch_size : (idx + 1) * self.batch_size]
                y_batch = y_set[idx * self.batch_size : (idx + 1) * self.batch_size]
                m_batch = m_set[idx * self.batch_size : (idx + 1) * self.batch_size]

                self.mask = (
                    torch.tanh(self.mask_tensor) / (2 - self.epsilon) + 0.5
                ).repeat(1, self.img_channels, 1, 1)
                self.pattern = (
                    torch.tanh(self.pattern_tensor) / (2 - self.epsilon) + 0.5
                )

                x_adv = m_batch[:, None, None, None] * (
                    (1 - self.mask[0]) * x_batch + self.mask[0] * self.pattern[0]
                ) + (1 - m_batch[:, None, None, None]) * (
                    (1 - self.mask[1]) * x_batch + self.mask[1] * self.pattern[1]
                )

                optimizer.zero_grad()

                output = self.model(x_adv)

                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(y_batch.view_as(pred)).squeeze()
                acc = [
                    ((m_batch * acc).sum() / m_batch.sum()).detach().cpu().numpy(),
                    (((1 - m_batch) * acc).sum() / (1 - m_batch).sum())
                    .detach()
                    .cpu()
                    .numpy(),
                ]

                loss_ce = criterion(output, y_batch)
                loss_ce_0 = (m_batch * loss_ce).sum().to(self.device)
                loss_ce_1 = ((1 - m_batch) * loss_ce).sum().to(self.device)
                loss_reg = (
                    torch.sum(torch.abs(self.mask), dim=(1, 2, 3)) / self.img_channels
                )
                loss_0 = loss_ce_0 + loss_reg[0] * cost[0]
                loss_1 = loss_ce_1 + loss_reg[1] * cost[1]
                loss = loss_0 + loss_1

                # loss.backward()
                loss.backward(retain_graph=True)
                optimizer.step()

                loss_ce_list.append(
                    [loss_ce_0.detach().cpu().numpy(), loss_ce_1.detach().cpu().numpy()]
                )
                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                loss_list.append(
                    [loss_0.detach().cpu().numpy(), loss_1.detach().cpu().numpy()]
                )
                acc_list.append(acc)

            avg_loss_ce = np.mean(loss_ce_list, axis=0)
            avg_loss_reg = np.mean(loss_reg_list, axis=0)
            avg_loss = np.mean(loss_list, axis=0)
            avg_acc = np.mean(acc_list, axis=0)

            for cb in range(2):
                if (
                    avg_acc[cb] >= self.attack_succ_threshold
                    and avg_loss_reg[cb] < reg_best[cb]
                ):
                    mask_best_local = self.mask
                    mask_best[cb] = mask_best_local[cb]
                    pattern_best_local = self.pattern
                    pattern_best[cb] = pattern_best_local[cb]
                    reg_best[cb] = avg_loss_reg[cb]

                    epsilon = 0.01
                    init_mask = mask_best_local[cb, :1, ...]
                    init_mask = init_mask + torch.distributions.Uniform(
                        low=-epsilon, high=epsilon
                    ).sample(init_mask.shape).to(self.device)
                    init_pattern = pattern_best_local[cb]
                    init_pattern = init_pattern + torch.distributions.Uniform(
                        low=-epsilon, high=epsilon
                    ).sample(init_pattern.shape).to(self.device)

                    otr_idx = (cb + 1) % 2
                    if cb == 0:
                        init_mask = torch.stack(
                            [init_mask, mask_best_local[otr_idx][:1, ...]]
                        )
                        init_pattern = torch.stack(
                            [init_pattern, pattern_best_local[otr_idx]]
                        )
                    else:
                        init_mask = torch.stack(
                            [mask_best_local[otr_idx][:1, ...], init_mask]
                        )
                        init_pattern = torch.stack(
                            [pattern_best_local[otr_idx], init_pattern]
                        )

                    init_mask = torch.clip(init_mask, 0.0, 1.0)
                    init_mask = torch.arctanh((init_mask - 0.5) * (2 - self.epsilon))
                    init_pattern = torch.clip(init_pattern, 0.0, 1.0)
                    init_pattern = torch.arctanh(
                        (init_pattern - 0.5) * (2 - self.epsilon)
                    )

                    with torch.no_grad():
                        self.mask_tensor.copy_(init_mask)
                        self.pattern_tensor.copy_(init_pattern)

                if avg_acc[cb] >= self.attack_succ_threshold:
                    cost_up_counter[cb] += 1
                    cost_down_counter[cb] = 0
                else:
                    cost_up_counter[cb] = 0
                    cost_down_counter[cb] += 1

                if cost_up_counter[cb] >= self.patience:
                    cost_up_counter[cb] = 0
                    if cost[cb] == 0:
                        cost[cb] = self.init_cost
                    else:
                        cost[cb] *= self.cost_multiplier_up
                elif cost_down_counter[cb] >= self.patience:
                    cost_down_counter[cb] = 0
                    cost[cb] /= self.cost_multiplier_down

            if step % 10 == 0:
                main.logger.info(
                    f"step: {step:3d}, attack: ({avg_acc[0]:.2f}, {avg_acc[1]:.2f}), "
                    + f"loss: ({avg_loss[0]:.2f}, {avg_loss[1]:.2f}), "
                    + f"ce: ({avg_loss_ce[0]:.2f}, {avg_loss_ce[1]:.2f}), "
                    + f"reg: ({avg_loss_reg[0]:.2f}, {avg_loss_reg[1]:.2f}), "
                    + f"reg_best: ({reg_best[0]:.2f}, {reg_best[1]:.2f})"
                )

        return mask_best, pattern_best


def trigger_fast_train(helper, model, data_iterator, start_epoch, agent_name_key):
    device = config.device
    model.train()
    model.to(device)

    num_classes = 10
    num_samples = 5
    learning_rate = 0.01
    args_iter = 100
    size_min = 1  ## minimum number of samples

    x_train = []
    y_train = []
    # get all the data on the client
    for batch_id, batch in enumerate(data_iterator):
        x_batch, y_batch = helper.get_batch(data_iterator, batch, evaluation=False)
        y_batch = y_batch.detach().cpu().numpy()
        if len(x_train) == 0:
            x_train = x_batch
        else:
            x_train = torch.cat((x_train, x_batch), 0)
        y_train = np.append(y_train, y_batch)

    # customize my dataloader
    my_x = x_train
    my_x = my_x.detach().cpu().numpy()
    tensor_x = torch.Tensor(my_x)
    tensor_y = torch.Tensor(y_train)

    my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    my_dataloader = DataLoader(
        my_dataset, batch_size=128, shuffle=True, num_workers=8
    )  # create your dataloader

    label_grt = [ele for ele, cnt in Counter(y_train).items() if cnt > size_min]
    indices = []
    for i in range(num_classes):
        if i in label_grt:
            idx = np.where(y_train == i)[0]
            indices.extend(list(idx[:size_min]))
    x_extra = x_train[indices]
    y_extra = y_train[indices]

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.9))

    if agent_name_key not in local_mat:
        mat_warm = np.zeros((num_classes, num_classes))
        mat_diff = np.full((num_classes, num_classes), -np.inf)
        mat_univ = np.full((num_classes, num_classes), -np.inf)
        mask_dict = {}
        pattern_dict = {}
        WARMUP = True
    else:
        mat_warm = local_mat[agent_name_key]["mat_warm"]
        mat_diff = local_mat[agent_name_key]["mat_diff"]
        mat_univ = local_mat[agent_name_key]["mat_univ"]
        mask_dict = local_mat[agent_name_key]["mask_dict"]
        pattern_dict = local_mat[agent_name_key]["pattern_dict"]
        WARMUP = False

    epochs = 5
    portion = 0.6
    trigger_steps = 600
    cost = 1e-3
    count = np.zeros(2)
    warmup_steps = 1
    batch_size = 128
    retrain = 2

    main.logger.info(f"parameters: portion: {portion}")

    # set up trigger generation
    trigger = Trigger(
        model, steps=trigger_steps, attack_succ_threshold=0.90, num_classes=num_classes
    )
    trigger_combo = TriggerCombo(model, steps=trigger_steps)

    max_warmup_steps = warmup_steps * num_classes
    max_steps = max_warmup_steps + args_iter
    source, target = 0, -1
    step = 0
    for epoch in range(epochs):
        dataset_size = 0
        batch_id = 0
        main.logger.info(f"__epoch: {epoch}")
        for x_batch, y_batch in my_dataloader:
            x_batch = x_batch.to(device)
            x_adv = x_batch
            x_adv_trigger = x_batch
            y_batch = y_batch.detach().cpu().numpy().astype("int")
            batch_id += 1
            dataset_size += len(x_batch)
            main.logger.info(
                f"start_epoch: {start_epoch}, internal epoch: {epoch}, batch_id: {batch_id}, len of data: {len(x_batch)}, len of dataset_size: {dataset_size}"
            )

            label_grt_min = [
                ele for ele, cnt in Counter(y_batch).items() if cnt > size_min
            ]

            if len(label_grt_min) == 0:
                if step + 1 >= max_steps:
                    break
                step += 1
                continue

            # trigger stamping
            if step >= max_warmup_steps:
                if WARMUP:
                    mat_diff /= np.max(mat_diff)
                WARMUP = False
                warmup_steps = 2

            if (WARMUP and step % warmup_steps == 0) or (
                (step - max_warmup_steps) % warmup_steps == 0
            ):
                if WARMUP:
                    target += 1
                    trigger_steps = 600
                else:
                    if np.random.rand() < 0.3:  # scheduler
                        source = np.random.choice(label_grt_min)
                        target_set = list(range(10))
                        target_set.remove(source)
                        target = np.random.choice(target_set)

                    else:
                        diff_sum = mat_diff
                        top_source_target = np.vstack(
                            np.unravel_index(
                                np.argsort(diff_sum.ravel())[::-1], diff_sum.shape
                            )
                        ).T
                        i = 0
                        while True:
                            if i == 100:
                                source = -1
                                break
                            source_new, target_new = top_source_target[i]
                            if source_new == source and target_new == target:
                                main.logger.info(
                                    f"source:{source}, target: {target} has been selected before, pass"
                                )
                            if source_new in label_grt_min and source_new != source:
                                source = source_new
                                target = target_new
                                break
                            i += 1
                        if source == -1:
                            if step + 1 >= max_steps:
                                break
                            step += 1
                            continue
                        if np.isnan(diff_sum[source, target]):
                            main.logger.info("encounter nan during selection!")
                            exit()
                    trigger_steps = 400

                key = f"{source}-{target}" if source < target else f"{target}-{source}"
                main.logger.info(f"source: {source}, target: {target}, key: {key}")

                if key in mask_dict:
                    init_mask = mask_dict[key]
                    init_pattern = pattern_dict[key]
                else:
                    init_mask = None
                    init_pattern = None

                cost = 1e-3
                count[...] = 0
                mask_size_list = []

            if WARMUP:
                trigger = Trigger(
                    model,
                    steps=trigger_steps,
                    attack_succ_threshold=0.90,
                    num_classes=num_classes,
                )

                indices = np.where(y_extra != target)[0]
                source_labels = y_extra[indices]
                x_set = x_extra[indices]
                y_set = torch.full((x_set.shape[0],), target)

                # generate universal trigger
                mask, pattern, speed = trigger.generate(
                    (None, target),
                    x_set,
                    y_set,
                    attack_size=len(indices),
                    steps=trigger_steps,
                    init_cost=cost,
                    init_m=init_mask,
                    init_p=init_pattern,
                )
                trigger_size = mask.abs().sum().detach().cpu().numpy()
                indices = np.where(y_batch != target)[0]
                length = int(len(indices) * portion)
                choice = np.random.choice(indices, length, replace=False)

                x_batch_adv = (1 - mask) * x_batch[
                    choice
                ] + 1.0 * mask * pattern  # change the image with mask
                x_batch_adv = torch.clip(x_batch_adv, 0.0, 1.0)
                x_adv[choice] = x_batch_adv

                main.logger.info(
                    f"mask shape: {mask.shape}, pattern shape: {pattern.shape}, speed shape: {speed.shape}"
                )

                mask = mask.detach().cpu().numpy()
                pattern = pattern.detach().cpu().numpy()
                for i in range(num_classes):
                    if i in source_labels and i != target:
                        source_position = list(set(source_labels)).index(i)
                        diff = np.mean(
                            speed[
                                source_position
                                * num_samples : (source_position + 1)
                                * num_samples
                            ]
                        )

                        mat_univ[i, target] = diff

                        src, tgt = i, target
                        key = f"{src}-{tgt}" if src < tgt else f"{tgt}-{src}"
                        if key not in mask_dict:
                            mask_dict[key] = mask[:1, ...]
                            pattern_dict[key] = pattern
                        else:
                            if src < tgt:
                                mask_dict[key] = np.stack(
                                    [mask[:1, ...], mask_dict[key]], axis=0
                                )
                                pattern_dict[key] = np.stack(
                                    [pattern, pattern_dict[key]], axis=0
                                )
                            else:
                                mask_dict[key] = np.stack(
                                    [mask_dict[key], mask[:1, ...]], axis=0
                                )
                                pattern_dict[key] = np.stack(
                                    [pattern_dict[key], pattern], axis=0
                                )

                        mat_warm[i, target] = trigger_size
                        mat_diff[i, target] = mat_warm[i, target]
                    elif i not in source_labels and i != target:
                        mat_warm[i, target] = 0
                        mat_diff[i, target] = 0
                        mat_univ[i, target] = 0
                    else:
                        pass

                x_batch = x_adv.detach()
                optimizer.zero_grad()
                output = model(x_batch)
                y_adv = torch.from_numpy(y_batch).to(device).long()
                loss = criterion(output, y_adv)
                loss.backward()
                optimizer.step()
            else:
                if (
                    source in label_grt_min and target in label_grt_min
                ):  # symmetric training
                    idx_source = np.where(y_batch == source)[0]
                    idx_target = np.where(y_batch == target)[0]
                    length = int(min(len(idx_source), len(idx_target)) * portion)
                    if length > 0:
                        if (step - max_warmup_steps) % warmup_steps > 0:
                            if count[0] > 0 or count[1] > 0:
                                trigger_steps = 400
                                cost = 1e-3
                                count[...] = 0
                            else:
                                trigger_steps = 100
                                cost = 1e-2

                        x_set = torch.cat((x_batch[idx_source], x_batch[idx_target]))
                        y_target = torch.full((len(idx_source),), target)
                        y_source = torch.full((len(idx_target),), source)
                        y_set = torch.cat((y_target, y_source))
                        m_set = torch.zeros(x_set.shape[0])
                        m_set[: len(idx_source)] = 1

                        if init_mask is not None:
                            if init_mask.ndim != 4:
                                init_mask = None
                                init_pattern = None
                            else:
                                pass

                        mask, pattern = trigger_combo.generate(
                            (source, target),
                            x_set,
                            y_set,
                            m_set,
                            attack_size=x_set.shape[0],
                            steps=trigger_steps,
                            init_cost=cost,
                            init_m=init_mask,
                            init_p=init_pattern,
                        )

                        trigger_size = (
                            mask.abs().sum(axis=(1, 2, 3)).detach().cpu().numpy()
                        )
                        if np.max(trigger_size) > 28 * 28 * 1 / 8:
                            if step + 1 >= max_steps:
                                break
                            step += 1
                            continue

                        for cb in range(2):
                            indices = idx_source if cb == 0 else idx_target
                            choice = np.random.choice(indices, length, replace=False)

                            x_batch_adv = (1 - mask[cb]) * x_batch[choice] + 1.0 * mask[
                                cb
                            ] * pattern[cb]
                            x_batch_adv = torch.clip(x_batch_adv, 0.0, 1.0)
                            x_adv[choice] = x_batch_adv

                        mask = mask.detach().cpu().numpy()
                        pattern = pattern.detach().cpu().numpy()
                        for cb in range(2):
                            if init_mask is None or key not in mask_dict:
                                init_mask = mask[:, :1, ...]
                                init_pattern = pattern

                                if key not in mask_dict:
                                    mask_dict[key] = init_mask
                                    pattern_dict[key] = init_pattern
                            else:
                                if np.sum(mask[cb]) > 0 and len(mask_dict[key]) > 1:
                                    init_mask[cb] = mask[cb, :1, ...]
                                    init_pattern[cb] = pattern[cb]
                                    if np.sum(init_mask[cb]) > np.sum(
                                        mask_dict[key][cb]
                                    ):
                                        mask_dict[key][cb] = init_mask[cb]
                                        pattern_dict[key][cb] = init_pattern[cb]
                                else:
                                    count[cb] += 1
                        mask_size_list.append(
                            list(np.sum(3 * np.abs(init_mask), axis=(1, 2, 3)))
                        )

                    if (step - max_warmup_steps) % warmup_steps == warmup_steps - 1:
                        if len(mask_size_list) <= 0:
                            if step + 1 >= max_steps:
                                break
                            step += 1
                            continue

                        mask_size_avg = np.mean(mask_size_list, axis=0)

                        if mat_warm[source, target] == 0:
                            mat_warm[source, target] = mask_size_avg[0]
                            mat_warm[target, source] = mask_size_avg[1]
                            mat_diff[source, target] = 0
                            mat_diff[target, source] = 0
                        else:
                            last_warm = mat_warm[source, target]
                            if last_warm != 0:
                                mat_diff[source, target] += (
                                    mask_size_avg[0] - last_warm
                                ) / last_warm
                            mat_diff[source, target] /= 2
                            last_warm = mat_warm[target, source]
                            if last_warm != 0:
                                mat_diff[target, source] += (
                                    mask_size_avg[1] - last_warm
                                ) / last_warm
                            mat_diff[target, source] /= 2
                            if mask_size_avg[0] != 0:
                                mat_warm[source, target] = mask_size_avg[0]
                            if mask_size_avg[1] != 0:
                                mat_warm[target, source] = mask_size_avg[1]

                    x_batch = x_adv.detach()
                    optimizer.zero_grad()
                    output = model(x_batch)
                    y_adv = torch.from_numpy(y_batch).to(device).long()
                    loss = criterion(output, y_adv)
                    loss.backward()
                    optimizer.step()
                elif (
                    source in label_grt_min and target not in label_grt_min
                ):  # single direction, asymmetric training
                    # use the generate trigger without combo
                    idx_source = np.where(y_batch == source)[0]
                    x_set = x_batch[idx_source]
                    y_set = torch.full((x_set.shape[0],), source)

                    length = int(len(idx_source) * portion)
                    if (
                        init_mask is not None
                    ):  # reshape the init mask and pattern from 4 dim to 3 dim
                        if init_mask.ndim == 3:
                            if source < target:
                                init_mask = init_mask
                                init_pattern = init_pattern
                            else:
                                init_mask = init_mask
                                init_pattern = init_pattern
                        if init_mask.ndim == 4:
                            if source < target and np.sum(init_mask[0]) > 0:
                                init_mask = init_mask[0]
                                init_pattern = init_pattern[0]
                            elif source > target and np.sum(init_mask[1]) > 0:
                                init_mask = init_mask[1]
                                init_pattern = init_pattern[1]
                            else:
                                init_mask = None
                                init_pattern = None
                    if length > 0:
                        mask, pattern, speed = trigger.generate(
                            (source, target),
                            x_set,
                            y_set,
                            attack_size=x_set.shape[0],
                            steps=trigger_steps,
                            init_cost=cost,
                            init_m=init_mask,
                            init_p=init_pattern,
                        )
                        trigger_size = mask.abs().sum().detach().cpu().numpy()
                        if trigger_size > 28 * 28 * 1 / 8:
                            if step + 1 >= max_steps:
                                break
                            step += 1
                            continue

                        choice = np.random.choice(idx_source, length, replace=False)

                        x_batch_adv = (1 - mask) * x_batch[
                            choice
                        ] + 1.0 * mask * pattern  # change the image with mask
                        x_batch_adv = torch.clip(x_batch_adv, 0.0, 1.0)
                        x_adv[choice] = x_batch_adv  # change the image

                        mask = mask.detach().cpu().numpy()
                        pattern = pattern.detach().cpu().numpy()

                        key_WARMUP_F = (
                            f"{source}-{target}"
                            if source < target
                            else f"{target}-{source}"
                        )

                        if init_mask is None or key_WARMUP_F not in mask_dict:
                            init_mask = mask[:1, ...]
                            init_pattern = pattern
                            if key_WARMUP_F not in mask_dict:
                                mask_dict[key_WARMUP_F] = init_mask
                                pattern_dict[key_WARMUP_F] = init_pattern
                        else:
                            if mask_dict[key_WARMUP_F].ndim == 3:
                                if source < target:
                                    mask_dict[key_WARMUP_F] = np.stack(
                                        [mask[:1, ...], mask_dict[key_WARMUP_F]], axis=0
                                    )
                                    pattern_dict[key_WARMUP_F] = np.stack(
                                        [pattern, pattern_dict[key_WARMUP_F]], axis=0
                                    )
                                else:
                                    mask_dict[key_WARMUP_F] = np.stack(
                                        [mask_dict[key_WARMUP_F], mask[:1, ...]], axis=0
                                    )
                                    pattern_dict[key_WARMUP_F] = np.stack(
                                        [pattern_dict[key_WARMUP_F], pattern], axis=0
                                    )
                            elif (
                                np.sum(mask[:1, ...]) > 0
                                and mask_dict[key_WARMUP_F].ndim == 4
                            ):
                                init_mask = mask[:1, ...]
                                init_pattern = pattern
                                if np.sum(init_mask) > np.sum(
                                    mask_dict[key_WARMUP_F][0]
                                ) or np.sum(init_mask) > np.sum(
                                    mask_dict[key_WARMUP_F][1]
                                ):
                                    if source < target:
                                        mask_dict[key_WARMUP_F] = np.stack(
                                            [mask[:1, ...], mask_dict[key_WARMUP_F][0]],
                                            axis=0,
                                        )
                                        pattern_dict[key_WARMUP_F] = np.stack(
                                            [pattern, pattern_dict[key_WARMUP_F][0]],
                                            axis=0,
                                        )
                                    else:
                                        mask_dict[key_WARMUP_F] = np.stack(
                                            [mask_dict[key_WARMUP_F][1], mask[:1, ...]],
                                            axis=0,
                                        )
                                        pattern_dict[key_WARMUP_F] = np.stack(
                                            [pattern_dict[key_WARMUP_F][1], pattern],
                                            axis=0,
                                        )
                            else:
                                pass

                        tmp = [np.sum(np.abs(init_mask))] * 2
                        mask_size_list.append(tmp)

                        if (step - max_warmup_steps) % warmup_steps == warmup_steps - 1:
                            if len(mask_size_list) <= 0:
                                if step + 1 >= max_steps:
                                    break
                                step += 1
                                continue
                            mask_size_avg = np.mean(mask_size_list)

                            if mat_warm[source, target] == 0:
                                mat_warm[source, target] = mask_size_avg
                                mat_warm[target, source] = 0
                                mat_diff[source, target] = 0
                                mat_diff[target, source] = 0
                            else:
                                last_warm = mat_warm[source, target]
                                if last_warm != 0:
                                    mat_diff[source, target] += (
                                        mask_size_avg - last_warm
                                    ) / last_warm
                                mat_diff[
                                    source, target
                                ] /= 2  # decrease the previous exponentially
                                mat_diff[target, source] = 0
                                if mask_size_avg != 0:
                                    mat_warm[source, target] = mask_size_avg
                                    mat_warm[target, source] = 0

                        x_batch = x_adv.detach()
                        optimizer.zero_grad()
                        output = model(x_batch)
                        y_adv = torch.from_numpy(y_batch).to(device).long()
                        loss = criterion(output, y_adv)
                        loss.backward()
                        optimizer.step()
                else:
                    main.logger.info(
                        f"Pass, source not in label_grt_min and target not in label_grt_min"
                    )
                    pass

            local_mat[agent_name_key]["mat_warm"] = mat_warm
            local_mat[agent_name_key]["mat_diff"] = mat_diff
            local_mat[agent_name_key]["mat_univ"] = mat_univ
            local_mat[agent_name_key]["mask_dict"] = mask_dict
            local_mat[agent_name_key]["pattern_dict"] = pattern_dict
            main.logger.info(
                f"Update mat_warm, mat_diff, mat_univ, mask_dict, pattern_dict. Breakout condition step: {step}, max_step: {max_steps} mat_diff: \n{mat_diff}"
            )
            if step + 1 >= max_steps:
                break
            step += 1
        if step + 1 >= max_steps:
            break
        return_dataset_size = dataset_size
    return model, return_dataset_size
