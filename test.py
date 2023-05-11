import torch
import torch.nn as nn

import config
import main


def Mytest(helper, epoch, model, is_poison=False, agent_name_key=""):
    model.eval()
    total_loss = 0
    was_correct = 0  # means the data should be correct, but due to threshold, part of it was rejected
    correct = 0
    dataset_size = 0
    if (
        helper.params["type"] == config.TYPE_CIFAR
        or helper.params["type"] == config.TYPE_MNIST
        or helper.params["type"] == config.TYPE_FASHION_MNIST
    ):
        data_iterator = helper.test_data
        for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_iterator, batch, evaluation=True)
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(
                output, targets, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability

            probabilities = torch.nn.functional.softmax(output)
            probabilities = torch.gather(probabilities, 1, pred.unsqueeze(1))

            for idx, is_correct in enumerate(pred.eq(targets.data.view_as(pred)).cpu()):
                if is_correct == True:
                    was_correct += 1
                    if probabilities[idx][0] >= helper.params["confidence_threshold"]:
                        correct += 1
    acc = 100.0 * (float(correct) / float(dataset_size)) if dataset_size != 0 else 0
    total_l = total_loss / dataset_size if dataset_size != 0 else 0

    main.logger.info(
        "___Test {} poisoned: {}, epoch: {}: , model {}, Average loss: {:.4f}, "
        "Accuracy: {}/{} ({:.4f}%)".format(
            model.name,
            is_poison,
            epoch,
            agent_name_key,
            total_l,
            correct,
            dataset_size,
            acc,
        )
    )
    model.train()
    return (total_l, acc, was_correct, correct, dataset_size)


def Mytest_poison(helper, epoch, model, is_poison=False, agent_name_key=""):
    model.eval()
    total_loss = 0.0
    was_correct = 0  # means the data should be correct, but due to threshold, part of it was rejected
    correct = 0
    dataset_size = 0
    poison_data_count = 0
    if (
        helper.params["type"] == config.TYPE_CIFAR
        or helper.params["type"] == config.TYPE_MNIST
        or helper.params["type"] == config.TYPE_FASHION_MNIST
    ):
        data_iterator = helper.test_data_poison
        for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(
                batch, adversarial_index=-1, evaluation=True
            )

            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(
                output, targets, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability

            probabilities = torch.nn.functional.softmax(output)
            probabilities = torch.gather(probabilities, 1, pred.unsqueeze(1))
            for idx, is_correct in enumerate(pred.eq(targets.data.view_as(pred)).cpu()):
                if is_correct == True:
                    was_correct += 1
                    if probabilities[idx][0] >= helper.params["confidence_threshold"]:
                        correct += 1

    acc = (
        100.0 * (float(correct) / float(poison_data_count))
        if poison_data_count != 0
        else 0
    )
    total_l = total_loss / poison_data_count if poison_data_count != 0 else 0
    main.logger.info(
        "___Test {} poisoned: {}, epoch: {}: , model {}, Average loss: {:.4f}, "
        "Accuracy: {}/{} ({:.4f}%)".format(
            model.name,
            is_poison,
            epoch,
            agent_name_key,
            total_l,
            correct,
            poison_data_count,
            acc,
        )
    )

    model.train()
    return total_l, acc, was_correct, correct, poison_data_count


def Mytest_poison_trigger(helper, model, adver_trigger_index):
    model.eval()
    total_loss = 0.0
    was_correct = 0  # means the data should be correct, but due to threshold, part of it was rejected
    correct = 0
    dataset_size = 0
    poison_data_count = 0

    if (
        helper.params["type"] == config.TYPE_CIFAR
        or helper.params["type"] == config.TYPE_MNIST
        or helper.params["type"] == config.TYPE_FASHION_MNIST
    ):
        data_iterator = helper.test_data_poison
        adv_index = adver_trigger_index
        for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(
                batch, adversarial_index=adv_index, evaluation=True
            )

            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(
                output, targets, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability

            probabilities = torch.nn.functional.softmax(output)
            probabilities = torch.gather(probabilities, 1, pred.unsqueeze(1))
            for idx, is_correct in enumerate(pred.eq(targets.data.view_as(pred)).cpu()):
                if is_correct == True:
                    was_correct += 1
                    if probabilities[idx][0] >= helper.params["confidence_threshold"]:
                        correct += 1
    acc = (
        100.0 * (float(correct) / float(poison_data_count))
        if poison_data_count != 0
        else 0
    )
    total_l = total_loss / poison_data_count if poison_data_count != 0 else 0

    model.train()
    return total_l, acc, was_correct, correct, poison_data_count


def Mytest_poison_agent_trigger(helper, model, agent_name_key):
    model.eval()
    total_loss = 0.0
    was_correct = 0  # means the data should be correct, but due to threshold, part of it was rejected
    correct = 0
    dataset_size = 0
    poison_data_count = 0

    if (
        helper.params["type"] == config.TYPE_CIFAR
        or helper.params["type"] == config.TYPE_MNIST
        or helper.params["type"] == config.TYPE_FASHION_MNIST
    ):
        data_iterator = helper.test_data_poison
        adv_index = -1
        for temp_index in range(0, len(helper.params["adversary_list"])):
            if int(agent_name_key) == helper.params["adversary_list"][temp_index]:
                adv_index = temp_index
                break
        for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(
                batch, adversarial_index=adv_index, evaluation=True
            )

            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(
                output, targets, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability

            probabilities = torch.nn.functional.softmax(output)
            probabilities = torch.gather(probabilities, 1, pred.unsqueeze(1))
            for idx, is_correct in enumerate(pred.eq(targets.data.view_as(pred)).cpu()):
                if is_correct == True:
                    was_correct += 1
                    if probabilities[idx][0] >= helper.params["confidence_threshold"]:
                        correct += 1
    acc = (
        100.0 * (float(correct) / float(poison_data_count))
        if poison_data_count != 0
        else 0
    )
    total_l = total_loss / poison_data_count if poison_data_count != 0 else 0

    model.train()
    return total_l, acc, was_correct, correct, poison_data_count
