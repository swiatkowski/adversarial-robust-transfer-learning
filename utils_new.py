"""
Utility code for the Colab:
https://colab.research.google.com/drive/14O1yyQaT1GEFx2BWj-BAkHLTuuJnC8AR?usp=sharing
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, NamedTuple
import matplotlib.pyplot as plt


from cox.store import Store
from robustness import imagenet_models
from robustness.attacker import AttackerModel
from robustness.datasets import ImageNet
from robustness.tools.helpers import AverageMeter

import torch
from torch import nn, optim
from tqdm import tqdm


import utils_original as utils


@dataclass(frozen=True)
class Config:
    name: str
    robust_model: bool
    adv_train: bool
    fine_tune: bool
    step_size: float = 0.001


class ConfigsGroup(NamedTuple):
    configs: List[Config]
    title: str


def train_loop(model, loader, optimizer, criterion, epoch="-", normalization=None,
               store=None, adv=False):
    acc_meter = AverageMeter()
    # iterator = tqdm(iter(loader), total=len(loader), position=0, leave=True)
    for data, target in loader:
        data, target = data.cuda(), target.cuda()
        model.train()
        if adv:
            data = utils.L2PGD(model, data, target, normalization,
                               step_size=0.5, Nsteps=20,
                               eps=1.25, targeted=False, use_tqdm=False)

        optimizer.zero_grad()
        logits = utils.forward_pass(model, data, normalization)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            val = utils.accuracy(model, data, target, normalization)
            acc_meter.update(val, data.shape[0])

            # Commented out to reduce the amount of logs in Colab
            # iterator.set_description(f"Epoch: {epoch}, Adv: {adv}, Train accuracy={acc_meter.avg:.2f}")
            # iterator.refresh()
    if store:
        store.tensorboard.add_scalar("train_accuracy", acc_meter.avg, epoch)


def eval_loop(model, loader, epoch="-", normalization=None, store=None, adv=False):
    acc_meter = AverageMeter()
    iterator = tqdm(iter(loader), total=len(loader), position=0, leave=True)
    model.eval()

    for data, target in iterator:
        data, target = data.cuda(), target.cuda()
        if adv:
            data = utils.L2PGD(model, data, target, normalization,
                               step_size=0.5, Nsteps=20,
                               eps=1.25, targeted=False, use_tqdm=False)

        val = utils.accuracy(model, data, target, normalization)
        acc_meter.update(val, data.shape[0])

        iterator.set_description(f"Epoch: {epoch}, Adv: {adv}, TEST accuracy={acc_meter.avg:.2f}")
        iterator.refresh()
    if store:
        store.tensorboard.add_scalar(f"test_accuracy_{str(adv)}", acc_meter.avg, epoch)
        print(f'test_accuracy_{str(adv)}')
        store['result'].update_row({f'test_accuracy_{str(adv)}': acc_meter.avg,
                                    'epoch': epoch})


def get_new_model(robust=False):
    # IMPORTANT - use the architecture corresponding to the model you've chosen!

    if robust:
        checkpoint = torch.load("pretrained-models/resnet-18-l2-eps3.ckpt", pickle_module=dill)
    else:
        checkpoint = torch.load("pretrained-models/resnet-18-l2-eps0.ckpt", pickle_module=dill)

    model = imagenet_models.resnet18()
    model = utils.load_imagenet_model_from_checkpoint(model, checkpoint)

    return model


def get_transfered_model(file_name: str, num_targets: int, save_dir_models: str):
    model = imagenet_models.resnet18()
    model.fc = nn.Linear(512, num_targets).cuda()
    model = AttackerModel(model, ImageNet(''))
    while hasattr(model, 'model'):
        model = model.model
    checkpoint = torch.load(f"{save_dir_models}/{file_name}.pt")
    model.load_state_dict(checkpoint)
    return model.cuda()


def plot_metric(metrics: List[str],
                metric_labels: List[str],
                labels: List[str],
                n_seeds: int,
                config_groups: List[ConfigsGroup],
                plot_name: str,
                save_dir_artifacts: str,
                run_time_str: str = None):
    # plt.figure(figsize=(8, 6))
    conf_group_n = len(config_groups)
    metrics_n = len(metrics)

    n_rows = metrics_n * conf_group_n // 2
    n_cols = 2 if len(config_groups) > 1 else 1
    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(6 * conf_group_n, 6 * metrics_n),
                            sharex=False, sharey='row')
    if n_rows == 1 and n_cols == 1:
        axs = [axs]
    elif n_rows > 1:
        axs = [ax for ax_row in axs for ax in ax_row]
    for metric_idx, metric in enumerate(metrics):
        for conf_group_idx, config_group in enumerate(config_groups):
            for config_idx, config in enumerate(config_group.configs):
                df_sum = None
                df_sum_squared = None
                K = None
                for seed in range(n_seeds):
                    dir_path = f'{config.name}_{seed}' if run_time_str is None else f'{config.name}_{run_time_str}_{seed}'
                    df = Store(save_dir_artifacts, dir_path)['result'].df
                    # plt.plot(df['epoch'], df['test_accuracy_False'], label=config.name)
                    if K is None:
                        K = df
                        df_sum = df - K
                        df_sum_squared = (df - K) ** 2
                    else:
                        df_sum += df - K
                        df_sum_squared += (df - K) ** 2

                df_mean = K + df_sum / 3
                df_var = (df_sum_squared - df_sum * df_sum / 3) / (3 - 1)
                mean = df_mean[metric]
                stddev = df_var[metric] ** (0.5)
                label = labels[config_idx] if labels is not None else dir_path
                linestyle = 'dashed' if 'STD_network' in config.name else 'solid'
                axs[metric_idx * conf_group_n + conf_group_idx].plot(df['epoch'], mean, label=label,
                                                                     linestyle=linestyle)
                axs[metric_idx * conf_group_n + conf_group_idx].fill_between(df['epoch'], mean - stddev, mean + stddev,
                                                                             alpha=0.2)
            axs[metric_idx * conf_group_n + conf_group_idx].set_title(config_group.title)
            axs[metric_idx * conf_group_n + conf_group_idx].set_xlabel('Epoch')
            if conf_group_idx == 0:
                axs[metric_idx * conf_group_n + conf_group_idx].set_ylabel(metric_labels[metric_idx])
            if metric_idx * conf_group_n + conf_group_idx == 1:
                axs[metric_idx * conf_group_n + conf_group_idx].legend(loc='best')
    fig.savefig(f'{save_dir_artifacts}/{plot_name}.pdf')


def train_configs(configs: List[Config],
                  n_seeds: List[int],
                  targets: List[int],
                  n_epochs: int,
                  save_dir_artifacts: str,
                  save_dir_models: str,
                  data_aug: bool = False,
                  save_models: bool = True,
                  adv_acc: bool = False,
                  run_time_str: str = None):

    loaders_BIN, normalization_function_BIN, label_map_BIN = utils.get_binary_dataset(
        batch_size=256, transfer=True, data_aug=data_aug, targets=targets, per_target=128, random=False
    )
    train_loader_BIN, test_loader_BIN = loaders_BIN

    run_time_str = run_time_str if run_time_str is not None else datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    for seed in n_seeds:
        for config in configs:
            dir_name = f"{config.name}_{run_time_str}_{seed}"
            store = Store(save_dir_artifacts, dir_name)
            writer = store.tensorboard

            if adv_acc:
                metrics = {
                    'test_accuracy_True': float,
                    'test_accuracy_False': float,
                    'epoch': int}
            else:
                metrics = {
                    'test_accuracy_False': float,
                    'epoch': int}

            store.add_table('result', metrics)

            # std_linear_net = utils.Linear(Nfeatures=3*32*32, Nclasses=2).cuda()
            model = get_new_model(config.robust_model)
            if not config.fine_tune:
                for param in model.parameters():
                    param.requires_grad = False
            model.fc = nn.Linear(512, len(targets)).cuda()
            model.train()

            train_model(model,
                        train_loader_BIN,
                        test_loader_BIN,
                        train_loop,
                        eval_loop,
                        step_size=config.step_size,
                        epochs=n_epochs,
                        normalization=normalization_function_BIN,
                        store=store,
                        adv_train=config.adv_train,
                        log_iterations=10,
                        adv_acc=adv_acc
                        )
            if save_models:
                path = f"{save_dir_models}/{dir_name}.pt"
                torch.save(model.state_dict(), path)


def train_model(model, train_loader, test_loader, train_loop, eval_loop, step_size,
                epochs, normalization, store, adv_train, log_iterations=5,
                adv_acc=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=step_size)

    eval_loop(model, test_loader, normalization=normalization, adv=False)
    # eval_loop(model, test_loader, normalization=normalization, adv=True)

    for epoch in range(epochs):

        train_loop(model, train_loader, optimizer, criterion, epoch, normalization, store, adv_train)

        if epoch % log_iterations == 0:
            eval_loop(model, test_loader, epoch, normalization, store, adv=False)
            if adv_acc:
                eval_loop(model, test_loader, epoch, normalization, store, adv=True)

            if store:
                store['result'].flush_row()
