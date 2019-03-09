#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: trainer.py
@time: 2019/3/9 15:45
"""
import torch
import numpy as np
from .base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class.
    Note:
        Inherited from BaseTrainer.
        ------
        realize the _train_epoch method.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, logger=None):
        """Trainer.
        :param model:
        :param loss:
        :param metrics:
        :param optimizer:
        :param resume:
        :param config:
        :param data_loader:
        :param logger:
        """
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, logger)
        # data loader
        self.data_loader = data_loader
        # if do validation
        self.do_validation = self.data_loader.eval_iter is not None

        # log step, in every epoch, every batches to log
        self.log_step = int(np.sqrt(data_loader.train_batch_size))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        total_loss = 0.
        total_acc = 0.
        # begin train
        for batch_idx, data in enumerate(self.data_loader.train_iter):
            # not use paths
            # sample = data.question, data.answer1, data.answer2, data.answer3
            # target = data.label
            # use paths
            sample = data["question"], data["answer1"], data["answer2"], data["answer3"], \
                     data["paths1"], data["paths2"], data["paths3"]
            target = data["label"].squeeze(-1)

            self.optimizer.zero_grad()
            output, max_idx = self.model(sample, self.device)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            # self.writer.set_step((epoch - 1) * len(self.data_loader.train_iter) + batch_idx)
            # self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item() * output.size()[0]
            acc = self.metrics[0](max_idx, target)
            total_acc += acc * output.size()[0]

            # log for several batches
            # if batch_idx % self.log_step == 0:
            #     self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}'.format(
            #         epoch,
            #         batch_idx * self.data_loader.train_batch_size,
            #         len(self.data_loader.train),
            #         100.0 * batch_idx / (len(self.data_loader.train) / self.data_loader.train_batch_size + 1.),
            #         loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # if train
        avg_loss = total_loss / (len(self.data_loader.train) + 0.)
        avg_acc = total_acc / (len(self.data_loader.train) + 0.)
        metrics = np.array([avg_loss])
        result = {
            "train_metrics": metrics
        }
        self.logger.info("Training epoch {} done, avg loss: {}, avg acc: {}".format(epoch, avg_loss, avg_acc))
        # if evaluate
        if self.do_validation:
            result = self._valid_epoch(epoch)
        return result

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_acc = 0.
        samples = 0
        total_loss = 0.
        with torch.no_grad():
            for batch_idx, data in enumerate(self.data_loader.eval_iter):
                # not use paths
                # sample = data.question, data.answer1, data.answer2, data.answer3
                # target = data.label

                # use paths
                sample = data["question"], data["answer1"], data["answer2"], data["answer3"], \
                         data["paths1"], data["paths2"], data["paths3"]
                target = data["label"].squeeze(-1)

                output, max_idx = self.model(sample, self.device)
                loss = self.loss(output, target)

                # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                # self.writer.add_scalar('loss', loss.item())
                acc = self.metrics[0](max_idx, target)
                total_acc += acc * output.size()[0]
                samples += output.size()[0]
                total_loss += loss.item() * output.size()[0]
        # evaluate
        val_acc = total_acc / samples
        val_loss = total_loss / samples
        self.logger.info('Val Epoch: {}, ACC: {:.6f}, loss: {:.6f}'.format(epoch, val_acc, val_loss))
        # metrics dict
        metrics = np.array([val_acc])
        return {
            "val_metrics": metrics
        }