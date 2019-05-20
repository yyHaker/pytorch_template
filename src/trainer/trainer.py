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
                 data_loader):
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
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config)
        # data loader
        self.data_loader = data_loader
        # if do validation
        self.do_validation = self.data_loader.eval_iter is not None

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch.

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        self.model.train()
        total_loss = 0.
        # begin train
        self.data_loader.train_iter.device = self.device
        for batch_idx, data in enumerate(self.data_loader.train_iter):
            # build data
            input_data, label = self._build_data(data)
            # forward and backward
            self.optimizer.zero_grad()
            output = self.model(input_data)
            loss = self.loss(output, label)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * output.size()[0]

            # log for several batches
            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.train_batch_size,
                    len(self.data_loader.train),
                    100.0 * batch_idx / (len(self.data_loader.train) / self.data_loader.train_batch_size + 1.),
                    loss.item()))

            # add scalar to writer
            global_step = (epoch - 1) * len(self.data_loader.train_iter) + batch_idx
            self.writer.add_scalar('train loss', loss.item(), global_step=global_step)

        # if train
        avg_loss = total_loss / (len(self.data_loader.train) + 0.)
        metrics = np.array([avg_loss])
        result = {
            "train_metrics": metrics
        }
        # if evaluate
        if self.do_validation:
            result = self._valid_epoch(epoch)
        self.logger.info("Training epoch {} done, avg loss: {}, {}: {}".format(epoch, avg_loss,
                                                                               self.monitor, result[self.monitor]))
        # add to writer
        self.writer.add_scalar("eval_" + self.monitor, result[self.monitor], global_step=epoch * len(self.data_loader.train))
        return result

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation
        """
        self.model.eval()
        total_loss = 0.
        with torch.no_grad():
            self.data_loader.eval_iter.device = self.device
            for batch_idx, data in enumerate(self.data_loader.eval_iter):
                input_data, label = self._build_data(data)
                output = self.model(input_data)
                loss = self.loss(output, label)

                total_loss += loss.item() * output.size()[0]

                # add scalar to writer
                global_step = (epoch - 1) * len(self.data_loader.eval_iter) + batch_idx
                self.writer.add_scalar('eval loss', loss.item(), global_step=global_step)

        # evaluate
        val_loss = total_loss / len(self.data_loader.eval_iter)
        # TODO: calc monitor value
        monitor_value = 0.
        self.logger.info('Valid Epoch: {}, loss: {:.6f}'.format(epoch, val_loss))
        # metrics dict
        metrics = {}
        metrics[self.monitor] = monitor_value
        return metrics

    def _build_data(self, batch):
        # TODO
        data, label = batch
        return data, label
