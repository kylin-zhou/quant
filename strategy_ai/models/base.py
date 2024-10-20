# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import abc
from typing import Text, Union

import numpy as np
from typing import Text, Union
import copy
from tqdm import tqdm

import torch
import torch.nn.functional as F

from .utils import get_or_create_path

class BaseModel():
    """Learnable Models"""

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        
        if self.loss == "bce":
            return F.binary_cross_entropy_with_logits(pred[mask], label[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):

        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, x_train, y_train):

        x_train_values = x_train
        y_train_values = np.squeeze(y_train)

        self.model.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        pbar = tqdm(range(len(indices))[:: self.batch_size])
        for i in pbar:
            pbar.set_description('Train')

            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_train_values[indices[i : i + self.batch_size]]).float().to(self.device)

            pred = self.model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):

        # prepare training data
        x_values = data_x
        y_values = np.squeeze(data_y)

        self.model.eval()

        scores = []
        losses = []

        indices = np.arange(len(x_values))

        pbar = tqdm(range(len(indices))[:: self.batch_size])
        for i in pbar:
            pbar.set_description('Valid')
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)

            pred = self.model(feature)
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            score = self.metric_fn(pred, label)
            scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        train_data,
        valid_data,
        evals_result=dict(),
        save_path=None,
    ):

        if not train_data or not valid_data:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = train_data
        x_valid, y_valid = valid_data

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            # self.logger.info("training...")
            self.train_epoch(x_train, y_train)
            # self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(x_train, y_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            self.logger.info("train loss %.6f, valid loss %.6f" % (train_loss, val_loss))
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
                
                self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
                self.model.load_state_dict(best_param)
                torch.save(best_param, save_path)
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        if self.use_gpu:
            torch.cuda.empty_cache()

        return evals_result

    def predict(self, test_data):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = test_data
        self.model.eval()
        x_values = x_test
        sample_num = x_values.shape[0]
        preds = []

        for begin in tqdm(range(sample_num)[:: self.batch_size]):
            if sample_num - begin < self.batch_size:
                end = sample_num
            else:
                end = begin + self.batch_size
            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
            with torch.no_grad():
                pred = self.model(x_batch).detach().cpu().numpy()
            preds.append(pred)

        return np.concatenate(preds)