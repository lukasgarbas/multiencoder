import torch
import os
import math
import datetime
from copy import deepcopy
from pathlib import Path
from typing import Union, List

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from utils import Corpus
from encoders import MultiEncoder
from task_heads import TextClassifier, TextRegressor, SequenceTagger


class ModelTrainer:
    def __init__(
            self,
            model: Union[TextClassifier, TextRegressor, SequenceTagger],
            corpus: Corpus,
            optimizer: torch.optim.Optimizer = AdamW,
    ):
        self.model = model
        self.corpus = corpus
        self.optimizer = optimizer
        self.evaluation_metrics = corpus.evaluation_metric
        self.main_metric = corpus.main_metric
        self.use_linear_scheduler = False
        self.scheduler = None
        self.best_model = None
        self.best_model_score = 0.
        self.epoch = 0

        self.device = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

        self.encoder_type = "single-transformer"

        if hasattr(model, "encoder") and isinstance(model.encoder, MultiEncoder):
            self.encoder_type = "multi-transformer"


    def train(
            self,
            learning_rate: Union[float, List[float]] = 2e-5,
            decoder_learning_rate: float = None,
            batch_size: int = 16,
            epochs: int = 10,
            shuffle_data: bool = False,
            num_workers: int = 0,
            base_path: Union[Path, str] = None,
            save_best_model: bool = True,
            use_linear_scheduler: bool = False,
            warmup_fraction: float = 0.1,
            weight_decay: float = 0.01,
    ):
        """
        :param learning_rate: pick a learning rate, 2e-5 is good for base models, large models need a lower lr
        :param decoder_learning_rate: allow decoder to use a different (e.g. larger) learning rate
        :param batch_size: mini batch size
        :param epochs: number of epochs to train your model
        :param shuffle_data: set tu True if you want tu shuffle the data after the first epoch
        :param num_workers: num_workers for torch data loader
        :param base_path: directory path to store trained model
        :param save_best_model: set to False if you do not want to save model in file
        :param use_linear_scheduler: linear decay lr scheduler with warmup (false by default)
        :param warmup_fraction: warmup proportion  for linear decay scheduler
        :param weight_decay: weight decay for AdamW optimizer
        """

        if type(learning_rate) is float:
            learning_rate = [learning_rate]

        if len(learning_rate) == 1 and not decoder_learning_rate:
            self.optimizer = self.optimizer(self.model.parameters(),
                                            lr=learning_rate[0],
                                            weight_decay=weight_decay)
        else:
            self.optimizer = self._create_optimizer_with_multiple_learning_rates(
                learning_rate,
                decoder_learning_rate,
            )

        num_updates_per_epoch = math.ceil(len(self.corpus.train)/batch_size)
        num_training_steps = math.ceil(epochs * num_updates_per_epoch)

        self.use_linear_scheduler = use_linear_scheduler
        if self.use_linear_scheduler:
            num_warmup_steps = math.ceil(num_training_steps*warmup_fraction)
            self.scheduler = self._create_linear_scheduler(num_training_steps,
                                                           num_warmup_steps)

        if self.encoder_type == 'multi-transformer':
            lm_names = []
            for i in range(self.model.encoder.num_lms):
                lm_names.append(getattr(self.model.encoder, f"lm_{i+1}").name)
        else:
            lm_names = [self.model.encoder.name]

        log_line(f"Model: {self.model}")
        log_line("----------------------------------------------------")
        log_line(f"Corpus:     {self.corpus.name}")
        log_line(f"Split:      {self.corpus.show_data_split()}")
        log_line(f"Task type:  {self.corpus.task_type}")
        log_line(f"Evaluation: {', '.join(self.evaluation_metrics)}")
        log_line("----------------------------------------------------")
        log_line("Language Models")
        for encoder_name in lm_names:
            log_line(f"- {encoder_name}")
        log_line("----------------------------------------------------")
        if self.encoder_type == "multi-transformer":
            log_line(f"Model type:")
            log_line(f" - multi-transformer")
            log_line(f" - {self.model.name}")
            log_line(f" - combine method:     {self.model.encoder.combine_method}")
            log_line(f" - combine dropout:    {self.model.encoder.combine_dropout}")
            log_line(f" - alternate freezing: {self.model.encoder.alternate_freezing_prob}")
        else:
            log_line(f"Model type:")
            log_line(f" - single-transformer")
            log_line(f" - {self.model.name}")
        log_line("----------------------------------------------------")
        log_line(f"Model parameters:")
        log_line(f" - device:          {self.device}")
        log_line(f" - epochs:          {epochs}")
        log_line(f" - batch size:      {batch_size}")
        log_line(f" - training steps:  {num_training_steps}")
        log_line(f" - optimizer:       {type(self.optimizer).__name__}")
        log_line(f" - learning rate:   {' '.join([str(x) for x in learning_rate])}")
        log_line(f" - linear schedule: {use_linear_scheduler}")
        if use_linear_scheduler:
            log_line(f" - warmup fraction: {warmup_fraction}")
        log_line(f" - weight decay:    {weight_decay}")
        log_line(f" - shuffle data:    {shuffle_data}")
        log_line(f" - save best model: {save_best_model}")
        log_line("-----------------------------------------------------")

        # create base path where to save the model if not provided
        if not base_path:
            corpus_prefix = f"{self.corpus.name[:4].strip('- ')}-" if self.corpus.name else ""
            base_path = f"models/{corpus_prefix}{self.encoder_type}-{self.model.name}"

        if type(base_path) is str:
            base_path = Path(base_path)

        # create batches for dev data
        dev_batch_loader = self.corpus.create_dev_dataloader(
            batch_size=batch_size,
            num_workers=num_workers,
        )

        train_loss = []
        dev_loss = []
        dev_score_history = []

        self.model.train()

        for epoch in range(epochs):

            # keep track of epochs for logging purposes
            self.epoch = epoch + 1

            # shuffle the data at each epoch except the first one
            # shuffle is false by default
            train_batch_loader = self.corpus.create_train_dataloader(
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle_data if self.epoch > 1 else False,
            )

            # training step
            train_epoch_loss = self.fit(train_batch_loader)
            train_loss.append(train_epoch_loss)

            # validation step
            dev_epoch_loss, metrics = self.validate(dev_batch_loader)
            dev_loss.append(dev_epoch_loss)
            dev_score_history.append(metrics[self.main_metric])

            log_line("-----------------------------------------------------")
            log_line(f"EPOCH {epoch+1} done: train loss avg {train_epoch_loss:.4f}")
            log_line(f"DEV: loss {dev_epoch_loss:.4f} {metrics_log(metrics)}")

            # save model with the highest score
            current_score = metrics[self.main_metric]
            if current_score > self.best_model_score:
                self.best_model_score = current_score
                self.best_model = deepcopy(self.model)

                # store trained models using torch.save()
                if save_best_model:
                    log_line("saving best model")
                    if not os.path.isdir(base_path):
                        os.makedirs(base_path)
                    torch.save(self.model, base_path / "best-model.pt")

            log_line("----------------------------------------------------")

        log_line("DONE TRAINING.")
        log_line(f"Best model score on dev set: {self.main_metric} {self.best_model_score}")

        # final evaluation of the model if corpus has a test set
        if self.corpus.test:
            log_line("----------------------------------------------------")
            log_line("Evaluating on test set...")
            test_batches = self.corpus.create_test_dataloader(batch_size=batch_size,
                                                              num_workers=num_workers)
            _, metrics = self.validate(test_batches, use_best_model=True)
            log_line(f"Corpus: {self.corpus.name}")
            log_line(f"TEST set performance: {metrics_log(metrics)}")
            log_line("----------------------------------------------------")

        history = {
            "train_loss": train_loss,
            "val_loss": dev_loss,
            "dev_score_history": dev_score_history
        }
        return history


    def fit(self, train_batches):
        self.model.train()
        train_loss = 0.0
        intermittent_loss = 0.0

        seen_batches = 0
        number_of_batches = len(train_batches)
        modulo = max(1, int(number_of_batches / 10))

        for data, targets in train_batches:

            data = list(data)
            targets = targets.to(self.device)

            # zero the gradients on the model and optimizer
            self.model.zero_grad()
            self.optimizer.zero_grad()

            # forward pass
            loss = self.model.forward_loss(data, targets)

            loss_item = loss.item()
            train_loss += loss_item
            intermittent_loss += loss_item
            seen_batches += 1

            loss.backward()

            if seen_batches % modulo == 0:
                # display current learning rate
                learning_rate_info = "lr - "
                for group in self.optimizer.param_groups:
                    learning_rate_info += f"{group['lr']:.0e} "

                intermittent_loss = intermittent_loss / modulo
                log_line(f"epoch {self.epoch} - iter {seen_batches}/{number_of_batches}"
                        f" - loss {intermittent_loss:.5f} - {learning_rate_info}")
                intermittent_loss = 0.0

            # do gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
  
            # perform optimizer step
            self.optimizer.step()

            # scheduler step: change the learning rate using linear scheduler
            if self.use_linear_scheduler:
                self.scheduler.step()

        avg_train_loss = float(train_loss/number_of_batches)
        return avg_train_loss


    def validate(self, batches, use_best_model=False):
        model = self.best_model if use_best_model else self.model

        # sets the module in evaluation mode
        model.eval()

        # sum up the losses after each forward pass
        validation_loss = 0.0
        predictions = []
        ground_truths = []

        for data, targets in batches:

            data = list(data)
            targets = targets.to(self.device)

            with torch.no_grad():
                loss = model.forward_loss(data, targets)

            validation_loss += loss.item()

            prediction = model.predict(data)
            predictions.extend(prediction.cpu())
            ground_truths.extend(targets.cpu())

        # validation loss average
        # val loss: sum of all batch losses / number of batches
        val_loss = float(validation_loss/len(batches))

        # get the evaluation score
        if isinstance(model, SequenceTagger):
            metrics = model.evaluation_score(predictions,
                                             ground_truths,
                                             self.evaluation_metrics,
                                             self.corpus.label_map)
        else:
            metrics = model.evaluation_score(predictions,
                                             ground_truths,
                                             self.evaluation_metrics)
        return val_loss, metrics


    def _create_optimizer_with_multiple_learning_rates(
            self,
            learning_rates: list,
            decoder_learning_rate: float
    ):
        num_encoders = 1 if self.encoder_type == 'single-transformer' else self.model.encoder.num_lms
        num_learning_rates = len(learning_rates)

        assert num_learning_rates == 1 or num_learning_rates == num_encoders, \
            f"Incorrect number of learning rates: you are using {num_encoders} LMs " \
            f"and {len(learning_rates)} learning rates"

        parameter_groups = []

        if num_learning_rates == 1:
            encoder = self.model.encoder
            params = {"params": encoder.parameters(), "lr": learning_rates[0]}
            parameter_groups.append(params)

        # attach learning rates to each encoder/language model
        if num_learning_rates > 1:
            for encoder_id in range(num_encoders):
                encoder = getattr(self.model.encoder, f"lm_{encoder_id+1}")
                params = {"params": encoder.parameters(), "lr": learning_rates[encoder_id]}
                parameter_groups.append(params)

        # attach learning rate for the decoder
        decoder_group = {
            "params": self.model.decoder.parameters(),
            "lr": decoder_learning_rate
        }

        parameter_groups.append(decoder_group)
        return self.optimizer(parameter_groups)


    def _create_linear_scheduler(self, num_training_steps, num_warmup_steps, last_epoch=-1):
        """
        Linear scheduler with warmup:
        increase the learning rate from zero to the given lr during warmup,
        decrease the learning during from the given lr to zero for the rest training steps.
        """
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        return LambdaLR(self.optimizer, lr_lambda, last_epoch)


def log_line(log_text):
    dt = datetime.datetime.now().isoformat(" ", "seconds")
    log_string = f"{dt} {log_text}"
    print(log_string)


def metrics_log(metrics):
    metrics_results = ""
    for metric, score in metrics.items():
        metrics_results += f"{str(metric)} {str(round(score, 4))} "
    return metrics_results

