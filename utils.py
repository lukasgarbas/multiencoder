import os
import re
import requests
import shutil
import tempfile
import warnings
from pathlib import Path
from tqdm import tqdm

import torch
from torch import tensor
from torch.utils.data import Dataset, DataLoader, dataset

from typing import Union, List, Dict


class Sentence:
    """
    This class represents a sentence or sentence pair object.
    Each Sentence has a text value and a true label.
    Text value can be a list of length two to for sentence pairs.
    """

    def __init__(
            self,
            text: Union[str, List[str]],
            true_label: Union[int, float, List[int]] = None,
            sentence_type: str = None,
    ):
        self._text = text
        self._true_label = true_label
        self._predicted_label = None
        self._sentence_type = sentence_type

        if type(text) == str:
            self._sentence_type = "single-sentence"

        if type(text) == list and len(text) == 2:
            self._sentence_type = "sentence-pair"

        if type(text) == list and len(text) > 2:
            raise ValueError(f"Corrupt sentence: we only solve single-sentence" \
                "and sentence-pair tasks \"{text}\"")

        if text is None:
            warnings.warn("Creating an empty sentence." \
                "Make sure that there are no empty sentences in your Dataset.")

    @property
    def text(self) -> str:
        return self._text

    @property
    def true_label(self) -> Union[int, float]:
        return self._true_label

    @property
    def predicted_label(self) -> Union[int, float]:
        return self._predicted_label

    @predicted_label.setter
    def predicted_label(self, label):
        self._predicted_label = label

    def remove_pedicted_label(self):
        self._predicted_label = None

    def __str__(self) -> str:
        return f"Sentence: {self.text} \nTrue label: {self.true_label} \n" \
            f"Predicted label: {self.predicted_label}"

    def __repr__(self) -> str:
        return f"Sentence: {self.text} \nTrue label: {self.true_label} \n" \
            f"Predicted label: {self.predicted_label}"


class TextDataset(Dataset):
    def __init__(
            self,
            sentences: List[Sentence],
    ):
        self.sentences = sentences
        self.targets = []
        self.data = []

        for i, sentence in enumerate(sentences):
            if sentence.text:
                self.targets.append(sentence.true_label)
                self.data.append(sentence.text)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        label = tensor(self.targets[idx])
        text = self.data[idx]
        return text, label

    def sentence_by_id(self, sentence_id):
        return self.sentences[sentence_id]

    def get_subset(self, start_id, end_id):
        return self.sentences[start_id:end_id]


class Corpus:
    def __init__(
            self,
            name: str,
            task_type: str,
            train: TextDataset,
            dev: TextDataset,
            test: TextDataset = None,
            label_map: Dict = None,
            evaluation_metric: Union[str, List[str]] = None,
    ):
        # set train dev and test data
        self.train = train
        self.dev = dev
        self.test = {} if test is None else test

        # additional info about the dataset
        self.name = name
        self.task_type = task_type
        self.label_map = label_map

        self.num_train_samples = len(train)
        self.num_dev_samples = len(dev)
        self.num_test_samples = len(test) if test else 0

        # set default metrics for the corpus if no metric is provided
        if not evaluation_metric:
            if "classification" in self.task_type:
                evaluation_metric = ['accuracy', 'f1_score']
            if "regression" in self.task_type:
                evaluation_metric = ['spearmanr', 'pearsonr']
        
        if isinstance(evaluation_metric, str):
            evaluation_metric = [evaluation_metric]

        self.evaluation_metric = evaluation_metric

        # main metric will be used to find the best model during optimization
        self.main_metric = self.evaluation_metric[0]

        # create label dictionary to know the number of unique labels
        self.label_dictionary = {}

        # regression tasks do not have class labels
        if "regression" in self.task_type:
            self.label_dictionary = {}
            self.num_classes = 1
        else:
            self.label_dictionary = self.create_label_dictionary()
            self.num_classes = len(self.label_dictionary)


    def create_label_dictionary(self) -> Dict:
        if "regression" in self.task_type:
            warnings.warn("You can not create label dictionary for regression task.")
            return {}

        label_dictionary = {}
        labels = self.train.targets + self.dev.targets
        if type(labels[0]) == list:
            labels = [j for sub in labels for j in sub]
        unique_labels = list(set(labels))

        for label in unique_labels:
            label_dictionary[label] = labels.count(label)

        return label_dictionary

    def statistics(self) -> Dict:
        stats = {
            "Dataset name": self.name,
            "Task type": self.task_type,
            "Train set": self.num_train_samples,
            "Dev set": self.num_dev_samples,
            "Test set": self.num_test_samples,
            "Evaluation metric": self.evaluation_metric
        }

        if "classification" in self.task_type:
            stats["label frequency"] = self.create_label_dictionary()

        if self.label_map:
            stats["label map"] = self.label_map

        return stats

    def _collate_data(self, dataset: TextDataset):
        data = [sample[0] for sample in dataset]
        targets = [sample[-1] for sample in dataset]
        target_lengths = [target.shape for target in targets]

        # pad targets for sequence labeling tasks
        if min(target_lengths) != max(target_lengths):
            padded_targets = []
            for target in targets:
                pad_offset = max(target_lengths)[0] - target.shape[0]
                target = torch.nn.functional.pad(target, (0, pad_offset), value=-1.)
                padded_targets.append(target)
            targets = padded_targets

        return data, torch.stack(targets)

    def create_train_dataloader(self, batch_size, num_workers, shuffle=False):
        return DataLoader(
                self.train,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=self._collate_data
            )

    def create_dev_dataloader(self, batch_size, num_workers, shuffle=False):
        return DataLoader(
                self.dev,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=self._collate_data
            )

    def create_test_dataloader(self, batch_size, num_workers, shuffle=False):
        return DataLoader(
                self.test,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=self._collate_data
            )

    def show_data_split(self) -> str:
        return f"train {len(self.train)} + dev {len(self.dev)} + test {len(self.test)}"

    def __str__(self) -> str:
        stats = self.statistics()
        log_string = ""
        for key, value in stats.items():
            log_string += f"{key}: {value}\n"
        return log_string

    def __repr__(self) -> str:
        stats = self.statistics()
        log_string = ""
        for key, value in stats.items():
            log_string += f"{key}: {value}\n"
        return log_string


class CSVCorpus(Corpus):
    def __init__(
            self,
            dataset_path: str,
            label_column: int,
            sentence_column: int,
            sentence_pair_column: int = None,
            skip_header: bool = True,
            label_map: str = None,
            task_type: str = 'Sentence classification',
            dataset_name: str = 'Custom dataset',
            evaluation_metric: str = ['accuracy', 'f1_score'],
    ):
        if type(dataset_path) == str:
            dataset_path = Path(dataset_path)

        train_file = Path(dataset_path / "train.csv")
        dev_file = Path(dataset_path / "dev.csv")
        test_file = Path(dataset_path / "test.csv")

        assert train_file.exists() and dev_file.exists(), \
            f"No train.csv or dev.csv files found in the directory {dataset_path}."

        train_set = read_dataset_from_tsv(train_file,
                                          label_column=label_column,
                                          sentence_column=sentence_column,
                                          sentence_pair_column=sentence_pair_column,
                                          skip_header=skip_header,
                                          label_map=label_map,
                                          task_type=task_type)

        dev_set = read_dataset_from_tsv(dev_file,
                                        label_column=label_column,
                                        sentence_column=sentence_column,
                                        sentence_pair_column=sentence_pair_column,
                                        skip_header=skip_header,
                                        label_map=label_map,
                                        task_type=task_type)

        if test_file.exists():
            test_set = read_dataset_from_tsv(dev_file,
                                             label_column=label_column,
                                             sentence_column=sentence_column,
                                             sentence_pair_column=sentence_pair_column,
                                             skip_header=skip_header,
                                             label_map=label_map,
                                             task_type=task_type)
        else:
            test_set = TextDataset([])

        super(CSVCorpus, self).__init__(
            name=dataset_name,
            task_type=task_type,
            train=train_set,
            dev=dev_set,
            test=test_set,
            label_map=label_map,
            evaluation_metric=evaluation_metric,
        )


def download_file(url: str, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)

    filename = re.sub(r".+/", "", url)
    cache_path = directory / filename

    # create temporary file
    fd, temp_filename = tempfile.mkstemp()

    # GET file object
    req = requests.get(url, stream=True)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    progress = tqdm(total=total)
    with open(temp_filename, "wb") as temp_file:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                temp_file.write(chunk)

    shutil.copyfile(temp_filename, str(cache_path))
    os.close(fd)
    os.remove(temp_filename)

    progress.close()


def unzip_file(zipped_data_path: Path, unzip_to: Path):
    if ".zip" in str(zipped_data_path):
        from zipfile import ZipFile
        with ZipFile(Path(zipped_data_path), "r") as zipObj:
            # extract all the contents of zip file in current directory
            zipObj.extractall(unzip_to)

    if ".gz" in str(zipped_data_path):
        import gzip
        with gzip.open(zipped_data_path, "rb") as f_in:
            with open(str(zipped_data_path).replace(".gz", ""), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    os.remove(str(zipped_data_path))


def read_dataset_from_tsv(
        file_path: Path,
        label_column: int,
        sentence_column: int,
        task_type: str,
        sentence_pair_column: int = None,
        skip_header: bool = True,
        label_map: Dict = None,
    ) -> TextDataset:

    sentences = []

    with open(file_path, mode="r") as tsv_file:
        if skip_header:
            next(tsv_file)

        for line in tsv_file:
            line = line.strip()
            row = line.split('\t')

            label = row[label_column]
            if sentence_pair_column is None:
                sentence_type = "single-sentence"
                text = row[sentence_column]
            else:
                sentence_type = "single-pair"
                text = [row[sentence_column], row[sentence_pair_column]]

            # try to convert labels to integers for classification tasks
            # try to convert labels to floats for regression tasks
            try:
                float(label)
            except ValueError:
                if label_map:
                    label = label_map[label]
                else:
                    raise ValueError(f'Please provide a label name map for this dataset: {file_path}')
            else:
                if 'regression' in task_type:
                    label = float(label)
                else:
                    label = int(label)

            sentence = Sentence(text=text,
                                true_label=label,
                                sentence_type=sentence_type)
            sentences.append(sentence)

    return TextDataset(sentences)


def read_conll_textfile(
        file_path: Path,
        label_map: Dict,
        word_position: int = 0,
        label_position: int = 3,
        delimiter: str = " ",
    ) -> TextDataset:

    sentences = []
    sentence_type = 'single-sentence'

    with open(file_path, mode='r') as txt_file:

        sentence = ""
        tags = []

        for i, line in enumerate(txt_file):
            line = line.strip()
            row = line.split(delimiter)

            if line in ["", "\n"]:
                # if sentence != "-DOCSTART- ": # ignore docstarts?
                sentence = Sentence(text=sentence,
                                    true_label=tags,
                                    sentence_type=sentence_type)
                sentences.append(sentence)
                sentence = ""
                tags = []
            else:
                word = row[word_position]
                label = row[label_position]
                label = label.split(",")[0]
                label = label_map[label]
                sentence += f"{word} "
                tags.append(label)
    return TextDataset(sentences)