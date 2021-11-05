from utils import Corpus
from utils import download_file, unzip_file
from utils import read_dataset_from_tsv
from pathlib import Path
import csv
import os

from typing import Union, List


class EMOBANK(Corpus):
    def __init__(
        self,
        base_path: str = "dataset_cache",
        evaluation_metric: Union[List[str], str] = ["spearmanr", "pearsonr"],
        label_type: str = "valence",
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        assert label_type.lower() in ["valence", "arousal", "dominance"], \
            f"This dataset has three continous labels: \"valence\", \"arousal\" , and \"dominance\"."\
            f"Requested label \"{label_type}\" is not provided."

        dataset_name = "emobank"
        data_folder = base_path / dataset_name
        data_file = data_folder / "train.csv"

        # if data is not downloaded yet, download it
        if not data_file.is_file():
            print("Downloading EMOBANK corpus...")

            # get the zip file
            download_file("https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv",
                          Path("dataset_cache") / dataset_name)

            train, dev, test = [], [], []

            with open(data_folder / "emobank.csv", mode="r") as original_file:
                emobank_reader = csv.reader(original_file, delimiter=',')
                next(emobank_reader)

                for line in emobank_reader:
                    split = line[1]
                    valence_score = line[2]
                    arousal_score = line[3]
                    dominance_score = line[4]
                    sentence = line[5]

                    if split == "train":
                        train.append((sentence, valence_score, arousal_score, dominance_score))
                    if split == "dev":
                        dev.append((sentence, valence_score, arousal_score, dominance_score))
                    if split == "test":
                        test.append((sentence, valence_score, arousal_score, dominance_score))

            with open(data_folder / "train.csv", mode="w") as new_file:
                for sample in train:
                    new_file.write(f"{sample[0]}\t{sample[1]}\t{sample[2]}\t{sample[3]}\n")

            with open(data_folder / "dev.csv", mode="w") as new_file:
                for sample in dev:
                    new_file.write(f"{sample[0]}\t{sample[1]}\t{sample[2]}\t{sample[3]}\n")

            with open(data_folder / "test.csv", mode="w") as new_file:
                for sample in dev:
                    new_file.write(f"{sample[0]}\t{sample[1]}\t{sample[2]}\t{sample[3]}\n")

            os.remove(data_folder / "emobank.csv")
            print(f"The corpus is stored in {data_folder} directory.")

        train_file = Path(data_folder / "train.csv")
        dev_file = Path(data_folder / "dev.csv")
        test_file = Path(data_folder / "test.csv")

        task_type = "Sentence regression"
        
        label_columns = {"valence": 1, "arousal": 2, "dominance": 3}
        label_column = label_columns[label_type]

        train_set = read_dataset_from_tsv(train_file,
                                          label_column=label_column,
                                          sentence_column=0,
                                          skip_header=False,
                                          task_type=task_type)

        dev_set = read_dataset_from_tsv(dev_file,
                                        label_column=label_column,
                                        sentence_column=0,
                                        skip_header=False,
                                        task_type=task_type)

        test_set = read_dataset_from_tsv(test_file,
                                        label_column=label_column,
                                        sentence_column=0,
                                        skip_header=False,
                                        task_type=task_type)

        super(EMOBANK, self).__init__(
            name='EmoBank emotions according to Valence-Arousal-Dominance scheme',
            task_type=task_type,
            train=train_set,
            dev=dev_set,
            test=test_set,
            evaluation_metric=evaluation_metric
        )


class FB_VALENCE_AROUSAL(Corpus):
    def __init__(
        self,
        base_path: str = "dataset_cache",
        evaluation_metric: Union[List[str], str] = ["spearmanr", "pearsonr"],
        label_type: str = "valence",
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        assert label_type.lower() in ["valence", "arousal"], \
            f"This dataset has two continous labels: \"valence\" and \"arousal\"."\
            f"Requested label \"{label_type}\" is not provided."

        dataset_name = "fb_valence_arousal"
        data_folder = base_path / dataset_name
        data_file = data_folder / "train.csv"

        # if data is not downloaded yet, download it
        if not data_file.is_file():
            print("Downloading VALENCE AROUSAL corpus...")

            # get the zip file
            download_file("http://wwbp.org/downloads/public_data/dataset-fb-valence-arousal-anon.csv",
                          Path("dataset_cache") / dataset_name)

            samples = []
            with open(data_folder / "dataset-fb-valence-arousal-anon.csv", mode="r") as original_file:
                va_reader = csv.reader(original_file, delimiter=",")
                next(va_reader)
                for line in va_reader:
                    valence_score = line[1]
                    arousal_score = line[3]
                    sentence = line[0].replace("\t", "")
                    if sentence:
                        samples.append((sentence, valence_score, arousal_score))

            num_dev_samples = 800
            dev = samples[:num_dev_samples]
            train = samples[num_dev_samples:]

            with open(data_folder / "train.csv", mode="w") as new_file:
                for sample in train:
                    new_file.write(f"{sample[0]}\t{sample[1]}\t{sample[2]}\n")

            with open(data_folder / "dev.csv", mode="w") as new_file:
                for sample in dev:
                    new_file.write(f"{sample[0]}\t{sample[1]}\t{sample[2]}\n")

            print(f"The corpus is stored in {data_folder}directory.")

        train_file = Path(data_folder / "train.csv")
        dev_file = Path(data_folder / "dev.csv")

        task_type = "Sentence regression"

        valence_column, arousal_column = 1, 2
        label_column = valence_column if label_type == "valence" else arousal_column

        train_set = read_dataset_from_tsv(train_file,
                                          label_column=label_column,
                                          sentence_column=0,
                                          skip_header=False,
                                          task_type=task_type)

        dev_set = read_dataset_from_tsv(dev_file,
                                        label_column=label_column,
                                        sentence_column=0,
                                        skip_header=False,
                                        task_type=task_type)

        super(FB_VALENCE_AROUSAL, self).__init__(
            name="FBVA Predicting Valence and Arousal in Facebook Posts",
            task_type=task_type,
            train=train_set,
            dev=dev_set,
            evaluation_metric=evaluation_metric
        )
