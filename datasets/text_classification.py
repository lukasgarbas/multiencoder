from utils import Corpus, CSVCorpus
from utils import download_file, unzip_file
from utils import read_dataset_from_tsv
from pathlib import Path
import random
import os
import csv
import re

from typing import Union, List


class ISEAR(Corpus):
    def __init__(
        self,
        base_path: str = "dataset_cache",
        evaluation_metric: Union[List[str], str] = "accuracy"
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        dataset_name = "isear"
        data_folder = base_path / dataset_name

        data_file = data_folder / "train.tsv"

        label_map = {"joy": 0, "anger": 1, "fear": 2, "disgust": 3, "shame": 4, "guilt": 5, "sadness": 6}

        # if data is not downloaded yet, download it
        if not data_file.is_file():
            print("Downloading ISEAR corpus...")

            # get the isear.csv file
            download_file(
                "https://raw.githubusercontent.com/lukasgarbas/nlp-text-emotion/master/data/datasets/isear.csv",
                base_path / dataset_name
            )
            
            file_path = Path(base_path / "isear/isear.csv")

            original_isear = []
            with open(file_path, mode="r") as isear_csv:
                isear_reader = csv.reader(isear_csv, delimiter=",")
                next(isear_reader)
                for line in isear_reader:
                    original_isear.append((line[0], line[1]))

            # create dev file
            dev = []
            dev_instances_per_label = 180
            label_counter = [0] * len(label_map)
            for isear_instance in original_isear:
                label = isear_instance[0]
                if label_counter[label_map[label]] < dev_instances_per_label:
                    label_counter[label_map[label]] += 1
                    dev.append(isear_instance)

            train = [isear_instance for isear_instance in original_isear if isear_instance not in dev]
            
            with open(data_folder / "train.tsv", mode="w") as train_file:
                for instance in train:
                    train_file.write(f'{instance[0]}\t{instance[1]}\n')

            with open(data_folder / "dev.tsv", mode="w") as train_file:
                for instance in dev:
                    train_file.write(f"{instance[0]}\t{instance[1]}\n")

            os.remove(file_path)
            print(f"The corpus is stored in {data_folder} data folder.")

        train_file = Path(data_folder / "train.tsv")
        dev_file = Path(data_folder / "dev.tsv")

        label_map = {"joy": 0, "anger": 1, "fear": 2, "disgust": 3, "shame": 4, "guilt": 5, "sadness": 6}

        task_type = "Sentence classification"

        train_set = read_dataset_from_tsv(train_file,
                                        label_column=0,
                                        sentence_column=1,
                                        skip_header=False,
                                        label_map=label_map,
                                        task_type=task_type)

        dev_set = read_dataset_from_tsv(dev_file,
                                        label_column=0,
                                        sentence_column=1,
                                        skip_header=False,
                                        label_map=label_map,
                                        task_type=task_type)

        super(ISEAR, self).__init__(
            name="ISEAR Emotion Classification",
            task_type=task_type,
            train=train_set,
            dev=dev_set,
            label_map=label_map,
            evaluation_metric=evaluation_metric
        )


class TREC(Corpus):
    def __init__(
        self,
        base_path: str = "dataset_cache",
        evaluation_metric: Union[List[str], str] = "accuracy"
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        dataset_name = "trec"
        data_folder = base_path / dataset_name

        data_file = data_folder / "train.tsv"

        label_map = {
            "abbreviation": 0,
            "entities": 1,
            "descriptions": 2,
            "human_beings": 3,
            "locations": 4,
            "numeric_values": 5
        }

        # if data is not downloaded yet, download it
        if not data_file.is_file():
            print("Downloading TREC corpus...")

            # get the train set
            download_file(
                "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label",
                base_path / dataset_name
            )
            
            # get test set
            download_file(
                "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label",
                base_path / dataset_name
            )

            old_filenames = ["train_5500.label", "TREC_10.label"]
            new_filenames = ["train.tsv", "dev.tsv"]
            
            label_names = {
                "ABBR": "abbreviation",
                "ENTY": "entities",
                "DESC": "descriptions",
                "HUM": "human_beings",
                "LOC": "locations",
                "NUM": "numeric_values"
            }
            
            for new_name, old_name in zip(new_filenames, old_filenames):
                samples = []
                with open(data_folder / old_name, mode="r", encoding = "ISO-8859-1") as original_file:
                    for line in original_file.readlines():
                        line = line.split(" ")
                        label = line[0].split(":")[0]
                        sentence = ' '.join(line[1:])
                        samples.append((label_names[label], sentence))
                
                with open(data_folder / new_name, mode="w") as new_file:
                    for sample in samples:
                        new_file.write(f"{sample[0]}\t{sample[1]}")

            os.remove(data_folder / old_filenames[0])
            os.remove(data_folder / old_filenames[1])
            print(f"The corpus is stored in {data_folder} data folder.")

        train_file = Path(data_folder / "train.tsv")
        dev_file = Path(data_folder / "dev.tsv")

        task_type = "Sentence classification"

        train_set = read_dataset_from_tsv(train_file,
                                        label_column=0,
                                        sentence_column=1,
                                        skip_header=False,
                                        label_map=label_map,
                                        task_type=task_type)

        dev_set = read_dataset_from_tsv(dev_file,
                                        label_column=0,
                                        sentence_column=1,
                                        skip_header=False,
                                        label_map=label_map,
                                        task_type=task_type)

        super(TREC, self).__init__(
            name="TREC Question Classification",
            task_type=task_type,
            train=train_set,
            dev=dev_set,
            label_map=label_map,
            evaluation_metric=evaluation_metric
        )


class SICK(Corpus):
    def __init__(
        self,
        base_path: str = "dataset_cache",
        evaluation_metric: Union[List[str], str] = "accuracy"
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        dataset_name = "sick"
        data_folder = base_path / dataset_name

        data_file = data_folder / "train.tsv"

        label_map = {
            "NEUTRAL": 0,
            "ENTAILMENT": 1,
            "CONTRADICTION": 2,
        }

        # if data is not downloaded yet, download it
        if not data_file.is_file():
            print("Downloading SICK corpus...")

            # get the zip file
            download_file(
                "https://zenodo.org/record/2787612/files/SICK.zip",
                base_path / dataset_name
            )

            unzip_file(base_path / dataset_name / "SICK.zip",
                       data_folder)

            samples = []
            with open(data_folder / "SICK.txt", mode="r") as original_file:
                sick_reader = csv.reader(original_file, delimiter='\t')
                next(sick_reader)
                for line in sick_reader:
                    samples.append((line[1], line[2], line[3]))

            dev = []
            dev_samples_per_label = 320
            label_counter = [0] * len(label_map)
            for sample in samples:
                label = sample[2]
                if label_counter[label_map[label]] <= dev_samples_per_label:
                    label_counter[label_map[label]] += 1
                    dev.append(sample)

            train = [sample for sample in samples if sample not in dev]

            with open(data_folder / "dev.tsv", mode="w") as new_file:
                for sample in dev:
                    new_file.write(f"{sample[2]}\t{sample[0]}\t{sample[1]}\n")

            with open(data_folder / "train.tsv", mode="w") as new_file:
                for sample in train:
                    new_file.write(f"{sample[2]}\t{sample[0]}\t{sample[1]}\n")

            # rename test file to eval_dataset, since it has no labels
            os.remove(str(data_folder / "SICK.txt"))
            os.remove(str(data_folder / "readme.txt"))
            print(f"The corpus is stored in {data_folder} data folder.")

        task_type = "Sentence-pair classification"

        train_file = Path(data_folder / "train.tsv")
        dev_file = Path(data_folder / "dev.tsv")

        train_set = read_dataset_from_tsv(train_file,
                                        label_column=0,
                                        sentence_column=1,
                                        sentence_pair_column=2,
                                        skip_header=True,
                                        label_map=label_map,
                                        task_type=task_type)

        dev_set = read_dataset_from_tsv(dev_file,
                                        label_column=0,
                                        sentence_column=1,
                                        sentence_pair_column=2,
                                        skip_header=True,
                                        label_map=label_map,
                                        task_type=task_type)

        super(SICK, self).__init__(
            name="SICK Sentences Involving Compositional Knowledge (NLI)",
            task_type=task_type,
            train=train_set,
            dev=dev_set,
            label_map=label_map,
            evaluation_metric=evaluation_metric
        )


class CLICKBAIT(Corpus):
    def __init__(
        self,
        base_path: str = "dataset_cache",
        evaluation_metric: Union[List[str], str] = "accuracy"
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        dataset_name = "clickbait"
        data_folder = base_path / dataset_name

        data_file = data_folder / "train.tsv"

        label_map = {
            "clickbait": 0,
            "non clickbait": 1,
        }

        # if data is not downloaded yet, download it
        if not data_file.is_file():
            print("Downloading CLICKBAIT corpus...")

            # get the data with clickbait sentences 
            download_file(
                "https://github.com/bhargaviparanjape/clickbait/raw/master/dataset/clickbait_data.gz",
                base_path / dataset_name
            )

            # get the data with non clickbait sentences 
            download_file(
                "https://github.com/bhargaviparanjape/clickbait/raw/master/dataset/non_clickbait_data.gz",
                base_path / dataset_name
            )

            unzip_file(base_path / dataset_name / "clickbait_data.gz", data_folder)
            unzip_file(base_path / dataset_name / "non_clickbait_data.gz", data_folder)

            samples = []
            with open(data_folder / "clickbait_data", mode='r') as original_file:
                clickbait_reader = csv.reader(original_file, delimiter='\t')
                for line in clickbait_reader:
                    if line:
                        samples.append(("clickbait", line[0]))

            with open(data_folder / "non_clickbait_data", mode='r') as original_file:
                clickbait_reader = csv.reader(original_file, delimiter='\t')
                for line in clickbait_reader:
                    if line:
                        samples.append(("non clickbait", line[0]))

            dev = []
            dev_samples_per_label = 2000
            label_counter = [0] * len(label_map)
            for sample in samples:
                label = sample[0]
                if label_counter[label_map[label]] < dev_samples_per_label:
                    label_counter[label_map[label]] += 1
                    dev.append(sample)

            train = [sample for sample in samples if sample not in dev]  
            random.shuffle(dev)
            random.shuffle(train)

            with open(data_folder / "dev.tsv", mode="w") as new_file:
                for sample in dev:
                    new_file.write(f"{sample[0]}\t{sample[1]}\n")

            with open(data_folder / "train.tsv", mode="w") as new_file:
                for sample in train:
                    new_file.write(f"{sample[0]}\t{sample[1]}\n")

            # rename test file to eval_dataset, since it has no labels
            os.remove(str(data_folder / "clickbait_data"))
            os.remove(str(data_folder / "non_clickbait_data"))
            print(f"The corpus is stored in {data_folder} data folder.")

        task_type = "Sentence classification"

        train_file = Path(data_folder / "train.tsv")
        dev_file = Path(data_folder / "dev.tsv")

        train_set = read_dataset_from_tsv(train_file,
                                          label_column=0,
                                          sentence_column=1,
                                          skip_header=False,
                                          label_map=label_map,
                                          task_type=task_type)

        dev_set = read_dataset_from_tsv(dev_file,
                                        label_column=0,
                                        sentence_column=1,
                                        skip_header=False,
                                        label_map=label_map,
                                        task_type=task_type)

        super(CLICKBAIT, self).__init__(
            name="CLICKBAIT Detecting Clickbaits in Online News Media",
            task_type=task_type,
            train=train_set,
            dev=dev_set,
            label_map=label_map,
            evaluation_metric=evaluation_metric
        )


class EMOTION_STIMULUS(Corpus):
    def __init__(
        self,
        base_path: str = "dataset_cache",
        evaluation_metric: Union[List[str], str] = ["accuracy", "f1_score"]
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        dataset_name = "emotion_stimulus"
        data_folder = base_path / dataset_name

        data_file = data_folder / "train.tsv"

        label_map = {"happy": 0, "sad": 1, "surprise": 2,
                     "disgust": 3, "anger": 4, "fear": 5, "shame": 6}

        # if data is not downloaded yet, download it
        if not data_file.is_file():
            print("Downloading EMOTION STIMULUS corpus...")

            # get the data with clickbait sentences
            download_file(
                "https://www.eecs.uottawa.ca/~diana/resources/emotion_stimulus_data/Dataset.zip",
                base_path / dataset_name
            )

            unzip_file(base_path / dataset_name / "Dataset.zip", data_folder)
            file_path = f"{data_folder}/Dataset/No Cause.txt"

            original_emo_stimulus = []
            with open(file_path, mode="r") as emo_stimulus:
                for line in emo_stimulus:
                    line = line.strip()
                    label = re.findall('<.*?>', line)[0]
                    label = label.replace('<', '').replace('>', '')
                    sentence = re.sub('<.*?>', '', line)
                    original_emo_stimulus.append((label, sentence))

            # create dev file
            dev = []
            dev_instances_per_label = 30
            label_counter = [0] * len(label_map)
            for isear_instance in original_emo_stimulus:
                label = isear_instance[0]
                if label_counter[label_map[label]] < dev_instances_per_label:
                    label_counter[label_map[label]] += 1
                    dev.append(isear_instance)

            train = [sample for sample in original_emo_stimulus if sample not in dev]
            random.shuffle(dev)
            random.shuffle(train)

            with open(data_folder / "train.tsv", mode="w") as train_file:
                for instance in train:
                    train_file.write(f"{instance[0]}\t{instance[1]}\n")

            with open(data_folder / "dev.tsv", mode="w") as train_file:
                for instance in dev:
                    train_file.write(f"{instance[0]}\t{instance[1]}\n")

            import shutil
            shutil.rmtree("dataset_cache/emotion_stimulus/Dataset")
            print(f"The corpus is stored in {data_folder} data folder.")

        train_file = Path(data_folder / "train.tsv")
        dev_file = Path(data_folder / "dev.tsv")

        task_type = "Sentence classification"

        train_set = read_dataset_from_tsv(train_file,
                                          label_column=0,
                                          sentence_column=1,
                                          skip_header=False,
                                          label_map=label_map,
                                          task_type=task_type)

        dev_set = read_dataset_from_tsv(dev_file,
                                        label_column=0,
                                        sentence_column=1,
                                        skip_header=False,
                                        label_map=label_map,
                                        task_type=task_type)

        super(EMOTION_STIMULUS, self).__init__(
            name="EMOTION STIMULUS 7-way emotion classification",
            task_type=task_type,
            train=train_set,
            dev=dev_set,
            label_map=label_map,
            evaluation_metric=evaluation_metric
        )