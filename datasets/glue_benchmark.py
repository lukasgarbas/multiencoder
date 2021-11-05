from utils import Corpus
from utils import download_file, unzip_file
from utils import read_dataset_from_tsv
from pathlib import Path
import os

from typing import Union, List


class GLUE_COLA(Corpus):
    def __init__(
        self,
        base_path: str = "dataset_cache",
        evaluation_metric: Union[List[str], str] = ["matthews_corr", "accuracy"]
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        dataset_name = "glue"
        data_folder = base_path / dataset_name

        data_file = data_folder / "CoLA/train.tsv"

        # if data is not downloaded yet, download it
        if not data_file.is_file():
            print("Downloading CoLA corpus...")

            # get the zip file
            download_file("https://dl.fbaipublicfiles.com/glue/data/CoLA.zip",
                          Path("dataset_cache") / dataset_name)

            unzip_file(Path("dataset_cache") / dataset_name / "CoLA.zip",
                       data_folder)

            # rename test file to eval_dataset, since it has no labels
            os.rename(str(data_folder / "CoLA/test.tsv"), str(data_folder / "CoLA/eval_dataset.tsv"))

            print(f"The corpus is stored in {data_folder}/CoLA directory.")

        train_file = Path(data_folder / "CoLA/train.tsv")
        dev_file = Path(data_folder / "CoLA/dev.tsv")

        label_map = {"not_acceptable": 0, "acceptable": 1}
        task_type = "Sentence classification"

        train_set = read_dataset_from_tsv(train_file,
                                          label_column=1,
                                          sentence_column=3,
                                          skip_header=False,
                                          label_map=label_map,
                                          task_type=task_type)

        dev_set = read_dataset_from_tsv(dev_file,
                                        label_column=1,
                                        sentence_column=3,
                                        skip_header=False,
                                        label_map=label_map,
                                        task_type=task_type)

        super(GLUE_COLA, self).__init__(
            name="CoLA Corpus of Linguistic Acceptability",
            task_type=task_type,
            train=train_set,
            dev=dev_set,
            label_map=label_map,
            evaluation_metric=evaluation_metric
        )


class GLUE_SST2(Corpus):
    def __init__(
        self,
        base_path: str = "dataset_cache",
        evaluation_metric: Union[List[str], str] = "accuracy"
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        dataset_name = "glue"
        data_folder = base_path / dataset_name

        data_file = data_folder / "SST-2/train.tsv"

        # if data is not downloaded yet, download it
        if not data_file.is_file():
            print("Downloading SST-2 corpus...")

            # get the zip file
            download_file("https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
                          Path("dataset_cache") / dataset_name)

            unzip_file(Path("dataset_cache") / dataset_name / "SST-2.zip",
                       data_folder)

            # rename test file to eval_dataset, since it has no labels
            os.rename(str(data_folder / "SST-2/test.tsv"), str(data_folder / "SST-2/eval_dataset.tsv"))

            print(f"The corpus is stored in {data_folder}/SST-2 data folder.")

        train_file = Path(data_folder / "SST-2/train.tsv")
        dev_file = Path(data_folder / "SST-2/dev.tsv")

        label_map = {"negative": 0, "positive": 1}
        task_type="Sentence classification"

        train_set = read_dataset_from_tsv(train_file,
                                          label_column=1,
                                          sentence_column=0,
                                          skip_header=True,
                                          label_map=label_map,
                                          task_type=task_type)

        dev_set = read_dataset_from_tsv(dev_file,
                                        label_column=1,
                                        sentence_column=0,
                                        skip_header=True,
                                        label_map=label_map,
                                        task_type=task_type)

        super(GLUE_SST2, self).__init__(
            name="SST-2 Stanford Sentiment Treebank",
            task_type=task_type,
            train=train_set,
            dev=dev_set,
            label_map=label_map,
            evaluation_metric=evaluation_metric
        )


class GLUE_MRPC(Corpus):
    def __init__(
        self,
        base_path: str = "dataset_cache",
        evaluation_metric: Union[List[str], str] = ["accuracy", "f1_score"],
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        dataset_name = "glue"
        data_folder = base_path / dataset_name

        data_file = data_folder / "MRPC/train.tsv"

        # if data is not downloaded yet, download it
        if not data_file.is_file():
            print("Downloading MRPC corpus...")

            # we will download MRPC from another (unofficial) git repository
            # microsoft keeps this dataset in not so friendly .msi format
            download_file(
                "https://raw.githubusercontent.com/MegEngine/Models/master/official/nlp/bert/glue_data/MRPC/train.tsv",
                Path("dataset_cache") / dataset_name / "MRPC"
            )

            # get the dev set
            download_file(
                "https://raw.githubusercontent.com/MegEngine/Models/master/official/nlp/bert/glue_data/MRPC/dev.tsv",
                Path("dataset_cache") / dataset_name / "MRPC"
            )

            # get the test set
            download_file(
                "https://raw.githubusercontent.com/MegEngine/Models/master/official/nlp/bert/glue_data/MRPC/test.tsv",
                Path("dataset_cache") / dataset_name / "MRPC"
            )

            os.rename(str(data_folder / "MRPC/test.tsv"), str(data_folder / "MRPC/eval_dataset.tsv"))
            print(f"The corpus is stored in {data_folder}/MRPC data folder.")

        # Read the file and load train and dev sets into Dataset object
        train_file = Path(data_folder / "MRPC/train.tsv")
        dev_file = Path(data_folder / "MRPC/dev.tsv")

        label_map = {"not_paraphrase": 0, "paraphrase": 1}
        task_type = "Sentence-pair classification"

        train_set = read_dataset_from_tsv(train_file,
                                          label_column=0,
                                          sentence_column=3,
                                          sentence_pair_column=4,
                                          skip_header=True,
                                          label_map=label_map,
                                          task_type=task_type)

        dev_set = read_dataset_from_tsv(dev_file,
                                        label_column=0,
                                        sentence_column=3,
                                        sentence_pair_column=4,
                                        skip_header=True,
                                        label_map=label_map,
                                        task_type=task_type)

        super(GLUE_MRPC, self).__init__(
            name="MRPC Microsoft Research Paraphrase Corpus",
            task_type=task_type,
            train=train_set,
            dev=dev_set,
            label_map=label_map,
            evaluation_metric=evaluation_metric
        )


class GLUE_STSB(Corpus):
    def __init__(
        self,
        base_path: str = "dataset_cache",
        evaluation_metric: Union[List[str], str] = ["pearsonr", "spearmanr"],
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        dataset_name = "glue"

        if not base_path:
            base_path = dataset_name / "dataset_cache"
        data_folder = base_path / dataset_name

        data_file = data_folder / "STS-B/train.tsv"

        # if data is not downloaded yet, download it
        if not data_file.is_file():
            print("Downloading STS-B corpus...")

            # get the zip file
            download_file("https://dl.fbaipublicfiles.com/glue/data/STS-B.zip",
                          Path("dataset_cache") / dataset_name)

            unzip_file(Path("dataset_cache") / dataset_name / "STS-B.zip",
                       data_folder)

            # rename test file to eval_dataset, since it has no labels
            os.rename(str(data_folder / "STS-B/test.tsv"), str(data_folder / "STS-B/eval_dataset.tsv"))

            print(f"The corpus is stored in {data_folder}/STS-B data folder.")

        train_file = Path(data_folder / "STS-B/train.tsv")
        dev_file = Path(data_folder / "STS-B/dev.tsv")

        task_type = "Sentence-pair regression"

        train_set = read_dataset_from_tsv(train_file,
                                          label_column=9,
                                          sentence_column=7,
                                          sentence_pair_column=8,
                                          skip_header=True,
                                          task_type=task_type)

        dev_set = read_dataset_from_tsv(dev_file,
                                        label_column=9,
                                        sentence_column=7,
                                        sentence_pair_column=8,
                                        skip_header=True,
                                        task_type=task_type)

        super(GLUE_STSB, self).__init__(
            name="STS-B Semantic Textual Similarity Benchmark",
            task_type=task_type,
            train=train_set,
            dev=dev_set,
            evaluation_metric=evaluation_metric
        )


class GLUE_QQP(Corpus):
    def __init__(
        self,
        base_path: str = "dataset_cache",
        evaluation_metric: Union[List[str], str] = ["accuracy", "f1_score"]
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        dataset_name = "glue"

        if not base_path:
            base_path = dataset_name / "dataset_cache"
        data_folder = base_path / dataset_name

        data_file = data_folder / "QQP/train.tsv"

        # if data is not downloaded yet, download it
        if not data_file.is_file():
            print("Downloading QQP corpus...")

            # get the zip file
            download_file(
                "https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip",
                Path("dataset_cache") / dataset_name
            )

            unzip_file(Path("dataset_cache") / dataset_name / "QQP-clean.zip",
                       data_folder)

            # rename test file to eval_dataset, since it has no labels
            os.rename(str(data_folder / "QQP/test.tsv"), str(data_folder / "QQP/eval_dataset.tsv"))

            print(f"The corpus is stored in {data_folder}/QQP data folder.")

        # Read the file and load train and dev sets into Dataset object
        train_file = Path(data_folder / "QQP/train.tsv")
        dev_file = Path(data_folder / "QQP/dev.tsv")

        label_map = {"not_paraphrase": 0, "paraphrase": 1}
        task_type = "Sentence-pair classification"

        train_set = read_dataset_from_tsv(train_file,
                                          label_column=5,
                                          sentence_column=3,
                                          sentence_pair_column=4,
                                          skip_header=True,
                                          label_map=label_map,
                                          task_type=task_type)

        dev_set = read_dataset_from_tsv(dev_file,
                                        label_column=5,
                                        sentence_column=3,
                                        sentence_pair_column=4,
                                        skip_header=True,
                                        label_map=label_map,
                                        task_type=task_type)

        super(GLUE_QQP, self).__init__(
            name="QQP Quora Question Pairs",
            task_type=task_type,
            train=train_set,
            dev=dev_set,
            label_map=label_map,
            evaluation_metric=evaluation_metric
        )


class GLUE_MNLI(Corpus):
    def __init__(
        self,
        base_path: str = "dataset_cache",
        evaluate_on_matched=True,
        evaluation_metric: Union[List[str], str] = "accuracy"
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        dataset_name = "glue"

        if not base_path:
            base_path = dataset_name / "dataset_cache"
        data_folder = base_path / dataset_name

        data_file = data_folder / "MNLI/train.tsv"

        # if data is not downloaded yet, download it
        if not data_file.is_file():
            print("Downloading MNLI corpus...")

            # get the zip file
            download_file(
                "https://dl.fbaipublicfiles.com/glue/data/MNLI.zip",
                Path("dataset_cache") / dataset_name
            )

            unzip_file(Path("dataset_cache") / dataset_name / "MNLI.zip",
                       data_folder)

            # rename test file eval_dataset, same as in other GLUE datasets
            os.rename(str(data_folder / "MNLI/test_matched.tsv"), str(data_folder / "MNLI/eval_dataset_matched.tsv"))
            os.rename(str(data_folder / "MNLI/test_mismatched.tsv"), str(data_folder / "MNLI/eval_dataset_mismatched.tsv"))

            print(f"The corpus is stored in {data_folder}/MNLI data folder.")

        # Read the file and load train and dev sets into Dataset object
        train_file = Path(data_folder / "MNLI/train.tsv")

        dev_set_name = "dev_matched.tsv" if evaluate_on_matched else "dev_mismatched.tsv"
        dev_file = Path(data_folder / "MNLI" / dev_set_name)

        label_map = {"contradiction": 0, "entailment": 1, "neutral": 2}
        task_type = "Sentence-pair classification"

        train_set = read_dataset_from_tsv(train_file,
                                          label_column=10,
                                          sentence_column=8,
                                          sentence_pair_column=9,
                                          skip_header=True,
                                          label_map=label_map,
                                          task_type=task_type)

        dev_set = read_dataset_from_tsv(dev_file,
                                        label_column=10,
                                        sentence_column=8,
                                        sentence_pair_column=9,
                                        skip_header=True,
                                        label_map=label_map,
                                        task_type=task_type)

        super(GLUE_MNLI, self).__init__(
            name="MNLI Multi-Genre Natural Language Inference",
            task_type=task_type,
            train=train_set,
            dev=dev_set,
            label_map=label_map,
            evaluation_metric=evaluation_metric
        )


class GLUE_QNLI(Corpus):
    def __init__(
        self,
        base_path: str = "dataset_cache",
        evaluation_metric: Union[List[str], str] = "accuracy"
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        dataset_name = "glue"
        data_folder = base_path / dataset_name

        data_file = data_folder / "QNLI/train.tsv"

        # if data is not downloaded yet, download it
        if not data_file.is_file():
            print("Downloading QNLI corpus...")

            # get the zip file
            download_file(
                "https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip",
                Path("dataset_cache") / dataset_name
            )

            os.rename(str(data_folder / "QNLIv2.zip"), str(data_folder / "QNLI.zip"))

            unzip_file(Path("dataset_cache") / dataset_name / "QNLI.zip",
                       data_folder)

            # rename test file to eval_dataset, since it has no labels
            os.rename(str(data_folder / "QNLI/test.tsv"), str(data_folder / "QNLI/eval_dataset.tsv"))

            print(f"The corpus is stored in {data_folder}/QNLI data folder.")

        # Read the file and load train and dev sets into Dataset object
        train_file = Path(data_folder / "QNLI/train.tsv")
        dev_file = Path(data_folder / "QNLI/dev.tsv")

        label_map = {"not_entailment": 0, "entailment": 1}
        task_type = "Sentence-pair classification"

        train_set = read_dataset_from_tsv(train_file,
                                          label_column=3,
                                          sentence_column=1,
                                          sentence_pair_column=2,
                                          skip_header=True,
                                          label_map=label_map,
                                          task_type=task_type)

        dev_set = read_dataset_from_tsv(dev_file,
                                        label_column=3,
                                        sentence_column=1,
                                        sentence_pair_column=2,
                                        skip_header=True,
                                        label_map=label_map,
                                        task_type=task_type)

        super(GLUE_QNLI, self).__init__(
            name="QNLI Question-answering Natural Language Inference",
            task_type=task_type,
            train=train_set,
            dev=dev_set,
            label_map=label_map,
            evaluation_metric=evaluation_metric
        )


class GLUE_RTE(Corpus):
    def __init__(
        self,
        base_path: str = "dataset_cache",
        evaluation_metric: Union[List[str], str] = "accuracy"
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        dataset_name = "glue"
        data_folder = base_path / dataset_name

        data_file = data_folder / "RTE/train.tsv"

        # if data is not downloaded yet, download it
        if not data_file.is_file():
            print("Downloading RTE corpus...")

            # get the zip file
            download_file(
                "https://dl.fbaipublicfiles.com/glue/data/RTE.zip",
                Path("dataset_cache") / dataset_name
            )

            unzip_file(Path("dataset_cache") / dataset_name / "RTE.zip",
                       data_folder)

            # rename test file to eval_dataset, since it has no labels
            os.rename(str(data_folder / "RTE/test.tsv"), str(data_folder / "RTE/eval_dataset.tsv"))

            print(f"The corpus is stored in {data_folder}/RTE data folder.")

        train_file = Path(data_folder / "RTE/train.tsv")
        dev_file = Path(data_folder / "RTE/dev.tsv")

        label_map = {"not_entailment": 0, "entailment": 1}
        task_type = "Sentence-pair classification"

        train_set = read_dataset_from_tsv(train_file,
                                          label_column=3,
                                          sentence_column=1,
                                          sentence_pair_column=2,
                                          skip_header=True,
                                          label_map=label_map,
                                          task_type=task_type)

        dev_set = read_dataset_from_tsv(dev_file,
                                        label_column=3,
                                        sentence_column=1,
                                        sentence_pair_column=2,
                                        skip_header=True,
                                        label_map=label_map,
                                        task_type=task_type)

        super(GLUE_RTE, self).__init__(
            name="RTE Recognizing Textual Entailment",
            task_type=task_type,
            train=train_set,
            dev=dev_set,
            label_map=label_map,
            evaluation_metric=evaluation_metric
        )


class GLUE_WNLI(Corpus):
    def __init__(
        self,
        base_path: str = "dataset_cache",
        evaluation_metric: Union[List[str], str] = "accuracy"
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        dataset_name = "glue"
        data_folder = base_path / dataset_name

        data_file = data_folder / "WNLI/train.tsv"

        # if data is not downloaded yet, download it
        if not data_file.is_file():
            print("Downloading WNLI corpus...")

            # get the zip file
            download_file(
                "https://dl.fbaipublicfiles.com/glue/data/WNLI.zip",
                Path("dataset_cache") / dataset_name
            )

            unzip_file(Path("dataset_cache") / dataset_name / "WNLI.zip",
                       data_folder)

            # rename test file to eval_dataset, since it has no labels
            os.rename(str(data_folder / "WNLI/test.tsv"), str(data_folder / "WNLI/eval_dataset.tsv"))

            print(f"The corpus is stored in {data_folder}/WNLI data folder.")

        train_file = Path(data_folder / "WNLI/train.tsv")
        dev_file = Path(data_folder / "WNLI/dev.tsv")

        label_map = {"not_entailment": 0, "entailment": 1}
        task_type = "Sentence-pair classification"

        train_set = read_dataset_from_tsv(train_file,
                                          label_column=3,
                                          sentence_column=1,
                                          sentence_pair_column=2,
                                          skip_header=True,
                                          label_map=label_map,
                                          task_type=task_type)

        dev_set = read_dataset_from_tsv(dev_file,
                                        label_column=3,
                                        sentence_column=1,
                                        sentence_pair_column=2,
                                        skip_header=True,
                                        label_map=label_map,
                                        task_type=task_type)

        super(GLUE_WNLI, self).__init__(
            name="WNLI Winograd Schema Challenge",
            task_type=task_type,
            train=train_set,
            dev=dev_set,
            label_map=label_map,
            evaluation_metric=evaluation_metric
        )
