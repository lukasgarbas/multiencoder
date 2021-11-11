from utils import Corpus
from utils import download_file, unzip_file
from utils import read_conll_textfile
from pathlib import Path
import os

from typing import Union, List


class CONLL_NER(Corpus):
    def __init__(
        self,
        base_path: str = "dataset_cache",
        evaluation_metric: Union[List[str], str] = "f1_score"
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        dataset_name = "conll2003"
        data_folder = base_path / dataset_name

        label_map = {
            "O": 0, 
            "B-PER": 1, "I-PER": 2,
            "B-ORG": 3, "I-ORG": 4,
            "B-LOC": 5, "I-LOC": 6,
            "B-MISC": 7, "I-MISC": 8,
        }

        data_file = data_folder / "train.txt"

        # if data is not downloaded yet, download it
        if not data_file.is_file():
            print("Downloading CONLL 2003 corpus...")

            download_file(
                "https://data.deepai.org/conll2003.zip",
                Path("dataset_cache") / dataset_name
            )

            unzip_file(Path("dataset_cache") / dataset_name / "conll2003.zip", data_folder)
            os.rename(str(data_folder / "valid.txt"), str(data_folder / "dev.txt"))
            print(f'The corpus is stored in {data_folder} data folder.')

        train_file = Path(data_folder / "train.txt")
        dev_file = Path(data_folder / "dev.txt")
        test_file = Path(data_folder / "test.txt")

        task_type = "Sequence labeling"

        train_set = read_conll_textfile(train_file,
                                        label_map=label_map)

        dev_set = read_conll_textfile(dev_file,
                                      label_map=label_map)

        test_set = read_conll_textfile(test_file,
                                      label_map=label_map)

        super(CONLL_NER, self).__init__(
            name="CONLL 4-way Named Entity Recognition",
            task_type=task_type,
            train=train_set,
            dev=dev_set,
            test=test_set,
            label_map=label_map,
            evaluation_metric=evaluation_metric
        )


class WNUT_NER(Corpus):
    def __init__(
        self,
        base_path: str = "dataset_cache",
        evaluation_metric: Union[List[str], str] = "f1_score"
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        dataset_name = "wnut17"
        data_folder = base_path / dataset_name

        label_map = {
            "O": 0, 
            "B-corporation": 1, "I-corporation": 2,
            "B-creative-work": 3, "I-creative-work": 4,
            "B-group": 5, "I-group": 6,
            "B-location": 7, "I-location": 8,
            "B-person": 9, "I-person": 10,
            "B-product": 11, "I-product": 12,
        }

        data_file = data_folder / "train.txt"

        # if data is not downloaded yet, download it
        if not data_file.is_file():
            print("Downloading WNUT 17 corpus...")
            
            download_file(
                'https://raw.githubusercontent.com/leondz/emerging_entities_17/master/wnut17train.conll',
                Path("dataset_cache") / dataset_name
            )

            download_file(
                'https://raw.githubusercontent.com/leondz/emerging_entities_17/master/emerging.dev.conll',
                Path("dataset_cache") / dataset_name
            )

            download_file(
                'https://raw.githubusercontent.com/leondz/emerging_entities_17/master/emerging.test.conll',
                Path("dataset_cache") / dataset_name
            )

            os.rename(str(data_folder / "wnut17train.conll"), str(data_folder / "train.txt"))
            os.rename(str(data_folder / "emerging.dev.conll"), str(data_folder / "dev.txt"))
            os.rename(str(data_folder / "emerging.test.conll"), str(data_folder / "test.txt"))
            print(f'The corpus is stored in {data_folder} data folder.')

        train_file = Path(data_folder / "train.txt")
        dev_file = Path(data_folder / "dev.txt")
        test_file = Path(data_folder / "test.txt")

        task_type = "Sequence labeling"

        train_set = read_conll_textfile(train_file,
                                        label_map=label_map,
                                        label_position=1,
                                        delimiter="\t")

        dev_set = read_conll_textfile(dev_file,
                                      label_map=label_map,
                                      label_position=1,
                                      delimiter="\t")

        test_set = read_conll_textfile(test_file,
                                      label_map=label_map,
                                      label_position=1,
                                      delimiter="\t")

        super(WNUT_NER, self).__init__(
            name="WNUT 17 Emerging Entities task",
            task_type=task_type,
            train=train_set,
            dev=dev_set,
            test=test_set,
            label_map=label_map,
            evaluation_metric=evaluation_metric
        )


class MIT_MOVIE_NER(Corpus):
    def __init__(
        self,
        base_path: str = "dataset_cache",
        evaluation_metric: Union[List[str], str] = ["accuracy", "f1_score"]
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        dataset_name = "mit_movie"
        data_folder = base_path / dataset_name

        label_map = {
            "O": 0, 
            "B-Actor": 1, "I-Actor": 2,
            "B-Award": 3, "I-Award": 4,
            "B-Character_Name": 5, "I-Character_Name": 6,
            "B-Director": 7, "I-Director": 8,
            "B-Genre": 9, "I-Genre": 10,
            "B-Opinion": 11, "I-Opinion": 12,
            "B-Origin": 13, "I-Origin": 14,
            "B-Plot": 15, "I-Plot": 16,
            "B-Quote": 17, "I-Quote": 18,
            "B-Relationship": 19, "I-Relationship": 20,
            "B-Soundtrack": 21, "I-Soundtrack": 22,
            "B-Year": 23, "I-Year": 24,
        }

        data_file = data_folder / "train.txt"

        # if data is not downloaded yet, download it
        if not data_file.is_file():
            print("Downloading MIT MOVIE NER corpus...")

            download_file(
                "https://groups.csail.mit.edu/sls/downloads/movie/trivia10k13train.bio",
                Path("dataset_cache") / dataset_name
            )

            download_file(
                "https://groups.csail.mit.edu/sls/downloads/movie/trivia10k13test.bio",
                Path("dataset_cache") / dataset_name
            )

            os.rename(str(data_folder / "trivia10k13train.bio"), str(data_folder / "train.txt"))
            os.rename(str(data_folder / "trivia10k13test.bio"), str(data_folder / "dev.txt"))
            print(f"The corpus is stored in {data_folder} data folder.")

        train_file = Path(data_folder / "train.txt")
        dev_file = Path(data_folder / "dev.txt")

        task_type = "Sequence labeling"

        train_set = read_conll_textfile(train_file,
                                        word_position=1,
                                        label_position=0,
                                        delimiter="\t",
                                        label_map=label_map)

        dev_set = read_conll_textfile(dev_file,
                                      word_position=1,
                                      label_position=0,
                                      delimiter="\t",
                                      label_map=label_map)

        super(MIT_MOVIE_NER, self).__init__(
            name="MIT MOVIE Named Entity Recognition",
            task_type=task_type,
            train=train_set,
            dev=dev_set,
            label_map=label_map,
            evaluation_metric=evaluation_metric
        )

class MIT_RESTAURANT_NER(Corpus):
    def __init__(
        self,
        base_path: str = "dataset_cache",
        evaluation_metric: Union[List[str], str] = ["accuracy", "f1_score"]
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        dataset_name = "mit_restaurant"
        data_folder = base_path / dataset_name

        label_map = {
            "O": 0, 
            "B-Amenity": 1, "I-Amenity": 2,
            "B-Cuisine": 3, "I-Cuisine": 4,
            "B-Dish": 5, "I-Dish": 6,
            "B-Hours": 7, "I-Hours": 8,
            "B-Location": 9, "I-Location": 10,
            "B-Price": 11, "I-Price": 12,
            "B-Rating": 13, "I-Rating": 14,
            "B-Restaurant_Name": 15, "I-Restaurant_Name": 16,
        }

        data_file = data_folder / "train.txt"

        # if data is not downloaded yet, download it
        if not data_file.is_file():
            print("Downloading MIT RESTAURANT NER corpus...")

            download_file(
                "https://groups.csail.mit.edu/sls/downloads/restaurant/restauranttrain.bio",
                Path("dataset_cache") / dataset_name
            )

            download_file(
                "https://groups.csail.mit.edu/sls/downloads/restaurant/restauranttest.bio",
                Path("dataset_cache") / dataset_name
            )

            os.rename(str(data_folder / "restauranttrain.bio"), str(data_folder / "train.txt"))
            os.rename(str(data_folder / "restauranttest.bio"), str(data_folder / "dev.txt"))
            print(f'The corpus is stored in {data_folder} data folder.')

        train_file = Path(data_folder / "train.txt")
        dev_file = Path(data_folder / "dev.txt")

        task_type = "Sequence labeling"

        train_set = read_conll_textfile(train_file,
                                        word_position=1,
                                        label_position=0,
                                        delimiter="\t",
                                        label_map=label_map)

        dev_set = read_conll_textfile(dev_file,
                                      word_position=1,
                                      label_position=0,
                                      delimiter="\t",
                                      label_map=label_map)

        super(MIT_RESTAURANT_NER, self).__init__(
            name="MIT RESTAURANT Named Entity Recognition",
            task_type=task_type,
            train=train_set,
            dev=dev_set,
            label_map=label_map,
            evaluation_metric=evaluation_metric
        )
