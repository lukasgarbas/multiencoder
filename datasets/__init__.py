# NER Named Entity Recognition datasets
from .sequence_labeling import CONLL_NER
from .sequence_labeling import WNUT_NER
from .sequence_labeling import MIT_MOVIE_NER
from .sequence_labeling import MIT_RESTAURANT_NER

# datasets in GLUE benchmark
from .glue_benchmark import GLUE_COLA
from .glue_benchmark import GLUE_SST2
from .glue_benchmark import GLUE_MRPC
from .glue_benchmark import GLUE_MNLI
from .glue_benchmark import GLUE_QNLI
from .glue_benchmark import GLUE_QQP
from .glue_benchmark import GLUE_STSB
from .glue_benchmark import GLUE_WNLI
from .glue_benchmark import GLUE_RTE

# additional text regression datasets
from .text_regression import EMOBANK
from .text_regression import FB_VALENCE_AROUSAL

# additional text classification datasets
from .text_classification import SICK
from .text_classification import CLICKBAIT
from .text_classification import ISEAR
from .text_classification import TREC
from .text_classification import EMOTION_STIMULUS

