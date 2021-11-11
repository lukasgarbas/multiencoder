# Can We Tune Together

Can we tune multiple language models together? There are many pre-trained transformer language models available on [HuggingFace model hub](https://huggingface.co/models). The current hype for sentence-level tasks is to pick one language model (i.e BERT, ELECTRA, deBERTa) and fine-tune it for the task at hand. Each LM has something different: either a different pretraining objective, a different corpus used for pretaining, or some other twists in the transformer architecture. When evaluating unalike models on the [GLUE benchmark](https://gluebenchmark.com/), they also score differently.

I couldn't find any works that combine multiple LMs and tune them together — so I'm running some experiments here! 🤷‍♂️. 

The initial idea was to concatenate different CLS token representations of each LM and let the complete model (multi-encoder) figure out how to combine them. Later on, I came across some other works ([Dynamic Meta-Embeddings](https://arxiv.org/abs/1804.07983) for Sentence Representations by Kiela et. al.) which applied attention to combine static GloVe and FastText embeddings. Even though the previous work with static/frozen embeddings does not directly translate to language models, the attention mechanism can still help us visualize the attention scores for each LM and look at what is happening inside the multi-encoder.

My code is very much based on [FlairNLP](https://github.com/flairNLP/flair) (shout-out to them! Look it up, it's a cool NLP framework 🙃). If combining and tuning multiple LMs together happens to work, I might create a PR later.


# Datasets

<details>
  <summary>Datasets for experiments</summary>

#### Text Classification

| Corpus Name | Sentences | Labels | Task type | Source |
| ----------- | ---- | ---------- | --------- | --------- |
| CLICKBAIT   | 32k  | 2          | Sentence classification | [clickbait] |
| ISEAR       | 7k   | 7          | Sentence classification | [isear] #6 only in .sav and .mdb formats |
| TREC        | 6k   | 6          | Sentence classification | [trec] |
| EMOTION_STIMULUS | 1.6k  | 7    | Sentence classification | [emotion stimulus] |
| GLUE_COLA   | 9.5k  | 2         | Sentence classification | [glue] |
| GLUE_SST2   | 68k   | 2         | Sentence classification | [glue] |
| GLUE_MRPC   | 4k    | 3         | Sentence-pair           | [glue] |
| GLUE_RTE    | 3k    | 3         | Sentence-pair           | [glue] |
| GLUE_MNLI   | 413k  | 3         | Sentence-pair           | [glue] |
| GLUE_QNLI   | 114k  | 3         | Sentence-pair           | [glue] |
| GLUE_QQP    | 404k  | 3         | Sentence-pair           | [glue] |
| GLUE_WNLI   | 700   | 2         | Sentence-pair           | [glue] |
| SICK        | 10k   | 3         | Sentence-pair           | [sick] |


#### Text Regression

| Corpus Name | Sentences | Labels | Task type | Reference |
| ----------- | ---- | ---------- | --------- | --------- |
| GLUE_STSB   | 8.5k   | Similarity score | Sentence regression | [glue] |
| EMOBANK     | 10k   | Valance, arousal, or dominance scores | Sentence regression | [emobank] |
| FB_VALENCE_AROUSAL | 6k   | Valence or arousal scores | Sentence regression | [valence arousal in fb posts] |


#### Sequence Labeling

| Corpus Name | Sentences | Labels | Task type | Reference |
| ----------- | ---- | ---------- | --------- | --------- |
| CONLL_NER   | 20k   | 4 | NER | [conll03] |
| WNUT_NER     | 5k   | 6 | NER | [wnut17] |
| MIT_MOVIE_NER | 10k   | 13 | NER | [mit movie] |
| MIT_RESTAURANT_NER | 9k   | 9 | NER | [mit restaurant] |


[clickbait]: https://github.com/bhargaviparanjape/clickbait
[isear]: https://www.unige.ch/cisa/research/materials-and-online-research/research-material/
[trec]: https://cogcomp.seas.upenn.edu/Data/QA/QC/
[emotion stimulus]: https://www.eecs.uottawa.ca/~diana/resources/emotion_stimulus_data/
[glue]: https://gluebenchmark.com/tasks
[sick]: https://zenodo.org/record/2787612/#.YYSThnko-qQ
[emobank]: https://github.com/JULIELab/EmoBank
[valence arousal in fb posts]: https://github.com/wwbp/additional_data_sets/tree/master/valence_arousal
[conll03]: https://www.clips.uantwerpen.be/conll2003/ner/
[wnut17]: https://github.com/leondz/emerging_entities_17
[mit movie]: https://groups.csail.mit.edu/sls/downloads/movie/
[mit restaurant]: https://groups.csail.mit.edu/sls/downloads/restaurant/

</details>

Tasks: Text Classification ✅ Text Regression ✅ Sequence Labeling (still working on SequenceTagger).

# Tuning a single Language Model

You can pick any transformer-based language model from the model hub and tune it for GLUE tasks:

```python
from datasets import GLUE_COLA
from encoders import LanguageModel
from task_heads import TextClassifier
from trainer import ModelTrainer

# 1. load any GLUE corpus (e.g. Corpus of Linguistic Acceptability)
corpus = GLUE_COLA()

print(corpus)

# 2. pick a language model
language_model = LanguageModel('google/electra-base-discriminator')

# 3. use classification or regression head
classifier = TextClassifier(encoder=language_model,
                            num_classes=corpus.num_classes)

# 4. create model trainer
trainer = ModelTrainer(model=classifier,
                      corpus=corpus)

# 5. start training
trainer.train(learning_rate=2e-5,
              batch_size=16,
              epochs=10,
              shuffle_data=True)
```

- Electra base scores 67.7 ± 1.2 (Matthews correlation coefficient) for CoLA dev set. You can look at the scores provided by the authors here: [expected electra results](https://github.com/google-research/electra).
- Roberta base scores 62.0 ± 1.3 [expected roberta results](https://github.com/pytorch/fairseq/tree/master/examples/roberta).
- SpanBERT scores 57.2 ± 1.0
- Large models need much smaller learning rates e.g. 5e-6

# Combining Language Models

```python
from datasets import GLUE_COLA
from encoders import LanguageModel, MultiEncoder
from task_heads import TextClassifier
from trainer import ModelTrainer

# 1. load any GLUE corpus (i.e. Corpus of Linguistic Acceptability)
corpus = GLUE_COLA()

print(corpus)

# 2. pick some language models
language_models = [
    LanguageModel('google/electra-base-discriminator'),
    LanguageModel('roberta-base'),
]

# 3. create multi-encoder and choose the combine method: 'dme' or 'concat'
multi_encoder = MultiEncoder(language_models=language_models,
                            combine_method='dme')

# 4. use classification or regression head
classifier = TextClassifier(encoder=multi_encoder,
                            num_classes=corpus.num_classes)

# 5. create model trainer
trainer = ModelTrainer(model=classifier,
                       corpus=corpus)

# 6. start training
trainer.train(learning_rate=2e-5,
              batch_size=16,
              epochs=10,
              shuffle_data=True)
```

- We can increase the score of Electra if we add Roberta and tune them together. The average (5 runs) Mcc score is 68.6.
- Concatenation scores a bit better than DME `MultiEncoder(combine_method="concat")`. Expected difference is ↑ 0.1 compared to DME.
- The increase in scores is still very minor. Most of the time it's in the range of standard deviation of Electra.
- You can pick a more stable dataset where the difference between runs is much smaller (e.g. GLUE_STSB regression task has stdev of 0.2). Combining [Electra](https://huggingface.co/google/electra-base-generator) with [Ernie](nghuyong/ernie-2.0-en) scores 91.6 Spearman's rank (↑ 0.5).

## Looking at attention weights

If you use DME as combine method, you can embedd a few sentences from CoLA dev set and look at the attention scores for each language model:

```python
# pick a few sentences from CoLA dev set
sentences = [
  "The government's imposition of a fine.",
  "Jason happens to appear to seem to be sick.",
  "Somebody just left - guess who.",
]

# let's load the best model after training
model_path = "models/CoLA-multi-transformer-electra-base-discriminator" \
    "-roberta-base-classifier/best-model.pt"
model = TextClassifier.from_checkpoint(model_path)

# classify sentences and look at the attention scores of the multi-encoder
predictions = model.predict_tags(sentences, corpus.label_map)

print(predictions)

"""
{'Sentences': 
     ["The government's imposition of a fine.", 
      "Somebody just left - guess who.",
      "He can will go"], 
 'DME scores': 
     [[0.7901, 0.2099],
      [0.8116, 0.1884],
      [0.6635, 0.3365]],
 'Predictions': 
     ['acceptable', 'acceptable', 'not_acceptable']}
"""

```

DME scores show how much weight does the multi-encoder assign to each cls representation when embedding a given sentence. Electra is a stronger model than roberta but roberta also adds something to the meta-embedding. We might find sentences where roberta gets more weight than electra.

```python
# if you did not save the model to file, you can predict with the best model directly from trainer
predictions = trainer.best_model.predict_tags(sentences, corpus.label_map)
```

## Combining more than two models

You can throw even more LMs to the mix:

```python
# 2. mix some language models
language_models = [
    LanguageModel('google/electra-base-discriminator'),
    LanguageModel('roberta-base'),
    LanguageModel('SpanBERT/spanbert-base-cased'),
]

# 3. create multi-encoder and choose the combine method: 'dme' or 'concat'
multi_encoder = MultiEncoder(
    language_models=language_models,
)
```

This scored 69.3. I tried it only once so the score might differ after taking the average of multiple runs. Combining a lot of language models suffers from overfitting and needs to be further analysed 🤷‍♂️.

## Notes on joint tuning and why it sometimes doesn't work

This approach is very sensitive to overfitting.
- There are cases where one language model can fit the training data much faster than others. DME scores show that some language models can simply be ignored in the classifier layer. One option would be to play with hidden dropout parameters in LMs: `LanguageModel('roberta-base', hidden_dropout_prob=0.4)`.
- You can also attach different learning rates for each language model: `trainer.train(learning_rate=[3e-5, 1e-5])`
- I am currently experimenting with larger learning rates just for linear decoder layer: `trainer.train(learning_rate=2e-5, decoder_learning_rate=1e-3)`. More examples are coming ✌️


