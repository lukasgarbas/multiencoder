## Text Classification tasks

TODO: add an example. No noticeable improvement for any tasks so far.

## Text Regression tasks

Here's one example where joint tuning works:

```python
from datasets import GLUE_STSB
from encoders import LanguageModel, MultiEncoder
from task_heads import TextRegressor
from trainer import ModelTrainer

# 1. load any GLUE corpus (i.e. Corpus of Linguistic Acceptability)
corpus = GLUE_STSB()

print(corpus)

# 2. pick some language models
language_models = [
    LanguageModel('google/electra-base-discriminator'),
    LanguageModel('nghuyong/ernie-2.0-en'),
]

# 3. create multi-encoder and choose the combine method: 'dme' or 'concat'
multi_encoder = MultiEncoder(language_models=language_models,
                             combine_method='concat')

# 4. use classification or regression head
classifier = TextRegressor(encoder=multi_encoder,
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

- Single Electra scores 91.2 Spearman's rank on dev set [expected electra results](https://github.com/google-research/electra).
- Combination of ELECTRA + ERNIE scores 91.6.
- Adding RoBERTa to the mix improves the score up to 91.8.
- Adding a 4th model does not improve further 😅.


## Named Entity Recognition

TODO: run more experiments.