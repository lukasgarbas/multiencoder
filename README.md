# Can We Tune Together

Can we tune multiple language models together? There are many pre-trained transformer language models available in [huggingface model hub](https://huggingface.co/models). The current hype for sentence-level tasks is to pick one language model (i.e BERT, ELECTRA, deBERTa) and fine-tune it for the task at hand. Each LM has something different: either a different pretraining objective, a different corpus used for pretaining, or some other twists in the transformer architecture. When evaluating unalike models on the [GLUE benchmark](https://gluebenchmark.com/), they also score differently.

I couldn't find any works that combine multiple LMs and tune them together — so I'm simply running some experiments here! 🤷‍♂️. 

The initial idea was to concatenate different CLS token representations of each LM and let the complete model (multi-encoder) figure out how to combine them. Later on, I came across some other works ([Dynamic Meta-Embeddings](https://arxiv.org/abs/1804.07983) for Sentence Representations by Kiela et. al.) which applied attention to combine static GloVe and FastText embeddings. Even though the previous work with static/frozen embeddings does not directly translate to language models, the attention mechanism can still help us visualize the attention scores for each LM and look at what is happening inside the multi-encoder.

My code is very much based on [FlairNLP](https://github.com/flairNLP/flair) (shout-out to them! Look it up, it's a cool NLP framework 🙃). If combining and tuning multiple LMs together happens to work, I might open a PR later.

##### 🔁 This is still an ongoing project...

# Tuning a single Language Model

You can pick any transformer-based language model from the model hub and tune it for GLUE tasks:

```python
from glue_benchmark import GLUE_COLA
from encoders import LanguageModel
from task_heads import TextClassifier
from trainer import ModelTrainer

# 1. load any GLUE corpus (i.e. Corpus of Linguistic Acceptability)
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

- Electra base scores around 67.7 (Matthews correlation coefficient) for CoLA dev set. You can look at the scores provided by the authors here: [expected electra results](https://github.com/google-research/electra).
- Roberta base scores around 63.6 [expected roberta results](https://github.com/pytorch/fairseq/tree/master/examples/roberta).
- SpanBERT scores around 60.1.

# Combining Language Models

```python
from glue_benchmark import GLUE_COLA
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

Looks like we can increase the score of Electra if we add Roberta and tune them together. I ran it a few times and the average Mcc score is 68.6.

## Looking at attention weights

If you use DME as the combine method, you can embedd a few sentences from CoLA dev set and look at the attention scores for each language model:

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
model = TextClassifier.from_pretrained(model_path)

# classify sentences and look at the attention scores of the multi-encoder
predictions, attn_scores = model.predict_with_attention(sentences)

print(predictions)
# [1, 0, 1]
print(attn_scores)
# [[0.7043, 0.2957], [0.7101, 0.2899], [0.6899, 0.3101]],

```

Attention coefficients show how much weight does the multi-encoder assign to each cls representation when embedding a given sentence. Electra is a stronger model than roberta but roberta also adds something to the score. We might find sentences where roberta is even overpowering Electra.


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

This scored 69.3. I tried it only once so the score might differ after taking the average of multiple runs.

## Notes on combining different LMs and why it sometimes doesn't work

It can be that the multi-encoder is not scoring better than the strongest LM. This mostly happens if both LMs have similar scores for the same task. Here is a 3-Step plan how to fix that:

- Step 1: Each LMs has its own best learning rate. If the strongest LM scores higher than the complete multi-encoder, this can be due to some weaker LM overpowering others on the training data. I added an option to assign different learning rates for each LM in the multi-encoder. You can now decrease the learning rate just for the weaker LM by providing a list of learning rates to the trainer: `learning_rate=[3e-5, 1e-5]`).
- Step 2: If you nailed down different learning rates and both LMs are fitting the training data at a similar pace, the combined representation may become very powerfull and can overfit quickly. You can then add dropout on top of the combined embedding: `MultiEncoder(language_models=lms, dropout=0.2)`.
- Step 3: (Just skip 1 & 2 and jump here if you are impatient) Increase hidden dropout of the  weaker LM. When loading a LanguageModel, set the hidden dropout parameter: `LanguageModel('roberta-base', hidden_dropout_prob=0.4)`.


