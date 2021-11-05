# Datasets

I am currently tuning multiple language models on various NLP tasks. Trying to understand where joint tuning works and where it doesn't 🤷‍♂️. Here's a list of datasets I've found over time:

## Text Classification

| Corpus Name | Sentences | Labels | Task type | Reference |
| ----------- | ---- | ---------- | --------- | --------- |
| CLICKBAIT   | 32k  | 2          | Sentence classification | [clickbait] |
| ISEAR       | 7k   | 7          | Sentence classification | [isear] #6 only in .sav and .mdb formats 😓 |
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


## Text Regression

| Corpus Name | Sentences | Labels | Task type | Reference |
| ----------- | ---- | ---------- | --------- | --------- |
| GLUE_STSB   | 8.5k   | Similarity score | Sentence regression | [glue] |
| EMOBANK     | 10k   | Valance, arousal, or dominance scores | Sentence regression | [emobank] |
| FB_VALANCE_AROUSAL | 6k   | Valance or arousal scores | Sentence regression | [valance arousal in fb posts] |


## Sequence Labeling

| Corpus Name | Sentences | Labels | Task type | Reference |
| ----------- | ---- | ---------- | --------- | --------- |
| CONLL_NER   | 20k   | 4 | NER | [conll03] |
| WNUT_NER     | 5k   | 6 | NER | [wnut17] |
| MIT_MOVIE_NER | 10k   | 13   | NER | [mit movie] |
| MIT_RESTAURANT_NER | 9k   | 9   | NER | [mit restaurant] |


[clickbait]: https://github.com/bhargaviparanjape/clickbait
[isear]: https://www.unige.ch/cisa/research/materials-and-online-research/research-material/
[trec]: https://cogcomp.seas.upenn.edu/Data/QA/QC/
[emotion stimulus]: https://www.eecs.uottawa.ca/~diana/resources/emotion_stimulus_data/
[glue]: https://gluebenchmark.com/tasks
[sick]: https://zenodo.org/record/2787612/#.YYSThnko-qQ
[emobank]: https://github.com/JULIELab/EmoBank
[valance arousal in fb posts]: https://github.com/wwbp/additional_data_sets/tree/master/valence_arousal
[conll03]: https://www.clips.uantwerpen.be/conll2003/ner/
[wnut17]: https://github.com/leondz/emerging_entities_17
[mit movie]: https://groups.csail.mit.edu/sls/downloads/movie/
[mit restaurant]: https://groups.csail.mit.edu/sls/downloads/restaurant/
