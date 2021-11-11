import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import logging
import warnings
import random
from typing import List, Union


class LanguageModel(torch.nn.Module):
    def __init__(
            self,
            model: str = "google/electra-base-discriminator",
            fine_tune: bool = True,
            pooling: str = "cls",
            hidden_dropout_prob: float = 0.1,
            keep_special_tokens: bool = True,
    ):
        """
        Downloads any pretrained transformer-based language model from Huggingface and
        uses CLS token as sentence or sentence pair embedding

        :param model: language model handle from huggingface model hub
        :param fine_tune: switch fine-tuning on/off
        :param pooling: "cls" token, "mean" average all tokens, "word-level" for word tagging tasks
        :param hidden_dropout_prob: set hidden dropout in language model, default is 0.1 in all models
        :param keep_special_tokens: only used for sequence labeling, set to false to remove sep and cls tokens
        """

        super().__init__()

        # GPU or CPU device
        self.device = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # three different ways to use last-layer representations of LMs
        if pooling not in ["cls", "word-level", "mean"]:
            raise ValueError(f"We only support cls or mean pooling for sentence-level"
                " tasks. `{pooling}` is currently not defined")

        self.pooling = pooling
        self.keep_special_tokens = keep_special_tokens

        # load auto tokenizer from huggingface transformers
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        # ignore warning messages: we will use models from huggingface model hub checkpoints
        # some layers won't be used (i.e. pooling layer) and this will throw warning messages 
        logging.set_verbosity_error()

        # load transformer config file and set different dropout if needed
        self.config = AutoConfig.from_pretrained(model, output_hidden_states=True)
        self.config.hidden_dropout_prob=hidden_dropout_prob

        # load the model using auto class
        self.model = AutoModel.from_pretrained(model, config=self.config)

        # name of the language model
        self.name = str(model).rsplit("/")[-1]

        # set fine-tuning mode on/off
        self.fine_tune = fine_tune

        # when initializing, models are in eval mode by default
        self.model.eval()
        self.model.to(self.device)

        # send a mini-token through to inspect the model
        model_output = self.model(torch.tensor([1], device=self.device).unsqueeze(0))

        # hidden size of the original model
        self.embedding_length = model_output.last_hidden_state.shape[2]

        # check whether CLS is at beginning or end
        # some models i.e. BERT, ELECTRA, BART use cls as initial token
        # other models i.e. XLM use cls at the end
        tokens = self.tokenizer.encode('a')
        self.cls_position = 0 if tokens[0] == self.tokenizer.cls_token_id else -1


    def forward(self, sentences):
        # sentences can be tuples for sentence-pair tasks
        if isinstance(sentences[0], tuple) and len(sentences) == 2:
            sentences = list(zip(sentences[0], sentences[1]))

        # tokenize sentences and sentence pairs
        # huggingface tokenizers already support batch tokenization
        tokenized_sentences = self.tokenizer(sentences,
                                            padding=True,
                                            truncation=True,
                                            return_tensors="pt")

        # tokenizer returns input ids: unique ids for subwords
        input_ids = tokenized_sentences["input_ids"].to(self.device)

        # and attention masks: 1's for for subwords, 0's for padding tokens
        mask = tokenized_sentences['attention_mask'].to(self.device)

        # some models use token type ids for sentence-pair tasks (i.e. next sentence prediction)
        use_token_type = True if "token_type_ids" in tokenized_sentences else False
        if use_token_type:
            token_type_ids = tokenized_sentences["token_type_ids"].to(self.device)

        # gradients are enabled if fine-tuning is enabled
        gradient_context = torch.enable_grad() if self.fine_tune else torch.no_grad()

        with gradient_context:
            if use_token_type:
                subword_embeddings = self.model(
                        input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=mask,
                ).last_hidden_state
            else:
                subword_embeddings = self.model(
                        input_ids=input_ids,
                        attention_mask=mask,
                ).last_hidden_state

        # option 1: use only the cls token representation
        # this is the most common way to tune language models on sentence-level tasks
        if self.pooling == "cls":
            cls_embedding  = torch.stack(
                [embedding[self.cls_position] for embedding in subword_embeddings]
            )
            return cls_embedding

        # option 2: mean pool (average) all last hidden state outputs
        # pooling the last layer might give us extra information from the last transformer layer 
        if self.pooling == "mean":
            return subword_embeddings.mean(dim=1)

        # option 3: word-level pooling. Use all word (first subword token) embeddings.
        # word level representations are needed for sequence labeling tasks e.g. NER, POS.
        if self.pooling == "word-level":
            word_start_idx, _ = self._estimate_word_offsets(sentences)
            word_representations = self._create_word_representations(subword_embeddings,
                                                                     word_start_idx)
            return word_representations

    def change_lm_pooling(self, pooling):
        self.pooling = pooling

    def _create_word_representations(self, subword_embeddings, word_start_idx):
        """
        Iterate over subword representations and create new tensors that have 
        representations of words instead of subwords. We will only use the first 
        subword as word representation (token pooling = first).
        """
        sentence_word_embeddings = []

        # iterate over sentence representations
        # and create new tensors that have representations of words instead of subwords
        for i, subword_embedding in enumerate(subword_embeddings):
            word_embeddings = []

            # move cls token at the start for each LM
            if self.keep_special_tokens:
                word_embeddings.append(subword_embedding[self.cls_position])

            # let's keep it simple: token pooling = first
            # we will only use the first subword representation of each word
            for word_start_id in word_start_idx[i]:
                word_embeddings.append(subword_embedding[word_start_id])

            sentence_word_embeddings.append(word_embeddings)

        # add padding (zero tensor)
        # the length of each vector has to be maximum number of words in a batch
        longest_sequence = len(max(sentence_word_embeddings, key=len))
        final_embedding = []
        for sentence_embedding in sentence_word_embeddings:
            while len(sentence_embedding) < longest_sequence:
                sentence_embedding.append(torch.zeros(subword_embeddings.shape[-1]).to(self.device))
            final_embedding.append(torch.stack(sentence_embedding))

        return torch.stack(final_embedding)


    def _estimate_word_offsets(self, sentences):
        """
        Word tokenize original sentences and estimate starting positions and offsets
        """
        # the start index for each word in the sentence
        word_start_idx = []
        # word offsets: number of subwords for each word
        word_offsets = []

        for sentence in sentences:
            if type(sentence) == tuple:
                sentence = " SEP ".join(sentence)

            # TODO: handle punctuation
            # current way to tokenize: whitespace tokenization using built-in split() on whitespace
            # it would be good to split 'yes!' into 'yes', '!', but '!!!' should stay as one word
            word_tokens = sentence.split()

            # count the number of subwords for each word token
            # tokenizing a single word results in a sequence e.g. CLS, subword1, subword2, SEP
            # some tokenizers (e.g. RoBERTa, BART) need whitespace before each word
            word_offsets.append(
                [(len(self.tokenizer(f" {word}")["input_ids"]) - 2) for word in word_tokens]
            )

        # estimate start indices for each word by looking at their offsets
        for word_offset in word_offsets:
            indices = []
            current_index = 1 if self.cls_position == 0 else 0
            max_index = self.tokenizer.model_max_length
            for offset in word_offset:
                if current_index < max_index:
                    indices.append(current_index)
                    current_index = current_index + offset
            word_start_idx.append(indices)

        return word_start_idx, word_offsets


class MultiEncoder(torch.nn.Module):
    def __init__(
            self,
            language_models: List[LanguageModel],
            combine_method: str = 'concat',
            use_layer_norm: bool = False,
            alternate_freezing_prob: float = 0.,
            combine_dropout: float = 0.,
    ):
        """
        Uses multiple language models as encoders, combines last layer cls representations
        and fine-tunes them simultaneously. Combine methods: concatenation of cls tokens,
        dynamic meta-embeddings (approach proposed by Kiela et al. Dynamic Meta-Embeddings for
        Improved Sentence Representations). Dropout can be applied on top of the combined embedding

        :param language_models: a list of LanguageModels to be combined
        :param combine_method: methods to combine language model representations "concat", "dme", or "mean"
        :param use_layer_norm: set True/False if LM representations should be normalized before combining them
        :param alternate_freezing_prob: EXPERIMENTAL randomly freeze one LM during training
        float in range of [0., 1.] zero means that all language models will be tuned,
        0.5 means that one language model (random choice) will be frozen half of the training time.
        0.1 means that one language model (random choice) will be frozen during all training.
        :param combine_dropout: standard dropout probability on top of the combined representation
        """

        super().__init__()

        self.num_lms = len(language_models)
        self.combine_method = combine_method
        self.alternate_freezing_prob = alternate_freezing_prob
        self.combine_dropout = combine_dropout
        self.use_layer_norm = use_layer_norm

        self.name = ""
        for i, model in enumerate(language_models):
            if i > 0: self.name += "-"
            self.name += model.name

        # each language model as class property
        for i, model in enumerate(language_models):
            setattr(self, f"lm_{i+1}", model)

        # make sure that all language models are using same pooling operation
        pooling = [lm.pooling for lm in language_models]
        assert len(set(pooling)) == 1, \
            f"Multi-encoder assertion: all language models have to use the same pooling operation." \
            f"Current pooling settings are: {pooling}."

        # different transformers might use different embedding sizes
        embedding_sizes = [model.embedding_length for model in language_models]

        # no need for combine method if one language model was given
        if self.num_lms == 1:
            self.combine_method = "none"
            self.embedding_length = embedding_sizes[0]

        if combine_method not in ["concat", "dme", "mean"]:
            raise ValueError(f"We only support concatenation and DME as methods to"
                f"combine different embedding types. `{combine_method}` is currently not defined")

        # all methods except concatenation require same size embeddings
        self.use_projection = False
        if combine_method != "concat":
            self.use_projection = True

        # add linear layers to project embeddings into same size
        if self.use_projection:
            projection_size = max(embedding_sizes)
            for i in range(self.num_lms):
                projection = torch.nn.Linear(
                    embedding_sizes[i], projection_size
                )
                setattr(self, f"lm_{i+1}_projection", projection)

        if self.use_layer_norm:
            norm_size = max(embedding_sizes)
            if self.use_projection:
                norm_size = projection_size
            self.layer_norm = torch.nn.LayerNorm(norm_size, eps=1e-3)

        # stacking: concatenate the representations of individual LMs
        # embedding length is a sum of all LM embedding sizes or their projections
        if self.combine_method == "concat":
            self.embedding_length = sum(embedding_sizes)
            if self.use_projection:
                self.embedding_length = self.num_lms * projection_size

        # DME Dynamic Meta Embeddings: learn attention coefficients to weight
        # sentence or token embeddings and use their weighted sum as meta-embedding.
        # this can be for both: cls or word-level representations
        if self.combine_method == "dme":
            self.attention = torch.nn.Linear(projection_size, 1)

        if self.combine_method in ["mean", "dme"]:
            self.embedding_length = projection_size

        # experiments to randomly freeze one of the LMs
        self.alternate_dist = torch.distributions.uniform.Uniform(0,1)

        if self.combine_dropout > 0.:
            self.dropout = torch.nn.Dropout(p=self.combine_dropout)


    def forward(
            self,
            sentences,
            return_attn_scores=False,
    ):
        embedding = []

        # alternate freezing: randomly freeze one of the LMs
        if self.alternate_dist.sample() < self.alternate_freezing_prob:
            dice = random.randint(0, self.num_lms-1)
            model = getattr(self, f"lm_{dice+1}")
            model.fine_tune = False

        # embed sentences with each language model
        for i in range(self.num_lms):
            model = getattr(self, f"lm_{i+1}")
            embedding.append(model(sentences))

        # alternate freezing: switch all LMs back to fine-tuning mode
        if self.alternate_freezing_prob  > 0.:
            for i in range(self.num_lms):
                model = getattr(self, f"lm_{i+1}")
                model.fine_tune = True

        # don't combine embeddings if only one language model is provided
        if self.num_lms == 1:
            return embedding[0]

        # project embeddings into same space
        if self.use_projection:
            projections = []
            for i in range(self.num_lms):
                projection = getattr(self, f"lm_{i+1}_projection")
                projections.append(projection(embedding[i]))
            embedding = projections

        # layer normalize embeddings or their projections
        if self.use_layer_norm:
            normalized_embeddings = []
            for emb in embedding:
                normalized_embeddings.append(self.layer_norm(emb))
            embedding = normalized_embeddings

        # concatenate CLS embeddings
        if self.combine_method in "concat":
            embedding = torch.cat(embedding, dim=-1)

        # average CLS embeddings
        if self.combine_method == "mean":
            embedding = torch.stack(embedding)
            embedding = torch.mean(embedding, dim=0)

        # use attention to learn weights for each cls embedding
        # and use weighted sum to create meta-embedding
        if self.combine_method == "dme":
            embedding = torch.stack(embedding, dim=1)
            attn_scores = torch.softmax(self.attention(embedding), dim=1)
            embedding = attn_scores*embedding
            embedding = embedding.sum(dim=1)

        if self.combine_dropout > 0.:
            embedding = self.dropout(embedding)

        if return_attn_scores:
            return embedding, attn_scores

        return embedding

    def change_lm_pooling(self, pooling):
        # change pooling option for all LMs in the mix
        for i in range(self.num_lms):
            language_model = getattr(self, f"lm_{i+1}")
            language_model.pooling = pooling

    def embed_with_attention(self, sentences):
        if self.num_lms < 2:
            warnings.warn(f"You need to train more than one language model" \
                "if you want to create meta-embedding")
            return

        if self.combine_method != "dme":
            warnings.warn(f"You need to choose DME as combine method and train the model." \
                f"Current combine method is {self.combine_method} which does not use attention.")
            return

        with torch.no_grad():
            meta_embedding, attn_scores = self.forward(sentences,
                                                       return_attn_scores=True)

        return meta_embedding, attn_scores.squeeze()
