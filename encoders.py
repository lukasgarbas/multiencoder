import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import logging
import warnings

from typing import List

class LanguageModel(torch.nn.Module):
    def __init__(
            self,
            model: str = "google/electra-small-discriminator",
            fine_tune: bool = True,
            pooling: str = "cls",
            hidden_dropout_prob: float = 0.1,
        ):
            '''Downloads any pretrained transformer-based language model from Huggingface and 
            uses CLS token as sentence or sentence pair embedding'''

            super().__init__()

            # GPU or CPU device
            self.device = None
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')

            self.pooling = pooling
            if pooling not in ['cls']:
                raise ValueError(f"We only support cls pooling for sentence-level" \
                    " tasks. `{pooling}` is currently not defined")

            # load auto tokenizer from huggingface transformers
            self.tokenizer = AutoTokenizer.from_pretrained(model)

            # ignore warning messages: we will use models from huggingface model hub checkpoints
            # some layers won't be used (i.e. pooling layer) and this will throw warning messages 
            logging.set_verbosity_error()

            # load transformer config file and set different dropout if needed
            config = AutoConfig.from_pretrained(model, output_hidden_states=True)
            config.hidden_dropout_prob=hidden_dropout_prob

            # load the model using auto class
            self.model = AutoModel.from_pretrained(model, config=config)

            # name of the language model
            self.name = str(model).rsplit('/')[-1]

            # set finetuning mode to true/flase
            self.fine_tune = fine_tune

            # when initializing, models are in eval mode by default
            self.model.eval()
            self.model.to(self.device)

            # send a mini-token through to inspect the model
            model_output = self.model(torch.tensor([1], device=self.device).unsqueeze(0))

            # hidden size of the original model
            self.embedding_length = model_output.last_hidden_state.shape[2]

            # check whether CLS is at beginning or end
            # Some models i.e. BERT, ELECTRA, BART use cls as initial token
            # Other models i.e. XLM use cls at the end
            tokens = self.tokenizer.encode('a')
            self.cls_position = 0 if tokens[0] == self.tokenizer.cls_token_id else -1


    def forward(self, sentences):

        # sentences can be tuples for sentence-pair tasks
        if isinstance(sentences[0], tuple) and len(sentences) == 2:
            sentences = list(zip(sentences[0], sentences[1]))

        # Tokenize sentences and sentence pairs
        # huggingface tokenizers already support batch tokenization
        tokenized_sentences = self.tokenizer(sentences,
                                            padding=True,
                                            truncation=True,
                                            return_tensors='pt')

        # tokenizers return input ids: unique ids of subwords
        input_ids = tokenized_sentences['input_ids'].to(self.device)

        # and attention masks: 1's for for subwords, 0's for padding tokens
        mask = tokenized_sentences['attention_mask'].to(self.device)

        # some models use token type ids for sentence-pair tasks (i.e. next sentence prediction)
        use_token_type = True if 'token_type_ids' in tokenized_sentences else False
        if use_token_type:
            token_type_ids = tokenized_sentences['token_type_ids'].to(self.device)

        # gradients are enabled if fine-tuning is enabled
        gradient_context = torch.enable_grad() if self.fine_tune else torch.no_grad()

        with gradient_context:
            if use_token_type:
                subtoken_embeddings = self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=mask,
                ).last_hidden_state
            else:
                subtoken_embeddings = self.model(
                    input_ids=input_ids,
                    attention_mask=mask,
                ).last_hidden_state

            # use cls embedding for sentence-level tasks
            cls_embedding = torch.stack(
                [embedding[self.cls_position] for embedding in subtoken_embeddings]
            )

        return cls_embedding



class MultiEncoder(torch.nn.Module):
    def __init__(
            self,
            language_models: List[LanguageModel],
            use_projection: bool = False,
            combine_method: str = 'concat',
            dropout: float = 0.0,
            similarity_constrain = False,
        ):
        ''' Uses multiple language models as encoders, combines last layer cls
        representations, and fine-tunes them simultaneously. Combine methods: 
        concatenation of cls tokens, dynamic meta-embedding (approach proposed by Kiela et al.
        Dynamic Meta-Embeddings for Improved Sentence Representations).
        Dropout can be applied on top of the combined embedding'''

        super().__init__()

        # multiple different language models
        self.num_lms = len(language_models)
        self.use_projection = use_projection
        self.combine_method = combine_method
        self.dropout_prob = dropout
        self.similarity_constrain = similarity_constrain

        self.name = ""
        for i, model in enumerate(language_models):
            if i > 0: self.name += "-"
            self.name += model.name

        # each language model as class property
        for i, model in enumerate(language_models):
            setattr(self, f'lm_{i+1}', model)

        # different transformers might use different embedding sizes
        embedding_sizes = [model.embedding_length for model in language_models]

        # no need for combine method if one language model was given
        if self.num_lms == 1:
            self.combine_method = "none"
            self.embedding_length = embedding_sizes[0]

        if combine_method not in ['concat', 'dme']:
            raise ValueError(f"We only support concatenation and DME as methods to" \
                f"combine different embedding types. `{combine_method}` is currently not defined")

        # all methods except concatenation require same size embeddings
        if combine_method != "concat":
            self.use_projection = True

        # prejecting into the same space is required if similarity constrain is enabled
        if self.similarity_constrain:
            self.use_projection = True

        # add linear layers to project embeddings into same size
        if self.use_projection:
            projection_size = max(embedding_sizes)
            for i in range(self.num_lms):
                projection = torch.nn.Linear(
                    embedding_sizes[i], projection_size
                )
                setattr(self, f'lm_{i+1}_projection', projection)
   
        # stacking: concatenate the representations of individual LMs
        # embedding length is a sum of all LM embedding sizes or their projections
        if self.combine_method == 'concat':
            self.embedding_length = sum(embedding_sizes)
            if use_projection:
                self.embedding_length = self.num_lms * projection_size
  
        # DME Dynamic Meta Embeddings: learn attention coefficients to weight 
        # token embeddings and use their weighted sum as meta-embedding. 
        # We will only apply it to different CLS embeddings.
        if self.combine_method == 'dme':
            self.attention = torch.nn.Linear(projection_size, 1)
            self.embedding_length = projection_size

        # adding dropout to concatinated/summed representations of multiple lms
        if self.dropout_prob > 0:
            self.dropout = torch.nn.Dropout(self.dropout_prob)

        # experimental: compute cosine similarity between different encoder outputs
        if self.similarity_constrain:
            self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)


    def forward(
        self,
        sentences,
        return_attn_scores=False,
        return_cos_similarity=False,
    ):
        
        embedding = []

        # embedd sentences with each language model
        for i in range(self.num_lms):
            model = getattr(self, f'lm_{i+1}')
            embedding.append(model(sentences))

        # project embeddings into the same space
        if self.use_projection:
            projections = []
            for i in range(self.num_lms):
                projection = getattr(self, f'lm_{i+1}_projection')
                projections.append(projection(embedding[i]))
            embedding = projections

        # don't combine embeddings if only one language model is provided
        if self.num_lms == 1:
            return embedding[0]

        # experimental - compute cosine similarity between different cls projections
        # average the value for complete batch
        # alternative: take the max similarity value
        if self.similarity_constrain:
            cos_similarity = torch.mean(self.cos(embedding[0], embedding[1]), dim=0)

        # concatenate CLS embeddings
        if self.combine_method in ['concat', 'attention_on_cat']:
            embedding = torch.cat(embedding, dim=-1)

        # use attention to learn weights for each cls embedding
        # and use a weighted sum to create meta-embedding
        if self.combine_method == 'dme':
            embedding = torch.stack(embedding, dim=1)
            attn_coeff = torch.softmax(self.attention(embedding), dim=1)
            embedding = attn_coeff*embedding
            embedding = embedding.sum(dim=1)
   
        # apply dropout
        if self.dropout_prob > 0:
            embedding = self.dropout(embedding)

        if return_cos_similarity:
            return embedding, cos_similarity

        if return_attn_scores:
            return embedding, attn_coeff

        return embedding


    def embedd_with_attention(self, sentences):
        
        if self.num_lms < 2:
            warnings.warn(f"You need to train more than one language model" \
                "if you want to create meta-embedding")
            return

        if self.combine_method != 'dme': 
            warnings.warn(f"You need to choose DME as combine method and train the model." \
                f"Current combine method is {self.combine_method} which does not use attention.")
            return

        with torch.no_grad():
            meta_embedding, attn_scores = self.forward(sentences,
                                                       return_attn_scores=True)

        return meta_embedding, attn_scores.squeeze()


    def embedd_with_similarities(self, sentences):

        with torch.no_grad():
            meta_embedding, cos_similarity = self.forward(sentences,
                                                          return_cos_similarity=True)

        return meta_embedding, cos_similarity 
