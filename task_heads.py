import torch
from encoders import LanguageModel, MultiEncoder
import warnings

from typing import Union


class TextClassifier(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        encoder: Union[LanguageModel, MultiEncoder],
    ):
        super().__init__()

        self.num_classes = num_classes
        self.encoder = encoder

        self.name = str(encoder.name) + '-classifier'
        self.encoder_embedding_length = self.encoder.embedding_length
        
        self.similarity_constrain = False
        if isinstance(self.encoder, MultiEncoder):
            self.similarity_constrain = self.encoder.similarity_constrain


        self.decoder = torch.nn.Linear(
            self.encoder_embedding_length, self.num_classes
        )

        # initialize decoder weights
        torch.nn.init.xavier_uniform_(self.decoder.weight)

        # loss function for classification
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, sentences):
        encoded_sentences = self.encoder(sentences)
        return self.decode(encoded_sentences)
    
    def decode(self, encoded_sentences):
        scores = self.decoder(encoded_sentences)
        return scores
    
    def forward_loss(self, sentences, true_labels):
        if self.similarity_constrain:
            encoded_sentences, cos_similarity = self.encoder(sentences,
                                                            return_cos_similarity=True)
            scores = self.decoder(encoded_sentences)
            
            # Loss with similarity constrain
            cos_distance = 1 -  cos_similarity
            multi_encoder_loss = self.loss_function(scores, true_labels) + cos_distance
            return multi_encoder_loss
        
        scores = self.forward(sentences)
        loss_value = self.loss_function(scores, true_labels)
        
        return loss_value
    
    def predict(self, sentences):
        with torch.no_grad():
            raw_output = self.forward(sentences)
            label_scores = torch.nn.functional.softmax(raw_output, dim=1)
            predicted_label = torch.argmax(label_scores, 1)
        return predicted_label


    def predict_with_attention(self, sentences):
        
        if not isinstance(self.encoder, MultiEncoder):
            warnings.warn(f"You need to use multi-encoder to create meta-embeddings")
            return

        if self.encoder.combine_method != 'dme': 
            warnings.warn(f"You need to choose DME as combine method for your multi-encoder " \
                f"and train the model. Current combine method is {self.encoder.combine_method} " \
                f"which does not use attention.")
            return

        with torch.no_grad():
            embedding, attn_scores = self.encoder(sentences,
                                                  return_attn_scores=True)
            decoder_output = self.decoder(embedding)
            label_scores = torch.nn.functional.softmax(decoder_output, dim=1)
            predicted_label = torch.argmax(label_scores, 1)

        return predicted_label, attn_scores.squeeze()
    
    @staticmethod
    def from_checkpoint(path):
        return torch.load(path)


class TextRegressor(torch.nn.Module):
    def __init__(
        self,
        encoder: Union[LanguageModel, MultiEncoder],
    ):   
        super().__init__()

        self.encoder = encoder

        self.encoder_embedding_length = self.encoder.embedding_length
        self.name: str = str(encoder.name) + '-regressor'

        self.similarity_constrain = False
        if isinstance(self.encoder, MultiEncoder):
            self.similarity_constrain = self.encoder.similarity_constrain

        self.decoder = torch.nn.Linear(
            self.encoder_embedding_length, 1
        )

        # initialize decoder weights
        torch.nn.init.xavier_uniform_(self.decoder.weight)

        # loss function for regression
        self.loss_function = torch.nn.MSELoss()

    def forward(self, sentences):
        encoded_sentences = self.encoder(sentences)
        return self.decode(encoded_sentences)
    
    def decode(self, encoded_sentences):
        logits = self.decoder(encoded_sentences).squeeze()
        return logits
    
    def forward_loss(self, sentences, true_labels):

        if self.similarity_constrain:
            encoded_sentences, cos_similarity = self.encoder(sentences,
                                                            return_cos_similarity=True)
            scores = self.decoder(encoded_sentences)

            # Loss with similarity constrain
            cos_distance = 1 - cos_similarity
            multi_encoder_loss = self.loss_function(scores, true_labels) + cos_distance

            return multi_encoder_loss
        
        scores = self.forward(sentences)
        loss_value = self.loss_function(scores, true_labels)
        return loss_value

    def predict(self, sentences):

        with torch.no_grad():
            score = self.forward(sentences)
        return score

    def predict_with_attention(self, sentences):

        if not isinstance(self.encoder, MultiEncoder):
            warnings.warn(f"You need to use multi-encoder to create meta-embeddings")
            return

        if self.encoder.combine_method != 'dme': 
            warnings.warn(f"You need to choose DME as combine method for your multi-encoder " \
                "and train the model. Current combine method is {self.encoder.combine_method} " \
                "which does not use attention.")
            return

        with torch.no_grad():
            embedding, attn_scores = self.encoder(sentences,
                                                  return_attn_scores=True)
            predicted_score = self.decoder(embedding)

        return predicted_score, attn_scores.squeeze()

    @staticmethod
    def from_checkpoint(path):
        return torch.load(path)
