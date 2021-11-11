import torch
from encoders import LanguageModel, MultiEncoder
import warnings
from typing import Union

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from seqeval.metrics import accuracy_score as seqeval_acc, f1_score as seqeval_f1


class TextClassifier(torch.nn.Module):
    """
    Text Classification Head for any sentence-level classification tasks.
    This class uses representations from one (LanguageModel) or multiple language models (MultiEncoder) 
    and adds additional linear layer + softmax to solve text classification tasks.
    """
    def __init__(
            self,
            encoder: Union[LanguageModel, MultiEncoder],
            num_classes: int,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.encoder = encoder
        self.name = str(encoder.name) + '-classifier'
        self.encoder_embedding_length = self.encoder.embedding_length

        if isinstance(encoder, LanguageModel):
            lm_pooling = encoder.pooling
            self.encoder_type = "single-transformer"
        else:
            lm_pooling = encoder.lm_1.pooling
            if encoder.num_lms > 1:
                self.encoder_type = "multi-transformer"

        # linear classifier layer
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
        logits = self.forward(sentences)
        loss_value = self.loss_function(logits, true_labels)
        return loss_value

    def predict(self, sentences):
        with torch.no_grad():
            logits = self.forward(sentences)
            log_probs = torch.nn.functional.softmax(logits, dim=1)
            categorical_labels = torch.argmax(log_probs, 1)
        return categorical_labels

    def evaluation_score(self, predictions, ground_truths, evaluation_metrics):
        # classification tasks usually use Accuracy, F1 (micro), or Matthews metrics
        metrics = {}
        for metric in evaluation_metrics:
            if metric == "accuracy":
                metrics["accuracy"] = accuracy_score(ground_truths, predictions)

            if metric == "f1_score":
                metrics["f1_score"] = f1_score(ground_truths, predictions, average="micro")

            if metric == "matthews_corr":
                metrics["matthews_corr"] = matthews_corrcoef(ground_truths, predictions)
        return metrics

    def predict_with_attention(self, sentences):
        if not isinstance(self.encoder, MultiEncoder):
            warnings.warn(f"You need to use multi-encoder to create meta-embeddings")
            return

        if self.encoder.combine_method != 'dme': 
            warnings.warn(f"You need to choose DME as combine method for your multi-encoder "
                f"and train the model. Current combine method is {self.encoder.combine_method} "
                f"which does not use attention.")
            return

        with torch.no_grad():
            embedding, attn_scores = self.encoder(sentences,
                                                  return_attn_scores=True)
            decoder_output = self.decoder(embedding)
            label_scores = torch.nn.functional.softmax(decoder_output, dim=1)
            predictions = torch.argmax(label_scores, 1)

        return predictions, attn_scores.squeeze()

    def predict_tags(self, sentences, label_map=None):
        result_dictionary = {
            "Sentences": sentences
        }

        if self.encoder_type == "multi-transformer" and \
                self.encoder.combine_method == "dme":
            predictions, dme_scores = self.predict_with_attention(sentences)
            result_dictionary['DME scores'] = dme_scores
        else:
            predictions = self.predict(sentences)
        predictions = predictions.cpu().tolist()

        # convert to readable strings if label map is provided
        if label_map:
            reversed_label_map = {v: k for k, v in label_map.items()}
            predictions = [reversed_label_map[label] for label in predictions]

        result_dictionary['Predictions'] = predictions
        return result_dictionary

    @staticmethod
    def from_checkpoint(path):
        return torch.load(path)


class TextRegressor(torch.nn.Module):
    """
    Text Regression Head for any sentence-level regression tasks.
    This class uses representations from one (LanguageModel) or multiple language models (MultiEncoder) 
    and adds additional linear layer to solve text regression tasks.
    """
    def __init__(
            self,
            encoder: Union[LanguageModel, MultiEncoder],
            num_classes: int = 1,
    ):   
        super().__init__()

        self.encoder = encoder
        self.encoder_embedding_length = self.encoder.embedding_length
        self.name: str = str(encoder.name) + "-regressor"

        if isinstance(encoder, LanguageModel):
            lm_pooling = encoder.pooling
            self.encoder_type = "single-transformer"
        else:
            lm_pooling = encoder.lm_1.pooling
            self.encoder_type = "multi-transformer"

        # linear layer with num classes = 1 for regression tasks
        self.decoder = torch.nn.Linear(
            self.encoder_embedding_length, num_classes
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
        scores = self.forward(sentences)
        loss_value = self.loss_function(scores, true_labels)
        return loss_value

    def predict(self, sentences):
        with torch.no_grad():
            score = self.forward(sentences)
        return score

    def evaluation_score(self, predictions, ground_truths, evaluation_metrics):
        # regression tasks usually use Pearson or Spearman coefficients
        metrics = {}
        for metric in evaluation_metrics:
            if metric == "pearsonr":
                metrics["pearsonr"] = pearsonr(ground_truths, predictions)[0]

            if metric == "spearmanr":
                metrics["spearmanr"] = spearmanr(ground_truths, predictions)[0]
        return metrics

    def predict_with_attention(self, sentences):
        if not isinstance(self.encoder, MultiEncoder):
            warnings.warn(f"You need to use multi-encoder to create weighted meta-embeddings")
            return

        if self.encoder.combine_method != "dme":
            warnings.warn(f"You need to choose DME as combine method for your multi-encoder "
                "and train the model. Current combine method is {self.encoder.combine_method} "
                "which does not use attention.")
            return

        with torch.no_grad():
            embedding, attn_scores = self.encoder(sentences,
                                                  return_attn_scores=True)
            predicted_score = self.decoder(embedding)

        return predicted_score, attn_scores.squeeze()

    def predict_tags(self, sentences, label_map=None):
        result_dictionary = {"Sentences": sentences}

        if self.encoder_type == "multi-transformer" and \
                self.encoder.combine_method == "dme":
            predictions, dme_scores = self.predict_with_attention(sentences)
            result_dictionary['DME scores'] = dme_scores.cpu().tolist()
        else:
            predictions = self.predict(sentences)
        predictions = predictions.cpu().tolist()

        # convert to readable strings if label map is provided
        if label_map:
            reversed_label_map = {v: k for k, v in label_map.items()}
            predictions = [reversed_label_map[label] for label in predictions]

        result_dictionary['Predictions'] = predictions
        return result_dictionary

    @staticmethod
    def from_checkpoint(path):
        return torch.load(path)


class SequenceTagger(torch.nn.Module):
    """
    Sequence Tagger Head for sequence labeling tasks.
    This class uses representations from one (LanguageModel) or multiple language models (MultiEncoder)
    and adds prediction head (linear layer) to solve word level tasks.
    Thus, it uses representations of words instead of sentences.
    """
    def __init__(
            self,
            num_classes: int,
            encoder: Union[LanguageModel, MultiEncoder],
    ):
        super().__init__()

        self.encoder = encoder
        self.encoder_embedding_length = self.encoder.embedding_length
        self.name: str = str(encoder.name) + "-sequence-tagger"

        if isinstance(encoder, LanguageModel):
            self.encoder_type = "single-transformer"
        else:
            if encoder.num_lms > 1:
                self.encoder_type = "multi-transformer"

        # sequence labeling requires token-level representations
        # switch to word-level pooling in all language models
        self.encoder.change_lm_pooling("word-level")

        # linear decoder layer to classify word tags
        self.decoder = torch.nn.Linear(
            self.encoder_embedding_length, num_classes
        )

        # initialize decoder weights
        torch.nn.init.xavier_uniform_(self.decoder.weight)

        # loss function for classification
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, sentences):
        encoded_sentences = self.encoder(sentences)
        return self.decode(encoded_sentences)

    def decode(self, encoded_sentences):
        logits = self.decoder(encoded_sentences)
        return logits

    def forward_loss(self, sentences, true_labels):
        scores = self.forward(sentences)
        loss_sum = 0.

        for predicted_tags, true_tags in zip(scores, true_labels):
            # remove paddings
            true_tags = true_tags[true_tags!=-1.]
            predicted_tags = predicted_tags[:true_tags.shape[0]]

            # sum losses for each sentence
            loss_sum += self.loss_function(predicted_tags, true_tags)

        return loss_sum/len(sentences)

    def predict(self, sentences):
        with torch.no_grad():
            raw_output = self.forward(sentences)
            label_scores = torch.nn.functional.softmax(raw_output, dim=2)
            categorical_labels = torch.argmax(label_scores, 2)
        return categorical_labels

    def predict_tags(self, sentences, label_map=None):
        result_dictionary = {"Sentences": sentences}

        if self.encoder_type == "multi-transformer" and \
                self.encoder.combine_method == "dme":
            predictions, dme_scores = self.predict_with_attention(sentences)
            result_dictionary['DME scores'] = dme_scores.cpu().tolist()
        else:
            predictions = self.predict(sentences)

        predicted_tags = []
        for i, (sentence, predicted_label) in enumerate(zip(sentences, predictions)):
            words = sentence.split()
            word_tags = []

            for word_id, word in enumerate(words):
                tag = predicted_label[word_id].cpu().item()

                # convert to readable strings if label map is provided
                if label_map:
                    reversed_label_map = {v: k for k, v in label_map.items()}
                    tag = reversed_label_map[tag]

                word_tags.append((word, tag))
            predicted_tags.append(word_tags)

        result_dictionary['Predictions'] = predictions
        return result_dictionary

    def evaluation_score(self, predictions, ground_truths, evaluation_metrics, label_map):
        # ignore paddings and bring float targets back to B-I-O (beginning, inside, outside) strings
        # use seqeval library to evaluate per entity level (seqeval uses B-I strings)

        reversed_label_map = {v: k for k, v in label_map.items()}
        pred_as_stings, gt_as_strings = [], []

        for tags_per_prediction, tags_per_gt in zip(predictions, ground_truths):
            pred, gt = [], []
            for predicted_tag, gt_tag in zip(tags_per_prediction, tags_per_gt):
                if gt_tag.item() != -1.:
                    pred.append(reversed_label_map[predicted_tag.item()])
                    gt.append(reversed_label_map[gt_tag.item()])
            pred_as_stings.append(pred)
            gt_as_strings.append(gt)

        predictions = pred_as_stings
        ground_truths = gt_as_strings

        metrics = {}
        # we will use seqeval library to estimate f1 on entity-level
        for metric in evaluation_metrics:
            if metric == "accuracy":
                metrics["accuracy"] = seqeval_acc(ground_truths, predictions)
            if metric == "f1_score":
                metrics["f1_score"] = seqeval_f1(ground_truths, predictions)

        return metrics

    def predict_with_attention(self, sentences):
        if not isinstance(self.encoder, MultiEncoder):
            warnings.warn(f"You need to use multi-encoder to create weighted meta-embeddings")
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
