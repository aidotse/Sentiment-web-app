from transformers import BertForSequenceClassification, BertTokenizerFast, BertForTokenClassification

def load_sentiment_classifier(model):
    classifier = BertForSequenceClassification.from_pretrained(
        model,
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    return classifier


def load_token_classifier(model):
    classifier = BertForTokenClassification.from_pretrained(
        model,
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    return classifier


# global tokenizer
# global classifier_violence
# global classifier_fear
# global classifier_fear_targets
# global classifier_violence_targets
# global classifier_swedish_ner

print(f"Loading tokenizer")
tokenizer = BertTokenizerFast.from_pretrained("RecordedFuture/Swedish-Sentiment-Fear")
print(f"Loading Violence Sentiment model")
classifier_violence = load_sentiment_classifier("RecordedFuture/Swedish-Sentiment-Violence")
print(f"Loading Fear Sentiment model")
classifier_fear = load_sentiment_classifier("RecordedFuture/Swedish-Sentiment-Fear")
print(f"Loading Fear sentiment target model")
classifier_fear_targets = load_token_classifier("RecordedFuture/Swedish-Sentiment-Fear-Targets")
print(f"Loading Violence sentiment target model")
classifier_violence_targets = load_token_classifier("RecordedFuture/Swedish-Sentiment-Violence-Targets")
print(f"Loading Swedish NER model")
classifier_swedish_ner = load_token_classifier("RecordedFuture/Swedish-NER")

