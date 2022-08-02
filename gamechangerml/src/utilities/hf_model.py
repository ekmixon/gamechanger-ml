from transformers import pipeline, __version__
from transformers import AutoTokenizer, AutoModelForTokenClassification
import logging

logger = logging.getLogger(__name__)


def ner_models():
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "dbmdz/bert-large-cased-finetuned-conll03-english"
        )
        model = AutoModelForTokenClassification.from_pretrained(
            "dbmdz/bert-large-cased-finetuned-conll03-english"
        )
        return tokenizer, model
    except (EnvironmentError, ValueError) as e:
        logger.exception(f"{type(e)}: {str(e)}", exc_info=True)
        raise


def ner_pipeline():
    tokenizer, model = ner_models()
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        grouped_entities=True,
    )
