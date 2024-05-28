"""Test some code snippets"""

from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, TaskType


model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"
