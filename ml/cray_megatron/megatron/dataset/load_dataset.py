from cray_infra.util.get_job_config import get_job_config

from cray_megatron.megatron.dataset.load_embedding_dataset import load_embedding_dataset
from cray_megatron.megatron.dataset.load_language_model_dataset import load_language_model_dataset
from cray_megatron.megatron.dataset.load_vlm_dataset import load_vlm_dataset

def load_dataset(model, tokenizer, epoch):
    """Load dataset for language model, vlm, or embedding model training."""
    job_config = get_job_config()
    training_mode = job_config["training_mode"]

    if training_mode == "embedding":
        return load_embedding_dataset(model, tokenizer, epoch)
    elif training_mode == "vlm":
        return load_vlm_dataset(model, tokenizer, epoch)
    else:
        return load_language_model_dataset(model, tokenizer, epoch)
