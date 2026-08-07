"""Vision-language dataset loader: image+text examples for multimodal fine-tuning.

Schema per jsonlines row:
    {"input": str, "output": str, "images": [relative_path, ...]}   # images optional

Image paths are resolved relative to the dataset file's directory (the job
directory), where the upload tar was extracted.

Differences from the text pipeline (load_language_model_dataset):
  - Prompts are rendered with the model's chat template via AutoProcessor so
    each architecture's image placeholder tokens are inserted correctly.
  - No packing: examples are batched individually and padded. Packing text
    blocks around per-example pixel tensors buys little for image-heavy rows
    (visual tokens dominate) and multimodal wrappers skip the 4D doc mask
    anyway.
  - The collate function carries pixel_values / image_grid_thw through to the
    training step when present.

v1 constraint: a batch must be homogeneous (all rows with images, or all rows
without). The default batch_size=1 satisfies this trivially.
"""

from cray_infra.util.get_job_config import get_job_config

from cray_megatron.collectives.data_parallelism import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
)

import datasets
import jsonlines
import os

import torch

import logging

logger = logging.getLogger(__name__)

_processor = None


def get_processor():
    global _processor
    if _processor is None:
        from transformers import AutoProcessor

        job_config = get_job_config()
        _processor = AutoProcessor.from_pretrained(
            job_config["llm_name"], trust_remote_code=True
        )
    return _processor


def load_vlm_dataset(model, tokenizer, epoch):
    hf_dataset = datasets.IterableDataset.from_generator(
        make_dataset_generator(),
        features=datasets.Features(
            {
                "input": datasets.Value(dtype="string"),
                "output": datasets.Value(dtype="string"),
                "images": datasets.Sequence(datasets.Value(dtype="string")),
            }
        ),
    )
    job_config = get_job_config()
    if job_config.get("shuffle_training_data", True):
        hf_dataset = hf_dataset.shuffle(seed=42 + epoch, buffer_size=256)
    split_dataset = split_dataset_by_node(hf_dataset)

    processed = split_dataset.map(
        get_process_function(model, tokenizer),
        remove_columns=["input", "output", "images"],
    )

    return processed.with_format("torch")


def make_dataset_generator():
    def read_dataset():
        dataset_path = get_dataset_path()
        with open(dataset_path) as dataset_file:
            reader = jsonlines.Reader(dataset_file)
            for obj in reader:
                obj.setdefault("images", [])
                yield obj

    return read_dataset


def get_dataset_path():
    job_config = get_job_config()
    return job_config["training_data_path"]


def split_dataset_by_node(dataset):
    data_parallel_rank = get_data_parallel_rank()
    data_parallel_world_size = get_data_parallel_world_size()

    return dataset.filter(
        lambda example, idx: idx % data_parallel_world_size == data_parallel_rank,
        with_indices=True,
    )


def get_process_function(model, tokenizer):
    processor = get_processor()
    dataset_dir = os.path.dirname(get_dataset_path())
    max_len = get_max_sequence_length(model)

    def process(example):
        from PIL import Image

        image_paths = example.get("images") or []
        images = [
            Image.open(os.path.join(dataset_dir, p)).convert("RGB")
            for p in image_paths
        ]

        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": example["input"]})
        messages = [{"role": "user", "content": content}]

        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if images:
            prompt_inputs = processor(
                text=[prompt_text], images=images, return_tensors=None
            )
        else:
            prompt_inputs = processor(text=[prompt_text], return_tensors=None)

        prompt_ids = prompt_inputs["input_ids"][0]

        output_ids = tokenizer(example["output"], add_special_tokens=False)[
            "input_ids"
        ]
        output_ids = output_ids + [get_eos_token(model, tokenizer)]

        input_ids = list(prompt_ids) + output_ids
        labels = [-100] * len(prompt_ids) + output_ids

        # Truncate from the left of the *output* is never desired; if the
        # prompt alone exceeds the model context there is no salvageable
        # example, so clip the whole sequence and mask everything clipped.
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]

        result = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

        for key in ("pixel_values", "image_grid_thw"):
            if key in prompt_inputs:
                result[key] = prompt_inputs[key]

        return result

    return process


def get_eos_token(model, tokenizer):
    if model.generation_config is not None:
        eos = model.generation_config.eos_token_id
        if isinstance(eos, list):
            return eos[-1]
        if eos is not None:
            return eos
    return tokenizer.eos_token_id


def get_max_sequence_length(model):
    job_config = get_job_config()
    config = model.config
    text_config = getattr(config, "text_config", None) or config
    max_pos = getattr(text_config, "max_position_embeddings", 32768)
    return min(max_pos, job_config["max_token_block_size"])


def make_vlm_collate_function(tokenizer):
    """Pad a list of processed examples into one batch.

    pixel_values / image_grid_thw are concatenated along dim 0 — vision towers
    that use grid metadata (Qwen2-VL family) split the patch stream back into
    images with image_grid_thw, so concatenation across batch rows is correct.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def collate(rows):
        max_len = max(len(r["input_ids"]) for r in rows)

        def pad(seq, value):
            seq = list(seq)
            return seq + [value] * (max_len - len(seq))

        batch = {
            "input_ids": torch.tensor(
                [pad(r["input_ids"], pad_id) for r in rows], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [pad(r["attention_mask"], 0) for r in rows], dtype=torch.long
            ),
            "labels": torch.tensor(
                [pad(r["labels"], -100) for r in rows], dtype=torch.long
            ),
        }

        with_pixels = [r for r in rows if "pixel_values" in r and r["pixel_values"] is not None]
        if with_pixels:
            if len(with_pixels) != len(rows):
                raise ValueError(
                    "Mixed batch: some rows have images and some do not. "
                    "Use batch_size=1 or homogeneous data."
                )
            batch["pixel_values"] = torch.cat(
                [torch.as_tensor(r["pixel_values"]) for r in rows], dim=0
            )
            if "image_grid_thw" in rows[0]:
                batch["image_grid_thw"] = torch.cat(
                    [torch.as_tensor(r["image_grid_thw"]) for r in rows], dim=0
                )

        return batch

    return collate
