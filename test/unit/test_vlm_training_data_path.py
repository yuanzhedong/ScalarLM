"""
Unit tests for the multimodal training data path:

  - SDK: examples referencing local image files are shipped in the training
    archive under images/ with paths rewritten (and deduplicated).
  - VLM collate: variable-length rows pad correctly (pad token, zero
    attention, -100 labels) and pixel tensors concatenate along dim 0;
    mixed image/text batches are rejected.
  - export_peft_adapter: exports through wrapped models, no-ops for
    non-LoRA jobs.
"""

import types

import pytest
import torch


# ---------------------------------------------------------------------------
# SDK image packing
# ---------------------------------------------------------------------------

def test_extract_image_references_rewrites_and_dedupes(tmp_path):
    from masint.engines.cray.submit_training_job import extract_image_references

    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    img_a.write_bytes(b"a")
    img_b.write_bytes(b"b")

    data = [
        {"input": "x", "output": "y", "images": [str(img_a)]},
        {"input": "x2", "output": "y2", "images": [str(img_a), str(img_b)]},
        {"input": "x3", "output": "y3"},
    ]
    rewritten, files = extract_image_references(data)

    # two unique files shipped, despite three references
    assert len(files) == 2
    arcnames = [arc for _, arc in files]
    assert all(arc.startswith("images/") for arc in arcnames)

    # references rewritten to archive-relative paths; duplicates share one
    assert rewritten[0]["images"][0] == rewritten[1]["images"][0]
    assert rewritten[1]["images"][1] != rewritten[1]["images"][0]
    # non-image example untouched
    assert rewritten[2] == data[2]


def test_extract_image_references_passthrough_for_non_list():
    from masint.engines.cray.submit_training_job import extract_image_references

    data, files = extract_image_references("/path/to/dataset.jsonl")
    assert data == "/path/to/dataset.jsonl"
    assert files == []


# ---------------------------------------------------------------------------
# VLM collate
# ---------------------------------------------------------------------------

class _StubTokenizer:
    pad_token_id = 0
    eos_token_id = 2


def _collate():
    from cray_megatron.megatron.dataset.load_vlm_dataset import (
        make_vlm_collate_function,
    )
    return make_vlm_collate_function(_StubTokenizer())


def test_collate_pads_to_longest_row():
    collate = _collate()
    rows = [
        {"input_ids": [5, 6], "attention_mask": [1, 1], "labels": [-100, 6]},
        {"input_ids": [7, 8, 9], "attention_mask": [1, 1, 1], "labels": [-100, 8, 9]},
    ]
    batch = collate(rows)
    assert batch["input_ids"].shape == (2, 3)
    assert batch["input_ids"][0].tolist() == [5, 6, 0]        # pad token
    assert batch["attention_mask"][0].tolist() == [1, 1, 0]   # zero attention
    assert batch["labels"][0].tolist() == [-100, 6, -100]     # ignored loss


def test_collate_concatenates_pixel_values():
    collate = _collate()
    rows = [
        {"input_ids": [1], "attention_mask": [1], "labels": [1],
         "pixel_values": torch.zeros(2, 3, 4, 4)},
        {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2],
         "pixel_values": torch.ones(3, 3, 4, 4)},
    ]
    batch = collate(rows)
    assert batch["pixel_values"].shape == (5, 3, 4, 4)


def test_collate_rejects_mixed_image_text_batches():
    collate = _collate()
    rows = [
        {"input_ids": [1], "attention_mask": [1], "labels": [1],
         "pixel_values": torch.zeros(1, 3, 4, 4)},
        {"input_ids": [1], "attention_mask": [1], "labels": [1]},
    ]
    with pytest.raises(ValueError):
        collate(rows)


# ---------------------------------------------------------------------------
# PEFT export
# ---------------------------------------------------------------------------

def _export(monkeypatch, job_config, model_obj):
    from cray_megatron.megatron import training_loop as tl

    monkeypatch.setattr(tl, "get_job_config", lambda: job_config)
    fake_self = types.SimpleNamespace(
        training_state=types.SimpleNamespace(model_info={"model": model_obj})
    )
    # call the undecorated function if main_rank_only wrapped it
    fn = tl.TrainingLoop.export_peft_adapter
    fn = getattr(fn, "__wrapped__", fn)
    fn(fake_self)


def test_export_peft_adapter_saves_through_wrappers(monkeypatch, tmp_path):
    calls = []

    class Peft:
        def save_pretrained(self, out):
            calls.append(out)

    wrapped = types.SimpleNamespace(module=types.SimpleNamespace(model=Peft()))
    _export(
        monkeypatch,
        {"adapter_type": "lora", "job_directory": str(tmp_path)},
        wrapped,
    )
    assert calls == [str(tmp_path)]


def test_export_peft_adapter_noop_for_tokenformer(monkeypatch, tmp_path):
    calls = []

    class Peft:
        def save_pretrained(self, out):
            calls.append(out)

    _export(
        monkeypatch,
        {"adapter_type": "tokenformer", "job_directory": str(tmp_path)},
        Peft(),
    )
    assert calls == []
