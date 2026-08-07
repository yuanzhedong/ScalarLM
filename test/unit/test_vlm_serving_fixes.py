"""
Unit tests for the multimodal serving-path fixes:

  1. Session-salted generate cache keys — identical request content must
     hash identically within a server session but the salt must be part of
     the digest, so a new session (new salt) cannot replay old cache files.
  2. convert_prompt_to_openai_format — image parts precede text (matching
     the training-side chat-template order), string branch is well-formed,
     unknown keys rejected.
  3. render_generate_entry — image-bearing entries pass through the queue
     as structured payloads instead of being rendered to a string.
  4. find_model dynamic discovery accepts HF PEFT-format adapter dirs.
"""

import importlib

import pytest


# ---------------------------------------------------------------------------
# 1. Cache-key session salting
# ---------------------------------------------------------------------------

def _generate_module():
    from cray_infra.api.fastapi.generate import generate
    return generate


def test_contents_hash_stable_within_session():
    generate = _generate_module()
    requests = [{"prompt": "hello", "model": "m", "max_tokens": 4}]
    a = generate.get_contents_hash(requests).hexdigest()
    b = generate.get_contents_hash(requests).hexdigest()
    assert a == b


def test_contents_hash_differs_for_different_content():
    generate = _generate_module()
    a = generate.get_contents_hash([{"prompt": "hello"}]).hexdigest()
    b = generate.get_contents_hash([{"prompt": "world"}]).hexdigest()
    assert a != b


def test_contents_hash_includes_session_salt(monkeypatch):
    """A new server session (different salt) must produce different cache
    keys for identical content — this is the property that prevents stale
    cross-session replay."""
    generate = _generate_module()
    requests = [{"prompt": "hello"}]
    original = generate.get_contents_hash(requests).hexdigest()
    monkeypatch.setattr(generate, "_SERVER_SESSION_SALT", "other-session")
    resalted = generate.get_contents_hash(requests).hexdigest()
    assert original != resalted


# ---------------------------------------------------------------------------
# 2. convert_prompt_to_openai_format ordering
# ---------------------------------------------------------------------------

def _convert():
    from cray_infra.one_server.create_generate_worker import (
        convert_prompt_to_openai_format,
    )
    return convert_prompt_to_openai_format


def test_images_precede_text_in_content_parts():
    convert = _convert()
    messages = convert({"text": "caption this", "images": ["data:image/jpeg;base64,AAA"]})
    content = messages[0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[-1]["type"] == "text"


def test_multiple_images_all_precede_text():
    convert = _convert()
    messages = convert(
        {"text": "t", "images": ["data:a", "data:b", "data:c"]}
    )
    types = [part["type"] for part in messages[0]["content"]]
    assert types == ["image_url", "image_url", "image_url", "text"]


def test_string_prompt_produces_plain_user_message():
    convert = _convert()
    messages = convert("plain prompt")
    assert messages == [{"role": "user", "content": "plain prompt"}]


def test_unknown_prompt_keys_rejected():
    convert = _convert()
    with pytest.raises(ValueError):
        convert({"text": "t", "bogus": 1})


# ---------------------------------------------------------------------------
# 3. render_generate_entry image passthrough
# ---------------------------------------------------------------------------

def _render():
    from cray_infra.api.fastapi.chat_completions.render_generate_entry import (
        render_generate_entry,
    )
    return render_generate_entry


def test_image_entry_passes_through_as_payload():
    render = _render()
    entry = {"text": "t", "images": ["data:image/jpeg;base64,AAA"]}
    out = render(entry, model="m")
    assert out == {"text": "t", "images": ["data:image/jpeg;base64,AAA"]}


def test_image_entry_without_text_is_rejected():
    from fastapi import HTTPException

    render = _render()
    with pytest.raises(HTTPException):
        render({"images": ["data:x"]}, model="m")


def test_plain_string_entry_unchanged():
    render = _render()
    assert render("raw prompt", model="m") == "raw prompt"


def test_prompt_dict_entry_unchanged():
    render = _render()
    assert render({"prompt": "raw prompt"}, model="m") == "raw prompt"


# ---------------------------------------------------------------------------
# 4. find_model discovers PEFT-format adapter dirs
# ---------------------------------------------------------------------------

def _make_manager(monkeypatch, tmp_path):
    from cray_infra.training import vllm_model_manager as vmm

    monkeypatch.setattr(
        vmm,
        "get_config",
        lambda: {"model": "base-model", "training_job_directory": str(tmp_path)},
    )
    manager = vmm.VLLMModelManager()
    return manager


def test_find_model_discovers_peft_dir(monkeypatch, tmp_path):
    adapter = tmp_path / ("a" * 64)
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"stub")
    manager = _make_manager(monkeypatch, tmp_path)
    assert manager.find_model("a" * 64) == "a" * 64


def test_find_model_discovers_pt_dir(monkeypatch, tmp_path):
    adapter = tmp_path / ("b" * 64)
    adapter.mkdir()
    (adapter / "checkpoint_1.pt").write_bytes(b"stub")
    manager = _make_manager(monkeypatch, tmp_path)
    assert manager.find_model("b" * 64) == "b" * 64


def test_find_model_rejects_empty_dir(monkeypatch, tmp_path):
    adapter = tmp_path / ("c" * 64)
    adapter.mkdir()
    manager = _make_manager(monkeypatch, tmp_path)
    assert manager.find_model("c" * 64) is None
