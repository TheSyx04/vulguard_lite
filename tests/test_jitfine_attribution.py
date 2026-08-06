import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from vulguard_lite.attribution.export import load_jsonl, write_jsonl, write_token_csv
from vulguard_lite.attribution.jitfine_attention import (
    aggregate_cls_attention,
    rank_code_tokens,
)
from vulguard_lite.models.jitfine.dataset import convert_examples_to_features
from vulguard_lite.models.jitfine.model import Model


class DummyTokenizer:
    cls_token = "<s>"
    sep_token = "</s>"

    def __init__(self):
        self.vocabulary = {"<s>": 0, "</s>": 2, "<ADD>": 3, "<REMOVE>": 4}

    def tokenize(self, text):
        return text.split() if text else []

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary) + 10
            ids.append(self.vocabulary[token])
        return ids


class DummyEncoderOutput:
    def __init__(self, hidden, attentions):
        self.last_hidden_state = hidden
        self.attentions = attentions

    def __getitem__(self, index):
        if index == 0:
            return self.last_hidden_state
        raise IndexError(index)


class DummyEncoder(nn.Module):
    def __init__(self, hidden_size=4, layers=2, heads=2):
        super().__init__()
        self.embedding = nn.Embedding(32, hidden_size)
        self.layers = layers
        self.heads = heads

    def forward(self, input_ids, attention_mask, output_attentions=None, return_dict=True):
        hidden = self.embedding(input_ids)
        attentions = None
        if output_attentions:
            batch, sequence_length = input_ids.shape
            base = torch.ones(
                batch, self.heads, sequence_length, sequence_length,
                dtype=hidden.dtype,
                device=hidden.device,
            )
            base = base / sequence_length
            attentions = tuple(base for _ in range(self.layers))
        return DummyEncoderOutput(hidden, attentions)


class JITFinePreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DummyTokenizer()
        self.item = (
            "abc",
            "<ADD>new value <REMOVE>old value",
            "fix bug",
            1,
            self.tokenizer,
            {"max_msg_length": 64},
            [0.0] * 14,
        )

    def test_regions_follow_exact_input_positions(self):
        feature = convert_examples_to_features(self.item, return_metadata=True)
        metadata = feature.attribution_metadata

        self.assertEqual(feature.input_tokens[:9], [
            "<s>", "fix", "bug", "<ADD>", "new", "value",
            "<REMOVE>", "old", "value",
        ])
        self.assertEqual(metadata["sequence_regions"][:10], [
            "cls", "message", "message", "add_marker", "added", "added",
            "remove_marker", "removed", "removed", "sep",
        ])
        self.assertEqual(metadata["add_marker_position"], 3)
        self.assertEqual(metadata["remove_marker_position"], 6)
        self.assertFalse(metadata["truncated"])
        self.assertEqual(len(feature.input_ids), 512)
        self.assertEqual(len(metadata["sequence_regions"]), 512)

    def test_metadata_does_not_change_legacy_tensors(self):
        legacy = convert_examples_to_features(self.item)
        attributed = convert_examples_to_features(self.item, return_metadata=True)
        self.assertEqual(legacy.input_ids, attributed.input_ids)
        self.assertEqual(legacy.input_mask, attributed.input_mask)
        self.assertEqual(legacy.input_tokens, attributed.input_tokens)
        self.assertIsNone(legacy.attribution_metadata)

    def test_truncated_tokens_are_counted_not_scored(self):
        long_item = list(self.item)
        long_item[1] = "<ADD>" + " ".join(f"a{i}" for i in range(600)) + " <REMOVE>old"
        feature = convert_examples_to_features(tuple(long_item), return_metadata=True)
        metadata = feature.attribution_metadata
        self.assertTrue(metadata["truncated"])
        self.assertGreater(metadata["truncated_content_tokens"], 0)
        self.assertGreater(metadata["truncated_added_tokens"], 0)
        self.assertNotIn("removed", metadata["sequence_regions"])

    def test_malformed_markers_are_explicit(self):
        malformed = list(self.item)
        malformed[1] = "new value without markers"
        with self.assertRaisesRegex(ValueError, "missing_or_malformed"):
            convert_examples_to_features(tuple(malformed), return_metadata=True)


class AttentionTests(unittest.TestCase):
    def test_attention_aggregation_shapes(self):
        first = torch.zeros(1, 2, 4, 4)
        second = torch.zeros(1, 2, 4, 4)
        first[:, :, 0, :] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        second[:, :, 0, :] = torch.tensor([4.0, 3.0, 2.0, 1.0])
        last = aggregate_cls_attention((first, second), "last_layer_cls_mean")
        all_layers = aggregate_cls_attention((first, second), "all_layers_cls_mean")
        self.assertEqual(tuple(last.shape), (1, 4))
        self.assertTrue(torch.equal(last[0], torch.tensor([4.0, 3.0, 2.0, 1.0])))
        self.assertTrue(torch.equal(all_layers[0], torch.tensor([2.5, 2.5, 2.5, 2.5])))

    def test_ranking_filters_and_preserves_occurrences(self):
        ranked = rank_code_tokens(
            input_tokens=["<s>", "msg", "<ADD>", "x", "x", "<REMOVE>", "y", "</s>"],
            input_ids=list(range(8)),
            sequence_regions=["cls", "message", "add_marker", "added", "added", "remove_marker", "removed", "sep"],
            token_scores=[0.0, 9.0, 8.0, 0.2, 0.9, 7.0, 0.5, 6.0],
            strategy="last_layer_cls_mean",
        )
        self.assertEqual([item.sequence_position for item in ranked], [4, 6, 3])
        self.assertEqual([item.token for item in ranked].count("x"), 2)
        self.assertEqual(ranked[0].normalized_score, 1.0)
        self.assertEqual(ranked[-1].normalized_score, 0.0)

    def test_structured_attention_preserves_probability(self):
        torch.manual_seed(1)
        config = SimpleNamespace(
            feature_size=2,
            hidden_size=4,
            hidden_dropout_prob=0.0,
            num_hidden_layers=2,
        )
        model = Model(DummyEncoder(), config, DummyTokenizer(), {})
        model.eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        mask = torch.ones_like(input_ids)
        manual = torch.tensor([[0.1, 0.2]])
        normal = model(input_ids, mask, manual)
        attributed = model(
            input_ids,
            mask,
            manual,
            output_attentions=True,
            return_attribution_data=True,
        )
        self.assertTrue(torch.equal(normal, attributed["probability"]))
        self.assertEqual(len(attributed["attentions"]), 2)


class ExportTests(unittest.TestCase):
    def test_jsonl_and_csv_escape_token_text(self):
        record = {
            "commit_id": "abc",
            "status": "succeeded",
            "model_name": "jitfine",
            "prediction_score": 0.8,
            "predicted_label": 1,
            "attention_strategy": "last_layer_cls_mean",
            "truncated": False,
            "ranked_tokens": [{
                "rank": 1,
                "sequence_position": 3,
                "token": "a,\"b",
                "token_id": 7,
                "change_type": "added",
                "raw_score": 0.5,
                "normalized_score": 1.0,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            jsonl_path = os.path.join(directory, "tokens.jsonl")
            csv_path = os.path.join(directory, "tokens.csv")
            write_jsonl(jsonl_path, [record])
            write_token_csv(csv_path, [record])
            self.assertEqual(load_jsonl(jsonl_path), [record])
            with open(csv_path, "r", encoding="utf-8") as handle:
                csv_content = handle.read()
            self.assertIn('"a,""b"', csv_content)


if __name__ == "__main__":
    unittest.main()
