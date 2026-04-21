# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from asyncio import Future
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
import yaml
from pydantic import ValidationError

from nemo_gym.base_resources_server import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.docent_utils import (
    DocentCollectionTarget,
    _build_docent_agent_run,
    is_docent_logging_requested,
    log_rollouts_to_docent,
    validate_docent_logging_requirements,
)
from nemo_gym.global_config import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper


class TestRolloutCollection:
    def test_docent_logging_args_are_mutually_exclusive(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="mutually exclusive"):
            RolloutCollectionConfig(
                agent_name="my_agent",
                input_jsonl_fpath=str(tmp_path / "input.jsonl"),
                output_jsonl_fpath=str(tmp_path / "out.jsonl"),
                docent_log_to_new_collection="new",
                docent_log_to_existing_collection="existing",
            )

    def test_preprocess_rows_with_prompt_config(self, tmp_path: Path) -> None:
        """prompt_config builds responses_create_params.input from template."""
        prompt_path = tmp_path / "prompt.yaml"
        prompt_path.write_text(yaml.dump({"system": "You are a math tutor.", "user": "Solve: {question}"}))

        fpath = tmp_path / "input.jsonl"
        rows = [
            {"question": "What is 2+2?", "expected_answer": "4"},
            {"question": "What is 3*5?", "expected_answer": "15"},
        ]
        fpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            prompt_config=str(prompt_path),
            num_repeats=1,
        )

        result = RolloutCollectionHelper._preprocess_rows_from_config(None, config)

        assert len(result) == 2
        assert result[0]["responses_create_params"]["input"] == [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "Solve: What is 2+2?"},
        ]
        assert result[0]["expected_answer"] == "4"
        assert result[1]["responses_create_params"]["input"][1]["content"] == "Solve: What is 3*5?"

    def test_preprocess_rows_prompt_config_rejects_prebaked(self, tmp_path: Path) -> None:
        """prompt_config raises when rows already have responses_create_params.input."""
        prompt_path = tmp_path / "prompt.yaml"
        prompt_path.write_text(yaml.dump({"user": "{question}"}))

        fpath = tmp_path / "input.jsonl"
        rows = [{"question": "test", "responses_create_params": {"input": [{"role": "user", "content": "baked"}]}}]
        fpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            prompt_config=str(prompt_path),
        )

        with pytest.raises(ValueError, match="mutually exclusive"):
            RolloutCollectionHelper._preprocess_rows_from_config(None, config)

    def test_preprocess_rows_prompt_config_preserves_rcp_fields(self, tmp_path: Path) -> None:
        """prompt_config preserves other responses_create_params fields like tools."""
        prompt_path = tmp_path / "prompt.yaml"
        prompt_path.write_text(yaml.dump({"user": "{question}"}))

        fpath = tmp_path / "input.jsonl"
        rows = [{"question": "test", "responses_create_params": {"tools": [{"type": "function", "name": "calc"}]}}]
        fpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            prompt_config=str(prompt_path),
            num_repeats=1,
        )

        result = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert result[0]["responses_create_params"]["tools"] == [{"type": "function", "name": "calc"}]
        assert result[0]["responses_create_params"]["input"] == [{"role": "user", "content": "test"}]

    def test_preprocess_rows_from_config(self, tmp_path: Path) -> None:
        fpath = tmp_path / "input.jsonl"
        samples = [json.dumps({"responses_create_params": {"input": []}, "x": i}) for i in range(10)]
        fpath.write_text("\n".join(samples) + "\n")

        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath="abcd",
            limit=3,
            num_repeats=2,
            num_repeats_add_seed=True,
            num_samples_in_parallel=None,
            responses_create_params=dict(temperature=0.1),
        )

        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert rows == [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "responses_create_params": {"input": [], "seed": 0, "temperature": 0.1},
                "x": 0,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "responses_create_params": {"input": [], "seed": 1, "temperature": 0.1},
                "x": 0,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 0,
                "responses_create_params": {"input": [], "seed": 0, "temperature": 0.1},
                "x": 1,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 1,
                "responses_create_params": {"input": [], "seed": 1, "temperature": 0.1},
                "x": 1,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 0,
                "responses_create_params": {"input": [], "seed": 0, "temperature": 0.1},
                "x": 2,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 1,
                "responses_create_params": {"input": [], "seed": 1, "temperature": 0.1},
                "x": 2,
                "agent_ref": {"name": "my_agent"},
            },
        ]

    async def test_run_from_config_sanity(self, tmp_path: Path) -> None:
        input_jsonl_fpath = tmp_path / "input.jsonl"
        samples = [
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my agent name"}, "x": i})
            for i in range(10)
        ]
        input_jsonl_fpath.write_text("\n".join(samples) + "\n")
        output_jsonl_fpath = tmp_path / "output.jsonl"

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            limit=3,
            num_repeats=2,
        )

        class TestRolloutCollectionHelper(RolloutCollectionHelper):
            def run_examples(
                self,
                examples: list[dict],
                *args,
                **kwargs,
            ):
                futures = []
                for example in examples:
                    future = Future()
                    # (row, result)
                    future.set_result((example, {"response": {"usage": {"abc usage": 1}}}))
                    futures.append(future)

                return futures

            async def _call_aggregate_metrics(self, results, rows, output_fpath):
                """Compute aggregate metrics locally (no server needed)."""
                stripped = [{k: v for k, v in r.items() if k not in ("responses_create_params",)} for r in results]
                agg = compute_aggregate_metrics(stripped)
                metrics_fpath = output_fpath.with_stem(output_fpath.stem + "_aggregate_metrics").with_suffix(".json")
                metrics_fpath.write_bytes(
                    orjson.dumps(
                        [{"agent_ref": {"name": "my agent name"}, **agg.model_dump()}], option=orjson.OPT_INDENT_2
                    )
                )
                return metrics_fpath

        actual_returned_results = await TestRolloutCollectionHelper().run_from_config(config)

        expected_results = [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 1,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 1,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
        ]

        assert expected_results == actual_returned_results

        expected_materialized_inputs_len = 6
        with (tmp_path / "output_materialized_inputs.jsonl").open() as f:
            actual_materialized_inputs_len = len(list(f))
        assert expected_materialized_inputs_len == actual_materialized_inputs_len

        with output_jsonl_fpath.open() as f:
            actual_written_results = [json.loads(line) for line in f]
        assert expected_results == actual_written_results

        aggregate_metrics_fpath = tmp_path / "output_aggregate_metrics.json"
        actual_aggregate_metrics = json.loads(aggregate_metrics_fpath.read_text())
        expected_aggregate_metrics = [
            {
                "agent_ref": {"name": "my agent name"},
                "agent_metrics": {
                    "mean/abc usage": 1.0,
                    "max/abc usage": 1,
                    "min/abc usage": 1,
                    "median/abc usage": 1.0,
                    "std/abc usage": 0.0,
                },
                "key_metrics": {"mean/abc usage": 1.0},
                "group_level_metrics": actual_aggregate_metrics[0]["group_level_metrics"],
            }
        ]
        assert expected_aggregate_metrics == actual_aggregate_metrics

    async def test_run_from_config_sorted(self, tmp_path: Path) -> None:
        input_jsonl_fpath = tmp_path / "input.jsonl"
        samples = [
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my agent name"}, "x": i})
            for i in range(10)
        ]
        input_jsonl_fpath.write_text("\n".join(samples) + "\n")
        output_jsonl_fpath = tmp_path / "output.jsonl"

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            limit=3,
            num_repeats=2,
        )

        class TestRolloutCollectionHelper(RolloutCollectionHelper):
            def run_examples(
                self,
                examples: list[dict],
                *args,
                **kwargs,
            ):
                futures = []
                for example in examples:
                    future = Future()
                    # (row, result)
                    future.set_result((example, {"response": {"usage": {"abc usage": 1}}}))
                    futures.append(future)

                # Reverse!
                futures = reversed(futures)

                return futures

            async def _call_aggregate_metrics(self, results, rows, output_fpath):
                return None

        actual_returned_results = await TestRolloutCollectionHelper().run_from_config(config)

        expected_results = [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 1,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 1,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
        ]

        assert expected_results == actual_returned_results

    def test_load_from_cache(self, tmp_path: Path) -> None:
        input_jsonl_fpath = tmp_path / "input.jsonl"
        materialized_inputs_jsonl_fpath = tmp_path / "output_materialized_inputs.jsonl"

        materialized_inputs = [
            {"_ng_task_index": 0, "_ng_rollout_index": 0, "input": True},
            {"_ng_task_index": 0, "_ng_rollout_index": 1, "input": True},
            {"_ng_task_index": 1, "_ng_rollout_index": 0, "input": True},
            {"_ng_task_index": 1, "_ng_rollout_index": 1, "input": True},
            {"_ng_task_index": 2, "_ng_rollout_index": 0, "input": True},
            {"_ng_task_index": 2, "_ng_rollout_index": 1, "input": True},
        ]
        materialized_inputs_jsonl_fpath.write_bytes(b"\n".join(map(orjson.dumps, materialized_inputs)) + b"\n")

        outputs = [
            {"_ng_task_index": 0, "_ng_rollout_index": 0, "output": True},
            {"_ng_task_index": 0, "_ng_rollout_index": 1, "output": True},
            {"_ng_task_index": 1, "_ng_rollout_index": 1, "output": True},
        ]
        output_jsonl_fpath = tmp_path / "output.jsonl"
        output_jsonl_fpath.write_bytes(b"\n".join(map(orjson.dumps, outputs)) + b"\n")

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            limit=3,
            num_repeats=2,
        )

        actual_returned_results = RolloutCollectionHelper()._load_from_cache(config)

        expected_results = (
            [
                {"_ng_task_index": 1, "_ng_rollout_index": 0, "input": True},
                {"_ng_task_index": 2, "_ng_rollout_index": 0, "input": True},
                {"_ng_task_index": 2, "_ng_rollout_index": 1, "input": True},
            ],
            [
                {"_ng_task_index": 0, "_ng_rollout_index": 0, "input": True},
                {"_ng_task_index": 0, "_ng_rollout_index": 1, "input": True},
                {"_ng_task_index": 1, "_ng_rollout_index": 1, "input": True},
            ],
            [
                {"_ng_task_index": 0, "_ng_rollout_index": 0, "output": True},
                {"_ng_task_index": 0, "_ng_rollout_index": 1, "output": True},
                {"_ng_task_index": 1, "_ng_rollout_index": 1, "output": True},
            ],
            [
                [orjson.dumps({"_ng_task_index": 0, "_ng_rollout_index": 0, "output": True})],
                [orjson.dumps({"_ng_task_index": 0, "_ng_rollout_index": 1, "output": True})],
                [orjson.dumps({"_ng_task_index": 1, "_ng_rollout_index": 1, "output": True})],
            ],
        )

        assert expected_results == actual_returned_results

    def test_build_docent_agent_run_with_transitions(self) -> None:
        result = {
            "_ng_task_index": 3,
            "_ng_rollout_index": 2,
            "agent_ref": {"name": "my_agent"},
            "responses_create_params": {
                "input": [{"role": "user", "content": "Solve this problem."}],
            },
            "response": {
                "model": "test-model",
                "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
                "output": [
                    [
                        {"role": "user", "type": "message", "content": "Solve this problem."},
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "calculator",
                            "arguments": '{"expr": "2+2"}',
                        },
                    ],
                    [
                        {"role": "user", "type": "message", "content": "Solve this problem."},
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "calculator",
                            "arguments": '{"expr": "2+2"}',
                        },
                        {
                            "type": "function_call_output",
                            "call_id": "call_1",
                            "output": "4",
                        },
                        {
                            "role": "assistant",
                            "type": "message",
                            "content": [{"type": "output_text", "text": "The answer is 4."}],
                        },
                    ],
                ],
            },
            "reward": 1.0,
        }

        agent_run = _build_docent_agent_run(result=result)

        assert agent_run.name == "my_agent/task/3/rollout/2"
        assert agent_run.description is None
        assert len(agent_run.transcripts) == 1

        messages = agent_run.transcripts[0].messages
        assert [message.role for message in messages] == ["user", "assistant", "tool", "assistant"]
        assert messages[0].text == "Solve this problem."
        assert messages[1].tool_calls is not None
        assert len(messages[1].tool_calls) == 1
        assert messages[1].tool_calls[0].id == "call_1"
        assert messages[1].tool_calls[0].function == "calculator"
        assert messages[1].tool_calls[0].arguments == {"expr": "2+2"}
        assert messages[2].function == "calculator"
        assert messages[2].tool_call_id == "call_1"
        assert messages[2].text == "4"
        assert messages[3].text == "The answer is 4."

        assert agent_run.metadata["scores"] == {"reward": 1.0}
        assert agent_run.metadata["nemogym"] == {
            "task_index": 3,
            "rollout_index": 2,
            "agent_ref": {"name": "my_agent"},
        }
        assert agent_run.transcripts[0].metadata["source"]["response_model"] == "test-model"
        assert agent_run.transcripts[0].metadata["source"]["input_item_count"] == 0

    async def test_run_from_config_docent_requires_api_key(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("DOCENT_API_KEY", raising=False)
        input_jsonl_fpath = tmp_path / "input.jsonl"
        input_jsonl_fpath.write_text(
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my agent name"}}) + "\n"
        )

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(tmp_path / "output.jsonl"),
            docent_log_to_existing_collection="collection-123",
        )

        class TestRolloutCollectionHelper(RolloutCollectionHelper):
            def run_examples(self, examples: list[dict], *args, **kwargs):
                raise AssertionError("run_examples should not be reached when Docent config is invalid")

        with pytest.raises(ValueError, match="DOCENT_API_KEY"):
            await TestRolloutCollectionHelper().run_from_config(config)

    def test_validate_docent_logging_requirements_requires_docent_dependency(self, monkeypatch) -> None:
        monkeypatch.setenv("DOCENT_API_KEY", "test-docent-key")

        def _raise_import_error():
            raise ImportError("Docent logging requires the optional `nemo-gym[docent]` dependency extra.")

        monkeypatch.setattr("nemo_gym.docent_utils._get_docent_client_class", _raise_import_error)

        with pytest.raises(ImportError, match="nemo-gym\\[docent\\]"):
            validate_docent_logging_requirements()

    def test_is_docent_logging_requested(self) -> None:
        assert is_docent_logging_requested(log_to_new_collection="", log_to_existing_collection=None) is True
        assert is_docent_logging_requested(log_to_new_collection=None, log_to_existing_collection="abc") is True
        assert is_docent_logging_requested(log_to_new_collection=None, log_to_existing_collection=None) is False

    async def test_run_from_config_uploads_to_docent_new_collection(self, tmp_path: Path, monkeypatch) -> None:
        input_jsonl_fpath = tmp_path / "input.jsonl"
        samples = [
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my agent name"}, "x": i})
            for i in range(3)
        ]
        input_jsonl_fpath.write_text("\n".join(samples) + "\n")

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(tmp_path / "output.jsonl"),
            num_repeats=2,
            docent_log_to_new_collection="",
        )

        validate_calls = []
        docent_calls = []

        monkeypatch.setattr(
            "nemo_gym.rollout_collection.validate_docent_logging_requirements",
            lambda: validate_calls.append(True),
        )
        monkeypatch.setattr(
            "nemo_gym.rollout_collection.log_rollouts_to_docent",
            lambda **kwargs: docent_calls.append(kwargs) or len(kwargs["results"]),
        )

        class TestRolloutCollectionHelper(RolloutCollectionHelper):
            def run_examples(self, examples: list[dict], *args, **kwargs):
                futures = []
                for example in examples:
                    future = Future()
                    future.set_result((example, {"response": {"usage": {"abc usage": 1}}}))
                    futures.append(future)
                return futures

            async def _call_aggregate_metrics(self, results, rows, output_fpath):
                return None

        actual_results = await TestRolloutCollectionHelper().run_from_config(config)

        assert len(actual_results) == 6
        assert len(validate_calls) == 1
        assert len(docent_calls) == 1
        assert docent_calls[0]["log_to_new_collection"] == ""
        assert docent_calls[0]["resume_from_cache"] is False
        assert docent_calls[0]["initial_result_count"] == 0
        assert len(docent_calls[0]["results"]) == 6

    async def test_run_from_config_docent_existing_collection_passes_resume_state(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        materialized_inputs = [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "responses_create_params": {"input": []},
                "agent_ref": {"name": "my agent"},
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "responses_create_params": {"input": []},
                "agent_ref": {"name": "my agent"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 0,
                "responses_create_params": {"input": []},
                "agent_ref": {"name": "my agent"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 1,
                "responses_create_params": {"input": []},
                "agent_ref": {"name": "my agent"},
            },
        ]
        materialized_inputs_jsonl_fpath = tmp_path / "output_materialized_inputs.jsonl"
        materialized_inputs_jsonl_fpath.write_bytes(b"\n".join(map(orjson.dumps, materialized_inputs)) + b"\n")

        cached_outputs = [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "responses_create_params": {"input": []},
                "response": {"usage": {"tokens": 10}},
                "agent_ref": {"name": "my agent"},
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "responses_create_params": {"input": []},
                "response": {"usage": {"tokens": 11}},
                "agent_ref": {"name": "my agent"},
            },
        ]
        output_jsonl_fpath = tmp_path / "output.jsonl"
        output_jsonl_fpath.write_bytes(b"\n".join(map(orjson.dumps, cached_outputs)) + b"\n")

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "input.jsonl"),
            output_jsonl_fpath=str(output_jsonl_fpath),
            resume_from_cache=True,
            docent_log_to_existing_collection="existing-collection",
        )

        validate_calls = []
        docent_calls = []

        monkeypatch.setattr(
            "nemo_gym.rollout_collection.validate_docent_logging_requirements",
            lambda: validate_calls.append(True),
        )
        monkeypatch.setattr(
            "nemo_gym.rollout_collection.log_rollouts_to_docent",
            lambda **kwargs: docent_calls.append(kwargs) or len(kwargs["results"]),
        )

        class TestRolloutCollectionHelper(RolloutCollectionHelper):
            def run_examples(self, examples: list[dict], *args, **kwargs):
                futures = []
                for example in examples:
                    future = Future()
                    future.set_result((example, {"response": {"usage": {"tokens": 99}}}))
                    futures.append(future)
                return futures

            async def _call_aggregate_metrics(self, results, rows, output_fpath):
                return None

        actual_results = await TestRolloutCollectionHelper().run_from_config(config)

        assert len(actual_results) == 4
        assert len(validate_calls) == 1
        assert len(docent_calls) == 1
        assert docent_calls[0]["log_to_existing_collection"] == "existing-collection"
        assert docent_calls[0]["resume_from_cache"] is True
        assert docent_calls[0]["initial_result_count"] == 2
        assert len(docent_calls[0]["results"]) == 4

    def test_log_rollouts_to_docent_existing_collection_skips_cached_results(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        results = [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "responses_create_params": {"input": []},
                "response": {"usage": {"tokens": 10}},
                "agent_ref": {"name": "my agent"},
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "responses_create_params": {"input": []},
                "response": {"usage": {"tokens": 11}},
                "agent_ref": {"name": "my agent"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 0,
                "responses_create_params": {"input": []},
                "response": {"usage": {"tokens": 12}},
                "agent_ref": {"name": "my agent"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 1,
                "responses_create_params": {"input": []},
                "response": {"usage": {"tokens": 13}},
                "agent_ref": {"name": "my agent"},
            },
        ]
        upload_calls = []

        monkeypatch.setattr(
            "nemo_gym.docent_utils._initialize_docent_collection_target",
            lambda **kwargs: DocentCollectionTarget(
                client=object(),
                collection_id="existing-collection",
                collection_name=None,
                is_new_collection=False,
            ),
        )
        monkeypatch.setattr(
            "nemo_gym.docent_utils._upload_rollouts_to_docent_collection",
            lambda **kwargs: upload_calls.append(kwargs) or len(kwargs["results"]),
        )

        uploaded_count = log_rollouts_to_docent(
            output_fpath=tmp_path / "output.jsonl",
            log_to_new_collection=None,
            log_to_existing_collection="existing-collection",
            results=results,
            resume_from_cache=True,
            initial_result_count=2,
        )

        assert uploaded_count == 2
        assert len(upload_calls) == 1
        assert upload_calls[0]["results"] == results[2:]

    async def test_call_aggregate_metrics(self, tmp_path: Path) -> None:
        """Test _call_aggregate_metrics with a mocked server client."""

        agg = AggregateMetrics(
            agent_metrics={"mean/reward": 0.5},
            key_metrics={"mean/reward": 0.5},
            group_level_metrics=[{"mean/reward": 1.0}, {"mean/reward": 0.0}],
        )

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.read = AsyncMock(return_value=orjson.dumps(agg.model_dump()))
        mock_response.status = 200

        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock(return_value=mock_response)

        class MockHelper(RolloutCollectionHelper):
            def setup_server_client(self):
                return mock_server_client

        helper = MockHelper()

        rows = [
            {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0},
            {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1},
            {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0},
            {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 1},
        ]
        results = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0, "response": {"usage": {"tokens": 10}}},
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1, "reward": 0.0, "response": {"usage": {"tokens": 12}}},
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0, "response": {"usage": {"tokens": 8}}},
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 1, "reward": 0.0, "response": {"usage": {"tokens": 15}}},
        ]

        output_fpath = tmp_path / "output.jsonl"
        metrics_fpath = await helper._call_aggregate_metrics(results, rows, output_fpath)

        # Verify file was written
        assert metrics_fpath is not None
        assert metrics_fpath.exists()
        written = json.loads(metrics_fpath.read_text())
        assert len(written) == 1
        assert written[0][AGENT_REF_KEY_NAME] == {"name": "my_agent"}
        assert written[0]["agent_metrics"]["mean/reward"] == 0.5
        assert written[0]["key_metrics"]["mean/reward"] == 0.5
        assert len(written[0]["group_level_metrics"]) == 2

        # Verify server_client.post was called with stripped data (usage preserved)
        call_kwargs = mock_server_client.post.call_args
        sent_request = call_kwargs.kwargs["json"]
        sent_data = (
            sent_request.verify_responses
            if isinstance(sent_request, AggregateMetricsRequest)
            else sent_request["verify_responses"]
        )
        for item in sent_data:
            assert "responses_create_params" not in item
            assert "usage" in item["response"]

    async def test_call_aggregate_metrics_multiple_agents(self, tmp_path: Path) -> None:
        """Test _call_aggregate_metrics with multiple agents runs concurrently via as_completed."""

        agg_a = AggregateMetrics(
            agent_metrics={"mean/reward": 1.0},
            key_metrics={"mean/reward": 1.0},
            group_level_metrics=[{"mean/reward": 1.0}],
        )
        agg_b = AggregateMetrics(
            agent_metrics={"mean/reward": 0.0},
            key_metrics={"mean/reward": 0.0},
            group_level_metrics=[{"mean/reward": 0.0}],
        )

        # Return different responses per agent based on server_name
        async def mock_post(server_name, **kwargs):
            agg = agg_a if server_name == "agent_a" else agg_b
            resp = AsyncMock()
            resp.raise_for_status = MagicMock()
            resp.read = AsyncMock(return_value=orjson.dumps(agg.model_dump()))
            resp.status = 200
            return resp

        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock(side_effect=mock_post)

        class MockHelper(RolloutCollectionHelper):
            def setup_server_client(self):
                return mock_server_client

        helper = MockHelper()

        rows = [
            {AGENT_REF_KEY_NAME: {"name": "agent_a"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0},
            {AGENT_REF_KEY_NAME: {"name": "agent_a"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1},
            {AGENT_REF_KEY_NAME: {"name": "agent_b"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0},
            {AGENT_REF_KEY_NAME: {"name": "agent_b"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1},
        ]
        results = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0, "response": {"usage": {"tokens": 10}}},
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1, "reward": 1.0, "response": {"usage": {"tokens": 12}}},
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 0.0, "response": {"usage": {"tokens": 8}}},
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1, "reward": 0.0, "response": {"usage": {"tokens": 15}}},
        ]

        output_fpath = tmp_path / "output.jsonl"
        metrics_fpath = await helper._call_aggregate_metrics(results, rows, output_fpath)

        written = json.loads(metrics_fpath.read_text())
        assert len(written) == 2

        # Both agents should be present (order may vary due to as_completed)
        agent_names = {entry[AGENT_REF_KEY_NAME]["name"] for entry in written}
        assert agent_names == {"agent_a", "agent_b"}

        for entry in written:
            if entry[AGENT_REF_KEY_NAME]["name"] == "agent_a":
                assert entry["agent_metrics"]["mean/reward"] == 1.0
            else:
                assert entry["agent_metrics"]["mean/reward"] == 0.0

        # Verify both agents were called
        assert mock_server_client.post.call_count == 2

    async def test_call_aggregate_metrics_empty(self, tmp_path: Path) -> None:
        """_call_aggregate_metrics returns None for empty results."""
        helper = RolloutCollectionHelper()
        output_fpath = tmp_path / "output.jsonl"
        result = await helper._call_aggregate_metrics([], [], output_fpath)
        assert result is None
