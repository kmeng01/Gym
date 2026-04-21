from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


DEFAULT_DOCENT_COLLECTION_PREFIX = "NeMo Gym rollouts"
DOCENT_DEPENDENCY_ERROR = "Docent logging requires the optional `nemo-gym[docent]` dependency extra."
DOCENT_AGENT_RUN_DESCRIPTION = "Rollout collected with NeMo Gym."
DOCENT_COLLECTION_DESCRIPTION = "Rollouts collected with NeMo Gym."


@dataclass
class DocentCollectionTarget:
    client: Any
    collection_id: str
    collection_name: Optional[str]
    is_new_collection: bool


def validate_docent_logging_requirements() -> None:
    """Fail fast if Docent logging was requested but its runtime requirements are missing."""
    _get_docent_api_key()
    _get_docent_client_class()
    _get_docent_nemogym_rollout_converter()


def log_rollouts_to_docent(
    *,
    output_fpath: Path,
    log_to_new_collection: Optional[str],
    log_to_existing_collection: Optional[str],
    results: list[dict[str, Any]],
    resume_from_cache: bool,
    initial_result_count: int,
) -> int:
    """Upload rollouts to a Docent collection.

    When a rollout run resumes from cache, ``results`` contains both:
    - cached rollouts loaded before new work started, and
    - new rollouts collected during the current invocation.

    ``initial_result_count`` records how many entries were already present in
    ``results`` before new rollouts were appended. That count is only used when
    logging to an existing Docent collection while ``resume_from_cache=True``:
    in that case we upload only ``results[initial_result_count:]`` so cached
    rollouts are not uploaded twice. When logging to a new collection, all
    results are uploaded even if the run resumed from cache, because the new
    collection starts empty.
    """
    collection_target = _initialize_docent_collection_target(
        output_fpath=output_fpath,
        log_to_new_collection=log_to_new_collection,
        log_to_existing_collection=log_to_existing_collection,
    )
    if collection_target.is_new_collection:
        print(f"Created Docent collection `{collection_target.collection_name}` ({collection_target.collection_id})")
    else:
        print(f"Using existing Docent collection {collection_target.collection_id}")

    results_to_upload = results
    if resume_from_cache and not collection_target.is_new_collection:
        # ``results`` begins with cached rollouts that are already present in the
        # existing Docent collection, so only upload the newly appended suffix.
        results_to_upload = results[initial_result_count:]

    print(f"Uploading {len(results_to_upload)} rollouts to Docent collection {collection_target.collection_id}")
    uploaded_count = _upload_rollouts_to_docent_collection(
        collection_target=collection_target,
        results=results_to_upload,
        output_fpath=output_fpath,
    )
    print(f"Uploaded {uploaded_count} rollouts to Docent")
    return uploaded_count


def _initialize_docent_collection_target(
    *,
    output_fpath: Path,
    log_to_new_collection: Optional[str],
    log_to_existing_collection: Optional[str],
) -> DocentCollectionTarget:
    api_key = _get_docent_api_key()
    Docent = _get_docent_client_class()

    client = Docent(api_key=api_key)

    if log_to_existing_collection is not None:
        if not client.collection_exists(log_to_existing_collection):
            raise ValueError(f"Docent collection `{log_to_existing_collection}` does not exist.")
        return DocentCollectionTarget(
            client=client,
            collection_id=log_to_existing_collection,
            collection_name=None,
            is_new_collection=False,
        )

    collection_name = log_to_new_collection or _default_collection_name(output_fpath)
    collection_id = client.create_collection(
        name=collection_name,
        description=DOCENT_COLLECTION_DESCRIPTION,
        metadata={
            "source": "nemo_gym",
            "output_jsonl_fpath": str(output_fpath),
        },
    )
    return DocentCollectionTarget(
        client=client,
        collection_id=collection_id,
        collection_name=collection_name,
        is_new_collection=True,
    )


def _upload_rollouts_to_docent_collection(
    *,
    collection_target: DocentCollectionTarget,
    results: list[dict[str, Any]],
    output_fpath: Path,
) -> int:
    if not results:
        return 0

    agent_runs = [_build_docent_agent_run(result=result, output_fpath=output_fpath) for result in results]

    collection_target.client.add_agent_runs(collection_target.collection_id, agent_runs)
    return len(agent_runs)


def is_docent_logging_requested(
    *,
    log_to_new_collection: Optional[str],
    log_to_existing_collection: Optional[str],
) -> bool:
    """Return whether rollout collection should initialize Docent logging at all."""
    return log_to_new_collection is not None or log_to_existing_collection is not None


def _get_docent_api_key() -> str:
    api_key = os.getenv("DOCENT_API_KEY")
    if not api_key:
        raise ValueError("Docent logging requires DOCENT_API_KEY to be set in the environment.")
    return api_key


def _get_docent_client_class() -> Any:
    try:
        from docent import Docent
    except ImportError as exc:  # pragma: no cover - exercised via tests with monkeypatching
        raise ImportError(DOCENT_DEPENDENCY_ERROR) from exc
    return Docent


def _get_docent_nemogym_rollout_converter() -> Any:
    try:
        from docent.sdk.integrations import convert_nemogym_rollout_to_agent_run
    except ImportError as exc:  # pragma: no cover - exercised via tests with monkeypatching
        raise ImportError(DOCENT_DEPENDENCY_ERROR) from exc
    return convert_nemogym_rollout_to_agent_run


def _build_docent_agent_run(*, result: dict[str, Any], output_fpath: Path) -> Any:
    convert_nemogym_rollout_to_agent_run = _get_docent_nemogym_rollout_converter()
    normalized_result = _normalize_rollout_for_docent_sdk(result=result)
    agent_run = convert_nemogym_rollout_to_agent_run(normalized_result)
    _apply_docent_agent_run_compatibility_patches(
        agent_run=agent_run,
        result=result,
        output_fpath=output_fpath,
    )
    return agent_run


def _normalize_rollout_for_docent_sdk(*, result: dict[str, Any]) -> dict[str, Any]:
    """Normalize NeMo Gym rollout payloads to the Docent SDK converter's expected shape.

    NeMo Gym transition rollouts store cumulative response snapshots as ``response.output``
    entries like ``[[...], [...]]``. The Docent SDK converter expects one flat list of
    output items. To preserve NeMo Gym's current semantics, we feed only the final snapshot
    to the converter and clear ``responses_create_params.input`` so the final snapshot
    remains the source of truth for the full transcript.
    """
    normalized_result = deepcopy(result)
    response = normalized_result.get("response")
    if not isinstance(response, dict):
        return normalized_result

    output_items = response.get("output")
    if _response_contains_transitions(response):
        final_snapshot = output_items[-1] if output_items else []
        response["output"] = deepcopy(final_snapshot)

        responses_create_params = normalized_result.get("responses_create_params")
        if not isinstance(responses_create_params, dict):
            normalized_result["responses_create_params"] = {"input": []}
        else:
            responses_create_params["input"] = []
        return normalized_result

    if isinstance(output_items, list):
        response["output"] = deepcopy(output_items)
    return normalized_result


def _apply_docent_agent_run_compatibility_patches(
    *,
    agent_run: Any,
    result: dict[str, Any],
    output_fpath: Path,
) -> None:
    agent_name = ((result.get("agent_ref") or {}).get("name")) or "unknown-agent"
    task_index = result.get("_ng_task_index")
    rollout_index = result.get("_ng_rollout_index")
    response = result.get("response") or {}
    response_usage = response.get("usage") or {}

    score_metadata = {}
    if result.get("reward") is not None:
        score_metadata["reward"] = result["reward"]

    total_tokens = response_usage.get("total_tokens")
    if isinstance(total_tokens, (int, float)):
        score_metadata["total_tokens"] = total_tokens

    input_tokens = response_usage.get("input_tokens")
    if isinstance(input_tokens, (int, float)):
        score_metadata["input_tokens"] = input_tokens

    output_tokens = response_usage.get("output_tokens")
    if isinstance(output_tokens, (int, float)):
        score_metadata["output_tokens"] = output_tokens

    agent_run.name = _build_legacy_docent_agent_run_name(
        agent_name=agent_name,
        task_index=task_index,
        rollout_index=rollout_index,
    )
    agent_run.description = DOCENT_AGENT_RUN_DESCRIPTION

    metadata = dict(agent_run.metadata or {})
    nemo_gym_metadata = dict(metadata.get("nemo_gym") or {})
    nemo_gym_metadata["agent_name"] = agent_name
    nemo_gym_metadata["task_index"] = task_index
    nemo_gym_metadata["rollout_index"] = rollout_index
    nemo_gym_metadata["output_jsonl_fpath"] = str(output_fpath)
    nemo_gym_metadata["raw_rollout"] = result
    if result.get("agent_ref") is not None:
        nemo_gym_metadata["agent_ref"] = result["agent_ref"]
    if response.get("model") is not None:
        nemo_gym_metadata["response_model"] = response["model"]
    metadata["nemo_gym"] = nemo_gym_metadata

    if score_metadata:
        scores = dict(metadata.get("scores") or {})
        scores.update(score_metadata)
        metadata["scores"] = scores
    agent_run.metadata = metadata

    if not getattr(agent_run, "transcripts", None):
        return

    transcript = agent_run.transcripts[0]
    transcript_metadata = dict(transcript.metadata or {})
    source_metadata = dict(transcript_metadata.get("source") or {})
    source_metadata["input_item_count"] = _count_docent_input_items(_get_responses_create_input_payload(result))
    transcript_metadata["source"] = source_metadata
    transcript.metadata = transcript_metadata


def _build_legacy_docent_agent_run_name(
    *,
    agent_name: str,
    task_index: Any,
    rollout_index: Any,
) -> str:
    return f"{agent_name}/task-{task_index}/rollout-{rollout_index}"


def _default_collection_name(output_fpath: Path) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{DEFAULT_DOCENT_COLLECTION_PREFIX} {output_fpath.stem} {timestamp}"


def _response_contains_transitions(response: dict[str, Any]) -> bool:
    output_items = response.get("output") or []
    return bool(output_items) and all(isinstance(item, list) for item in output_items)


def _get_responses_create_input_payload(result: dict[str, Any]) -> Any:
    responses_create_params = result.get("responses_create_params")
    if not isinstance(responses_create_params, dict):
        return None
    return responses_create_params.get("input")


def _count_docent_input_items(input_payload: Any) -> int:
    if input_payload is None:
        return 0
    if isinstance(input_payload, list):
        return len(input_payload)
    return 1
