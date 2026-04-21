from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


DEFAULT_DOCENT_COLLECTION_PREFIX = "NeMo Gym rollouts"
DOCENT_DEPENDENCY_ERROR = "Docent logging requires the optional `nemo-gym[docent]` dependency extra."
DOCENT_COLLECTION_DESCRIPTION = "Rollouts collected with NeMo Gym."


@dataclass
class DocentCollectionTarget:
    client: Any
    collection_id: str
    collection_name: Optional[str]
    is_new_collection: bool


@dataclass
class DocentLoggingStats:
    attempted_rollouts: int = 0
    skipped_rollouts: int = 0
    uploaded_rollouts: int = 0


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
) -> DocentLoggingStats:
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
    results_to_upload = results
    if resume_from_cache and log_to_existing_collection is not None:
        # ``results`` begins with cached rollouts that are already present in the
        # existing Docent collection, so only upload the newly appended suffix.
        results_to_upload = results[initial_result_count:]

    stats = DocentLoggingStats(attempted_rollouts=len(results_to_upload))

    try:
        collection_target = _initialize_docent_collection_target(
            output_fpath=output_fpath,
            log_to_new_collection=log_to_new_collection,
            log_to_existing_collection=log_to_existing_collection,
        )
    except Exception as exc:
        print(f"Warning: failed to initialize Docent logging: {exc}")
        _print_docent_logging_summary(stats)
        return stats

    if collection_target.is_new_collection:
        print(f"Created Docent collection `{collection_target.collection_name}` ({collection_target.collection_id})")
    else:
        print(f"Using existing Docent collection {collection_target.collection_id}")

    print(f"Preparing {len(results_to_upload)} rollouts for Docent collection {collection_target.collection_id}")
    agent_runs, skipped_conversion_rollouts = _build_docent_agent_runs(results=results_to_upload)
    stats.skipped_rollouts = skipped_conversion_rollouts

    if not agent_runs:
        print(
            f"Skipping Docent upload because no rollouts were converted successfully for {collection_target.collection_id}"
        )
        _print_docent_logging_summary(stats)
        return stats

    print(f"Uploading {len(agent_runs)} rollouts to Docent collection {collection_target.collection_id}")
    try:
        uploaded_count = _upload_rollouts_to_docent_collection(
            collection_target=collection_target,
            agent_runs=agent_runs,
        )
    except Exception as exc:
        print(f"Warning: failed to upload rollouts to Docent collection {collection_target.collection_id}: {exc}")
        _print_docent_logging_summary(stats)
        return stats

    stats.uploaded_rollouts = uploaded_count
    print(f"Uploaded {uploaded_count} rollouts to Docent")
    _print_docent_logging_summary(stats)
    return stats


def _build_docent_agent_runs(
    *,
    results: list[dict[str, Any]],
) -> tuple[list[Any], int]:
    agent_runs: list[Any] = []
    skipped_conversion_rollouts = 0

    for result in results:
        try:
            agent_runs.append(_build_docent_agent_run(result=result))
        except Exception as exc:
            skipped_conversion_rollouts += 1
            task_index = result.get("_ng_task_index", "?")
            rollout_index = result.get("_ng_rollout_index", "?")
            agent_name = (result.get("agent_ref") or {}).get("name", "?")
            print(
                "Warning: failed to convert rollout "
                f"task={task_index}, rollout={rollout_index}, agent={agent_name} "
                f"for Docent upload: {exc}"
            )

    return agent_runs, skipped_conversion_rollouts


def _print_docent_logging_summary(stats: DocentLoggingStats) -> None:
    summary_parts = [
        f"attempted={stats.attempted_rollouts}",
        f"skipped={stats.skipped_rollouts}",
        f"uploaded={stats.uploaded_rollouts}",
    ]

    print("Docent logging summary: " + ", ".join(summary_parts))


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

    collection_name = log_to_new_collection or (
        f"{DEFAULT_DOCENT_COLLECTION_PREFIX} {output_fpath.stem} {datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
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
    agent_runs: list[Any],
) -> int:
    if not agent_runs:
        return 0

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


def _build_docent_agent_run(*, result: dict[str, Any]) -> Any:
    convert_nemogym_rollout_to_agent_run = _get_docent_nemogym_rollout_converter()
    normalized_result = _normalize_rollout_for_docent_sdk(result=result)
    return convert_nemogym_rollout_to_agent_run(normalized_result)


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
    if bool(output_items) and all(isinstance(item, list) for item in output_items):
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
