from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import NAMESPACE_URL, uuid5


DEFAULT_DOCENT_COLLECTION_PREFIX = "NeMo Gym rollouts"
DOCENT_DEPENDENCY_ERROR = "Docent logging requires the optional `nemo-gym[docent]` dependency extra."


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
    _get_docent_upload_classes()


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
        description="Rollouts collected with NeMo Gym.",
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

    AgentRun, Transcript, UserMessage, AssistantMessage = _get_docent_upload_classes()

    agent_runs = []
    for result in results:
        payload = _build_docent_agent_run_payload(result=result, output_fpath=output_fpath)
        transcript = Transcript(
            messages=[_build_docent_message(msg, UserMessage, AssistantMessage) for msg in payload["messages"]]
        )
        agent_runs.append(
            AgentRun(
                id=payload["id"],
                name=payload["name"],
                description=payload["description"],
                transcripts=[transcript],
                metadata=payload["metadata"],
            )
        )

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


def _get_docent_upload_classes() -> tuple[Any, Any, Any, Any]:
    try:
        from docent.data_models import AgentRun, Transcript
        from docent.data_models.chat import AssistantMessage, UserMessage
    except ImportError as exc:  # pragma: no cover - exercised via tests with monkeypatching
        raise ImportError(DOCENT_DEPENDENCY_ERROR) from exc
    return AgentRun, Transcript, UserMessage, AssistantMessage


def _build_docent_agent_run_payload(*, result: dict[str, Any], output_fpath: Path) -> dict[str, Any]:
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

    metadata: dict[str, Any] = {
        "nemo_gym": {
            "agent_name": agent_name,
            "task_index": task_index,
            "rollout_index": rollout_index,
            "output_jsonl_fpath": str(output_fpath),
            "response_model": response.get("model"),
            "raw_rollout": result,
        }
    }
    if score_metadata:
        metadata["scores"] = score_metadata

    return {
        "id": _build_agent_run_id(
            output_fpath=output_fpath,
            agent_name=agent_name,
            task_index=task_index,
            rollout_index=rollout_index,
        ),
        "name": f"{agent_name}/task-{task_index}/rollout-{rollout_index}",
        "description": "Rollout collected with NeMo Gym.",
        "messages": _build_docent_messages(result),
        "metadata": metadata,
    }


def _default_collection_name(output_fpath: Path) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{DEFAULT_DOCENT_COLLECTION_PREFIX} {output_fpath.stem} {timestamp}"


def _build_agent_run_id(
    *,
    output_fpath: Path,
    agent_name: str,
    task_index: Any,
    rollout_index: Any,
) -> str:
    output_path = output_fpath.resolve(strict=False)
    return str(uuid5(NAMESPACE_URL, f"nemo-gym:{output_path}:{agent_name}:{task_index}:{rollout_index}"))


def _build_docent_messages(result: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    response = result.get("response") or {}

    if _response_contains_transitions(response):
        messages.extend(_messages_from_output_payload(response))
    else:
        input_payload = (result.get("responses_create_params") or {}).get("input")
        messages.extend(_messages_from_input_payload(input_payload))
        messages.extend(_messages_from_output_payload(response))

    if not messages:
        messages.append({"role": "assistant", "content": "[empty rollout]"})

    return messages


def _response_contains_transitions(response: dict[str, Any]) -> bool:
    output_items = response.get("output") or []
    return bool(output_items) and all(isinstance(item, list) for item in output_items)


def _messages_from_input_payload(input_payload: Any) -> list[dict[str, str]]:
    if input_payload is None:
        return []

    if isinstance(input_payload, str):
        return [{"role": "user", "content": input_payload}]

    if not isinstance(input_payload, list):
        return [{"role": "user", "content": _stringify_unknown(input_payload)}]

    messages = []
    for item in input_payload:
        if isinstance(item, dict) and (item.get("type") == "message" or "role" in item):
            role = item.get("role") or "user"
            content = _content_to_text(item.get("content"))
            if content:
                messages.append(_normalize_role_message(role=role, content=content))
            continue

        messages.append({"role": "user", "content": _stringify_unknown(item)})

    return messages


def _messages_from_output_payload(response: dict[str, Any]) -> list[dict[str, str]]:
    output_items = response.get("output") or []
    if output_items and all(isinstance(item, list) for item in output_items):
        output_items = output_items[-1]

    messages = []
    for item in output_items:
        if not isinstance(item, dict):
            messages.append({"role": "assistant", "content": _stringify_unknown(item)})
            continue

        item_type = item.get("type")
        if item_type == "message":
            role = item.get("role") or "assistant"
            content = _content_to_text(item.get("content"))
            if content:
                messages.append(_normalize_role_message(role=role, content=content))
            continue

        if item_type == "function_call":
            name = item.get("name") or "unknown_tool"
            arguments = item.get("arguments")
            messages.append({"role": "assistant", "content": f"[tool call] {name}({arguments or ''})"})
            continue

        if item_type == "function_call_output":
            call_id = item.get("call_id") or "unknown_call"
            output = item.get("output")
            messages.append({"role": "assistant", "content": f"[tool result {call_id}] {output}"})
            continue

        if item_type == "reasoning":
            summary = _extract_reasoning_summary(item)
            if summary:
                messages.append({"role": "assistant", "content": f"[reasoning]\n{summary}"})
            continue

        messages.append({"role": "assistant", "content": _stringify_unknown(item)})

    return messages


def _normalize_role_message(*, role: str, content: str) -> dict[str, str]:
    if role == "assistant":
        return {"role": "assistant", "content": content}
    if role == "user":
        return {"role": "user", "content": content}
    return {"role": "user", "content": f"[{role}] {content}"}


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return _stringify_unknown(content)

    chunks: list[str] = []
    for part in content:
        if isinstance(part, str):
            chunks.append(part)
            continue

        if not isinstance(part, dict):
            chunks.append(_stringify_unknown(part))
            continue

        if isinstance(part.get("text"), str):
            chunks.append(part["text"])
            continue

        if isinstance(part.get("refusal"), str):
            chunks.append(part["refusal"])
            continue

        if part.get("type") in {"input_image", "image"}:
            chunks.append("[image]")
            continue

        chunks.append(_stringify_unknown(part))

    return "\n".join(chunk for chunk in chunks if chunk)


def _extract_reasoning_summary(item: dict[str, Any]) -> str:
    summary = item.get("summary")
    if not isinstance(summary, list):
        return ""

    chunks = []
    for part in summary:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            chunks.append(part["text"])
        elif isinstance(part, str):
            chunks.append(part)

    return "\n".join(chunk for chunk in chunks if chunk)


def _stringify_unknown(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        return str(value)


def _build_docent_message(message: dict[str, str], user_cls: Any, assistant_cls: Any) -> Any:
    if message["role"] == "assistant":
        return assistant_cls(content=message["content"])
    return user_cls(content=message["content"])
