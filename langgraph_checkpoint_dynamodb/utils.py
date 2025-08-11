import asyncio
import time
from typing import Any, Dict, Optional, Tuple

from boto3.dynamodb.types import Binary
from botocore.exceptions import ClientError

from .config import DynamoDBConfig
from .constants import CheckpointItem, WriteItem
from .errors import DynamoDBCheckpointError, DynamoDBValidationError


def create_ttl_filter(ttl_attribute: str) -> Tuple[str, Dict[str, int]]:
    """
    Create a filter expression and attribute values for TTL.

    Args:
        ttl_attribute: Name of the TTL attribute

    Returns:
        Tuple of (filter expression, expression attribute values)
    """
    current_time = int(time.time())
    return (
        f"attribute_not_exists({ttl_attribute}) OR {ttl_attribute} > :current_time",
        {":current_time": current_time},
    )


def make_key(
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: Optional[str] = None,
    *,
    write_task_id: Optional[str] = None,
    write_idx: Optional[int] = None,
) -> Dict[str, str]:
    """
    Create DynamoDB key structure for items.

    Args:
        thread_id: Thread identifier
        checkpoint_ns: Checkpoint namespace
        checkpoint_id: Optional checkpoint identifier
        write_task_id: Optional task ID for writes
        write_idx: Optional index for writes

    Returns:
        Dictionary containing PK and SK for DynamoDB
    """
    pk = thread_id
    if write_task_id is not None and write_idx is not None:
        # Key for writes
        if checkpoint_id is None:
            raise ValueError("checkpoint_id required for writes")
        sk = f"{checkpoint_ns}#write#{checkpoint_id}#{write_task_id}#{write_idx:010d}"
    else:
        # Key for checkpoints
        sk = (
            f"{checkpoint_ns}#checkpoint#{checkpoint_id}"
            if checkpoint_id
            else checkpoint_ns
        )

    return {"PK": pk, "SK": sk}


def deserialize_dynamodb_binary(value: Any) -> bytes:
    """Convert DynamoDB Binary to bytes."""
    if isinstance(value, Binary):
        return bytes(value)
    return value


async def execute_with_retry(
    operation: Any,
    config: DynamoDBConfig,
    error_context: str = "",
) -> Any:
    """
    Execute DynamoDB operation with exponential backoff retry.

    Args:
        operation: Async callable to execute
        config: DynamoDB configuration
        error_context: Context for error messages

    Returns:
        Operation result

    Raises:
        DynamoDBCheckpointError: On operation failure after retries
    """
    delay = config.initial_retry_delay
    last_exception = None

    for attempt in range(config.max_retries):
        try:
            return await operation()
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ProvisionedThroughputExceededException":
                if attempt == config.max_retries - 1:
                    raise DynamoDBCheckpointError(
                        f"{error_context}: Max retries exceeded"
                    ) from e
                delay = min(delay * 2, config.max_retry_delay)
                await asyncio.sleep(delay)
                last_exception = e
                continue
            raise DynamoDBCheckpointError(f"{error_context}: {error_code}") from e
        except Exception as e:
            raise DynamoDBCheckpointError(f"{error_context}: Unexpected error") from e

    if last_exception:
        raise DynamoDBCheckpointError(
            f"{error_context}: Max retries exceeded"
        ) from last_exception


def create_checkpoint_item(
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    type_: str,
    checkpoint_data: str,
    metadata_data: str,
    parent_checkpoint_id: Optional[str] = None,
    ttl_minutes: Optional[int] = None,
    ttl_attribute: str = "expireAt",
) -> CheckpointItem:
    """Create and validate checkpoint item."""
    # Create item with Binary type for binary data
    item = {
        **make_key(thread_id, checkpoint_ns, checkpoint_id),
        "type": type_,
        "checkpoint_id": checkpoint_id,
        "checkpoint": (
            Binary(checkpoint_data)
            if isinstance(checkpoint_data, bytes)
            else checkpoint_data
        ),
        "metadata": (
            Binary(metadata_data) if isinstance(metadata_data, bytes) else metadata_data
        ),
    }

    if parent_checkpoint_id:
        item["parent_checkpoint_id"] = parent_checkpoint_id

    # Add TTL attribute if ttl_minutes is set
    if ttl_minutes is not None:
        expiration_time = int(time.time()) + (ttl_minutes * 60)
        item[ttl_attribute] = expiration_time

    return validate_checkpoint_item(item)


def create_write_item(
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    task_id: str,
    idx: int,
    channel: str,
    type_: str,
    value_data: str,
    ttl_minutes: Optional[int] = None,
    ttl_attribute: str = "expireAt",
) -> WriteItem:
    """Create and validate write item."""
    item = {
        **make_key(
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            write_task_id=task_id,
            write_idx=idx,
        ),
        "type": type_,
        "channel": channel,
        "value": Binary(value_data) if isinstance(value_data, bytes) else value_data,
        "task_id": task_id,
        "idx": idx,
    }

    # Add TTL attribute if ttl_minutes is set
    if ttl_minutes is not None:
        expiration_time = int(time.time()) + (ttl_minutes * 60)
        item[ttl_attribute] = expiration_time

    return validate_write_item(item)


def validate_checkpoint_item(item: Dict[str, Any]) -> CheckpointItem:
    """
    Validate checkpoint item structure.

    Args:
        item: DynamoDB item

    Returns:
        Validated CheckpointItem

    Raises:
        DynamoDBValidationError: If validation fails
    """
    required_fields = {"PK", "SK", "type", "checkpoint_id", "checkpoint", "metadata"}
    missing_fields = required_fields - set(item.keys())
    if missing_fields:
        raise DynamoDBValidationError(
            f"Checkpoint item missing required fields: {missing_fields}"
        )

    if "#checkpoint#" not in item["SK"]:
        raise DynamoDBValidationError(f"Invalid checkpoint SK format: {item['SK']}")

    return CheckpointItem(**dict(item))


def validate_write_item(item: Dict[str, Any]) -> WriteItem:
    """
    Validate write item structure.

    Args:
        item: DynamoDB item

    Returns:
        Validated WriteItem

    Raises:
        DynamoDBValidationError: If validation fails
    """
    required_fields = {"PK", "SK", "type", "task_id", "channel", "value", "idx"}
    missing_fields = required_fields - set(item.keys())
    if missing_fields:
        raise DynamoDBValidationError(
            f"Write item missing required fields: {missing_fields}"
        )

    if "#write#" not in item["SK"]:
        raise DynamoDBValidationError(f"Invalid write SK format: {item['SK']}")

    # Handle idx conversion if needed
    if not isinstance(item["idx"], int):
        item = dict(item)
        item["idx"] = int(item["idx"])

    return WriteItem(**dict(item))
