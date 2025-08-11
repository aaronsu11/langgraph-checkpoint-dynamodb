import logging
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Sequence, Tuple
import time

import aioboto3
import boto3
from boto3.dynamodb.conditions import Key
from boto3.dynamodb.types import TypeSerializer
from botocore.exceptions import ClientError
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)

from .config import BillingMode, DynamoDBConfig
from .errors import (
    DynamoDBCheckpointError,
    DynamoDBConfigurationError,
    DynamoDBValidationError,
)
from .utils import (
    create_checkpoint_item,
    create_ttl_filter,
    create_write_item,
    deserialize_dynamodb_binary,
    execute_with_retry,
    make_key,
    validate_checkpoint_item,
    validate_write_item,
)

logger = logging.getLogger(__name__)


class DynamoDBSaver(BaseCheckpointSaver):
    """
    DynamoDB implementation of checkpoint saver.

    Stores checkpoints and writes in a single DynamoDB table using a composite
    key structure that enables efficient queries and lookups.
    """

    def __init__(
        self,
        config: Optional[DynamoDBConfig] = None,
        deploy: bool = False,
    ) -> None:
        """
        Initialize DynamoDB checkpoint saver.

        Args:
            config: DynamoDB configuration
            deploy: If True, creates/updates the table. If False, only checks if table exists.
        """
        super().__init__()
        self.config = config or DynamoDBConfig()

        # Sync clients
        self.dynamodb = boto3.resource("dynamodb", **self.config.get_client_config())
        self.client = boto3.client("dynamodb", **self.config.get_client_config())
        self.table = self.dynamodb.Table(self.config.table_config.table_name)

        # Async clients - initialize lazily
        self.async_session = aioboto3.Session()
        self._async_client = None
        self._async_resource = None
        self._async_table = None

        # Handle table deployment/verification
        region = self.client.meta.region_name
        if deploy:
            try:
                self._create_or_update_table()
            except self.client.exceptions.ResourceInUseException:
                logger.info(
                    f"Table {self.config.table_config.table_name} in region {region} is being created by another process"
                )
            except Exception as e:
                logger.error(f"Failed to create or update table: {e}")
                raise DynamoDBCheckpointError(
                    f"Failed to create or update table in region {region}"
                ) from e
        else:
            try:
                self.client.describe_table(
                    TableName=self.config.table_config.table_name
                )
                logger.info(
                    f"Found table {self.config.table_config.table_name} in region {region}"
                )
            except self.client.exceptions.ResourceNotFoundException:
                logger.error(
                    f"Table {self.config.table_config.table_name} does not exist in region {region}."
                )
                raise DynamoDBCheckpointError(
                    f"Table {self.config.table_config.table_name} does not exist in region {region}. Set deploy=True to create it."
                )
            except Exception as e:
                logger.error(f"Error checking table: {e}")
                raise DynamoDBCheckpointError(f"Failed to check table existence: {e}")

    async def _ensure_async_clients(self):
        """Ensure async clients are initialized."""
        if self._async_client is None:
            self._async_resource = await self.async_session.resource(
                "dynamodb", **self.config.get_client_config()
            ).__aenter__()
            self._async_client = await self.async_session.client(
                "dynamodb", **self.config.get_client_config()
            ).__aenter__()
            self._async_table = await self._async_resource.Table(
                self.config.table_config.table_name
            )

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get checkpoint tuple by config."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        try:
            # Get checkpoint
            if checkpoint_id:
                # Get specific checkpoint
                key = make_key(thread_id, checkpoint_ns, checkpoint_id)
                response = self.table.get_item(Key=key, ConsistentRead=True)
                item = response.get("Item")
                if not item:
                    return None
                try:
                    checkpoint_item = validate_checkpoint_item(item)
                except DynamoDBValidationError as e:
                    logger.error(f"Invalid checkpoint item: {e}")
                    return None
            else:
                # Get latest checkpoint
                # Build query params with filter for non-expired items
                query_params = {
                    "KeyConditionExpression": Key("PK").eq(thread_id)
                    & Key("SK").begins_with(f"{checkpoint_ns}#checkpoint#"),
                    "ScanIndexForward": False,
                    "Limit": 1,
                    "ConsistentRead": True,
                }

                # Add filter for TTL if attribute is used
                if self.config.table_config.ttl_minutes is not None:
                    filter_expr, expr_values = create_ttl_filter(
                        self.config.table_config.ttl_attribute
                    )
                    query_params["FilterExpression"] = filter_expr
                    query_params["ExpressionAttributeValues"] = expr_values

                response = self.table.query(**query_params)
                if not response["Items"]:
                    return None
                try:
                    checkpoint_item = validate_checkpoint_item(response["Items"][0])
                except DynamoDBValidationError as e:
                    logger.error(f"Invalid checkpoint item: {e}")
                    return None

            # Get writes for the checkpoint
            write_prefix = f"{checkpoint_ns}#write#{checkpoint_item['checkpoint_id']}"
            writes_query_params = {
                "KeyConditionExpression": Key("PK").eq(thread_id)
                & Key("SK").begins_with(write_prefix),
            }

            # Add filter for TTL if attribute is used
            if self.config.table_config.ttl_minutes is not None:
                filter_expr, expr_values = create_ttl_filter(
                    self.config.table_config.ttl_attribute
                )
                writes_query_params["FilterExpression"] = filter_expr
                writes_query_params["ExpressionAttributeValues"] = expr_values

            writes_response = self.table.query(**writes_query_params)

            writes = []
            for write_item in writes_response.get("Items", []):
                try:
                    writes.append(validate_write_item(write_item))
                except DynamoDBValidationError as e:
                    logger.error(f"Invalid write item: {e}")
                    continue

            # Process validated writes
            pending_writes = [
                (
                    write["task_id"],
                    write["channel"],
                    self.serde.loads_typed(
                        (write["type"], deserialize_dynamodb_binary(write["value"]))
                    ),
                )
                for write in writes
            ]

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_item["checkpoint_id"],
                    }
                },
                checkpoint=self.serde.loads_typed(
                    (
                        checkpoint_item["type"],
                        deserialize_dynamodb_binary(checkpoint_item["checkpoint"]),
                    )
                ),
                metadata=self.serde.loads_typed(
                    (
                        checkpoint_item["type"],
                        deserialize_dynamodb_binary(checkpoint_item["metadata"]),
                    )
                ),
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_item["parent_checkpoint_id"],
                        }
                    }
                    if checkpoint_item.get("parent_checkpoint_id")
                    else None
                ),
                pending_writes=pending_writes,
            )

        except ClientError as e:
            raise DynamoDBCheckpointError("Failed to get checkpoint") from e

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get checkpoint tuple asynchronously with validation."""
        await self._ensure_async_clients()
        if not self._async_table:
            raise DynamoDBConfigurationError(
                "Saver must be used as async context manager"
            )

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        try:
            # Get checkpoint
            if checkpoint_id:
                # Get specific checkpoint
                key = make_key(thread_id, checkpoint_ns, checkpoint_id)
                response = await execute_with_retry(
                    lambda: self._async_table.get_item(Key=key, ConsistentRead=True),
                    self.config,
                    "Failed to get checkpoint",
                )
                item = response.get("Item")
                if not item:
                    return None
                try:
                    checkpoint_item = validate_checkpoint_item(item)
                except DynamoDBValidationError as e:
                    logger.error(f"Invalid checkpoint item: {e}")
                    return None
            else:
                # Get latest checkpoint
                # Build query params with filter for non-expired items
                query_params = {
                    "KeyConditionExpression": Key("PK").eq(thread_id)
                    & Key("SK").begins_with(f"{checkpoint_ns}#checkpoint#"),
                    "ScanIndexForward": False,
                    "Limit": 1,
                    "ConsistentRead": True,
                }

                # Add filter for TTL if attribute is used
                if self.config.table_config.ttl_minutes is not None:
                    filter_expr, expr_values = create_ttl_filter(
                        self.config.table_config.ttl_attribute
                    )
                    query_params["FilterExpression"] = filter_expr
                    query_params["ExpressionAttributeValues"] = expr_values

                response = await execute_with_retry(
                    lambda: self._async_table.query(**query_params),
                    self.config,
                    "Failed to get latest checkpoint",
                )
                if not response["Items"]:
                    return None
                try:
                    checkpoint_item = validate_checkpoint_item(response["Items"][0])
                except DynamoDBValidationError as e:
                    logger.error(f"Invalid checkpoint item: {e}")
                    return None

            # Get writes for the checkpoint
            write_prefix = f"{checkpoint_ns}#write#{checkpoint_item['checkpoint_id']}"
            writes_query_params = {
                "KeyConditionExpression": Key("PK").eq(thread_id)
                & Key("SK").begins_with(write_prefix),
            }

            # Add filter for TTL if attribute is used
            if self.config.table_config.ttl_minutes is not None:
                filter_expr, expr_values = create_ttl_filter(
                    self.config.table_config.ttl_attribute
                )
                writes_query_params["FilterExpression"] = filter_expr
                writes_query_params["ExpressionAttributeValues"] = expr_values

            writes_response = await execute_with_retry(
                lambda: self._async_table.query(**writes_query_params),
                self.config,
                "Failed to get writes",
            )

            writes = []
            for write_item in writes_response.get("Items", []):
                try:
                    writes.append(validate_write_item(write_item))
                except DynamoDBValidationError as e:
                    logger.error(f"Invalid write item: {e}")
                    continue

            # Process validated writes
            pending_writes = [
                (
                    write["task_id"],
                    write["channel"],
                    self.serde.loads_typed(
                        (write["type"], deserialize_dynamodb_binary(write["value"]))
                    ),
                )
                for write in writes
            ]

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_item["checkpoint_id"],
                    }
                },
                checkpoint=self.serde.loads_typed(
                    (
                        checkpoint_item["type"],
                        deserialize_dynamodb_binary(checkpoint_item["checkpoint"]),
                    )
                ),
                metadata=self.serde.loads_typed(
                    (
                        checkpoint_item["type"],
                        deserialize_dynamodb_binary(checkpoint_item["metadata"]),
                    )
                ),
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_item["parent_checkpoint_id"],
                        }
                    }
                    if checkpoint_item.get("parent_checkpoint_id")
                    else None
                ),
                pending_writes=pending_writes,
            )

        except ClientError as e:
            raise DynamoDBCheckpointError("Failed to get checkpoint") from e

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints matching criteria."""
        if not config:
            raise DynamoDBConfigurationError("Config required for listing checkpoints")

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        try:
            # Build query for checkpoints only
            condition = Key("PK").eq(thread_id)
            checkpoint_prefix = f"{checkpoint_ns}#checkpoint#"

            if before and (before_id := get_checkpoint_id(before)):
                condition = condition & Key("SK").lt(
                    f"{checkpoint_ns}#checkpoint#{before_id}"
                )

            query_params = {
                "KeyConditionExpression": condition
                & Key("SK").begins_with(checkpoint_prefix),
                "ScanIndexForward": False,
            }

            if limit:
                query_params["Limit"] = limit

            # Add filter for TTL if attribute is used
            if self.config.table_config.ttl_minutes is not None:
                filter_expr, expr_values = create_ttl_filter(
                    self.config.table_config.ttl_attribute
                )
                query_params["FilterExpression"] = filter_expr
                query_params["ExpressionAttributeValues"] = expr_values

            response = self.table.query(**query_params)
            items = response.get("Items", [])

            # Process each checkpoint
            for item in items:
                try:
                    checkpoint_item = validate_checkpoint_item(item)

                    # Apply metadata filter if specified
                    if filter:
                        metadata = self.serde.loads_typed(
                            (
                                checkpoint_item["type"],
                                deserialize_dynamodb_binary(
                                    checkpoint_item["metadata"]
                                ),
                            )
                        )
                        if not all(metadata.get(k) == v for k, v in filter.items()):
                            continue

                    # Get writes for this checkpoint
                    write_prefix = (
                        f"{checkpoint_ns}#write#{checkpoint_item['checkpoint_id']}"
                    )
                    writes_query_params = {
                        "KeyConditionExpression": Key("PK").eq(thread_id)
                        & Key("SK").begins_with(write_prefix)
                    }

                    # Add filter for TTL if attribute is used
                    if self.config.table_config.ttl_minutes is not None:
                        filter_expr, expr_values = create_ttl_filter(
                            self.config.table_config.ttl_attribute
                        )
                        writes_query_params["FilterExpression"] = filter_expr
                        writes_query_params["ExpressionAttributeValues"] = expr_values

                    writes_response = self.table.query(**writes_query_params)

                    writes = []
                    for write_item in writes_response.get("Items", []):
                        try:
                            writes.append(validate_write_item(write_item))
                        except DynamoDBValidationError as e:
                            logger.error(f"Invalid write item found: {e}")
                            continue

                    pending_writes = [
                        (
                            write["task_id"],
                            write["channel"],
                            self.serde.loads_typed(
                                (
                                    write["type"],
                                    deserialize_dynamodb_binary(write["value"]),
                                )
                            ),
                        )
                        for write in writes
                    ]

                    yield CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": checkpoint_item["checkpoint_id"],
                            }
                        },
                        checkpoint=self.serde.loads_typed(
                            (
                                checkpoint_item["type"],
                                deserialize_dynamodb_binary(
                                    checkpoint_item["checkpoint"]
                                ),
                            )
                        ),
                        metadata=self.serde.loads_typed(
                            (
                                checkpoint_item["type"],
                                deserialize_dynamodb_binary(
                                    checkpoint_item["metadata"]
                                ),
                            )
                        ),
                        parent_config=(
                            {
                                "configurable": {
                                    "thread_id": thread_id,
                                    "checkpoint_ns": checkpoint_ns,
                                    "checkpoint_id": checkpoint_item[
                                        "parent_checkpoint_id"
                                    ],
                                }
                            }
                            if checkpoint_item.get("parent_checkpoint_id")
                            else None
                        ),
                        pending_writes=pending_writes,
                    )

                except DynamoDBValidationError as e:
                    logger.error(f"Invalid checkpoint item found: {e}")
                    continue

        except ClientError as e:
            raise DynamoDBCheckpointError("Failed to list checkpoints") from e

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints asynchronously."""
        await self._ensure_async_clients()
        if not self._async_table:
            raise DynamoDBConfigurationError(
                "Saver must be used as async context manager"
            )

        if not config:
            raise DynamoDBConfigurationError("Config required for listing checkpoints")

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        try:
            # Build query for checkpoints
            condition = Key("PK").eq(thread_id)
            checkpoint_prefix = f"{checkpoint_ns}#checkpoint#"

            if before and (before_id := get_checkpoint_id(before)):
                condition = condition & Key("SK").lt(
                    f"{checkpoint_ns}#checkpoint#{before_id}"
                )

            query_params = {
                "KeyConditionExpression": condition
                & Key("SK").begins_with(checkpoint_prefix),
                "ScanIndexForward": False,
            }

            if limit:
                query_params["Limit"] = limit

            # Add filter for TTL if attribute is used
            if self.config.table_config.ttl_minutes is not None:
                filter_expr, expr_values = create_ttl_filter(
                    self.config.table_config.ttl_attribute
                )
                query_params["FilterExpression"] = filter_expr
                query_params["ExpressionAttributeValues"] = expr_values

            response = await execute_with_retry(
                lambda: self._async_table.query(**query_params),
                self.config,
                "Failed to list checkpoints",
            )

            items = response.get("Items", [])

            for item in items:
                try:
                    checkpoint_item = validate_checkpoint_item(item)

                    if filter:
                        metadata = self.serde.loads_typed(
                            (
                                checkpoint_item["type"],
                                deserialize_dynamodb_binary(
                                    checkpoint_item["metadata"]
                                ),
                            )
                        )
                        if not all(metadata.get(k) == v for k, v in filter.items()):
                            continue

                    # Get writes for this checkpoint
                    write_prefix = (
                        f"{checkpoint_ns}#write#{checkpoint_item['checkpoint_id']}"
                    )
                    writes_query_params = {
                        "KeyConditionExpression": Key("PK").eq(thread_id)
                        & Key("SK").begins_with(write_prefix)
                    }

                    # Add filter for TTL if attribute is used
                    if self.config.table_config.ttl_minutes is not None:
                        filter_expr, expr_values = create_ttl_filter(
                            self.config.table_config.ttl_attribute
                        )
                        writes_query_params["FilterExpression"] = filter_expr
                        writes_query_params["ExpressionAttributeValues"] = expr_values

                    writes_response = await execute_with_retry(
                        lambda: self._async_table.query(**writes_query_params),
                        self.config,
                        "Failed to get writes",
                    )

                    writes = []
                    for write_item in writes_response.get("Items", []):
                        try:
                            writes.append(validate_write_item(write_item))
                        except DynamoDBValidationError as e:
                            logger.error(f"Invalid write item found: {e}")
                            continue

                    pending_writes = [
                        (
                            write["task_id"],
                            write["channel"],
                            self.serde.loads_typed(
                                (
                                    write["type"],
                                    deserialize_dynamodb_binary(write["value"]),
                                )
                            ),
                        )
                        for write in writes
                    ]

                    yield CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": checkpoint_item["checkpoint_id"],
                            }
                        },
                        checkpoint=self.serde.loads_typed(
                            (
                                checkpoint_item["type"],
                                deserialize_dynamodb_binary(
                                    checkpoint_item["checkpoint"]
                                ),
                            )
                        ),
                        metadata=self.serde.loads_typed(
                            (
                                checkpoint_item["type"],
                                deserialize_dynamodb_binary(
                                    checkpoint_item["metadata"]
                                ),
                            )
                        ),
                        parent_config=(
                            {
                                "configurable": {
                                    "thread_id": thread_id,
                                    "checkpoint_ns": checkpoint_ns,
                                    "checkpoint_id": checkpoint_item[
                                        "parent_checkpoint_id"
                                    ],
                                }
                            }
                            if checkpoint_item.get("parent_checkpoint_id")
                            else None
                        ),
                        pending_writes=pending_writes,
                    )

                except DynamoDBValidationError as e:
                    logger.error(f"Invalid checkpoint item found: {e}")
                    continue

        except ClientError as e:
            raise DynamoDBCheckpointError("Failed to list checkpoints") from e

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store checkpoint with metadata."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]

        # Serialize data
        type_, checkpoint_data = self.serde.dumps_typed(checkpoint)
        _, metadata_data = self.serde.dumps_typed(metadata)

        # Create and validate item
        item = create_checkpoint_item(
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            type_,
            checkpoint_data,
            metadata_data,
            config["configurable"].get("checkpoint_id"),
            self.config.table_config.ttl_minutes,
            self.config.table_config.ttl_attribute,
        )

        try:
            self.table.put_item(Item=item)
            return {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }
        except ClientError as e:
            raise DynamoDBCheckpointError("Failed to put checkpoint") from e

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store checkpoint asynchronously."""
        await self._ensure_async_clients()
        if not self._async_table:
            raise DynamoDBConfigurationError(
                "Saver must be used as async context manager"
            )

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]

        # Serialize data
        type_, checkpoint_data = self.serde.dumps_typed(checkpoint)
        _, metadata_data = self.serde.dumps_typed(metadata)

        # Create and validate item
        item = create_checkpoint_item(
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            type_,
            checkpoint_data,
            metadata_data,
            config["configurable"].get("checkpoint_id"),
            self.config.table_config.ttl_minutes,
            self.config.table_config.ttl_attribute,
        )

        try:
            # Put item with retries
            await execute_with_retry(
                lambda: self._async_table.put_item(Item=item),
                self.config,
                "Failed to put checkpoint",
            )

            return {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }
        except ClientError as e:
            raise DynamoDBCheckpointError("Failed to put checkpoint") from e

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """
        Store writes for a checkpoint.

        Implements batch writing with automatic chunking and retry handling.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        try:
            with self.table.batch_writer() as batch:
                for idx, (channel, value) in enumerate(writes):
                    type_, value_data = self.serde.dumps_typed(value)

                    # Create and validate write item
                    item = create_write_item(
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        idx,
                        channel,
                        type_,
                        value_data,
                        self.config.table_config.ttl_minutes,
                        self.config.table_config.ttl_attribute,
                    )

                    batch.put_item(Item=item)

        except ClientError as e:
            raise DynamoDBCheckpointError("Failed to put writes") from e

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store writes asynchronously with efficient batch processing."""
        await self._ensure_async_clients()
        if not self._async_table:
            raise DynamoDBConfigurationError(
                "Saver must be used as async context manager"
            )

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        try:
            # Convert deque to list for slicing
            writes_list = list(writes)
            # Create TypeSerializer instance
            serializer = TypeSerializer()

            # Process writes in batches
            batch_size = 25  # DynamoDB batch write limit
            for i in range(0, len(writes_list), batch_size):
                batch_writes = writes_list[i : i + batch_size]
                batch_items = []

                for idx, (channel, value) in enumerate(batch_writes, start=i):
                    type_, value_data = self.serde.dumps_typed(value)

                    # Create and validate write item
                    item = create_write_item(
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        idx,
                        channel,
                        type_,
                        value_data,
                        self.config.table_config.ttl_minutes,
                        self.config.table_config.ttl_attribute,
                    )

                    # Convert item to DynamoDB format
                    dynamodb_item = {
                        k: serializer.serialize(v) for k, v in item.items()
                    }
                    batch_items.append({"PutRequest": {"Item": dynamodb_item}})

                # Execute batch write with retry
                await execute_with_retry(
                    lambda: self._async_client.batch_write_item(
                        RequestItems={self.config.table_config.table_name: batch_items}
                    ),
                    self.config,
                    "Failed to put writes",
                )

        except ClientError as e:
            raise DynamoDBCheckpointError("Failed to put writes") from e

    @classmethod
    def create(
        cls,
        config: Optional[DynamoDBConfig] = None,
    ) -> "DynamoDBSaver":
        """Deprecated: use deploy instead."""
        import warnings

        warnings.warn(
            "create() is deprecated, use DynamoDBSaver(config, deploy=True) or DynamoDBSaver.deploy(config) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.deploy(config)

    @classmethod
    def deploy(
        cls,
        config: Optional[DynamoDBConfig] = None,
    ) -> "DynamoDBSaver":
        """
        Create a DynamoDBSaver instance and ensure table exists.

        Args:
            config: DynamoDB configuration

        Returns:
            DynamoDBSaver instance with table created/updated

        Raises:
            DynamoDBCheckpointError: If table creation fails
        """
        saver = cls(config, deploy=True)
        return saver

    def _create_or_update_table(self) -> None:
        """Create or update DynamoDB table based on configuration."""
        table_config = self.config.table_config

        try:
            # Check if table exists
            existing_table = self.client.describe_table(
                TableName=table_config.table_name
            )["Table"]

            logger.info(f"Table {table_config.table_name} already exists")

            # Update if needed
            updates_needed = []

            # Check billing mode
            if (
                existing_table.get("BillingModeSummary", {}).get("BillingMode")
                != table_config.billing_mode
            ):
                updates_needed.append(self._update_billing_mode)

            # Check capacity if provisioned
            if table_config.billing_mode == BillingMode.PROVISIONED:
                current_capacity = existing_table["ProvisionedThroughput"]
                if (
                    current_capacity["ReadCapacityUnits"] != table_config.read_capacity
                    or current_capacity["WriteCapacityUnits"]
                    != table_config.write_capacity
                ):
                    updates_needed.append(self._update_capacity)

            # Check point-in-time recovery
            if table_config.enable_point_in_time_recovery:
                pitr_status = self.client.describe_continuous_backups(
                    TableName=table_config.table_name
                )["ContinuousBackupsDescription"]["PointInTimeRecoveryDescription"][
                    "PointInTimeRecoveryStatus"
                ]
                if pitr_status != "ENABLED":
                    updates_needed.append(self._enable_pitr)

            # Check TTL configuration
            if table_config.ttl_minutes is not None:
                ttl_status = self.client.describe_time_to_live(
                    TableName=table_config.table_name
                )["TimeToLiveDescription"]["TimeToLiveStatus"]
                if ttl_status != "ENABLED":
                    updates_needed.append(self._enable_ttl)

            # Execute updates if needed
            for update in updates_needed:
                update()

        except self.client.exceptions.ResourceNotFoundException:
            # Create new table
            create_args = {
                "TableName": table_config.table_name,
                "KeySchema": [
                    {"AttributeName": "PK", "KeyType": "HASH"},
                    {"AttributeName": "SK", "KeyType": "RANGE"},
                ],
                "AttributeDefinitions": [
                    {"AttributeName": "PK", "AttributeType": "S"},
                    {"AttributeName": "SK", "AttributeType": "S"},
                ],
                "BillingMode": table_config.billing_mode,
            }

            if table_config.billing_mode == BillingMode.PROVISIONED:
                create_args["ProvisionedThroughput"] = {
                    "ReadCapacityUnits": table_config.read_capacity,
                    "WriteCapacityUnits": table_config.write_capacity,
                }

            if table_config.enable_encryption:
                create_args["SSESpecification"] = {"Enabled": True, "SSEType": "KMS"}

            self.client.create_table(**create_args)

            # Wait for table to be active
            waiter = self.client.get_waiter("table_exists")
            waiter.wait(TableName=table_config.table_name)

            # Enable PITR if configured
            if table_config.enable_point_in_time_recovery:
                self._enable_pitr()

            # Enable TTL if configured
            if table_config.ttl_minutes is not None:
                self._enable_ttl()

            logger.info(f"Table {table_config.table_name} created")

    def _update_billing_mode(self) -> None:
        """Update table billing mode."""

        logger.info(
            f"Updating table {self.config.table_config.table_name} billing mode to {self.config.table_config.billing_mode}"
        )
        self.client.update_table(
            TableName=self.config.table_config.table_name,
            BillingMode=self.config.table_config.billing_mode,
        )

    def _update_capacity(self) -> None:
        """Update provisioned capacity."""
        logger.info(
            f"Updating table {self.config.table_config.table_name} capacity to {self.config.table_config.read_capacity} read and {self.config.table_config.write_capacity} write"
        )
        self.client.update_table(
            TableName=self.config.table_config.table_name,
            ProvisionedThroughput={
                "ReadCapacityUnits": self.config.table_config.read_capacity,
                "WriteCapacityUnits": self.config.table_config.write_capacity,
            },
        )

    def _enable_pitr(self) -> None:
        """Enable point-in-time recovery."""
        logger.info(
            f"Enabling point-in-time recovery for table {self.config.table_config.table_name}"
        )
        self.client.update_continuous_backups(
            TableName=self.config.table_config.table_name,
            PointInTimeRecoverySpecification={"PointInTimeRecoveryEnabled": True},
        )

    def _enable_ttl(self) -> None:
        """Enable TTL for the table and update existing items."""
        logger.info(
            f"Enabling TTL for table {self.config.table_config.table_name} with duration {self.config.table_config.ttl_minutes} minutes"
        )

        # First enable TTL on the table
        self.client.update_time_to_live(
            TableName=self.config.table_config.table_name,
            TimeToLiveSpecification={
                "Enabled": True,
                "AttributeName": self.config.table_config.ttl_attribute,
            },
        )

        # Then update existing items with TTL, but only those without the TTL attribute
        if self.config.table_config.ttl_minutes is not None:
            logger.info("Updating existing items that don't have TTL attribute...")
            expiration_time = int(time.time()) + (
                    self.config.table_config.ttl_minutes * 60
            )
            ttl_attr = self.config.table_config.ttl_attribute
            batch_size = 25  # DynamoDB batch write limit

            # Function to process items in batches
            def process_items_in_batches(scan_filter, item_type):
                total_updated = 0
                last_evaluated_key = None

                while True:
                    # Prepare scan parameters
                    scan_params = {
                        "FilterExpression": scan_filter,
                        "ExpressionAttributeNames": {"#ttl": ttl_attr},
                        "ExpressionAttributeValues": {":prefix": f"#{item_type}#"},
                    }

                    if last_evaluated_key:
                        scan_params["ExclusiveStartKey"] = last_evaluated_key

                    # Scan for items without TTL attribute
                    response = self.table.scan(**scan_params)
                    items = response.get("Items", [])

                    # Process items in smaller batches to avoid throughput issues
                    items_to_update = []
                    for item in items:
                        items_to_update.append(item)

                        if len(items_to_update) >= batch_size:
                            self._batch_update_ttl(
                                items_to_update, ttl_attr, expiration_time
                            )
                            total_updated += len(items_to_update)
                            items_to_update = []

                    # Process any remaining items
                    if items_to_update:
                        self._batch_update_ttl(
                            items_to_update, ttl_attr, expiration_time
                        )
                        total_updated += len(items_to_update)

                    # Check if we need to continue scanning
                    last_evaluated_key = response.get("LastEvaluatedKey")
                    if not last_evaluated_key:
                        break

                return total_updated

            # Process checkpoints and writes separately
            checkpoint_filter = (
                "begins_with(SK, :prefix) AND attribute_not_exists(#ttl)"
            )
            checkpoint_updated = process_items_in_batches(
                checkpoint_filter, "checkpoint"
            )

            write_filter = "begins_with(SK, :prefix) AND attribute_not_exists(#ttl)"
            write_updated = process_items_in_batches(write_filter, "write")

            logger.info(
                f"Updated {checkpoint_updated} checkpoints and {write_updated} writes with TTL attribute"
            )

    def _batch_update_ttl(self, items, ttl_attr, expiration_time):
        """Helper to update TTL attribute on a batch of items."""
        with self.table.batch_writer() as batch:
            for item in items:
                updated_item = {**item, ttl_attr: expiration_time}
                batch.put_item(Item=updated_item)

    def destroy(self) -> None:
        """
        Delete the DynamoDB table and clean up resources.

        Raises:
            DynamoDBCheckpointError: If table deletion fails
        """
        try:
            # Delete the table
            self.client.delete_table(TableName=self.config.table_config.table_name)

            # Wait for table to be deleted
            waiter = self.client.get_waiter("table_not_exists")
            waiter.wait(TableName=self.config.table_config.table_name)

            logger.info(
                f"Table {self.config.table_config.table_name} deleted successfully"
            )

        except self.client.exceptions.ResourceNotFoundException:
            logger.info(f"Table {self.config.table_config.table_name} does not exist")
        except ClientError as e:
            raise DynamoDBCheckpointError("Failed to delete table") from e
