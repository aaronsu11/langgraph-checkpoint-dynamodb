"""Integration tests for basic DynamoDB checkpoint operations."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointTuple

from langgraph_checkpoint_dynamodb import DynamoDBSaver

pytestmark = pytest.mark.integration


class TestBasicOperationsSync:
    """Test basic sync operations."""

    def test_put_and_get_checkpoint_sync(self, checkpointer, sample_config):
        """Test putting and getting a checkpoint synchronously."""
        checkpoint = {
            "id": "checkpoint_1",
            "data": "test_data",
        }
        metadata = {"step": 1, "custom": "value"}
        new_versions = {}

        # Put checkpoint
        result_config = checkpointer.put(
            sample_config, checkpoint, metadata, new_versions
        )
        assert result_config["configurable"]["checkpoint_id"] == "checkpoint_1"

        # Get checkpoint
        config_with_id = RunnableConfig(
            configurable={
                "thread_id": sample_config["configurable"]["thread_id"],
                "checkpoint_id": "checkpoint_1",
            }
        )
        result = checkpointer.get_tuple(config_with_id)
        assert result is not None
        assert isinstance(result, CheckpointTuple)
        assert result.checkpoint["id"] == "checkpoint_1"
        assert result.checkpoint["data"] == "test_data"
        assert result.metadata["step"] == 1
        assert result.metadata["custom"] == "value"

    def test_put_and_get_checkpoint_with_messages_sync(
        self, checkpointer, sample_config
    ):
        """Test putting and getting a checkpoint with messages synchronously."""
        messages = [HumanMessage(content="Hi!"), AIMessage(content="Hello!")]
        checkpoint = {
            "id": "checkpoint_msg_1",
            "messages": messages,
        }
        metadata = {"step": 1}
        new_versions = {}

        # Put checkpoint
        checkpointer.put(sample_config, checkpoint, metadata, new_versions)

        # Get checkpoint
        config_with_id = RunnableConfig(
            configurable={
                "thread_id": sample_config["configurable"]["thread_id"],
                "checkpoint_id": "checkpoint_msg_1",
            }
        )
        result = checkpointer.get_tuple(config_with_id)
        assert result is not None
        assert len(result.checkpoint["messages"]) == 2
        assert result.checkpoint["messages"][0].content == "Hi!"
        assert result.checkpoint["messages"][1].content == "Hello!"

    def test_put_writes_sync(self, checkpointer, sample_config):
        """Test putting writes synchronously."""
        # First create a checkpoint
        checkpoint = {"id": "checkpoint_write_1"}
        metadata = {"step": 1}
        config = checkpointer.put(sample_config, checkpoint, metadata, {})

        # Then add writes
        writes = [
            ("channel1", "value1"),
            ("channel2", "value2"),
            ("channel3", "value3"),
        ]
        task_id = "test_task_1"
        checkpointer.put_writes(config, writes, task_id)

        # Get checkpoint and verify writes
        result = checkpointer.get_tuple(config)
        assert result is not None
        assert len(result.pending_writes) == 3
        assert all(write[0] == task_id for write in result.pending_writes)
        assert result.pending_writes[0][1] == "channel1"
        assert result.pending_writes[1][1] == "channel2"
        assert result.pending_writes[2][1] == "channel3"

    def test_list_checkpoints_sync(self, checkpointer, sample_config):
        """Test listing checkpoints synchronously."""
        # Create multiple checkpoints
        for i in range(5):
            checkpoint_config = RunnableConfig(
                configurable={
                    "thread_id": sample_config["configurable"]["thread_id"],
                    "checkpoint_id": f"checkpoint_{i}",
                }
            )
            checkpoint = {"id": f"checkpoint_{i}"}
            metadata = {"step": i}
            checkpointer.put(checkpoint_config, checkpoint, metadata, {})

        # List checkpoints
        results = list(checkpointer.list(sample_config))
        assert len(results) == 5
        assert all(isinstance(r, CheckpointTuple) for r in results)

        # Verify order (most recent first)
        assert results[0].checkpoint["id"] == "checkpoint_4"
        assert results[-1].checkpoint["id"] == "checkpoint_0"

    def test_list_with_limit_sync(self, checkpointer, sample_config):
        """Test listing with limit parameter."""
        # Create multiple checkpoints
        for i in range(5):
            checkpoint_config = RunnableConfig(
                configurable={
                    "thread_id": sample_config["configurable"]["thread_id"],
                    "checkpoint_id": f"checkpoint_{i}",
                }
            )
            checkpoint = {"id": f"checkpoint_{i}"}
            metadata = {"step": i}
            checkpointer.put(checkpoint_config, checkpoint, metadata, {})

        # List with limit
        results = list(checkpointer.list(sample_config, limit=3))
        assert len(results) == 3
        assert results[0].checkpoint["id"] == "checkpoint_4"
        assert results[1].checkpoint["id"] == "checkpoint_3"
        assert results[2].checkpoint["id"] == "checkpoint_2"

    def test_get_latest_checkpoint_sync(self, checkpointer, sample_config):
        """Test getting latest checkpoint without checkpoint_id."""
        # Create multiple checkpoints
        parent_id = None
        for i in range(3):
            checkpoint_config = RunnableConfig(
                configurable={
                    "thread_id": sample_config["configurable"]["thread_id"],
                    "checkpoint_id": parent_id,
                }
            )
            checkpoint = {"id": f"checkpoint_{i}"}
            metadata = {"step": i}
            config = checkpointer.put(checkpoint_config, checkpoint, metadata, {})
            parent_id = config["configurable"]["checkpoint_id"]

        # Get latest without checkpoint_id
        result = checkpointer.get_tuple(sample_config)
        assert result is not None
        assert result.checkpoint["id"] == "checkpoint_2"


class TestBasicOperationsAsync:
    """Test basic async operations."""

    @pytest.mark.asyncio
    async def test_put_and_get_checkpoint_async(self, checkpointer, sample_config):
        """Test putting and getting a checkpoint asynchronously."""
        checkpoint = {
            "id": "checkpoint_async_1",
            "data": "async_test_data",
        }
        metadata = {"step": 1, "async": True}
        new_versions = {}

        # Put checkpoint
        result_config = await checkpointer.aput(
            sample_config, checkpoint, metadata, new_versions
        )
        assert result_config["configurable"]["checkpoint_id"] == "checkpoint_async_1"

        # Get checkpoint
        config_with_id = RunnableConfig(
            configurable={
                "thread_id": sample_config["configurable"]["thread_id"],
                "checkpoint_id": "checkpoint_async_1",
            }
        )
        result = await checkpointer.aget_tuple(config_with_id)
        assert result is not None
        assert isinstance(result, CheckpointTuple)
        assert result.checkpoint["id"] == "checkpoint_async_1"
        assert result.checkpoint["data"] == "async_test_data"
        assert result.metadata["step"] == 1
        assert result.metadata["async"] is True

    @pytest.mark.asyncio
    async def test_put_and_get_checkpoint_with_messages_async(
        self, checkpointer, sample_config
    ):
        """Test putting and getting a checkpoint with messages asynchronously."""
        messages = [
            HumanMessage(content="Async Hi!"),
            AIMessage(content="Async Hello!"),
        ]
        checkpoint = {
            "id": "checkpoint_async_msg_1",
            "messages": messages,
        }
        metadata = {"step": 1}
        new_versions = {}

        # Put checkpoint
        await checkpointer.aput(sample_config, checkpoint, metadata, new_versions)

        # Get checkpoint
        config_with_id = RunnableConfig(
            configurable={
                "thread_id": sample_config["configurable"]["thread_id"],
                "checkpoint_id": "checkpoint_async_msg_1",
            }
        )
        result = await checkpointer.aget_tuple(config_with_id)
        assert result is not None
        assert len(result.checkpoint["messages"]) == 2
        assert result.checkpoint["messages"][0].content == "Async Hi!"
        assert result.checkpoint["messages"][1].content == "Async Hello!"

    @pytest.mark.asyncio
    async def test_put_writes_async(self, checkpointer, sample_config):
        """Test putting writes asynchronously."""
        # First create a checkpoint
        checkpoint = {"id": "checkpoint_async_write_1"}
        metadata = {"step": 1}
        config = await checkpointer.aput(sample_config, checkpoint, metadata, {})

        # Then add writes
        writes = [
            ("channel1", "async_value1"),
            ("channel2", "async_value2"),
            ("channel3", "async_value3"),
            ("channel4", "async_value4"),
            ("channel5", "async_value5"),
        ]
        task_id = "test_task_async_1"
        await checkpointer.aput_writes(config, writes, task_id)

        # Get checkpoint and verify writes
        result = await checkpointer.aget_tuple(config)
        assert result is not None
        assert len(result.pending_writes) == 5
        assert all(write[0] == task_id for write in result.pending_writes)
        assert result.pending_writes[0][1] == "channel1"
        assert result.pending_writes[-1][1] == "channel5"

    @pytest.mark.asyncio
    async def test_put_writes_large_batch_async(self, checkpointer, sample_config):
        """Test putting large batch of writes asynchronously (exceeds 25 limit)."""
        # First create a checkpoint
        checkpoint = {"id": "checkpoint_large_batch"}
        metadata = {"step": 1}
        config = await checkpointer.aput(sample_config, checkpoint, metadata, {})

        # Create 50 writes (exceeds DynamoDB batch limit of 25)
        writes = [(f"channel{i}", f"value{i}") for i in range(50)]
        task_id = "test_task_large_batch"
        await checkpointer.aput_writes(config, writes, task_id)

        # Get checkpoint and verify all writes
        result = await checkpointer.aget_tuple(config)
        assert result is not None
        assert len(result.pending_writes) == 50
        assert all(write[0] == task_id for write in result.pending_writes)

    @pytest.mark.asyncio
    async def test_list_checkpoints_async(self, checkpointer, sample_config):
        """Test listing checkpoints asynchronously."""
        # Create multiple checkpoints
        for i in range(5):
            checkpoint_config = RunnableConfig(
                configurable={
                    "thread_id": sample_config["configurable"]["thread_id"],
                    "checkpoint_id": f"checkpoint_async_{i}",
                }
            )
            checkpoint = {"id": f"checkpoint_async_{i}"}
            metadata = {"step": i}
            await checkpointer.aput(checkpoint_config, checkpoint, metadata, {})

        # List checkpoints
        results = []
        async for result in checkpointer.alist(sample_config):
            results.append(result)

        assert len(results) == 5
        assert all(isinstance(r, CheckpointTuple) for r in results)
        assert results[0].checkpoint["id"] == "checkpoint_async_4"
        assert results[-1].checkpoint["id"] == "checkpoint_async_0"

    @pytest.mark.asyncio
    async def test_list_with_limit_async(self, checkpointer, sample_config):
        """Test listing with limit parameter asynchronously."""
        # Create multiple checkpoints
        for i in range(5):
            checkpoint_config = RunnableConfig(
                configurable={
                    "thread_id": sample_config["configurable"]["thread_id"],
                    "checkpoint_id": f"checkpoint_limit_{i}",
                }
            )
            checkpoint = {"id": f"checkpoint_limit_{i}"}
            metadata = {"step": i}
            await checkpointer.aput(checkpoint_config, checkpoint, metadata, {})

        # List with limit
        results = []
        async for result in checkpointer.alist(sample_config, limit=2):
            results.append(result)

        assert len(results) == 2
        assert results[0].checkpoint["id"] == "checkpoint_limit_4"
        assert results[1].checkpoint["id"] == "checkpoint_limit_3"

    @pytest.mark.asyncio
    async def test_get_latest_checkpoint_async(self, checkpointer, sample_config):
        """Test getting latest checkpoint without checkpoint_id asynchronously."""
        # Create multiple checkpoints
        parent_id = None
        for i in range(3):
            checkpoint_config = RunnableConfig(
                configurable={
                    "thread_id": sample_config["configurable"]["thread_id"],
                    "checkpoint_id": parent_id,
                }
            )
            checkpoint = {"id": f"checkpoint_async_latest_{i}"}
            metadata = {"step": i}
            config = await checkpointer.aput(
                checkpoint_config, checkpoint, metadata, {}
            )
            parent_id = config["configurable"]["checkpoint_id"]

        # Get latest without checkpoint_id
        result = await checkpointer.aget_tuple(sample_config)
        assert result is not None
        assert result.checkpoint["id"] == "checkpoint_async_latest_2"


class TestParentCheckpointChain:
    """Test parent checkpoint chain functionality."""

    def test_parent_checkpoint_chain_sync(self, checkpointer, sample_config):
        """Test parent checkpoint chain with sync methods."""
        parent_id = None
        checkpoints = []

        for i in range(3):
            checkpoint_config = RunnableConfig(
                configurable={
                    "thread_id": sample_config["configurable"]["thread_id"],
                    "checkpoint_id": parent_id,
                }
            )
            checkpoint = {"id": f"checkpoint_chain_{i}"}
            metadata = {"step": i}
            config = checkpointer.put(checkpoint_config, checkpoint, metadata, {})
            checkpoints.append(config)
            parent_id = config["configurable"]["checkpoint_id"]

        # Verify parent chain
        for i in range(1, len(checkpoints)):
            config = RunnableConfig(
                configurable={
                    "thread_id": sample_config["configurable"]["thread_id"],
                    "checkpoint_id": checkpoints[i]["configurable"]["checkpoint_id"],
                }
            )
            result = checkpointer.get_tuple(config)
            if i > 0:
                assert result.parent_config is not None
                assert (
                    result.parent_config["configurable"]["checkpoint_id"]
                    == checkpoints[i - 1]["configurable"]["checkpoint_id"]
                )

    @pytest.mark.asyncio
    async def test_parent_checkpoint_chain_async(self, checkpointer, sample_config):
        """Test parent checkpoint chain with async methods."""
        parent_id = None
        checkpoints = []

        for i in range(3):
            checkpoint_config = RunnableConfig(
                configurable={
                    "thread_id": sample_config["configurable"]["thread_id"],
                    "checkpoint_id": parent_id,
                }
            )
            checkpoint = {"id": f"checkpoint_async_chain_{i}"}
            metadata = {"step": i}
            config = await checkpointer.aput(
                checkpoint_config, checkpoint, metadata, {}
            )
            checkpoints.append(config)
            parent_id = config["configurable"]["checkpoint_id"]

        # Verify parent chain
        for i in range(1, len(checkpoints)):
            config = RunnableConfig(
                configurable={
                    "thread_id": sample_config["configurable"]["thread_id"],
                    "checkpoint_id": checkpoints[i]["configurable"]["checkpoint_id"],
                }
            )
            result = await checkpointer.aget_tuple(config)
            if i > 0:
                assert result.parent_config is not None
                assert (
                    result.parent_config["configurable"]["checkpoint_id"]
                    == checkpoints[i - 1]["configurable"]["checkpoint_id"]
                )


class TestTTLFunctionality:
    """Test TTL (Time-To-Live) functionality with integration tests."""

    @pytest.mark.asyncio
    async def test_ttl_with_async_operations(self, dynamodb_config, sample_config):
        """Test TTL functionality with async checkpoint operations."""
        from langgraph_checkpoint_dynamodb import DynamoDBTableConfig
        from langgraph_checkpoint_dynamodb.config import BillingMode
        import uuid

        # Create a new checkpointer with TTL enabled, reusing LocalStack config
        table_name = f"test-ttl-async-{uuid.uuid4().hex[:8]}"
        config = dynamodb_config
        config.table_config = DynamoDBTableConfig(
            table_name=table_name,
            billing_mode=BillingMode.PAY_PER_REQUEST,
            ttl_days=7,
        )
        ttl_checkpointer = DynamoDBSaver(config=config, deploy=True)

        try:
            # Create checkpoint with TTL using async method
            checkpoint = {"id": "checkpoint_ttl_async_1"}
            metadata = {"step": 1}
            result_config = await ttl_checkpointer.aput(
                sample_config, checkpoint, metadata, {}
            )

            # Get checkpoint - TTL filter should be applied
            result = await ttl_checkpointer.aget_tuple(result_config)
            assert result is not None
            assert result.checkpoint["id"] == "checkpoint_ttl_async_1"

            # Add writes with TTL using async method
            writes = [("channel1", "value1"), ("channel2", "value2")]
            task_id = "test_task_ttl_async"
            await ttl_checkpointer.aput_writes(result_config, writes, task_id)

            # Verify writes are retrieved with TTL filtering
            result = await ttl_checkpointer.aget_tuple(result_config)
            assert result is not None
            assert len(result.pending_writes) == 2

            # Test async list with TTL filtering
            results = []
            async for item in ttl_checkpointer.alist(sample_config):
                results.append(item)
            assert len(results) == 1
        finally:
            ttl_checkpointer.destroy()

    def test_ttl_with_sync_operations(self, dynamodb_config, sample_config):
        """Test TTL functionality with sync checkpoint operations."""
        from langgraph_checkpoint_dynamodb import DynamoDBTableConfig
        from langgraph_checkpoint_dynamodb.config import BillingMode
        import uuid

        # Create a new checkpointer with TTL enabled, reusing LocalStack config
        table_name = f"test-ttl-sync-{uuid.uuid4().hex[:8]}"
        config = dynamodb_config
        config.table_config = DynamoDBTableConfig(
            table_name=table_name,
            billing_mode=BillingMode.PAY_PER_REQUEST,
            ttl_days=7,
        )
        ttl_checkpointer = DynamoDBSaver(config=config, deploy=True)

        try:
            # Create checkpoint with TTL
            checkpoint = {"id": "checkpoint_ttl_sync_1"}
            metadata = {"step": 1}
            result_config = ttl_checkpointer.put(
                sample_config, checkpoint, metadata, {}
            )

            # Get checkpoint - TTL filter should be applied
            result = ttl_checkpointer.get_tuple(result_config)
            assert result is not None
            assert result.checkpoint["id"] == "checkpoint_ttl_sync_1"

            # Add writes with TTL
            writes = [
                ("channel1", "value1"),
                ("channel2", "value2"),
                ("channel3", "value3"),
            ]
            task_id = "test_task_ttl_sync"
            ttl_checkpointer.put_writes(result_config, writes, task_id)

            # Verify writes are retrieved with TTL filtering
            result = ttl_checkpointer.get_tuple(result_config)
            assert result is not None
            assert len(result.pending_writes) == 3

            # Test list with TTL filtering
            results = list(ttl_checkpointer.list(sample_config))
            assert len(results) == 1
            assert results[0].checkpoint["id"] == "checkpoint_ttl_sync_1"
        finally:
            ttl_checkpointer.destroy()

    def test_ttl_with_custom_attribute_name(self, dynamodb_config, sample_config):
        """Test TTL with custom attribute name."""
        from langgraph_checkpoint_dynamodb import DynamoDBTableConfig
        from langgraph_checkpoint_dynamodb.config import BillingMode
        import uuid

        # Create a new checkpointer with TTL and custom attribute name, reusing LocalStack config
        table_name = f"test-ttl-custom-{uuid.uuid4().hex[:8]}"
        config = dynamodb_config
        config.table_config = DynamoDBTableConfig(
            table_name=table_name,
            billing_mode=BillingMode.PAY_PER_REQUEST,
            ttl_days=30,
            ttl_attribute="customExpireTime",
        )
        ttl_checkpointer = DynamoDBSaver(config=config, deploy=True)

        try:
            # Create checkpoint with custom TTL attribute
            checkpoint = {"id": "checkpoint_ttl_custom_1"}
            metadata = {"step": 1}
            result_config = ttl_checkpointer.put(
                sample_config, checkpoint, metadata, {}
            )

            # Get checkpoint - should work with custom TTL attribute
            result = ttl_checkpointer.get_tuple(result_config)
            assert result is not None
            assert result.checkpoint["id"] == "checkpoint_ttl_custom_1"

            # Add writes with custom TTL attribute
            writes = [("channel1", "value1")]
            task_id = "test_task_ttl_custom"
            ttl_checkpointer.put_writes(result_config, writes, task_id)

            # Verify writes work with custom TTL attribute
            result = ttl_checkpointer.get_tuple(result_config)
            assert result is not None
            assert len(result.pending_writes) == 1
        finally:
            ttl_checkpointer.destroy()
