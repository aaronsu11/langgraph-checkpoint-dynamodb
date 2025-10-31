import operator

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.graph import END, START, MessageGraph, StateGraph
from moto import mock_aws
from typing_extensions import Annotated, TypedDict

from langgraph_checkpoint_dynamodb import (
    DynamoDBConfig,
    DynamoDBSaver,
    DynamoDBTableConfig,
)
from langgraph_checkpoint_dynamodb.config import BillingMode
from langgraph_checkpoint_dynamodb.errors import DynamoDBCheckpointError

# Configure pytest-asyncio to use function scope for event loops
# Configuration is handled in pyproject.toml


@pytest.fixture(scope="function")
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    import os

    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture(scope="function")
def dynamodb_table(aws_credentials):
    """Create a mock DynamoDB table."""
    with mock_aws():
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)
        yield saver


@pytest.fixture(scope="function")
def sample_config():
    """Sample runnable config for testing."""
    return RunnableConfig(
        configurable={
            "thread_id": "test_thread",
            "checkpoint_ns": "",
            "checkpoint_id": "test_checkpoint_1",
        }
    )


@pytest.fixture(scope="function")
def sample_messages():
    """Sample messages for testing."""
    return [HumanMessage(content="Hi!"), AIMessage(content="Hello!")]


class TestDynamoDBSaverBasics:
    """Basic functionality tests for DynamoDBSaver."""

    @mock_aws
    def test_initialization(self, aws_credentials):
        """Test basic initialization of DynamoDBSaver."""
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)
        assert isinstance(saver, DynamoDBSaver)

    @mock_aws
    def test_put_checkpoint(self, aws_credentials, sample_config):
        """Test putting a simple checkpoint."""
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        checkpoint = {
            "id": sample_config["configurable"]["checkpoint_id"],
            "data": "test",
        }
        metadata = {"step": 1}
        new_versions = {}

        config = saver.put(sample_config, checkpoint, metadata, new_versions)
        assert (
            config["configurable"]["checkpoint_id"]
            == sample_config["configurable"]["checkpoint_id"]
        )

    @mock_aws
    def test_get_nonexistent_checkpoint(self, aws_credentials):
        """Test getting a checkpoint that doesn't exist."""
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        config = RunnableConfig(
            configurable={"thread_id": "nonexistent", "checkpoint_id": "nonexistent"}
        )
        result = saver.get_tuple(config)
        assert result is None


class TestDynamoDBSaverCheckpoints:
    """Tests for checkpoint operations."""

    @mock_aws
    def test_put_and_get_checkpoint_with_messages(
        self, aws_credentials, sample_config, sample_messages
    ):
        """Test putting and getting a checkpoint with messages."""
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        checkpoint = {
            "id": sample_config["configurable"]["checkpoint_id"],
            "messages": sample_messages,
        }
        metadata = {"step": 1}
        new_versions = {}

        # Put checkpoint
        config = saver.put(sample_config, checkpoint, metadata, new_versions)

        # Get and verify
        result = saver.get_tuple(sample_config)
        assert isinstance(result, CheckpointTuple)
        assert len(result.checkpoint["messages"]) == len(sample_messages)
        assert result.metadata["step"] == 1

    @mock_aws
    def test_checkpoint_metadata(self, aws_credentials, sample_config):
        """Test checkpoint metadata handling."""
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        checkpoint = {"id": sample_config["configurable"]["checkpoint_id"]}
        metadata = {"step": 1, "custom_field": "test_value"}
        new_versions = {}

        saver.put(sample_config, checkpoint, metadata, new_versions)
        result = saver.get_tuple(sample_config)

        assert result.metadata["step"] == 1
        assert result.metadata["custom_field"] == "test_value"


class TestDynamoDBSaverWrites:
    """Tests for write operations."""

    @mock_aws
    def test_put_single_write(self, aws_credentials, sample_config):
        """Test putting a single write."""
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        # First create a checkpoint
        checkpoint = {"id": sample_config["configurable"]["checkpoint_id"]}
        metadata = {"step": 1}
        saver.put(sample_config, checkpoint, metadata, {})

        # Then add writes
        writes = [("channel1", "value1")]
        task_id = "test_task"

        saver.put_writes(sample_config, writes, task_id)
        result = saver.get_tuple(sample_config)

        assert result is not None
        assert len(result.pending_writes) == 1
        assert result.pending_writes[0][0] == task_id
        assert result.pending_writes[0][1] == "channel1"

    @mock_aws
    def test_put_multiple_writes(self, aws_credentials, sample_config):
        """Test putting multiple writes."""
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        # First create a checkpoint
        checkpoint = {"id": sample_config["configurable"]["checkpoint_id"]}
        metadata = {"step": 1}
        saver.put(sample_config, checkpoint, metadata, {})

        writes = [
            ("channel1", "value1"),
            ("channel2", "value2"),
            ("channel3", "value3"),
        ]
        task_id = "test_task"

        saver.put_writes(sample_config, writes, task_id)
        result = saver.get_tuple(sample_config)

        assert result is not None
        assert len(result.pending_writes) == 3
        assert all(write[0] == task_id for write in result.pending_writes)


class TestDynamoDBSaverListing:
    """Tests for listing operations."""

    @mock_aws
    def test_list_empty(self, aws_credentials, sample_config):
        """Test listing when no checkpoints exist."""
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        results = list(saver.list(sample_config))
        assert len(results) == 0

    @mock_aws
    def test_list_multiple_checkpoints(self, aws_credentials, sample_config):
        """Test listing multiple checkpoints."""
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        # Create multiple checkpoints
        for i in range(3):
            config = sample_config.copy()
            config["configurable"]["checkpoint_id"] = f"test_checkpoint_{i}"
            checkpoint = {"id": config["configurable"]["checkpoint_id"]}
            metadata = {"step": i}
            saver.put(config, checkpoint, metadata, {})

        results = list(saver.list(sample_config))
        assert len(results) == 3
        assert all(isinstance(r, CheckpointTuple) for r in results)

    @mock_aws
    def test_list_with_filter(self, aws_credentials, sample_config):
        """Test listing checkpoints with filter."""
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        # Create checkpoints with different metadata
        for i in range(3):
            config = sample_config.copy()
            config["configurable"]["checkpoint_id"] = f"test_checkpoint_{i}"
            checkpoint = {"id": config["configurable"]["checkpoint_id"]}
            metadata = {"step": i, "type": "test" if i == 1 else "other"}
            saver.put(config, checkpoint, metadata, {})

        filtered_results = list(saver.list(sample_config, filter={"type": "test"}))
        assert len(filtered_results) == 1


class TestLangGraphUsage:
    """Tests for high-level usage with LangGraph."""

    @mock_aws
    def test_message_graph_basic(self, aws_credentials):
        """Test basic MessageGraph usage."""
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        workflow = MessageGraph()
        workflow.add_node(
            "chatbot", lambda state: [{"role": "ai", "content": "Hello!"}]
        )
        workflow.add_edge(START, "chatbot")
        workflow.add_edge("chatbot", END)

        graph = workflow.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "test_thread_msg"}}

        result = graph.invoke([{"role": "human", "content": "Hi!"}], config)
        assert len(result) == 2
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)

    @mock_aws
    def test_state_graph_with_channels(self, aws_credentials):
        """Test StateGraph with multiple channels."""

        class State(TypedDict):
            count: int
            messages: Annotated[list[str], operator.add]

        workflow = StateGraph(State)
        workflow.add_node("counter", lambda state: {"count": state["count"] + 1})
        workflow.add_node(
            "messenger", lambda state: {"messages": ["msg" + str(state["count"])]}
        )
        workflow.add_edge(START, "counter")
        workflow.add_edge("counter", "messenger")
        workflow.add_edge("messenger", END)

        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        graph = workflow.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "test_thread_state"}}

        result = graph.invoke({"count": 0, "messages": []}, config)
        assert result["count"] == 1
        assert result["messages"] == ["msg1"]

    @mock_aws
    def test_graph_state_management(self, aws_credentials):
        """Test graph state management and retrieval."""
        workflow = MessageGraph()
        workflow.add_node(
            "chatbot", lambda state: [{"role": "ai", "content": "Hello!"}]
        )
        workflow.add_edge(START, "chatbot")
        workflow.add_edge("chatbot", END)

        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        graph = workflow.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "test_thread_state_mgmt"}}

        # Initial invocation
        graph.invoke([{"role": "human", "content": "Hi!"}], config)

        # Get state
        state = graph.get_state(config)
        assert len(state.values) == 2
        assert state.values[0].content == "Hi!"
        assert state.values[1].content == "Hello!"

        # Get state history
        history = list(graph.get_state_history(config))
        assert len(history) > 0
        assert all(hasattr(state, "values") for state in history)

    @mock_aws
    def test_graph_streaming(self, aws_credentials):
        """Test graph streaming capabilities."""
        workflow = MessageGraph()
        workflow.add_node(
            "chatbot", lambda state: [{"role": "ai", "content": "Hello!"}]
        )
        workflow.add_edge(START, "chatbot")
        workflow.add_edge("chatbot", END)

        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        graph = workflow.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "test_thread_stream"}}

        # Test values stream mode
        values_updates = list(
            graph.stream(
                [{"role": "human", "content": "Hi!"}], config, stream_mode="values"
            )
        )
        assert len(values_updates) > 0

        # Test updates stream mode
        updates = list(
            graph.stream(
                [{"role": "human", "content": "Hi!"}], config, stream_mode="updates"
            )
        )
        assert len(updates) > 0


class TestDynamoDBSaverErrorHandling:
    """Tests for error handling and edge cases."""

    @mock_aws
    def test_initialization_without_deploy_fails(self, aws_credentials):
        """
        Test initialization without deploy fails for non-existent table.

        Covers:
        - __init__ error path when deploy=False and table doesn't exist (lines 89-105)
        """
        table_config = DynamoDBTableConfig(
            table_name="nonexistent_table", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)

        # Should raise DynamoDBCheckpointError when table doesn't exist and deploy=False
        with pytest.raises(DynamoDBCheckpointError) as exc_info:
            DynamoDBSaver(config=config, deploy=False)

        # Verify the error message mentions the table doesn't exist
        assert "does not exist" in str(exc_info.value)

    @mock_aws
    def test_get_latest_checkpoint(self, aws_credentials, sample_config):
        """
        Test getting latest checkpoint without checkpoint_id.

        Covers:
        - get_tuple() without checkpoint_id (gets latest) (lines 140-166)
        - Parent checkpoint handling
        """
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        # Create multiple checkpoints
        checkpoint_configs = []
        parent_id = None
        for i in range(3):
            checkpoint_config = RunnableConfig(
                configurable={
                    "thread_id": sample_config["configurable"]["thread_id"],
                    "checkpoint_ns": "",
                    "checkpoint_id": parent_id,  # Set parent checkpoint_id
                }
            )
            checkpoint = {"id": f"checkpoint_{i}"}
            metadata = {"step": i}
            saver.put(checkpoint_config, checkpoint, metadata, {})
            checkpoint_configs.append(checkpoint_config)
            # Update parent_id for next checkpoint
            parent_id = f"checkpoint_{i}"

        # Get latest checkpoint without checkpoint_id
        latest_config = RunnableConfig(
            configurable={
                "thread_id": sample_config["configurable"]["thread_id"],
                "checkpoint_ns": "",
                # No checkpoint_id specified - should get latest
            }
        )
        result = saver.get_tuple(latest_config)

        # Should return the most recent checkpoint (checkpoint_2)
        assert result is not None
        assert result.checkpoint["id"] == "checkpoint_2"

    @mock_aws
    def test_list_with_pagination(self, aws_credentials, sample_config):
        """
        Test list() with before and limit parameters.

        Covers:
        - list() with before parameter (lines 402-405)
        - list() with limit parameter (lines 413-414)
        """
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints", billing_mode=BillingMode.PAY_PER_REQUEST
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        # Create multiple checkpoints
        checkpoint_configs = []
        for i in range(5):
            checkpoint_config = RunnableConfig(
                configurable={
                    "thread_id": sample_config["configurable"]["thread_id"],
                    "checkpoint_ns": "",
                    "checkpoint_id": f"checkpoint_{i}",
                }
            )
            checkpoint = {"id": f"checkpoint_{i}"}
            metadata = {"step": i}
            saver.put(checkpoint_config, checkpoint, metadata, {})
            checkpoint_configs.append(checkpoint_config)

        # Test with limit
        results = list(saver.list(sample_config, limit=3))
        assert len(results) == 3
        # Should get the 3 most recent checkpoints
        assert results[0].checkpoint["id"] == "checkpoint_4"
        assert results[1].checkpoint["id"] == "checkpoint_3"
        assert results[2].checkpoint["id"] == "checkpoint_2"

        # Test with before parameter
        before_config = RunnableConfig(
            configurable={
                "thread_id": sample_config["configurable"]["thread_id"],
                "checkpoint_ns": "",
                "checkpoint_id": "checkpoint_3",
            }
        )
        results_before = list(saver.list(sample_config, before=before_config))
        # Should get checkpoints before checkpoint_3 (checkpoint_2, checkpoint_1, checkpoint_0)
        assert len(results_before) == 3
        assert results_before[0].checkpoint["id"] == "checkpoint_2"
        assert results_before[1].checkpoint["id"] == "checkpoint_1"
        assert results_before[2].checkpoint["id"] == "checkpoint_0"

    @mock_aws
    def test_ttl_functionality(self, aws_credentials, sample_config):
        """
        Test checkpoint creation with TTL and TTL filtering in queries.

        Covers:
        - TTL paths in create_checkpoint_item (lines 152-154 in utils.py)
        - TTL filtering in queries (lines 152-157, 176-181 in saver.py)
        """
        # Configure table with TTL
        table_config = DynamoDBTableConfig(
            table_name="test_checkpoints",
            billing_mode=BillingMode.PAY_PER_REQUEST,
            ttl_days=7,
        )
        config = DynamoDBConfig(table_config=table_config)
        saver = DynamoDBSaver(config=config, deploy=True)

        # Create checkpoint with TTL
        checkpoint = {"id": sample_config["configurable"]["checkpoint_id"]}
        metadata = {"step": 1}
        saver.put(sample_config, checkpoint, metadata, {})

        # Get checkpoint and verify TTL was set
        result = saver.get_tuple(sample_config)
        assert result is not None
        assert result.checkpoint["id"] == sample_config["configurable"]["checkpoint_id"]

        # Verify TTL filter is applied in queries
        # Create another checkpoint
        checkpoint_config2 = RunnableConfig(
            configurable={
                "thread_id": sample_config["configurable"]["thread_id"],
                "checkpoint_ns": "",
                "checkpoint_id": "checkpoint_with_ttl",
            }
        )
        checkpoint2 = {"id": "checkpoint_with_ttl"}
        metadata2 = {"step": 2}
        saver.put(checkpoint_config2, checkpoint2, metadata2, {})

        # List checkpoints - should filter out expired ones
        results = list(saver.list(sample_config))
        assert len(results) == 2
        # Both checkpoints should be returned as they haven't expired yet
        checkpoint_ids = {r.checkpoint["id"] for r in results}
        assert "checkpoint_with_ttl" in checkpoint_ids
        assert sample_config["configurable"]["checkpoint_id"] in checkpoint_ids
