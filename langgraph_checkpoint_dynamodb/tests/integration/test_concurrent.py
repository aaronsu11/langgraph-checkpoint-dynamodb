"""Integration tests for concurrent operations and thread safety."""

import asyncio
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.graph import END, START, MessagesState, StateGraph

from langgraph_checkpoint_dynamodb import DynamoDBSaver

pytestmark = pytest.mark.integration


def create_simple_agent_workflow():
    """Create a simple agent workflow for testing."""
    def assistant(state: MessagesState):
        messages = state["messages"]
        last_msg = messages[-1]
        if isinstance(last_msg, HumanMessage):
            return {
                "messages": [
                    AIMessage(content=f"Response to: {last_msg.content}")
                ]
            }
        return {"messages": [AIMessage(content="I understand.")]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("assistant", assistant)
    workflow.add_edge(START, "assistant")
    workflow.add_edge("assistant", END)
    return workflow


class TestThreadIsolation:
    """Test thread isolation with concurrent operations."""

    def test_multiple_threads_isolation(self, checkpointer):
        """Test that different thread_ids don't interfere."""
        workflow = create_simple_agent_workflow()
        graph = workflow.compile(checkpointer=checkpointer)

        # Create multiple threads
        threads = [f"thread_isolate_{i}" for i in range(5)]
        configs = [
            {"configurable": {"thread_id": thread_id}} for thread_id in threads
        ]

        # Interleave operations across threads
        for i in range(3):
            for j, config in enumerate(configs):
                graph.invoke(
                    {
                        "messages": [
                            HumanMessage(content=f"Thread {j}, Step {i}")
                        ]
                    },
                    config,
                )

        # Verify threads are isolated
        for i, config in enumerate(configs):
            state = graph.get_state(config)
            messages = state.values["messages"]
            assert len(messages) >= 6  # 3 human + 3 AI messages

            # Verify messages belong to this thread
            human_msgs = [
                m for m in messages if isinstance(m, HumanMessage)
            ]
            assert all(f"Thread {i}" in msg.content for msg in human_msgs)

    @pytest.mark.asyncio
    async def test_multiple_threads_isolation_async(self, checkpointer):
        """Test thread isolation with async operations."""
        workflow = create_simple_agent_workflow()
        graph = workflow.compile(checkpointer=checkpointer)

        # Create multiple threads
        threads = [f"thread_async_isolate_{i}" for i in range(3)]
        configs = [
            {"configurable": {"thread_id": thread_id}} for thread_id in threads
        ]

        # Async interleave operations
        async def run_thread(config, thread_idx):
            for i in range(2):
                await graph.ainvoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=f"Async Thread {thread_idx}, Step {i}"
                            )
                        ]
                    },
                    config,
                )

        # Run all threads concurrently
        tasks = [run_thread(config, i) for i, config in enumerate(configs)]
        await asyncio.gather(*tasks)

        # Verify threads are isolated
        for i, config in enumerate(configs):
            state = await graph.aget_state(config)
            messages = state.values["messages"]
            assert len(messages) >= 4  # 2 human + 2 AI messages


class TestConcurrentWrites:
    """Test concurrent writes to the same thread."""

    def test_concurrent_writes_same_thread_sync(self, checkpointer):
        """Test handling concurrent writes to same thread (sequential)."""
        from langchain_core.runnables import RunnableConfig

        config = RunnableConfig(
            configurable={"thread_id": "thread_concurrent_sync"}
        )

        # Create checkpoint
        checkpoint = {"id": "checkpoint_concurrent"}
        metadata = {"step": 0}
        config_with_id = checkpointer.put(config, checkpoint, metadata, {})

        # Add writes sequentially (simulating concurrent access)
        writes1 = [("channel1", "value1"), ("channel2", "value2")]
        writes2 = [("channel3", "value3"), ("channel4", "value4")]

        checkpointer.put_writes(config_with_id, writes1, "task1")
        checkpointer.put_writes(config_with_id, writes2, "task2")

        # Verify all writes stored
        result = checkpointer.get_tuple(config_with_id)
        assert result is not None
        assert len(result.pending_writes) == 4  # All writes

    @pytest.mark.asyncio
    async def test_concurrent_writes_same_thread_async(self, checkpointer):
        """Test concurrent writes to same thread with async."""
        from langchain_core.runnables import RunnableConfig

        config = RunnableConfig(
            configurable={"thread_id": "thread_concurrent_async"}
        )

        # Create checkpoint
        checkpoint = {"id": "checkpoint_concurrent_async"}
        metadata = {"step": 0}
        config_with_id = await checkpointer.aput(
            config, checkpoint, metadata, {}
        )

        # Add writes concurrently
        writes1 = [("channel1", "value1"), ("channel2", "value2")]
        writes2 = [("channel3", "value3"), ("channel4", "value4")]
        writes3 = [("channel5", "value5"), ("channel6", "value6")]

        # Write concurrently
        await asyncio.gather(
            checkpointer.aput_writes(config_with_id, writes1, "task1"),
            checkpointer.aput_writes(config_with_id, writes2, "task2"),
            checkpointer.aput_writes(config_with_id, writes3, "task3"),
        )

        # Verify all writes stored
        result = await checkpointer.aget_tuple(config_with_id)
        assert result is not None
        assert len(result.pending_writes) == 6  # All writes

    @pytest.mark.asyncio
    async def test_concurrent_checkpoint_creation(self, checkpointer):
        """Test concurrent checkpoint creation."""
        from langchain_core.runnables import RunnableConfig

        base_thread = "thread_concurrent_create"

        async def create_checkpoint(idx):
            config = RunnableConfig(
                configurable={
                    "thread_id": base_thread,
                    "checkpoint_id": f"checkpoint_{idx}",
                }
            )
            checkpoint = {"id": f"checkpoint_{idx}"}
            metadata = {"step": idx}
            await checkpointer.aput(config, checkpoint, metadata, {})

        # Create checkpoints concurrently
        await asyncio.gather(*[create_checkpoint(i) for i in range(5)])

        # Verify all checkpoints created
        config = RunnableConfig(configurable={"thread_id": base_thread})
        results = []
        async for result in checkpointer.alist(config):
            results.append(result)

        assert len(results) >= 5
        checkpoint_ids = {r.checkpoint["id"] for r in results}
        assert all(f"checkpoint_{i}" in checkpoint_ids for i in range(5))


class TestAsyncConcurrentOperations:
    """Test async concurrent operations."""

    @pytest.mark.asyncio
    async def test_async_concurrent_operations(self, checkpointer):
        """Test multiple async tasks concurrently."""
        workflow = create_simple_agent_workflow()
        graph = workflow.compile(checkpointer=checkpointer)

        async def run_agent(thread_id, steps):
            config = {"configurable": {"thread_id": thread_id}}
            for i in range(steps):
                await graph.ainvoke(
                    {"messages": [HumanMessage(content=f"Step {i}")]}, config
                )

        # Run multiple agents concurrently
        await asyncio.gather(
            run_agent("thread_async_1", 3),
            run_agent("thread_async_2", 3),
            run_agent("thread_async_3", 3),
        )

        # Verify all agents completed
        for thread_id in ["thread_async_1", "thread_async_2", "thread_async_3"]:
            config = {"configurable": {"thread_id": thread_id}}
            state = await graph.aget_state(config)
            assert len(state.values["messages"]) >= 6  # 3 human + 3 AI

    @pytest.mark.asyncio
    async def test_async_streaming_concurrent(self, checkpointer):
        """Test concurrent async streaming."""
        workflow = create_simple_agent_workflow()
        graph = workflow.compile(checkpointer=checkpointer)

        async def stream_agent(thread_id):
            config = {"configurable": {"thread_id": thread_id}}
            events = []
            async for event in graph.astream(
                {"messages": [HumanMessage(content="Start")]}, config
            ):
                events.append(event)
            return events

        # Stream multiple agents concurrently
        results = await asyncio.gather(
            stream_agent("thread_stream_1"),
            stream_agent("thread_stream_2"),
            stream_agent("thread_stream_3"),
        )

        # Verify all streams completed
        for result in results:
            assert len(result) > 0


class TestBatchWritePerformance:
    """Test batch write performance."""

    @pytest.mark.asyncio
    async def test_batch_write_performance(self, checkpointer):
        """Test aput_writes with large batches."""
        from langchain_core.runnables import RunnableConfig

        config = RunnableConfig(
            configurable={"thread_id": "thread_batch_perf"}
        )

        # Create checkpoint
        checkpoint = {"id": "checkpoint_batch"}
        metadata = {"step": 0}
        config_with_id = await checkpointer.aput(
            config, checkpoint, metadata, {}
        )

        # Create large batch (100 writes, exceeds DynamoDB limit of 25)
        writes = [(f"channel{i}", f"value{i}") for i in range(100)]

        # Write batch
        await checkpointer.aput_writes(config_with_id, writes, "large_task")

        # Verify all writes stored
        result = await checkpointer.aget_tuple(config_with_id)
        assert result is not None
        assert len(result.pending_writes) == 100

        # Verify write order maintained
        for i in range(100):
            assert result.pending_writes[i][1] == f"channel{i}"

    @pytest.mark.asyncio
    async def test_multiple_batch_writes(self, checkpointer):
        """Test multiple large batch writes concurrently."""
        from langchain_core.runnables import RunnableConfig

        config = RunnableConfig(
            configurable={"thread_id": "thread_multi_batch"}
        )

        # Create checkpoint
        checkpoint = {"id": "checkpoint_multi_batch"}
        metadata = {"step": 0}
        config_with_id = await checkpointer.aput(
            config, checkpoint, metadata, {}
        )

        # Create multiple large batches
        batch1 = [(f"channel1_{i}", f"value1_{i}") for i in range(50)]
        batch2 = [(f"channel2_{i}", f"value2_{i}") for i in range(50)]
        batch3 = [(f"channel3_{i}", f"value3_{i}") for i in range(50)]

        # Write concurrently
        await asyncio.gather(
            checkpointer.aput_writes(config_with_id, batch1, "task1"),
            checkpointer.aput_writes(config_with_id, batch2, "task2"),
            checkpointer.aput_writes(config_with_id, batch3, "task3"),
        )

        # Verify all writes stored
        result = await checkpointer.aget_tuple(config_with_id)
        assert result is not None
        assert len(result.pending_writes) == 150  # All batches


class TestConcurrentListOperations:
    """Test concurrent list operations."""

    @pytest.mark.asyncio
    async def test_concurrent_list_operations(self, checkpointer):
        """Test listing checkpoints concurrently."""
        from langchain_core.runnables import RunnableConfig

        base_thread = "thread_concurrent_list"

        # Create checkpoints
        for i in range(10):
            config = RunnableConfig(
                configurable={
                    "thread_id": base_thread,
                    "checkpoint_id": f"checkpoint_{i}",
                }
            )
            checkpoint = {"id": f"checkpoint_{i}"}
            metadata = {"step": i}
            await checkpointer.aput(config, checkpoint, metadata, {})

        # List concurrently
        list_config = RunnableConfig(configurable={"thread_id": base_thread})

        async def list_checkpoints():
            results = []
            async for result in checkpointer.alist(list_config):
                results.append(result)
            return results

        # Run multiple list operations concurrently
        results_list = await asyncio.gather(
            list_checkpoints(),
            list_checkpoints(),
            list_checkpoints(),
        )

        # All should return same results
        for results in results_list:
            assert len(results) == 10
            checkpoint_ids = {r.checkpoint["id"] for r in results}
            assert all(f"checkpoint_{i}" in checkpoint_ids for i in range(10))

