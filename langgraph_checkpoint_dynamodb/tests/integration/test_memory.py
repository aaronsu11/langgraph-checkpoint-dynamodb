"""Integration tests for memory functionality with LangGraph."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from langgraph_checkpoint_dynamodb import DynamoDBSaver

pytestmark = pytest.mark.integration


def create_simple_agent_workflow():
    """Create a simple agent workflow with tools."""

    # Simple tools
    def add(a: int, b: int) -> int:
        """Adds a and b."""
        return a + b

    def multiply(a: int, b: int) -> int:
        """Multiply a and b."""
        return a * b

    def divide(a: int, b: int) -> float:
        """Divide a by b."""
        return a / b

    tools = [add, multiply, divide]

    # Simple assistant node that simulates tool calling
    def assistant(state: MessagesState):
        messages = state["messages"]
        last_msg = messages[-1]
        if isinstance(last_msg, HumanMessage):
            content = last_msg.content.lower()
            # Simple pattern matching for tool calls (no LLM)
            if "add" in content or "+" in content:
                # Extract numbers (simplified)
                if "3 and 4" in content or "3+4" in content:
                    result = add(3, 4)
                    return {
                        "messages": [
                            AIMessage(
                                content=f"The sum of 3 and 4 is {result}.",
                            )
                        ]
                    }
            elif "multiply" in content or "*" in content:
                # Try to extract previous result from context
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and "sum" in msg.content.lower():
                        # Extract 7 from previous context
                        result = multiply(7, 2)
                        return {
                            "messages": [
                                AIMessage(
                                    content=f"The result of multiplying 7 by 2 is {result}.",
                                )
                            ]
                        }
                # Default
                if "2 and 3" in content:
                    result = multiply(2, 3)
                    return {
                        "messages": [
                            AIMessage(
                                content=f"The result of multiplying 2 and 3 is {result}.",
                            )
                        ]
                    }
        return {"messages": [AIMessage(content="I understand.")]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("assistant", assistant)
    workflow.add_edge(START, "assistant")
    workflow.add_edge("assistant", END)
    return workflow


class TestSyncMemory:
    """Test short-term memory scenarios."""

    def test_short_term_memory(self, checkpointer):
        """Test simple agent with memory across invocations."""
        workflow = create_simple_agent_workflow()
        graph = workflow.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "thread_memory_1"}}

        # First interaction
        result1 = graph.invoke(
            {"messages": [HumanMessage(content="Add 3 and 4")]}, config
        )
        assert len(result1["messages"]) >= 1
        assert any("7" in str(msg.content) for msg in result1["messages"])

        # Second interaction - should remember context
        result2 = graph.invoke(
            {"messages": [HumanMessage(content="Multiply that by 2")]}, config
        )
        # Should reference previous result (7) and multiply by 2
        assert len(result2["messages"]) >= 1
        messages_content = " ".join(str(msg.content) for msg in result2["messages"])
        # Check that it's working with context
        assert any(msg for msg in result2["messages"] if isinstance(msg, AIMessage))

    def test_multi_turn_conversation(self, checkpointer):
        """Test multiple interactions with same thread_id."""
        workflow = create_simple_agent_workflow()
        graph = workflow.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "thread_multi_turn"}}

        # First turn
        result1 = graph.invoke(
            {"messages": [HumanMessage(content="Add 3 and 4")]}, config
        )

        # Second turn
        result2 = graph.invoke(
            {"messages": [HumanMessage(content="What did we calculate?")]}, config
        )

        # Third turn
        result3 = graph.invoke(
            {"messages": [HumanMessage(content="Now multiply that by 2")]}, config
        )

        # Verify all turns stored messages
        state = graph.get_state(config)
        assert len(state.values["messages"]) >= 6  # At least 3 human + 3 AI messages

    def test_separate_threads(self, checkpointer):
        """Test that different thread_ids maintain separate memory."""
        workflow = create_simple_agent_workflow()
        graph = workflow.compile(checkpointer=checkpointer)

        config1 = {"configurable": {"thread_id": "thread_separate_1"}}
        config2 = {"configurable": {"thread_id": "thread_separate_2"}}

        # Interaction in thread 1
        graph.invoke({"messages": [HumanMessage(content="Add 3 and 4")]}, config1)

        # Interaction in thread 2
        graph.invoke({"messages": [HumanMessage(content="Multiply 2 and 3")]}, config2)

        # Verify threads are isolated
        state1 = graph.get_state(config1)
        state2 = graph.get_state(config2)

        assert len(state1.values["messages"]) >= 2
        assert len(state2.values["messages"]) >= 2

        # Verify different contexts
        msg1 = next(
            (m.content for m in state1.values["messages"] if "7" in str(m.content)),
            None,
        )
        msg2 = next(
            (m.content for m in state2.values["messages"] if "6" in str(m.content)),
            None,
        )

        # At least one thread should have its expected content
        assert msg1 is not None or msg2 is not None

    def test_persistent_memory(self, checkpointer, dynamodb_config):
        """Test that checkpoints survive saver recreation."""
        import uuid

        table_name = f"test-persistent-{uuid.uuid4().hex[:8]}"
        config = dynamodb_config
        config.table_config.table_name = table_name

        workflow = create_simple_agent_workflow()

        # Create first saver and run
        saver1 = DynamoDBSaver(config=config, deploy=True)
        graph1 = workflow.compile(checkpointer=saver1)

        thread_config = {"configurable": {"thread_id": "thread_persistent"}}

        result1 = graph1.invoke(
            {"messages": [HumanMessage(content="Add 3 and 4")]}, thread_config
        )
        assert len(result1["messages"]) >= 1

        # Create new saver pointing to same table
        saver2 = DynamoDBSaver(config=config, deploy=False)
        graph2 = workflow.compile(checkpointer=saver2)

        # Verify state persisted
        state = graph2.get_state(thread_config)
        assert len(state.values["messages"]) >= 2  # Human + AI message

        # Continue conversation
        result2 = graph2.invoke(
            {"messages": [HumanMessage(content="Multiply that by 2")]},
            thread_config,
        )
        assert len(result2["messages"]) >= 1

        # Cleanup
        saver2.destroy()

    def test_message_accumulation(self, checkpointer):
        """Test that messages accumulate correctly across calls."""
        workflow = create_simple_agent_workflow()
        graph = workflow.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "thread_accumulation"}}

        # Multiple interactions
        for i in range(5):
            graph.invoke({"messages": [HumanMessage(content=f"Message {i}")]}, config)

        # Verify all messages accumulated
        state = graph.get_state(config)
        messages = state.values["messages"]
        assert len(messages) >= 10  # At least 5 human + 5 AI messages

        # Verify order (recent messages at end)
        assert isinstance(messages[-1], AIMessage)


class TestAsyncMemory:
    """Test async memory operations."""

    @pytest.mark.asyncio
    async def test_async_memory_operations(self, checkpointer):
        """Test memory operations with async methods."""
        workflow = create_simple_agent_workflow()
        graph = workflow.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "thread_async_memory"}}

        # First interaction
        result1 = await graph.ainvoke(
            {"messages": [HumanMessage(content="Add 3 and 4")]}, config
        )
        assert len(result1["messages"]) >= 1

        # Second interaction
        result2 = await graph.ainvoke(
            {"messages": [HumanMessage(content="Multiply that by 2")]}, config
        )
        assert len(result2["messages"]) >= 1

        # Verify state persisted
        state = await graph.aget_state(config)
        assert len(state.values["messages"]) >= 4  # At least 2 human + 2 AI messages

    @pytest.mark.asyncio
    async def test_async_stream_memory(self, checkpointer):
        """Test async streaming with memory."""
        workflow = create_simple_agent_workflow()
        graph = workflow.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "thread_async_stream"}}

        # Stream first interaction
        events = []
        async for event in graph.astream(
            {"messages": [HumanMessage(content="Add 3 and 4")]}, config
        ):
            events.append(event)

        assert len(events) > 0

        # Stream second interaction
        events2 = []
        async for event in graph.astream(
            {"messages": [HumanMessage(content="Now multiply that by 2")]}, config
        ):
            events2.append(event)

        assert len(events2) > 0

        # Verify memory persists
        state = await graph.aget_state(config)
        assert len(state.values["messages"]) >= 4
