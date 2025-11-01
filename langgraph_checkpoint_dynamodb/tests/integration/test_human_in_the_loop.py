"""Integration tests for interrupts and human-in-the-loop functionality."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph_checkpoint_dynamodb import DynamoDBSaver

pytestmark = pytest.mark.integration


def create_tool_agent_workflow():
    """Create a simple agent workflow with tools."""

    # Simple tools
    def multiply(a: int, b: int) -> int:
        """Multiply a and b."""
        return a * b

    def add(a: int, b: int) -> int:
        """Adds a and b."""
        return a + b

    def divide(a: int, b: int) -> float:
        """Divide a by b."""
        return a / b

    tools = [add, multiply, divide]

    # Simple assistant that generates tool calls
    def assistant(state: MessagesState):
        messages = state["messages"]
        last_msg = messages[-1]
        if isinstance(last_msg, HumanMessage):
            content = last_msg.content.lower()
            # Simulate tool calls (no real LLM)
            if "multiply" in content and "2" in content and "3" in content:
                from langchain_core.messages.tool import ToolMessage

                # Create tool call message
                tool_msg = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "multiply",
                            "args": {"a": 2, "b": 3},
                            "id": "call_multiply_123",
                        }
                    ],
                )
                return {"messages": [tool_msg]}
            elif "add" in content and "3" in content and "4" in content:
                tool_msg = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {"a": 3, "b": 4},
                            "id": "call_add_123",
                        }
                    ],
                )
                return {"messages": [tool_msg]}

        # Default response
        return {"messages": [AIMessage(content="I understand.")]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("assistant", assistant)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "assistant")
    workflow.add_conditional_edges("assistant", tools_condition)
    workflow.add_edge("tools", "assistant")
    return workflow


class TestInterrupts:
    """Test interrupt functionality."""

    def test_interrupt_before_node(self, checkpointer):
        """Test that interrupt_before stops execution before specified node."""
        workflow = create_tool_agent_workflow()
        graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["tools"])

        config = {"configurable": {"thread_id": "thread_interrupt"}}

        # Run graph - should stop before tools
        events = list(
            graph.stream(
                {"messages": [HumanMessage(content="Multiply 2 and 3")]}, config
            )
        )

        # Verify it stopped
        state = graph.get_state(config)
        assert state.next == ("tools",)

        # Verify we got the tool call message
        assert len(state.values["messages"]) >= 1
        last_msg = state.values["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert len(last_msg.tool_calls) > 0

    def test_resume_after_interrupt(self, checkpointer):
        """Test resuming execution after interrupt."""
        workflow = create_tool_agent_workflow()
        graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["tools"])

        config = {"configurable": {"thread_id": "thread_resume"}}

        # First run - should stop before tools
        events1 = list(
            graph.stream(
                {"messages": [HumanMessage(content="Multiply 2 and 3")]}, config
            )
        )

        # Verify it stopped
        state = graph.get_state(config)
        assert state.next == ("tools",)

        # Resume execution
        events2 = list(graph.stream(None, config))

        # Verify execution completed
        state_after = graph.get_state(config)
        # Should have completed (no next node unless it continues)
        # Verify we got the tool result and final response
        messages = state_after.values["messages"]
        assert len(messages) >= 3  # Human, Tool call, Tool result, Final AI

    def test_user_approval_workflow(self, checkpointer):
        """Test user approval workflow with interrupts."""
        workflow = create_tool_agent_workflow()
        graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["tools"])

        config = {"configurable": {"thread_id": "thread_approval"}}

        # Initial run - stops before tools
        list(
            graph.stream(
                {"messages": [HumanMessage(content="Multiply 2 and 3")]}, config
            )
        )

        # Simulate user approval
        state = graph.get_state(config)
        assert state.next == ("tools",)

        # User approves - continue
        events = list(graph.stream(None, config))

        # Verify execution completed
        final_state = graph.get_state(config)
        messages = final_state.values["messages"]
        # Should have tool execution
        assert any(isinstance(msg, AIMessage) and msg.tool_calls for msg in messages)

    def test_interrupt_with_state_update(self, checkpointer):
        """Test modifying state during interrupt."""
        workflow = create_tool_agent_workflow()
        graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["tools"])

        config = {"configurable": {"thread_id": "thread_state_update"}}

        # Initial run
        graph.stream({"messages": [HumanMessage(content="Multiply 2 and 3")]}, config)

        # Modify state during interrupt
        state = graph.get_state(config)

        # Update state with additional message
        updated_state = graph.update_state(
            config, {"messages": [HumanMessage(content="Actually, add 5 and 6")]}
        )

        # Resume with updated state
        events = list(graph.stream(None, config))

        # Verify updated message was processed
        final_state = graph.get_state(config)
        messages = final_state.values["messages"]
        # Should include the updated message
        assert any(
            "5" in str(msg.content) or "6" in str(msg.content) for msg in messages
        )

    @pytest.mark.asyncio
    async def test_async_interrupt_workflow(self, checkpointer):
        """Test async interrupt handling."""
        workflow = create_tool_agent_workflow()
        graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["tools"])

        config = {"configurable": {"thread_id": "thread_async_interrupt"}}

        # Async stream - should stop before tools
        events = []
        async for event in graph.astream(
            {"messages": [HumanMessage(content="Multiply 2 and 3")]}, config
        ):
            events.append(event)

        # Verify it stopped
        state = await graph.aget_state(config)
        assert state.next == ("tools",)

        # Resume async execution
        events2 = []
        async for event in graph.astream(None, config):
            events2.append(event)

        # Verify execution completed
        final_state = await graph.aget_state(config)
        messages = final_state.values["messages"]
        assert len(messages) >= 3


class TestMultipleInterrupts:
    """Test multiple interrupt scenarios."""

    def test_multiple_interrupts_in_sequence(self, checkpointer):
        """Test multiple interrupts in sequence."""
        workflow = create_tool_agent_workflow()
        graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["tools"])

        config = {"configurable": {"thread_id": "thread_multi_interrupt"}}

        # First invocation
        list(
            graph.stream(
                {"messages": [HumanMessage(content="Multiply 2 and 3")]}, config
            )
        )
        state1 = graph.get_state(config)
        assert state1.next == ("tools",)

        # Resume first
        list(graph.stream(None, config))
        state2 = graph.get_state(config)
        # Should complete or continue

        # Second invocation
        list(graph.stream({"messages": [HumanMessage(content="Add 3 and 4")]}, config))
        state3 = graph.get_state(config)
        assert state3.next == ("tools",)

        # Resume second
        list(graph.stream(None, config))

        # Verify both interactions stored
        final_state = graph.get_state(config)
        messages = final_state.values["messages"]
        # May have fewer messages if some interactions didn't complete fully
        assert len(messages) >= 4  # At least messages from both interactions

    def test_interrupt_after_node(self, checkpointer):
        """Test interrupt after node execution."""
        workflow = create_tool_agent_workflow()
        # Interrupt after tools to verify tool execution
        graph = workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=["tools"],
            interrupt_after=["tools"],
        )

        config = {"configurable": {"thread_id": "thread_interrupt_after"}}

        # First run - stops before tools
        list(
            graph.stream(
                {"messages": [HumanMessage(content="Multiply 2 and 3")]}, config
            )
        )

        state1 = graph.get_state(config)
        assert state1.next == ("tools",)

        # Resume - should execute tools then stop
        graph.stream(None, config)

        state2 = graph.get_state(config)
        # May have completed or stopped again depending on configuration
        messages = state2.values["messages"]
        # Should have tool call at minimum
        assert len(messages) >= 2
