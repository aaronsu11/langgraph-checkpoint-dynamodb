# LangGraph Checkpoint DynamoDB

A single table DynamoDB implementation of the [LangGraph checkpointer interface](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.base.BaseCheckpointSaver) for [persisting graph state](https://langchain-ai.github.io/langgraph/concepts/persistence/) and enabling features like human-in-the-loop, memory, time travel, and fault tolerance. Support sync and async methods with efficient DynamoDB queries and custom table configuration.

## Installation

```bash
pip install langgraph-checkpoint-amazon-dynamodb
```

## Quick Start

> **Note**: For the default configuration to work, you need to have the AWS credentials configured for your environment. See the [AWS Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html) for more details.

Create the DynamoDB table and compile the checkpointer with default settings:
```python
from langgraph.graph import END, START, MessageGraph
from langgraph_checkpoint_dynamodb import DynamoDBSaver

# Set deploy=True to create the table with default settings if it doesn't exist
checkpointer = DynamoDBSaver(deploy=True)

# Use with your LangGraph, below is a simple example
workflow = MessageGraph()
workflow.add_node("chatbot", lambda state: [{"role": "ai", "content": "Hello!"}])
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)
graph = workflow.compile(checkpointer=checkpointer)
```

Use a thread ID to retain state between invocations:
```python
config={"configurable": {"thread_id": "1"}}
# Checkpoints will be saved with the configured thread_id as DynamoDB partition key
print(graph.invoke([{"role": "human", "content": "Hi!"}], config))
# Review last saved checkpoint
print(graph.get_state(config))
```

## Configuration

To customize the DynamoDB table or client configuration, use the `DynamoDBConfig` and `DynamoDBTableConfig` classes:
```python
from langgraph_checkpoint_dynamodb import DynamoDBSaver, DynamoDBConfig, DynamoDBTableConfig

config = DynamoDBConfig(
    table_config=DynamoDBTableConfig(
        # Customize table name as needed
        table_name="langgraph-checkpoint",
    ),
    # Optional AWS credentials (if not using default credentials)
    aws_access_key_id="your-access-key-id",
    aws_secret_access_key="your-secret-access-key",
    aws_session_token="your-session-token"
)

# Create a table with custom configuration
checkpointer = DynamoDBSaver(config, deploy=True)

# Using an existing table with custom configuration
# checkpointer = DynamoDBSaver(config)
```

### DynamoDB Table Configuration Options

The `DynamoDBTableConfig` class provides configuration options for the DynamoDB table:

```python
from langgraph_checkpoint_dynamodb import DynamoDBTableConfig, BillingMode

table_config = DynamoDBTableConfig(
    table_name="langgraph-checkpoint",  # Name of the DynamoDB table
    billing_mode=BillingMode.PAY_PER_REQUEST,  # PAY_PER_REQUEST or PROVISIONED
    enable_encryption=True,  # Enable server-side encryption
    enable_point_in_time_recovery=False,  # Enable point-in-time recovery
    ttl_minutes=30,  # Enable TTL with 30 minutes expiration (set to None to disable)
    ttl_attribute="expireAt",  # TTL attribute name

    # For PROVISIONED billing mode only:
    read_capacity=None,  # Provisioned read capacity units
    write_capacity=None,  # Provisioned write capacity units

    # Optional auto-scaling configuration
    min_read_capacity=None,
    max_read_capacity=None,
    min_write_capacity=None,
    max_write_capacity=None
)
```

### DynamoDB Client Configuration Options

The `DynamoDBConfig` class provides configuration for the DynamoDB client:

```python
from langgraph_checkpoint_dynamodb import DynamoDBConfig

config = DynamoDBConfig(
    table_config=table_config,  # DynamoDBTableConfig instance
    region_name="us-west-2",  # AWS region selection
    endpoint_url=None,  # Custom endpoint URL (e.g., for local DynamoDB)
    max_retries=3,  # Maximum number of retries
    initial_retry_delay=0.1,  # Initial retry delay in seconds
    max_retry_delay=1.0,  # Maximum retry delay in seconds
    
    # Optional AWS credentials (if not using default credentials)
    aws_access_key_id=None,
    aws_secret_access_key=None,
    aws_session_token=None
)
```

## Infrastructure Setup

There are three ways to set up the required DynamoDB infrastructure:

### 1. Using DynamoDBSaver Methods

The simplest way to create and manage the table:

```python
from langgraph_checkpoint_dynamodb import DynamoDBSaver, DynamoDBConfig

config = DynamoDBConfig(...)

# Create table
checkpointer = DynamoDBSaver(config, deploy=True)

# Delete table when no longer needed
checkpointer.destroy()
```

### 2. Using CloudFormation

For simple deployments using CloudFormation:

```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'DynamoDB table for LangGraph checkpoint storage'

Parameters:
  TableName:
    Type: String
    Default: langgraph-checkpoint
  Environment:
    Type: String
    Default: dev
    AllowedValues: [dev, staging, prod]

Resources:
  CheckpointTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: !Ref TableName
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: PK
          AttributeType: S
        - AttributeName: SK
          AttributeType: S
      KeySchema:
        - AttributeName: PK
          KeyType: HASH
        - AttributeName: SK
          KeyType: RANGE
      PointInTimeRecoverySpecification:
        PointInTimeRecoveryEnabled: true
      SSESpecification:
        SSEEnabled: true
```

Deploy using AWS CLI:
```bash
aws cloudformation deploy \
  --template-file template.yaml \
  --stack-name langgraph-checkpoint \
  --parameter-overrides TableName=langgraph-checkpoint Environment=prod
```

### 3. Using AWS CDK

For more advanced infrastructure management using AWS CDK Python, install the `infra` optional dependency:

```bash
pip install "langgraph-checkpoint-amazon-dynamodb[infra]"
```

Then import the `DynamoDBCheckpointStack` to your CDK app:
```python
from aws_cdk import App
from langgraph_checkpoint_dynamodb.infra import DynamoDBCheckpointStack
from langgraph_checkpoint_dynamodb import DynamoDBTableConfig

app = App()

# Create the stack with custom configuration
table_config = DynamoDBTableConfig(
    table_name="langgraph-checkpoint",
    enable_point_in_time_recovery=True,
    enable_encryption=True
)

DynamoDBCheckpointStack(
    app,
    "LangGraphCheckpoint",
    table_config=table_config,
    tags={"Environment": "prod", "Tenant": "tenant-1"},
)

app.synth()
```

## Example usage with LangGraph in production

In production, it's recommended to deploy the table independently and reference it in the `DynamoDBConfig` class (deploy=False):

```python
from langgraph.graph import StateGraph
from langgraph_checkpoint_dynamodb import DynamoDBSaver, DynamoDBConfig

# Use an existing table as checkpointer
config = DynamoDBConfig(...)
checkpointer = DynamoDBSaver(config)

# Create your graph
workflow = StateGraph(...)
# ... configure your graph ...

# Compile with checkpointer
graph = workflow.compile(checkpointer=checkpointer)

# Use thread_id for persistence
config = {"configurable": {"thread_id": "<unique-thread-id>"}}

# Run the graph and get the final output state
graph.invoke({"messages": [{"type": "user", "content": "Hello!"}]}, config)

# Run with async methods and stream updates after every node
async for chunk in graph.astream(
    {"messages": [{"type": "user", "content": "Hello!"}]},
    config
):
    chunk["messages"][-1].pretty_print()
```

## Features

The DynamoDB checkpointer enables all LangGraph persistence features:

- **Human-in-the-loop**: Inspect, interrupt, and approve graph steps
- **Memory**: Retain state between interactions in the same thread
- **Time Travel**: Replay and debug specific graph steps
- **Fault Tolerance**: Recover from failures and resume from last successful step
- **Sync and Async**: Support sync and async methods using efficient DynamoDB KeyConditionExpressions for cost-effective queries

## License

MIT