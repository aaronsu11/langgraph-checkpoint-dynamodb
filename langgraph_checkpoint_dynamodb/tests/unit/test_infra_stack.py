import aws_cdk as core
from aws_cdk import assertions

from langgraph_checkpoint_dynamodb.infra.checkpoint_stack import DynamoDBCheckpointStack
from langgraph_checkpoint_dynamodb import DynamoDBTableConfig
from langgraph_checkpoint_dynamodb.config import BillingMode


def test_dynamodb_table_created_with_default_config():
    """Test DynamoDB table creation with default configuration."""
    app = core.App()
    stack = DynamoDBCheckpointStack(app, "DynamoDBCheckpointStack")
    template = assertions.Template.from_stack(stack)

    # Verify table is created
    template.has_resource_properties(
        "AWS::DynamoDB::Table",
        {
            "TableName": "langgraph-checkpoint",
            "BillingMode": "PAY_PER_REQUEST",
            "KeySchema": [
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            "AttributeDefinitions": [
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
        },
    )


def test_dynamodb_table_created_with_custom_config():
    """Test DynamoDB table creation with custom configuration."""
    table_config = DynamoDBTableConfig(
        table_name="custom-checkpoint-table",
        billing_mode=BillingMode.PAY_PER_REQUEST,
        enable_encryption=True,
        enable_point_in_time_recovery=True,
        ttl_days=7,
    )

    app = core.App()
    stack = DynamoDBCheckpointStack(
        app, "CustomCheckpointStack", table_config=table_config
    )
    template = assertions.Template.from_stack(stack)

    # Verify table is created with custom config
    template.has_resource_properties(
        "AWS::DynamoDB::Table",
        {
            "TableName": "custom-checkpoint-table",
            "BillingMode": "PAY_PER_REQUEST",
        },
    )

    # Verify point-in-time recovery (check separately as it might be in a different format)
    template.has_resource_properties(
        "AWS::DynamoDB::Table",
        {
            "PointInTimeRecoverySpecification": {"PointInTimeRecoveryEnabled": True},
        },
    )

    # Verify TTL is configured
    template.has_resource_properties(
        "AWS::DynamoDB::Table",
        {
            "TimeToLiveSpecification": {"Enabled": True, "AttributeName": "expireAt"},
        },
    )


def test_dynamodb_table_provisioned_mode():
    """Test DynamoDB table creation with provisioned billing mode."""
    table_config = DynamoDBTableConfig(
        table_name="provisioned-table",
        billing_mode=BillingMode.PROVISIONED,
        read_capacity=5,
        write_capacity=5,
        min_read_capacity=2,
        max_read_capacity=10,
        min_write_capacity=2,
        max_write_capacity=10,
    )

    app = core.App()
    stack = DynamoDBCheckpointStack(app, "ProvisionedStack", table_config=table_config)
    template = assertions.Template.from_stack(stack)

    # Verify table is created with provisioned capacity
    # Note: In PROVISIONED mode, CDK doesn't explicitly set BillingMode in the template
    template.has_resource_properties(
        "AWS::DynamoDB::Table",
        {
            "TableName": "provisioned-table",
            "ProvisionedThroughput": {
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5,
            },
        },
    )


def test_dynamodb_table_outputs():
    """Test that CloudFormation outputs are created."""
    app = core.App()
    stack = DynamoDBCheckpointStack(app, "OutputsStack")
    template = assertions.Template.from_stack(stack)

    # Verify outputs exist
    template.has_output("TableName", {})
    template.has_output("TableArn", {})


def test_dynamodb_table_with_encryption():
    """Test DynamoDB table with encryption enabled."""
    table_config = DynamoDBTableConfig(
        table_name="encrypted-table",
        billing_mode=BillingMode.PAY_PER_REQUEST,
        enable_encryption=True,
    )

    app = core.App()
    stack = DynamoDBCheckpointStack(app, "EncryptedStack", table_config=table_config)
    template = assertions.Template.from_stack(stack)

    # Verify encryption is enabled (SSE specification should be present)
    template.has_resource_properties(
        "AWS::DynamoDB::Table",
        {
            "SSESpecification": {"SSEEnabled": True},
        },
    )
