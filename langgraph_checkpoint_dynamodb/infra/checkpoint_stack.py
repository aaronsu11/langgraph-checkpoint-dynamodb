from aws_cdk import (
    Stack,
    Duration,
    aws_dynamodb as dynamodb,
    RemovalPolicy,
    CfnOutput,
    Tags,
)
from constructs import Construct
from typing import Optional

from ..config import DynamoDBTableConfig


class DynamoDBCheckpointStack(Stack):
    """CDK Stack for LangGraph Checkpoint DynamoDB infrastructure."""

    def __init__(
        self,
        scope: Construct,
        id: str,
        table_config: Optional[DynamoDBTableConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, id, **kwargs)

        # Use default props if none provided
        self.props = table_config or DynamoDBTableConfig()

        # Create the table
        self.table = self._create_table()

        # Configure auto-scaling if needed
        if (
            dynamodb.BillingMode(self.props.billing_mode)
            == dynamodb.BillingMode.PROVISIONED
        ):
            self._configure_auto_scaling()

        # Add outputs
        self._add_outputs()

        # Add tags
        self._add_tags()

    def _create_table(self) -> dynamodb.Table:
        """Create the DynamoDB table with configured properties."""
        table_config = {
            "table_name": self.props.table_name,
            "billing_mode": dynamodb.BillingMode(self.props.billing_mode),
            "partition_key": dynamodb.Attribute(
                name="PK", type=dynamodb.AttributeType.STRING
            ),
            "sort_key": dynamodb.Attribute(
                name="SK", type=dynamodb.AttributeType.STRING
            ),
            "removal_policy": RemovalPolicy.RETAIN,
        }

        # Add point-in-time recovery if enabled
        if self.props.enable_point_in_time_recovery:
            table_config["point_in_time_recovery_specification"] = (
                dynamodb.PointInTimeRecoverySpecification(
                    point_in_time_recovery_enabled=True
                )
            )

        # Add encryption if enabled
        if self.props.enable_encryption:
            table_config["encryption"] = dynamodb.TableEncryption.AWS_MANAGED

        # Add capacity if provisioned
        if (
            dynamodb.BillingMode(self.props.billing_mode)
            == dynamodb.BillingMode.PROVISIONED
        ):
            if not self.props.read_capacity or not self.props.write_capacity:
                raise ValueError(
                    "read_capacity and write_capacity required for PROVISIONED mode"
                )
            table_config["read_capacity"] = self.props.read_capacity
            table_config["write_capacity"] = self.props.write_capacity

        # Create table
        table = dynamodb.Table(self, "CheckpointTable", **table_config)

        # Enable TTL if configured via CloudFormation property
        if self.props.ttl_days is not None:
            if not self.props.ttl_attribute:
                raise ValueError("ttl_attribute is required when ttl_days is set")
            # Configure TTL via the underlying CloudFormation resource
            cfn_table = table.node.default_child
            if cfn_table:
                cfn_table.add_property_override(
                    "TimeToLiveSpecification",
                    {
                        "Enabled": True,
                        "AttributeName": self.props.ttl_attribute,
                    },
                )

        return table

    def _configure_auto_scaling(self) -> None:
        """Configure auto-scaling for provisioned capacity mode."""
        if not all(
            [
                self.props.min_read_capacity,
                self.props.max_read_capacity,
                self.props.min_write_capacity,
                self.props.max_write_capacity,
            ]
        ):
            return

        # Read auto-scaling
        read_scaling = self.table.auto_scale_read_capacity(
            min_capacity=self.props.min_read_capacity,
            max_capacity=self.props.max_read_capacity,
        )

        read_scaling.scale_on_utilization(
            target_utilization_percent=70,
            scale_in_cooldown=Duration.minutes(5),
            scale_out_cooldown=Duration.minutes(1),
        )

        # Write auto-scaling
        write_scaling = self.table.auto_scale_write_capacity(
            min_capacity=self.props.min_write_capacity,
            max_capacity=self.props.max_write_capacity,
        )

        write_scaling.scale_on_utilization(
            target_utilization_percent=70,
            scale_in_cooldown=Duration.minutes(5),
            scale_out_cooldown=Duration.minutes(1),
        )

    def _add_outputs(self) -> None:
        """Add CloudFormation outputs."""
        CfnOutput(
            self,
            "TableName",
            value=self.table.table_name,
            description="Name of the DynamoDB table",
        )

        CfnOutput(
            self,
            "TableArn",
            value=self.table.table_arn,
            description="ARN of the DynamoDB table",
        )

    def _add_tags(self) -> None:
        """Add resource tags."""
        Tags.of(self.table).add("Component", "Checkpoint")
