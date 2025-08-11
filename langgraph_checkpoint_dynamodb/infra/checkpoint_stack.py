from aws_cdk import (
    Stack,
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
        **kwargs
    ) -> None:
        super().__init__(scope, id, **kwargs)
        
        # Use default props if none provided
        self.props = table_config or DynamoDBTableConfig()
        
        # Create the table
        self.table = self._create_table()
        
        # Configure auto-scaling if needed
        if dynamodb.BillingMode(self.props.billing_mode) == dynamodb.BillingMode.PROVISIONED:
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
                name="PK",
                type=dynamodb.AttributeType.STRING
            ),
            "sort_key": dynamodb.Attribute(
                name="SK",
                type=dynamodb.AttributeType.STRING
            ),
            "removal_policy": RemovalPolicy.RETAIN,
            "point_in_time_recovery": self.props.enable_point_in_time_recovery,
        }
        
        # Add encryption if enabled
        if self.props.enable_encryption:
            table_config["encryption"] = dynamodb.TableEncryption.AWS_MANAGED
            
        # Add capacity if provisioned
        if dynamodb.BillingMode(self.props.billing_mode) == dynamodb.BillingMode.PROVISIONED:
            if not self.props.read_capacity or not self.props.write_capacity:
                raise ValueError(
                    "read_capacity and write_capacity required for PROVISIONED mode"
                )
            table_config["read_capacity"] = self.props.read_capacity
            table_config["write_capacity"] = self.props.write_capacity
            
        # Create table
        table = dynamodb.Table(self, "CheckpointTable", **table_config)
        
        # Enable TTL if configured
        if self.props.ttl_minutes is not None:
            if not self.props.ttl_attribute:
                raise ValueError("ttl_attribute is required when ttl_minutes is set")
            table.add_time_to_live_attribute(self.props.ttl_attribute)
            
        return table
    
    def _configure_auto_scaling(self) -> None:
        """Configure auto-scaling for provisioned capacity mode."""
        if not all([
            self.props.min_read_capacity,
            self.props.max_read_capacity,
            self.props.min_write_capacity,
            self.props.max_write_capacity
        ]):
            return
            
        # Read auto-scaling
        read_scaling = self.table.auto_scale_read_capacity(
            min_capacity=self.props.min_read_capacity,
            max_capacity=self.props.max_read_capacity
        )
        
        read_scaling.scale_on_utilization(
            target_utilization_percent=70,
            scale_in_cooldown=300,  # 5 minutes
            scale_out_cooldown=60   # 1 minute
        )
        
        # Write auto-scaling
        write_scaling = self.table.auto_scale_write_capacity(
            min_capacity=self.props.min_write_capacity,
            max_capacity=self.props.max_write_capacity
        )
        
        write_scaling.scale_on_utilization(
            target_utilization_percent=70,
            scale_in_cooldown=300,  # 5 minutes
            scale_out_cooldown=60   # 1 minute
        )
    
    def _add_outputs(self) -> None:
        """Add CloudFormation outputs."""
        CfnOutput(
            self,
            "TableName",
            value=self.table.table_name,
            description="Name of the DynamoDB table"
        )
        
        CfnOutput(
            self,
            "TableArn",
            value=self.table.table_arn,
            description="ARN of the DynamoDB table"
        )
    
    def _add_tags(self) -> None:
        """Add resource tags."""
        Tags.of(self.table).add("Component", "Checkpoint")