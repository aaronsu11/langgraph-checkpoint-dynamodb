from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BillingMode(str, Enum):
    PROVISIONED = "PROVISIONED"
    PAY_PER_REQUEST = "PAY_PER_REQUEST"


@dataclass
class DynamoDBTableConfig:
    """Configuration for DynamoDB table creation/usage."""

    table_name: str = "langgraph-checkpoint"
    billing_mode: BillingMode = BillingMode.PAY_PER_REQUEST
    enable_encryption: bool = True
    enable_point_in_time_recovery: bool = False

    # CAUTION: once TTL is enabled, attribute will be added to all items without it
    ttl_attribute: str = "expireAt"
    ttl_minutes: Optional[int] = None  # set a positive number to enable TTL

    # For provisioned capacity mode
    read_capacity: Optional[int] = None
    write_capacity: Optional[int] = None

    # Optional auto-scaling configuration
    min_read_capacity: Optional[int] = None
    max_read_capacity: Optional[int] = None
    min_write_capacity: Optional[int] = None
    max_write_capacity: Optional[int] = None

    def validate(self) -> None:
        """Validate table configuration."""
        if self.billing_mode == BillingMode.PROVISIONED:
            if not self.read_capacity or not self.write_capacity:
                raise ValueError(
                    "read_capacity and write_capacity required for PROVISIONED mode"
                )

        if self.ttl_minutes is not None and self.ttl_minutes <= 0:
            raise ValueError("ttl_minutes must be positive when specified")


@dataclass
class DynamoDBConfig:
    """Configuration for DynamoDB checkpoint saver."""

    table_config: DynamoDBTableConfig = field(default_factory=DynamoDBTableConfig)
    region_name: Optional[str] = None
    endpoint_url: Optional[str] = None
    max_retries: int = 3
    initial_retry_delay: float = 0.1
    max_retry_delay: float = 1.0
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None

    def get_client_config(self) -> dict:
        """Get boto3/aioboto3 client configuration."""
        config = {
            "region_name": self.region_name,
            "endpoint_url": self.endpoint_url,
        }
        if self.aws_access_key_id:
            config["aws_access_key_id"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            config["aws_secret_access_key"] = self.aws_secret_access_key
        if self.aws_session_token:
            config["aws_session_token"] = self.aws_session_token
        return {k: v for k, v in config.items() if v is not None}
