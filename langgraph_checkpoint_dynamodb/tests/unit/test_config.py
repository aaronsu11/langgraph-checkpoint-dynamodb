import pytest

from langgraph_checkpoint_dynamodb import DynamoDBTableConfig
from langgraph_checkpoint_dynamodb.config import BillingMode


class TestTableConfigValidation:
    """Tests for DynamoDBTableConfig validation."""

    def test_table_config_validation_provisioned_without_capacity(self):
        """
        Test validate() with provisioned mode without capacity.

        Covers:
        - DynamoDBTableConfig.validate() provisioned mode error path (lines 36-40 in config.py)
        """
        # Provisioned mode without read/write capacity
        table_config = DynamoDBTableConfig(
            table_name="test_table",
            billing_mode=BillingMode.PROVISIONED,
            # Missing read_capacity and write_capacity
        )

        with pytest.raises(ValueError) as exc_info:
            table_config.validate()

        assert "read_capacity" in str(exc_info.value).lower()
        assert "write_capacity" in str(exc_info.value).lower()
        assert "provisioned" in str(exc_info.value).lower()

    def test_table_config_validation_provisioned_with_capacity(self):
        """
        Test validate() with provisioned mode and capacity (should pass).

        Covers:
        - DynamoDBTableConfig.validate() provisioned mode success path
        """
        # Provisioned mode with capacity should validate successfully
        table_config = DynamoDBTableConfig(
            table_name="test_table",
            billing_mode=BillingMode.PROVISIONED,
            read_capacity=5,
            write_capacity=5,
        )

        # Should not raise an exception
        table_config.validate()

    def test_table_config_validation_negative_ttl_days(self):
        """
        Test validate() with negative TTL days.

        Covers:
        - DynamoDBTableConfig.validate() negative TTL error path (lines 42-43 in config.py)
        """
        # Negative TTL days
        table_config = DynamoDBTableConfig(
            table_name="test_table",
            billing_mode=BillingMode.PAY_PER_REQUEST,
            ttl_days=-1,  # Invalid: negative
        )

        with pytest.raises(ValueError) as exc_info:
            table_config.validate()

        assert "ttl_days must be positive" in str(exc_info.value).lower()

    def test_table_config_validation_zero_ttl_days(self):
        """
        Test validate() with zero TTL days.

        Covers:
        - DynamoDBTableConfig.validate() zero TTL error path
        """
        # Zero TTL days
        table_config = DynamoDBTableConfig(
            table_name="test_table",
            billing_mode=BillingMode.PAY_PER_REQUEST,
            ttl_days=0,  # Invalid: zero
        )

        with pytest.raises(ValueError) as exc_info:
            table_config.validate()

        assert "ttl_days must be positive" in str(exc_info.value).lower()

    def test_table_config_validation_valid_ttl_days(self):
        """
        Test validate() with valid TTL days (should pass).

        Covers:
        - DynamoDBTableConfig.validate() valid TTL success path
        """
        # Valid TTL days
        table_config = DynamoDBTableConfig(
            table_name="test_table",
            billing_mode=BillingMode.PAY_PER_REQUEST,
            ttl_days=7,  # Valid: positive
        )

        # Should not raise an exception
        table_config.validate()

    def test_table_config_validation_none_ttl_days(self):
        """
        Test validate() with None TTL days (should pass).

        Covers:
        - DynamoDBTableConfig.validate() None TTL success path
        """
        # None TTL days (disabled)
        table_config = DynamoDBTableConfig(
            table_name="test_table",
            billing_mode=BillingMode.PAY_PER_REQUEST,
            ttl_days=None,  # Valid: TTL disabled
        )

        # Should not raise an exception
        table_config.validate()

