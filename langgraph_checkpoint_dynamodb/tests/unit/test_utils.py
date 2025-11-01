import pytest
import time
from boto3.dynamodb.types import Binary

from langgraph_checkpoint_dynamodb.errors import DynamoDBValidationError
from langgraph_checkpoint_dynamodb.utils import (
    create_ttl_filter,
    deserialize_dynamodb_binary,
    validate_checkpoint_item,
    validate_write_item,
)


class TestTTLFilter:
    """Tests for TTL filter creation."""

    def test_ttl_filter_creation(self):
        """
        Test create_ttl_filter() function.

        Covers:
        - create_ttl_filter() function (lines 13-27 in utils.py)
        """
        ttl_attribute = "expireAt"
        filter_expr, expr_values = create_ttl_filter(ttl_attribute)

        # Verify filter expression format
        assert "attribute_not_exists" in filter_expr
        assert ttl_attribute in filter_expr
        assert ":current_time" in filter_expr

        # Verify expression attribute values
        assert ":current_time" in expr_values
        current_time = int(time.time())
        # Allow some time variance (within 5 seconds)
        assert abs(expr_values[":current_time"] - current_time) <= 5


class TestValidationErrors:
    """Tests for validation error handling."""

    def test_validate_checkpoint_item_missing_fields(self):
        """
        Test validate_checkpoint_item() with missing required fields.

        Covers:
        - validate_checkpoint_item() error path (line 211 in utils.py)
        """
        # Missing required fields
        invalid_item = {
            "PK": "test_thread",
            "SK": "test#checkpoint#test_id",
            # Missing: type, checkpoint_id, checkpoint, metadata
        }

        with pytest.raises(DynamoDBValidationError) as exc_info:
            validate_checkpoint_item(invalid_item)

        assert "missing required fields" in str(exc_info.value).lower()

    def test_validate_checkpoint_item_invalid_sk(self):
        """
        Test validate_checkpoint_item() with invalid SK format.

        Covers:
        - validate_checkpoint_item() error path (line 216 in utils.py)
        """
        # Invalid SK format (missing #checkpoint#)
        invalid_item = {
            "PK": "test_thread",
            "SK": "wrong_format",  # Should contain "#checkpoint#"
            "type": "test",
            "checkpoint_id": "test_id",
            "checkpoint": b"test",
            "metadata": b"test",
        }

        with pytest.raises(DynamoDBValidationError) as exc_info:
            validate_checkpoint_item(invalid_item)

        assert "invalid checkpoint sk format" in str(exc_info.value).lower()

    def test_validate_write_item_missing_fields(self):
        """
        Test validate_write_item() with missing required fields.

        Covers:
        - validate_write_item() error path (line 237 in utils.py)
        """
        # Missing required fields
        invalid_item = {
            "PK": "test_thread",
            "SK": "test#write#test_id#task#0000000000",
            # Missing: type, task_id, channel, value, idx
        }

        with pytest.raises(DynamoDBValidationError) as exc_info:
            validate_write_item(invalid_item)

        assert "missing required fields" in str(exc_info.value).lower()

    def test_validate_write_item_invalid_sk(self):
        """
        Test validate_write_item() with invalid SK format.

        Covers:
        - validate_write_item() error path (line 242 in utils.py)
        """
        # Invalid SK format (missing #write#)
        invalid_item = {
            "PK": "test_thread",
            "SK": "wrong_format",  # Should contain "#write#"
            "type": "test",
            "task_id": "task1",
            "channel": "channel1",
            "value": b"test",
            "idx": 0,
        }

        with pytest.raises(DynamoDBValidationError) as exc_info:
            validate_write_item(invalid_item)

        assert "invalid write sk format" in str(exc_info.value).lower()

    def test_deserialize_dynamodb_binary(self):
        """
        Test deserialize_dynamodb_binary() with Binary type.

        Covers:
        - deserialize_dynamodb_binary() Binary type path (line 72 in utils.py)
        """
        # Test with Binary type
        binary_data = Binary(b"test_data")
        result = deserialize_dynamodb_binary(binary_data)
        assert isinstance(result, bytes)
        assert result == b"test_data"

        # Test with bytes (should return as-is)
        bytes_data = b"test_data"
        result2 = deserialize_dynamodb_binary(bytes_data)
        assert isinstance(result2, bytes)
        assert result2 == bytes_data

