"""Shared fixtures for integration tests with LocalStack."""

import logging
import uuid

import pytest
from testcontainers.localstack import LocalStackContainer

from langgraph_checkpoint_dynamodb import (
    DynamoDBConfig,
    DynamoDBTableConfig,
    DynamoDBSaver,
)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def localstack():
    """Start LocalStack container for the test session."""
    logger.info("Starting LocalStack container for integration tests...")
    try:
        with LocalStackContainer(image="localstack/localstack:latest") as localstack:
            endpoint_url = localstack.get_url()
            logger.info(f"LocalStack container started successfully at {endpoint_url}")
            # Verify we can connect to LocalStack by checking the endpoint
            assert endpoint_url is not None, "LocalStack endpoint URL is None"
            assert (
                "localhost" in endpoint_url or "127.0.0.1" in endpoint_url
            ), f"LocalStack endpoint {endpoint_url} doesn't look like a local endpoint"
            yield localstack
            logger.info("LocalStack container stopped")
    except Exception as e:
        logger.error(f"Failed to start LocalStack container: {e}")
        if "docker" in str(e).lower() or "container" in str(e).lower():
            pytest.fail(
                "Failed to start LocalStack. Is Docker running? "
                "Integration tests require Docker to be installed and running. "
                f"Original error: {e}"
            )
        raise


@pytest.fixture(scope="function")
def dynamodb_config(localstack):
    """Create DynamoDB config pointing to LocalStack."""
    endpoint_url = localstack.get_url()
    # Verify endpoint is set correctly
    assert endpoint_url is not None, "LocalStack endpoint URL is None"
    return DynamoDBConfig(
        endpoint_url=endpoint_url,
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )


@pytest.fixture(scope="function")
def checkpointer(dynamodb_config):
    """Create DynamoDBSaver with real table in LocalStack."""
    config = dynamodb_config
    config.table_config = DynamoDBTableConfig(
        table_name=f"test-checkpoints-{uuid.uuid4().hex[:8]}",
    )
    saver = DynamoDBSaver(config=config, deploy=True)
    yield saver
    # Cleanup
    try:
        saver.destroy()
    except Exception:
        pass


@pytest.fixture(scope="function")
def sample_config():
    """Sample runnable config for testing."""
    from langchain_core.runnables import RunnableConfig

    return RunnableConfig(
        configurable={
            "thread_id": f"test_thread_{uuid.uuid4().hex[:8]}",
            "checkpoint_ns": "",
        }
    )
