# Testing Guide

This document provides comprehensive information about testing in LangGraph Checkpoint DynamoDB.

## Overview

All tests use local/mock AWS services and **never create cloud resources**. This ensures:
- Fast, reliable test execution
- No AWS costs or credentials required
- Tests can run in CI/CD environments
- No risk of affecting production resources

## Test Types

The test suite consists of two types of tests:

1. **Unit Tests** - Fast, in-memory tests using `moto` for mocking AWS DynamoDB
2. **Integration Tests** - Real DynamoDB API tests using LocalStack in Docker

## Unit Tests (Mock AWS with moto)

Unit tests use `moto` to mock AWS DynamoDB entirely in-memory. They're fast and don't require Docker.

### Running Unit Tests

```bash
# Install dev dependencies first
pip install -e ".[dev]"

# Run only unit tests (fast, no Docker required)
pytest -m "not integration"

# Or explicitly run unit tests
pytest langgraph_checkpoint_dynamodb/tests/unit/
```

### What Unit Tests Verify

- Checkpoint CRUD operations
- Write operations
- List and filtering
- Configuration validation
- Error handling

## Integration Tests (LocalStack Docker Container)

Integration tests use testcontainers to spin up a LocalStack Docker container, providing a real DynamoDB-compatible API locally.

### Prerequisites

- **Docker must be running** - Integration tests require Docker to spin up LocalStack containers
- Docker will automatically pull the LocalStack image if needed

### Running Integration Tests

```bash
# Install dev dependencies first
pip install -e ".[dev]"

# Run only integration tests
pytest -m integration

# Run all tests (unit + integration)
pytest

# Skip integration tests (faster for development)
pytest -m "not integration"
```

### What Integration Tests Verify

- Sync and async checkpoint operations against real DynamoDB API
- Table deployment and configuration (TTL, billing modes, encryption)
- Memory persistence across invocations
- Interrupts and human-in-the-loop workflows
- Time travel and state history
- Concurrent operations and thread safety

### Test Isolation

All tests are isolated:
- Each test gets its own DynamoDB table
- Tables are automatically created before tests
- Tables are automatically destroyed after tests
- No cleanup needed - everything is handled automatically

### Verifying Tests Use LocalStack

When you run `pytest -m integration`, you can verify tests are running against LocalStack (not real AWS):

1. **Check pytest output** - Look for log messages:
   - "Starting LocalStack container..."
   - "LocalStack container started successfully at http://..."

2. **Run verification test first**:
   ```bash
   pytest -m integration test_localstack_verification.py -v
   ```
   This confirms the endpoint is localhost, not AWS.

3. **Check Docker containers**:
   ```bash
   docker ps
   ```
   You should see a LocalStack container running during tests.

4. **Automatic verification** - Each test automatically verifies it's using a localhost endpoint, not amazonaws.com.

### Safety Guarantees

Tests include multiple safeguards:
- **Automatic endpoint verification** - Tests fail immediately if the endpoint looks like real AWS
- **Container management** - LocalStack containers are automatically started and stopped
- **Table isolation** - Each test uses unique table names
- **No cloud resources** - Tests will fail if they detect any attempt to use real AWS

## Running Specific Tests

```bash
# Run a specific test file
pytest langgraph_checkpoint_dynamodb/tests/unit/test_saver.py

# Run a specific test function
pytest langgraph_checkpoint_dynamodb/tests/integration/test_basic_operations.py::test_put_get_checkpoint

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=langgraph_checkpoint_dynamodb --cov-report=html
```

## Continuous Integration

Tests are designed to run in CI/CD environments:
- No AWS credentials required
- No cloud resources created
- Docker support required for integration tests (most CI platforms provide this)
- Fast unit tests for quick feedback
- Comprehensive integration tests for thorough validation

## Troubleshooting

### Integration Tests Fail to Start

If integration tests fail to start:
1. **Check Docker is running**: `docker ps` should work without errors
2. **Check Docker permissions**: Ensure your user can run Docker commands
3. **Check port availability**: LocalStack needs ports 4566-4571 (default)
4. **Check logs**: Look at pytest output for LocalStack container startup messages

### Tests Run Slowly

- Unit tests should be fast (< 1 second per test typically)
- Integration tests are slower but should complete in reasonable time
- If tests are very slow, check Docker resource allocation

### LocalStack Container Issues

If LocalStack containers aren't starting:
1. **Pull the image manually**: `docker pull localstack/localstack:latest`
2. **Check disk space**: Ensure Docker has enough disk space
3. **Check Docker logs**: `docker logs <container-id>` for errors

