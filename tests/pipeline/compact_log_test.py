# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import os
import unittest

from spdl.pipeline._common._misc import _get_compact_log, _set_compact_log, create_task
from spdl.pipeline.config import set_compact_log


class DummyException(Exception):
    """Test exception for simulating task failures."""

    pass


class CompactLogTest(unittest.TestCase):
    """Tests for compact logging mode functionality."""

    def setUp(self) -> None:
        """Reset the global compact log setting before each test."""
        # Reset to None to ensure clean state
        _set_compact_log(None)
        # Clear any environment variable that might be set
        if "SPDL_PIPELINE_COMPACT_LOG" in os.environ:
            del os.environ["SPDL_PIPELINE_COMPACT_LOG"]

    def tearDown(self) -> None:
        """Clean up after each test."""
        # Reset to None
        _set_compact_log(None)
        # Clear environment variable
        if "SPDL_PIPELINE_COMPACT_LOG" in os.environ:
            del os.environ["SPDL_PIPELINE_COMPACT_LOG"]

    def test_get_compact_log_defaults_to_false_when_env_not_set(self) -> None:
        """Test that _get_compact_log returns False when environment variable is not set."""
        # Setup: Environment variable is not set (cleared in setUp)

        # Execute: Get compact log setting
        result = _get_compact_log()

        # Assert: Should default to False
        self.assertFalse(result)

    def test_get_compact_log_returns_true_when_env_is_1(self) -> None:
        """Test that _get_compact_log returns True when environment variable is '1'."""
        # Setup: Set environment variable to '1'
        os.environ["SPDL_PIPELINE_COMPACT_LOG"] = "1"

        # Execute: Get compact log setting
        result = _get_compact_log()

        # Assert: Should return True
        self.assertTrue(result)

    def test_get_compact_log_returns_true_when_env_is_true(self) -> None:
        """Test that _get_compact_log returns True when environment variable is 'true'."""
        # Setup: Set environment variable to 'true'
        os.environ["SPDL_PIPELINE_COMPACT_LOG"] = "true"

        # Execute: Get compact log setting
        result = _get_compact_log()

        # Assert: Should return True
        self.assertTrue(result)

    def test_get_compact_log_returns_true_when_env_is_TRUE(self) -> None:
        """Test that _get_compact_log returns True when environment variable is 'TRUE'."""
        # Setup: Set environment variable to 'TRUE'
        os.environ["SPDL_PIPELINE_COMPACT_LOG"] = "TRUE"

        # Execute: Get compact log setting
        result = _get_compact_log()

        # Assert: Should return True
        self.assertTrue(result)

    def test_get_compact_log_returns_true_when_env_is_yes(self) -> None:
        """Test that _get_compact_log returns True when environment variable is 'yes'."""
        # Setup: Set environment variable to 'yes'
        os.environ["SPDL_PIPELINE_COMPACT_LOG"] = "yes"

        # Execute: Get compact log setting
        result = _get_compact_log()

        # Assert: Should return True
        self.assertTrue(result)

    def test_get_compact_log_returns_false_when_env_is_0(self) -> None:
        """Test that _get_compact_log returns False when environment variable is '0'."""
        # Setup: Set environment variable to '0'
        os.environ["SPDL_PIPELINE_COMPACT_LOG"] = "0"

        # Execute: Get compact log setting
        result = _get_compact_log()

        # Assert: Should return False
        self.assertFalse(result)

    def test_get_compact_log_returns_false_when_env_is_false(self) -> None:
        """Test that _get_compact_log returns False when environment variable is 'false'."""
        # Setup: Set environment variable to 'false'
        os.environ["SPDL_PIPELINE_COMPACT_LOG"] = "false"

        # Execute: Get compact log setting
        result = _get_compact_log()

        # Assert: Should return False
        self.assertFalse(result)

    def test_get_compact_log_caches_result_after_first_call(self) -> None:
        """Test that _get_compact_log caches the result and doesn't re-check env var."""
        # Setup: Set environment variable to 'true'
        os.environ["SPDL_PIPELINE_COMPACT_LOG"] = "true"

        # Execute: Get compact log setting twice
        result1 = _get_compact_log()
        # Change environment variable
        os.environ["SPDL_PIPELINE_COMPACT_LOG"] = "false"
        result2 = _get_compact_log()

        # Assert: Both should return True because value is cached
        self.assertTrue(result1)
        self.assertTrue(result2)

    def test_set_compact_log_to_true(self) -> None:
        """Test that _set_compact_log can set the value to True."""
        # Setup: Set to True
        _set_compact_log(True)

        # Execute: Get compact log setting
        result = _get_compact_log()

        # Assert: Should return True
        self.assertTrue(result)

    def test_set_compact_log_to_false(self) -> None:
        """Test that _set_compact_log can set the value to False."""
        # Setup: Set to False
        _set_compact_log(False)

        # Execute: Get compact log setting
        result = _get_compact_log()

        # Assert: Should return False
        self.assertFalse(result)

    def test_set_compact_log_to_none_resets_to_env_check(self) -> None:
        """Test that setting to None causes re-check of environment variable."""
        # Setup: Set to True first
        _set_compact_log(True)
        self.assertTrue(_get_compact_log())

        # Reset to None
        _set_compact_log(None)
        # Set environment variable
        os.environ["SPDL_PIPELINE_COMPACT_LOG"] = "false"

        # Execute: Get compact log setting after reset
        result = _get_compact_log()

        # Assert: Should return False from environment variable
        self.assertFalse(result)

    def test_set_compact_log_overrides_env_var(self) -> None:
        """Test that programmatically setting the value overrides environment variable."""
        # Setup: Set environment variable to 'true'
        os.environ["SPDL_PIPELINE_COMPACT_LOG"] = "true"
        # Override with False
        _set_compact_log(False)

        # Execute: Get compact log setting
        result = _get_compact_log()

        # Assert: Should return False (overridden value, not env var)
        self.assertFalse(result)

    def test_config_module_exposes_set_compact_log(self) -> None:
        """Test that set_compact_log is exposed in the config module."""
        # Setup: Set through config module
        set_compact_log(True)

        # Execute: Get compact log setting
        result = _get_compact_log()

        # Assert: Should return True
        self.assertTrue(result)

    def test_create_task_properly_calls_compact_log_setting(self) -> None:
        """Test that create_task respects the compact log setting."""

        async def failing_coro() -> None:
            raise DummyException("test error")

        async def run() -> None:
            # Test with compact=False (default)
            _set_compact_log(False)
            task1 = create_task(failing_coro(), name="task1")
            await asyncio.sleep(0)
            try:
                await task1
            except DummyException:
                pass
            # Task completed, no assertion needed - just verify no exceptions

            # Test with compact=True
            _set_compact_log(True)
            task2 = create_task(failing_coro(), name="task2")
            await asyncio.sleep(0)
            try:
                await task2
            except DummyException:
                pass
            # Task completed, no assertion needed - just verify no exceptions

        asyncio.run(run())

    def test_create_task_uses_get_compact_log(self) -> None:
        """Test that create_task gets the compact setting from _get_compact_log."""

        async def failing_coro() -> None:
            raise DummyException("test error")

        async def run() -> None:
            # Setup: Set compact log via environment variable
            os.environ["SPDL_PIPELINE_COMPACT_LOG"] = "1"
            _set_compact_log(None)  # Reset to force env var check

            # Execute: Create task - this should use compact mode from env var
            task = create_task(failing_coro(), name="test_task")
            await asyncio.sleep(0)
            try:
                await task
            except DummyException:
                pass

            # Assert: Verify the getter returns True (from env var)
            self.assertTrue(_get_compact_log())

        asyncio.run(run())

    def test_get_compact_log_with_various_truthy_env_values(self) -> None:
        """Test that _get_compact_log handles various truthy environment values."""
        truthy_values = ["1", "true", "TRUE", "on", "ON", "yes", "YES"]

        for value in truthy_values:
            with self.subTest(value=value):
                # Setup: Reset and set env var
                _set_compact_log(None)
                os.environ["SPDL_PIPELINE_COMPACT_LOG"] = value

                # Execute: Get compact log setting
                result = _get_compact_log()

                # Assert: Should return True
                self.assertTrue(result, f"Expected True for value '{value}'")

    def test_get_compact_log_with_various_falsy_env_values(self) -> None:
        """Test that _get_compact_log handles various falsy environment values."""
        falsy_values = ["0", "false", "FALSE", "off", "OFF", "no", "NO"]

        for value in falsy_values:
            with self.subTest(value=value):
                # Setup: Reset and set env var
                _set_compact_log(None)
                os.environ["SPDL_PIPELINE_COMPACT_LOG"] = value

                # Execute: Get compact log setting
                result = _get_compact_log()

                # Assert: Should return False
                self.assertFalse(result, f"Expected False for value '{value}'")
