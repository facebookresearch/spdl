# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from collections.abc import Iterator
from contextlib import contextmanager

from spdl.pipeline import (
    AsyncQueue,
    ProfileHook,
    ProfileResult,
    StatsQueue,
    TaskHook,
    TaskStatsHook,
)
from spdl.pipeline._config import (
    get_default_hook_class,
    get_default_profile_callback,
    get_default_profile_hook,
    get_default_queue_class,
    set_default_hook_class,
    set_default_profile_callback,
    set_default_profile_hook,
    set_default_queue_class,
)


class ConfigTest(unittest.TestCase):
    """Test the configuration setter/getter functions."""

    def setUp(self) -> None:
        """Reset all configuration state before each test."""
        set_default_hook_class()
        set_default_queue_class()
        set_default_profile_hook()
        set_default_profile_callback()

    def test_hook_class_default_is_none(self) -> None:
        """Test that default hook class is None when not configured."""
        result = get_default_hook_class()
        self.assertIs(result, TaskStatsHook)

    def test_hook_class_can_be_set_and_retrieved(self) -> None:
        """Test that hook class can be set and retrieved correctly."""

        class CustomHook(TaskHook):
            pass

        set_default_hook_class(CustomHook)
        result = get_default_hook_class()

        self.assertIs(result, CustomHook)

    def test_hook_class_via_config_module(self) -> None:
        """Test that hook class can be accessed via _config module."""

        class CustomHook(TaskHook):
            pass

        set_default_hook_class(CustomHook)
        result = get_default_hook_class()

        self.assertIs(result, CustomHook)

    def test_queue_class_default_is_none(self) -> None:
        """Test that default queue class is None when not configured."""
        result = get_default_queue_class()
        self.assertIs(result, StatsQueue)

    def test_queue_class_can_be_set_and_retrieved(self) -> None:
        """Test that queue class can be set and retrieved correctly."""

        class CustomQueue(StatsQueue[int]):
            pass

        set_default_queue_class(CustomQueue)
        result = get_default_queue_class()
        self.assertIs(result, CustomQueue)

    def test_queue_class_via_config_module(self) -> None:
        """Test that queue class can be accessed via _config module."""

        class CustomQueue(StatsQueue[int]):
            pass

        set_default_queue_class(CustomQueue)
        result = get_default_queue_class()
        self.assertIs(result, CustomQueue)

    def test_profile_hook_default_is_none(self) -> None:
        """Test that default profile hook is None when not configured."""
        result = get_default_profile_hook()
        self.assertIsNone(result)

    def test_profile_hook_can_be_set_and_retrieved(self) -> None:
        """Test that profile hook can be set and retrieved correctly."""

        class MockProfileHook(ProfileHook):
            @contextmanager
            def stage_profile_hook(
                self,
                stage: str,  # noqa: ARG002
                concurrency: int,  # noqa: ARG002
            ) -> Iterator[None]:
                yield

            @contextmanager
            def pipeline_profile_hook(self) -> Iterator[None]:
                yield

        hook_instance = MockProfileHook()
        set_default_profile_hook(hook_instance)
        result = get_default_profile_hook()
        self.assertIs(result, hook_instance)

    def test_profile_hook_via_config_module(self) -> None:
        """Test that profile hook can be accessed via _config module."""

        class MockProfileHook(ProfileHook):
            @contextmanager
            def stage_profile_hook(
                self,
                stage: str,  # noqa: ARG002
                concurrency: int,  # noqa: ARG002
            ) -> Iterator[None]:
                yield

            @contextmanager
            def pipeline_profile_hook(self) -> Iterator[None]:
                yield

        hook_instance = MockProfileHook()
        set_default_profile_hook(hook_instance)
        result = get_default_profile_hook()
        self.assertIs(result, hook_instance)

    def test_profile_callback_default_is_none(self) -> None:
        """Test that default profile callback is None when not configured."""
        result = get_default_profile_callback()
        self.assertIsNone(result)

    def test_profile_callback_can_be_set_and_retrieved(self) -> None:
        """Test that profile callback can be set and retrieved correctly."""

        def mock_callback(_: ProfileResult) -> None:
            pass

        set_default_profile_callback(mock_callback)
        result = get_default_profile_callback()
        self.assertIs(result, mock_callback)

    def test_profile_callback_via_config_module(self) -> None:
        """Test that profile callback can be accessed via _config module."""

        def mock_callback(_: object) -> None:
            pass

        set_default_profile_callback(mock_callback)
        result = get_default_profile_callback()
        self.assertIs(result, mock_callback)

    def test_multiple_configurations_independent(self) -> None:
        """Test that different configuration settings are independent."""

        class CustomHook(TaskHook):
            pass

        class CustomQueue(AsyncQueue[int]):
            pass

        def custom_callback(_: object) -> None:
            pass

        set_default_hook_class(CustomHook)
        set_default_queue_class(CustomQueue)
        set_default_profile_callback(custom_callback)

        self.assertIs(get_default_hook_class(), CustomHook)
        self.assertIs(get_default_queue_class(), CustomQueue)
        self.assertIs(get_default_profile_callback(), custom_callback)

    def test_configuration_can_be_updated(self) -> None:
        """Test that configuration can be updated to new values."""

        class FirstHook(TaskHook):
            pass

        class SecondHook(TaskHook):
            pass

        # Set first value, then update to second value
        set_default_hook_class(FirstHook)
        set_default_hook_class(SecondHook)
        result = get_default_hook_class()
        self.assertIs(result, SecondHook)
