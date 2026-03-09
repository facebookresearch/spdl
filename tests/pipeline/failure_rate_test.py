# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from fractions import Fraction

from parameterized import parameterized
from spdl.pipeline import PipelineBuilder, PipelineFailure


class PipelineFailureRateTest(unittest.TestCase):
    """Tests for Fraction-based failure rate thresholds in SPDL pipeline.

    Key design: A fixed probation period of 100 invocations is used before
    rate-based checking kicks in. This prevents early false positives when
    sample size is too small to be statistically meaningful.

    The pipeline stops when failure rate strictly exceeds the threshold (>).
    """

    @parameterized.expand(
        [
            ("completion",),
            ("input",),
        ]
    )
    def test_pipeline_failure_rate_basic(self, output_order: str) -> None:
        """Pipeline fails when failure rate exceeds Fraction threshold.

        Uses 1000 items. Fails on multiples of 100 (10 out of 1000 = 1%).
        With 0.1% threshold and fixed probation of 100, should fail.
        After probation: rate = 1% > 0.1% threshold -> fails.
        """

        def fail_on_hundred(x):
            if x % 100 == 0:  # Fails on 0, 100, 200, ..., 900 (10 out of 1000)
                raise ValueError(f"Multiple of 100: {x}")
            return x

        # 0.1% threshold - should fail because actual rate is 1%
        pipeline = (
            PipelineBuilder()
            .add_source(range(1000))
            .pipe(fail_on_hundred, output_order=output_order)
            .add_sink(1)
            .build(num_threads=1, max_failures=Fraction(1, 1000))
        )

        vals = []
        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                vals = list(pipeline.get_iterator(timeout=30))

        all_expected = {x for x in range(1000) if x % 100 != 0}
        self.assertTrue(len(vals) > 0)
        self.assertTrue(set(vals).issubset(all_expected))

    @parameterized.expand(
        [
            ("completion",),
            ("input",),
        ]
    )
    def test_pipeline_failure_rate_passes(self, output_order: str) -> None:
        """Pipeline succeeds when failure rate stays below threshold.

        Uses 100 items. Fails on multiples of 10 (10 failures = 10%).
        With 15% threshold (Fraction(3, 20)) and fixed probation of 100.
        After probation: rate = 10% < 15% -> succeeds.
        """

        def fail_on_ten(x):
            if x % 10 == 0:  # Fails on 0, 10, 20, ..., 90 (10 out of 100 = 10%)
                raise ValueError(f"Multiple of 10: {x}")
            return x

        # Allow 15% failure rate - should succeed because actual rate is 10%
        pipeline = (
            PipelineBuilder()
            .add_source(range(100))
            .pipe(fail_on_ten, output_order=output_order)
            .add_sink(1)
            .build(num_threads=1, max_failures=Fraction(3, 20))
        )

        with pipeline.auto_stop():
            vals = list(pipeline.get_iterator(timeout=10))

        # Should get all non-multiples of 10
        expected = [x for x in range(100) if x % 10 != 0]
        self.assertEqual(expected, vals)

    @parameterized.expand(
        [
            ("completion",),
            ("input",),
        ]
    )
    def test_pipeline_failure_rate_just_below_threshold(
        self, output_order: str
    ) -> None:
        """Pipeline succeeds when failure rate is just below threshold.

        Uses 100 items, fails on items >= 90 (10 failures = 10%).
        With 11% threshold and fixed probation of 100.
        After probation: rate = 10% < 11% -> succeeds.
        """

        def fail_late(x):
            if x >= 90:  # Fails on 90-99 (10 out of 100 = 10%)
                raise ValueError(f"Item >= 90: {x}")
            return x

        # Allow 11% failure rate - should succeed because actual rate is 10%
        pipeline = (
            PipelineBuilder()
            .add_source(range(100))
            .pipe(fail_late, output_order=output_order)
            .add_sink(1)
            .build(num_threads=1, max_failures=Fraction(11, 100))
        )

        with pipeline.auto_stop():
            vals = list(pipeline.get_iterator(timeout=10))

        # Should get values 0-89 (90 successful items)
        self.assertEqual(list(range(90)), vals)

    @parameterized.expand(
        [
            ("completion",),
            ("input",),
        ]
    )
    def test_pipeline_failure_rate_above_threshold(self, output_order: str) -> None:
        """Pipeline fails when failure rate exceeds threshold.

        Uses 100 items, fails on items >= 90 (10 failures = 10%).
        With 9% threshold and fixed probation of 100.
        After probation: rate = 10% > 9% -> fails.
        """

        def fail_late(x):
            if x >= 90:  # Fails on 90-99 (10 out of 100 = 10%)
                raise ValueError(f"Item >= 90: {x}")
            return x

        # 9% threshold - should fail because actual rate is 10%
        pipeline = (
            PipelineBuilder()
            .add_source(range(100))
            .pipe(fail_late, output_order=output_order)
            .add_sink(1)
            .build(num_threads=1, max_failures=Fraction(9, 100))
        )

        vals = []
        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                vals = list(pipeline.get_iterator(timeout=10))

        # Should get values 0-89 (90 successful items)
        self.assertEqual(list(range(90)), vals)

    @parameterized.expand(
        [
            ("completion",),
            ("input",),
        ]
    )
    def test_pipeline_failure_rate_probation_period(self, output_order: str) -> None:
        """Early failures don't trigger threshold due to fixed probation period (100).

        Uses 20 items. Fails on multiples of 5 (4 out of 20 = 20%).
        With 1% threshold but fixed probation of 100, only 20 items processed.
        Since probation never completes, pipeline succeeds despite 20% > 1%.
        """

        def fail_on_five(x):
            if x % 5 == 0:  # Fails on 0, 5, 10, 15 (4 out of 20 = 20%)
                raise ValueError(f"Multiple of 5: {x}")
            return x

        # 1% threshold with fixed probation=100
        # Since we only have 20 items, probation never completes
        pipeline = (
            PipelineBuilder()
            .add_source(range(20))
            .pipe(fail_on_five, output_order=output_order)
            .add_sink(1)
            .build(num_threads=1, max_failures=Fraction(1, 100))
        )

        with pipeline.auto_stop():
            vals = list(pipeline.get_iterator(timeout=10))

        # Should get all non-multiples of 5: 1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19
        expected = [x for x in range(20) if x % 5 != 0]
        self.assertEqual(expected, vals)

    @parameterized.expand(
        [
            ("completion",),
            ("input",),
        ]
    )
    def test_pipeline_failure_rate_pipe_override(self, output_order: str) -> None:
        """Per-pipe Fraction override works correctly.

        Uses 100 items. Fails on multiples of 7 (15 out of 100 = 15%).
        Pipe-level: 11% threshold - should fail (15% > 11%).
        Global: 19% threshold - would pass.
        Since pipe-level is stricter, pipeline should fail.
        """

        def fail_on_seven(x):
            if x % 7 == 0:  # Fails on 0, 7, 14, ..., 98 (15 out of 100 = 15%)
                raise ValueError(f"Multiple of 7: {x}")
            return x

        # Global allows 19% but pipe-level restricts to 11% - should fail
        pipeline = (
            PipelineBuilder()
            .add_source(range(100))
            .pipe(
                fail_on_seven,
                output_order=output_order,
                max_failures=Fraction(11, 100),
            )
            .add_sink(1)
            .build(num_threads=1, max_failures=Fraction(19, 100))
        )

        vals = []
        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                vals = list(pipeline.get_iterator(timeout=10))

        # Should get all non-multiples of 7
        expected = [x for x in range(100) if x % 7 != 0]
        self.assertEqual(expected, vals)

    @parameterized.expand(
        [
            ("completion",),
            ("input",),
        ]
    )
    def test_pipeline_failure_rate_multiple_stages(self, output_order: str) -> None:
        """Multiple stages with different Fraction thresholds.

        First stage: fails on odd numbers (50%) but allowed unlimited.
        Second stage: receives 50 even numbers, fails on divisible by 12 (9/50 = 18%).
        With 22% threshold (Fraction(11, 50)), 18% < 22% -> should pass.
        Note: probation is fixed at 100, but only 50 items reach second stage.
        """

        def fail_odd(x):
            if x % 2:
                raise ValueError(f"Odd number: {x}")
            return x

        def fail_twelve(x):
            if (x % 12) == 0:
                raise ValueError(f"Divisible by 12: {x}")
            return x

        # First stage fails 50% (odd numbers) but allowed unlimited failures
        # Second stage: 18% failure rate with 22% threshold -> should succeed
        pipeline = (
            PipelineBuilder()
            .add_source(range(100))
            .pipe(fail_odd, output_order=output_order, max_failures=-1)
            .pipe(fail_twelve, output_order=output_order, max_failures=Fraction(11, 50))
            .add_sink(1)
            .build(num_threads=1, max_failures=-1)
        )

        # Second stage receives 50 even numbers (0,2,4,...98)
        # Fails on 0,12,24,36,48,60,72,84,96 = 9 failures out of 50 = 18%
        # With 22% threshold, should succeed
        with pipeline.auto_stop():
            vals = list(pipeline.get_iterator(timeout=10))

        # Even numbers not divisible by 12: 2,4,6,8,10,14,16,...
        expected = [x for x in range(100) if x % 2 == 0 and x % 12 != 0]
        self.assertEqual(expected, vals)

    @parameterized.expand(
        [
            ("completion",),
            ("input",),
        ]
    )
    def test_pipeline_failure_rate_vs_count(self, output_order: str) -> None:
        """Verify int count-based behavior unchanged.

        Fails on multiples of 10 (10 failures).
        Count-based: Allow 15 failures - should succeed.
        Count-based: Allow 5 failures - should fail.
        """

        def fail_on_ten(x):
            if x % 10 == 0:  # Fails on 0, 10, 20, ..., 90 (10 failures)
                raise ValueError(f"Multiple of 10: {x}")
            return x

        # Count-based: Allow 15 failures - should succeed
        pipeline = (
            PipelineBuilder()
            .add_source(range(100))
            .pipe(fail_on_ten, output_order=output_order)
            .add_sink(1)
            .build(num_threads=1, max_failures=15)
        )

        with pipeline.auto_stop():
            vals = list(pipeline.get_iterator(timeout=10))

        # Should get all non-multiples of 10
        expected = [x for x in range(100) if x % 10 != 0]
        self.assertEqual(expected, vals)

        # Count-based: Allow 5 failures - should fail after 5 failures
        pipeline = (
            PipelineBuilder()
            .add_source(range(100))
            .pipe(fail_on_ten, output_order=output_order)
            .add_sink(1)
            .build(num_threads=1, max_failures=5)
        )

        vals = []
        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                vals = list(pipeline.get_iterator(timeout=10))

        # Pipeline stops early after 5 failures; vals is a subset of expected
        all_expected = {x for x in range(100) if x % 10 != 0}
        self.assertTrue(len(vals) > 0)
        self.assertTrue(set(vals).issubset(all_expected))

    def test_pipeline_failure_rate_ordered_pipe(self) -> None:
        """Test with output_order='input'.

        Uses 100 items. Fails on multiples of 7 (15 out of 100 = 15%).
        With 11% threshold and fixed probation of 100.
        After probation: rate = 15% > 11% -> should fail.
        """

        def fail_on_seven(x):
            if x % 7 == 0:  # Fails on 0, 7, 14, ..., 98 (15 out of 100 = 15%)
                raise ValueError(f"Multiple of 7: {x}")
            return x

        # 11% threshold - should fail because actual rate is ~15%
        pipeline = (
            PipelineBuilder()
            .add_source(range(100))
            .pipe(fail_on_seven, output_order="input", max_failures=Fraction(11, 100))
            .add_sink(1)
            .build(num_threads=1)
        )

        vals = []
        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                vals = list(pipeline.get_iterator(timeout=10))

        # Should get all non-multiples of 7
        expected = [x for x in range(100) if x % 7 != 0]
        self.assertEqual(expected, vals)

    @parameterized.expand(
        [
            ("completion",),
            ("input",),
        ]
    )
    def test_pipeline_failure_rate_high_concurrency(self, output_order: str) -> None:
        """Test failure rate with high concurrency.

        Uses 100 items. Fails on multiples of 5 (20 out of 100 = 20%).
        With 15% threshold (Fraction(3, 20)) and fixed probation of 100.
        After probation: rate = 20% > 15% -> should fail.
        With high concurrency, exact processing order is nondeterministic.
        """

        def fail_on_five(x):
            if x % 5 == 0:  # Fails on 0, 5, 10, ..., 95 (20 out of 100 = 20%)
                raise ValueError(f"Multiple of 5: {x}")
            return x

        # 15% threshold - should fail because actual rate is 20%
        pipeline = (
            PipelineBuilder()
            .add_source(range(100))
            .pipe(
                fail_on_five,
                output_order=output_order,
                concurrency=5,
                max_failures=Fraction(3, 20),
            )
            .add_sink(1)
            .build(num_threads=5, max_failures=Fraction(1, 2))
        )

        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                list(pipeline.get_iterator(timeout=10))

    @parameterized.expand(
        [
            ("completion",),
            ("input",),
        ]
    )
    def test_pipeline_failure_rate_zero_failures(self, output_order: str) -> None:
        """Test with zero failures.

        0% failure rate with 10% threshold - should succeed.
        """

        def no_fail(x):
            return x * 2

        # 0% failure rate with 10% threshold - should succeed
        pipeline = (
            PipelineBuilder()
            .add_source(range(100))
            .pipe(no_fail, output_order=output_order)
            .add_sink(1)
            .build(num_threads=1, max_failures=Fraction(1, 10))
        )

        with pipeline.auto_stop():
            vals = list(pipeline.get_iterator(timeout=10))

        # Should get all values doubled
        self.assertEqual([x * 2 for x in range(100)], vals)

    @parameterized.expand(
        [
            ("completion",),
            ("input",),
        ]
    )
    def test_pipeline_failure_rate_under_probation_always_succeeds(
        self, output_order: str
    ) -> None:
        """With fixed probation of 100, pipelines with < 100 items always succeed.

        Uses 10 items with 40% failure rate (multiples of 3 fail).
        With 30% threshold but fixed probation of 100.
        Since only 10 items processed, probation not reached -> succeeds.
        """

        def fail_on_three(x):
            if x % 3 == 0:  # Fails on 0, 3, 6, 9 (4 out of 10 = 40%)
                raise ValueError(f"Multiple of 3: {x}")
            return x

        # 30% threshold, but only 10 items (under probation)
        # Should succeed because probation (100) not reached
        pipeline = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(fail_on_three, output_order=output_order)
            .add_sink(1)
            .build(num_threads=1, max_failures=Fraction(3, 10))
        )

        # Should succeed despite 40% > 30% because probation not complete
        with pipeline.auto_stop():
            vals = list(pipeline.get_iterator(timeout=10))

        # Should get all non-multiples of 3: 1, 2, 4, 5, 7, 8
        expected = [x for x in range(10) if x % 3 != 0]
        self.assertEqual(expected, vals)

    def test_pipeline_failure_rate_invalid_fraction_zero(self) -> None:
        """Building pipeline with Fraction <= 0 raises ValueError."""

        def noop(x):
            return x

        # Zero Fraction should raise ValueError
        with self.assertRaises(ValueError) as ctx:
            PipelineBuilder().add_source(range(10)).pipe(noop).add_sink(1).build(
                num_threads=1, max_failures=Fraction(0, 100)
            )

        self.assertIn("must be in range (0, 1]", str(ctx.exception))

    def test_pipeline_failure_rate_invalid_fraction_negative(self) -> None:
        """Building pipeline with negative Fraction raises ValueError."""

        def noop(x):
            return x

        # Negative Fraction should raise ValueError
        with self.assertRaises(ValueError) as ctx:
            PipelineBuilder().add_source(range(10)).pipe(noop).add_sink(1).build(
                num_threads=1, max_failures=Fraction(-1, 10)
            )

        self.assertIn("must be in range (0, 1]", str(ctx.exception))

    def test_pipeline_failure_rate_invalid_fraction_greater_than_one(self) -> None:
        """Building pipeline with Fraction > 1 raises ValueError."""

        def noop(x):
            return x

        # Fraction > 1 (e.g., 150%) should raise ValueError
        with self.assertRaises(ValueError) as ctx:
            PipelineBuilder().add_source(range(10)).pipe(noop).add_sink(1).build(
                num_threads=1, max_failures=Fraction(15, 10)
            )

        self.assertIn("must be in range (0, 1]", str(ctx.exception))

    def test_pipeline_failure_rate_invalid_pipe_fraction_zero(self) -> None:
        """Building pipeline with zero Fraction at pipe level raises ValueError."""

        def noop(x):
            return x

        # Zero Fraction at pipe level should raise ValueError
        with self.assertRaises(ValueError) as ctx:
            PipelineBuilder().add_source(range(10)).pipe(
                noop, max_failures=Fraction(0, 100)
            ).add_sink(1).build(num_threads=1)

        self.assertIn("must be in range (0, 1]", str(ctx.exception))

    def test_pipeline_failure_rate_invalid_pipe_fraction_greater_than_one(self) -> None:
        """Building pipeline with Fraction > 1 at pipe level raises ValueError."""

        def noop(x):
            return x

        # Fraction > 1 at pipe level should raise ValueError
        with self.assertRaises(ValueError) as ctx:
            PipelineBuilder().add_source(range(10)).pipe(
                noop, max_failures=Fraction(200, 100)
            ).add_sink(1).build(num_threads=1)

        self.assertIn("must be in range (0, 1]", str(ctx.exception))

    def test_pipeline_failure_rate_valid_fraction_one(self) -> None:
        """Fraction(1, 1) = 100% is valid (allows all failures).

        With > comparison, rate can never exceed 100%, so pipeline always succeeds.
        """

        def always_fail(x):
            raise ValueError(f"Always fail: {x}")

        # 100% failure rate threshold - should never fail
        pipeline = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(always_fail)
            .add_sink(1)
            .build(num_threads=1, max_failures=Fraction(1, 1))
        )

        # Should complete without PipelineFailure (100% failures allowed)
        with pipeline.auto_stop():
            vals = list(pipeline.get_iterator(timeout=10))

        # No items should pass through since all fail
        self.assertEqual([], vals)

    def test_pipeline_failure_rate_valid_small_fraction(self) -> None:
        """Very small Fraction like Fraction(1, 1000) is valid."""

        def fail_on_hundred(x):
            if x % 100 == 0:  # Fails on 0, 100, 200, ..., 900 (10 out of 1000 = 1%)
                raise ValueError(f"Multiple of 100: {x}")
            return x

        # 0.1% threshold (Fraction(1, 1000)) - should fail because actual rate is 1%
        pipeline = (
            PipelineBuilder()
            .add_source(range(1000))
            .pipe(fail_on_hundred)
            .add_sink(1)
            .build(num_threads=1, max_failures=Fraction(1, 1000))
        )

        vals = []
        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                vals = list(pipeline.get_iterator(timeout=30))

        all_expected = {x for x in range(1000) if x % 100 != 0}
        self.assertTrue(len(vals) > 0)
        self.assertTrue(set(vals).issubset(all_expected))

    @parameterized.expand(
        [
            (Fraction(0, 1), "zero numerator"),
            (Fraction(0, 100), "zero with large denominator"),
            (Fraction(-1, 10), "negative numerator"),
            (Fraction(1, -10), "negative denominator"),
            (Fraction(-1, -10), "double negative (positive > 0 but > 1)"),
        ]
    )
    def test_pipeline_failure_rate_invalid_fractions_parameterized(
        self, fraction: Fraction, description: str
    ) -> None:
        """Parameterized test for various invalid Fraction values."""

        def noop(x):
            return x

        # Note: Fraction(-1, -10) normalizes to Fraction(1, 10) which is valid
        # But Fraction(0, x) and negative fractions should fail
        if fraction <= 0 or fraction > 1:
            with self.assertRaises(ValueError):
                PipelineBuilder().add_source(range(10)).pipe(noop).add_sink(1).build(
                    num_threads=1, max_failures=fraction
                )
        else:
            # This should not raise - it's a valid fraction
            pipeline = (
                PipelineBuilder()
                .add_source(range(10))
                .pipe(noop)
                .add_sink(1)
                .build(num_threads=1, max_failures=fraction)
            )
            with pipeline.auto_stop():
                vals = list(pipeline.get_iterator(timeout=10))
            self.assertEqual(list(range(10)), vals)

    @parameterized.expand(
        [
            (Fraction(11, 10), "110%"),
            (Fraction(2, 1), "200%"),
            (Fraction(150, 100), "150%"),
            (Fraction(101, 100), "101%"),
        ]
    )
    def test_pipeline_failure_rate_invalid_fractions_greater_than_one_parameterized(
        self, fraction: Fraction, description: str
    ) -> None:
        """Parameterized test for Fraction values > 1 (greater than 100%)."""

        def noop(x):
            return x

        with self.assertRaises(ValueError) as ctx:
            PipelineBuilder().add_source(range(10)).pipe(noop).add_sink(1).build(
                num_threads=1, max_failures=fraction
            )

        self.assertIn("must be in range (0, 1]", str(ctx.exception))

    @parameterized.expand(
        [
            (Fraction(1, 100), "1%"),
            (Fraction(1, 10), "10%"),
            (Fraction(1, 2), "50%"),
            (Fraction(99, 100), "99%"),
            (Fraction(1, 1), "100%"),
        ]
    )
    def test_pipeline_failure_rate_valid_fractions_parameterized(
        self, fraction: Fraction, description: str
    ) -> None:
        """Parameterized test for valid Fraction values in range (0, 1]."""

        def noop(x):
            return x

        # Should not raise - these are all valid fractions
        pipeline = (
            PipelineBuilder()
            .add_source(range(100))
            .pipe(noop)
            .add_sink(1)
            .build(num_threads=1, max_failures=fraction)
        )

        with pipeline.auto_stop():
            vals = list(pipeline.get_iterator(timeout=10))

        # All values should pass through
        self.assertEqual(list(range(100)), vals)

    def test_pipeline_failure_rate_probation_prevents_early_trigger(self) -> None:
        """Probation period (fixed at 100) prevents triggering even when rate is high.

        Uses 10 items, fails on multiples of 5 (2/10 = 20%).
        With 1% threshold but fixed probation of 100, only 10 items processed.
        Since probation not reached, pipeline succeeds despite 20% > 1%.
        """

        def fail_on_five(x):
            if x % 5 == 0:  # Fails on 0, 5 (2 out of 10 = 20%)
                raise ValueError(f"Multiple of 5: {x}")
            return x

        # 1% threshold with fixed probation=100. Only 10 items -> no check runs.
        pipeline = (
            PipelineBuilder()
            .add_source(range(10))
            .pipe(fail_on_five)
            .add_sink(1)
            .build(num_threads=1, max_failures=Fraction(1, 100))
        )

        with pipeline.auto_stop():
            vals = list(pipeline.get_iterator(timeout=10))

        # Should get all non-multiples of 5: 1, 2, 3, 4, 6, 7, 8, 9
        expected = [x for x in range(10) if x % 5 != 0]
        self.assertEqual(expected, vals)

    def test_pipeline_failure_rate_probation_triggers_after_warmup(self) -> None:
        """Once probation period (100) is met, threshold check runs.

        Uses 100 items, fails on multiples of 5 (20/100 = 20%).
        With 10% threshold and fixed probation of 100.
        After 100 invocations: rate = 20% > 10% -> fails.
        """

        def fail_on_five(x):
            if x % 5 == 0:  # Fails on 0, 5, ..., 95 (20 out of 100 = 20%)
                raise ValueError(f"Multiple of 5: {x}")
            return x

        # 10% threshold with fixed probation=100. All 100 items are processed.
        pipeline = (
            PipelineBuilder()
            .add_source(range(100))
            .pipe(fail_on_five)
            .add_sink(1)
            .build(num_threads=1, max_failures=Fraction(1, 10))
        )

        vals = []
        with self.assertRaises(PipelineFailure):
            with pipeline.auto_stop():
                vals = list(pipeline.get_iterator(timeout=10))

        # Should get all non-multiples of 5
        expected = [x for x in range(100) if x % 5 != 0]
        self.assertEqual(expected, vals)
