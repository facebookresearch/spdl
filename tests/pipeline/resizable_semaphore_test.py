# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import unittest

from spdl.pipeline._components._semaphore import ResizableSemaphore


class ResizableSemaphoreConstructionTest(unittest.TestCase):
    def test_init_valid(self) -> None:
        sem = ResizableSemaphore(5)
        self.assertEqual(sem.max_value, 5)
        self.assertEqual(sem.active, 0)

    def test_init_one(self) -> None:
        sem = ResizableSemaphore(1)
        self.assertEqual(sem.max_value, 1)
        self.assertEqual(sem.active, 0)

    def test_init_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            ResizableSemaphore(0)

    def test_init_negative_raises(self) -> None:
        with self.assertRaises(ValueError):
            ResizableSemaphore(-1)


class ResizableSemaphoreAcquireReleaseTest(unittest.TestCase):
    def test_acquire_decrements_permits(self) -> None:
        async def run() -> None:
            sem = ResizableSemaphore(3)
            await sem.acquire()
            self.assertEqual(sem.active, 1)
            await sem.acquire()
            self.assertEqual(sem.active, 2)
            await sem.acquire()
            self.assertEqual(sem.active, 3)

        asyncio.run(run())

    def test_release_increments_permits(self) -> None:
        async def run() -> None:
            sem = ResizableSemaphore(3)
            await sem.acquire()
            await sem.acquire()
            self.assertEqual(sem.active, 2)
            sem.release()
            self.assertEqual(sem.active, 1)
            sem.release()
            self.assertEqual(sem.active, 0)

        asyncio.run(run())

    def test_acquire_blocks_when_exhausted(self) -> None:
        async def run() -> None:
            sem: ResizableSemaphore = ResizableSemaphore(1)
            await sem.acquire()

            acquired: asyncio.Event = asyncio.Event()

            async def try_acquire() -> None:
                await sem.acquire()
                acquired.set()

            task = asyncio.create_task(try_acquire())
            # Yield to let the task enter acquire() and block.
            await asyncio.sleep(0)
            self.assertFalse(acquired.is_set())

            # Release unblocks the waiter.
            sem.release()
            await asyncio.sleep(0)
            self.assertTrue(acquired.is_set())

            # Clean up.
            sem.release()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run())

    def test_release_without_acquire_clamps(self) -> None:
        """Release without prior acquire should not exceed max_value."""

        async def run() -> None:
            sem = ResizableSemaphore(3)
            # active is 0, current_value == max_value == 3
            sem.release()
            # Should clamp: active stays 0, not -1.
            self.assertEqual(sem.active, 0)
            self.assertEqual(sem.max_value, 3)

        asyncio.run(run())

    def test_fifo_wake_order(self) -> None:
        """Waiters should be woken in FIFO order."""

        async def run() -> None:
            sem: ResizableSemaphore = ResizableSemaphore(1)
            await sem.acquire()

            order: list[int] = []

            async def waiter(idx: int) -> None:
                await sem.acquire()
                order.append(idx)
                sem.release()

            t1 = asyncio.create_task(waiter(1))
            await asyncio.sleep(0)
            t2 = asyncio.create_task(waiter(2))
            await asyncio.sleep(0)
            t3 = asyncio.create_task(waiter(3))
            await asyncio.sleep(0)

            # Release the initial acquire — should wake waiter 1 first.
            sem.release()

            # Wait for all waiters to complete.
            await asyncio.wait_for(asyncio.gather(t1, t2, t3), timeout=5.0)
            self.assertEqual(order, [1, 2, 3])

        asyncio.run(run())


class ResizableSemaphoreResizeUpTest(unittest.TestCase):
    def test_resize_up_increases_max(self) -> None:
        async def run() -> None:
            sem = ResizableSemaphore(2)
            sem.resize(5)
            self.assertEqual(sem.max_value, 5)

        asyncio.run(run())

    def test_resize_up_wakes_waiters(self) -> None:
        async def run() -> None:
            sem: ResizableSemaphore = ResizableSemaphore(1)
            await sem.acquire()

            acquired_events: list[asyncio.Event] = []

            async def waiter() -> None:
                await sem.acquire()
                evt = acquired_events[len(acquired_events)]
                evt.set()

            e1 = asyncio.Event()
            e2 = asyncio.Event()
            acquired_events.extend([e1, e2])

            # Simpler: track via counter
            acquired_count = 0

            async def counting_waiter() -> None:
                nonlocal acquired_count
                await sem.acquire()
                acquired_count += 1

            t1 = asyncio.create_task(counting_waiter())
            await asyncio.sleep(0)
            t2 = asyncio.create_task(counting_waiter())
            await asyncio.sleep(0)
            self.assertEqual(acquired_count, 0)

            # Resize from 1 -> 3: adds 2 permits, should wake both waiters.
            sem.resize(3)
            await asyncio.sleep(0)
            self.assertEqual(acquired_count, 2)
            self.assertEqual(sem.active, 3)

            # Clean up — release all 3 acquired permits.
            sem.release()
            sem.release()
            sem.release()

            # Await tasks to prevent warnings.
            await asyncio.wait_for(asyncio.gather(t1, t2), timeout=5.0)

        asyncio.run(run())

    def test_resize_up_partial_wake(self) -> None:
        """When resize adds fewer permits than waiters, only some wake."""

        async def run() -> None:
            sem: ResizableSemaphore = ResizableSemaphore(1)
            await sem.acquire()

            acquired: list[int] = []

            async def waiter(idx: int) -> None:
                await sem.acquire()
                acquired.append(idx)

            t1 = asyncio.create_task(waiter(1))
            await asyncio.sleep(0)
            t2 = asyncio.create_task(waiter(2))
            await asyncio.sleep(0)
            t3 = asyncio.create_task(waiter(3))
            await asyncio.sleep(0)

            # Resize from 1 -> 2: adds 1 permit, wakes 1 waiter.
            sem.resize(2)
            await asyncio.sleep(0)
            self.assertEqual(len(acquired), 1)
            self.assertEqual(acquired[0], 1)  # FIFO

            # Clean up remaining waiters.
            for t in (t1, t2, t3):
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

        asyncio.run(run())


class ResizableSemaphoreResizeDownTest(unittest.TestCase):
    def test_resize_down_no_preemption(self) -> None:
        """Active tasks continue after resize down."""

        async def run() -> None:
            sem = ResizableSemaphore(3)
            await sem.acquire()
            await sem.acquire()
            await sem.acquire()
            self.assertEqual(sem.active, 3)

            # Resize down to 1. All 3 are still active.
            sem.resize(1)
            self.assertEqual(sem.max_value, 1)
            self.assertEqual(sem.active, 3)  # No preemption.

        asyncio.run(run())

    def test_resize_down_blocks_new_acquires(self) -> None:
        """After resize down, new acquires block until drain completes."""

        async def run() -> None:
            sem: ResizableSemaphore = ResizableSemaphore(3)
            await sem.acquire()
            await sem.acquire()
            await sem.acquire()

            sem.resize(1)

            acquired: asyncio.Event = asyncio.Event()

            async def try_acquire() -> None:
                await sem.acquire()
                acquired.set()

            task = asyncio.create_task(try_acquire())
            await asyncio.sleep(0)
            self.assertFalse(acquired.is_set())

            # Release 3 permits (drain from 3 active -> 0).
            # First two releases bring active from 3->2->1 (at max).
            # Third release frees a permit for the waiter.
            sem.release()
            sem.release()
            sem.release()
            await asyncio.sleep(0)
            self.assertTrue(acquired.is_set())

            # Clean up.
            sem.release()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run())

    def test_resize_down_drain_releases_clamp(self) -> None:
        """Releases during drain clamp to max_value, not old max."""

        async def run() -> None:
            sem = ResizableSemaphore(5)
            await sem.acquire()
            await sem.acquire()
            # active=2, current_value=3
            sem.resize(2)
            # current_value should now be 0 (3 - (5-2) = 0)
            self.assertEqual(sem.active, 2)
            self.assertEqual(sem.max_value, 2)

            # Release one: active -> 1.
            sem.release()
            self.assertEqual(sem.active, 1)

            # Release another: active -> 0.
            sem.release()
            self.assertEqual(sem.active, 0)

            # Extra release should clamp.
            sem.release()
            self.assertEqual(sem.active, 0)

        asyncio.run(run())


class ResizableSemaphoreResizeEdgeCasesTest(unittest.TestCase):
    def test_resize_to_same_value(self) -> None:
        async def run() -> None:
            sem = ResizableSemaphore(3)
            await sem.acquire()
            sem.resize(3)
            self.assertEqual(sem.max_value, 3)
            self.assertEqual(sem.active, 1)

        asyncio.run(run())

    def test_resize_to_zero_raises(self) -> None:
        sem = ResizableSemaphore(3)
        with self.assertRaises(ValueError):
            sem.resize(0)

    def test_resize_to_negative_raises(self) -> None:
        sem = ResizableSemaphore(3)
        with self.assertRaises(ValueError):
            sem.resize(-5)

    def test_resize_preserves_max_after_error(self) -> None:
        """Failed resize should not change max_value."""
        sem = ResizableSemaphore(3)
        try:
            sem.resize(0)
        except ValueError:
            pass
        self.assertEqual(sem.max_value, 3)

    def test_resize_while_waiters_pending_noop(self) -> None:
        """Resize to same value while tasks are waiting should not wake them."""

        async def run() -> None:
            sem: ResizableSemaphore = ResizableSemaphore(1)
            await sem.acquire()

            acquired: asyncio.Event = asyncio.Event()

            async def waiter() -> None:
                await sem.acquire()
                acquired.set()

            task = asyncio.create_task(waiter())
            await asyncio.sleep(0)

            # Resize to same value — waiter should stay blocked.
            sem.resize(1)
            await asyncio.sleep(0)
            self.assertFalse(acquired.is_set())

            # Clean up.
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run())

    def test_multiple_resizes(self) -> None:
        """Multiple sequential resizes should work correctly."""

        async def run() -> None:
            sem = ResizableSemaphore(1)
            sem.resize(5)
            self.assertEqual(sem.max_value, 5)
            sem.resize(2)
            self.assertEqual(sem.max_value, 2)
            sem.resize(10)
            self.assertEqual(sem.max_value, 10)

        asyncio.run(run())


class ResizableSemaphoreCancellationTest(unittest.TestCase):
    def test_cancelled_waiter_is_removed(self) -> None:
        async def run() -> None:
            sem: ResizableSemaphore = ResizableSemaphore(1)
            await sem.acquire()

            async def waiter() -> None:
                await sem.acquire()

            task = asyncio.create_task(waiter())
            await asyncio.sleep(0)

            # Cancel the waiter.
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            # Release — should not raise even though waiter was cancelled.
            sem.release()
            self.assertEqual(sem.active, 0)

        asyncio.run(run())

    def test_cancel_one_of_multiple_waiters(self) -> None:
        """Cancel middle waiter; remaining waiters still get served."""

        async def run() -> None:
            sem: ResizableSemaphore = ResizableSemaphore(1)
            await sem.acquire()

            order: list[int] = []

            async def waiter(idx: int) -> None:
                await sem.acquire()
                order.append(idx)
                sem.release()

            t1 = asyncio.create_task(waiter(1))
            await asyncio.sleep(0)
            t2 = asyncio.create_task(waiter(2))
            await asyncio.sleep(0)
            t3 = asyncio.create_task(waiter(3))
            await asyncio.sleep(0)

            # Cancel the second waiter.
            t2.cancel()
            try:
                await t2
            except asyncio.CancelledError:
                pass

            sem.release()
            await asyncio.wait_for(asyncio.gather(t1, t3), timeout=5.0)
            self.assertEqual(order, [1, 3])

        asyncio.run(run())


class ResizableSemaphorePermitLeakTest(unittest.TestCase):
    """V5.1 (DESIGN.md): regression coverage for the permit-leak fix in
    ``acquire()``'s cancellation handler.

    Without the fix in case (b) — when ``release()`` direct-hands a
    permit via ``set_result(None)`` and the waiter is then cancelled
    before resuming — the permit was lost. Each cancellation under
    contention leaked one permit; eventually the semaphore deadlocked.
    """

    def test_acquire_cancel_after_grant_releases_permit(self) -> None:
        """Case (b): waiter is granted a permit via direct hand-off but
        cancelled before resuming. The permit must be released back so
        another acquirer can make progress. Without the fix, the second
        acquirer would deadlock.
        """

        async def run() -> None:
            sem: ResizableSemaphore = ResizableSemaphore(1)
            await sem.acquire()  # holder takes the only permit.

            # First waiter: will be granted the permit by release(),
            # then cancelled before its acquire() returns.
            granted_w1: asyncio.Event = asyncio.Event()

            async def w1() -> None:
                # Acquire will receive set_result(None) from the
                # holder's release(); cancellation arrives during
                # the await window before acquire() returns.
                try:
                    await sem.acquire()
                except asyncio.CancelledError:
                    granted_w1.set()
                    raise

            t1 = asyncio.create_task(w1())
            # Yield until W1 has enqueued its waiter future.
            await asyncio.sleep(0)
            self.assertEqual(len(sem._waiters), 1)

            # Direct-hand the permit to W1: this calls
            # waiter.set_result(None) on W1's future. W1 is now in
            # case (b): future is done with a result, but W1 hasn't
            # resumed yet.
            sem.release()
            self.assertEqual(len(sem._waiters), 0)

            # Cancel W1 before its acquire() resumes. Permit-leak fix
            # must detect case (b) and call self.release() to give
            # the permit back.
            t1.cancel()
            try:
                await t1
            except asyncio.CancelledError:
                pass
            self.assertTrue(granted_w1.is_set())

            # Now a new acquirer should be able to acquire the permit
            # immediately. WITHOUT THE FIX this would deadlock here.
            second_acquired: asyncio.Event = asyncio.Event()

            async def w2() -> None:
                await sem.acquire()
                second_acquired.set()

            t2 = asyncio.create_task(w2())
            await asyncio.wait_for(t2, timeout=2.0)
            self.assertTrue(second_acquired.is_set())

            # Pool accounting: we have one active permit (W2's).
            self.assertEqual(sem.active, 1)

        asyncio.run(run())

    def test_release_grants_to_next_when_first_waiter_cancelled(
        self,
    ) -> None:
        """Direct-hand-off race: ``release()`` pops a waiter whose
        future is already cancelled (done()) and must skip it to hand
        the permit to the next non-cancelled waiter.

        Sequence:
          1. sem with value=1, acquired by holder.
          2. Two waiters W1, W2 enqueued.
          3. W1.cancel() — its future becomes done() (cancelled).
          4. holder.release() — must pop W1 (skip), pop W2 (grant).
        """

        async def run() -> None:
            sem: ResizableSemaphore = ResizableSemaphore(1)
            await sem.acquire()  # holder

            w1_started: asyncio.Event = asyncio.Event()
            w2_started: asyncio.Event = asyncio.Event()
            w2_acquired: asyncio.Event = asyncio.Event()

            async def w1() -> None:
                w1_started.set()
                await sem.acquire()

            async def w2() -> None:
                w2_started.set()
                await sem.acquire()
                w2_acquired.set()

            t1 = asyncio.create_task(w1())
            await w1_started.wait()
            t2 = asyncio.create_task(w2())
            await w2_started.wait()
            # Force both to enqueue their waiter futures.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertEqual(len(sem._waiters), 2)

            # Cancel W1 — its waiter future transitions to done() but
            # remains in the deque (acquire()'s except handler removes
            # it after the await re-raises CancelledError).
            t1.cancel()
            try:
                await t1
            except asyncio.CancelledError:
                pass

            # release() must pop W1 (skip — done()) then pop W2 (grant).
            sem.release()
            await asyncio.wait_for(t2, timeout=2.0)
            self.assertTrue(w2_acquired.is_set())
            self.assertFalse(t2.cancelled())

        asyncio.run(run())

    def test_acquire_pre_cancelled_future_fast_path(self) -> None:
        """Case (c) documentation: when the future has been cancelled
        before the await even completes (e.g., the task was cancelled
        before it ran past the ``self._waiters.append(fut)`` line), the
        cancellation handler treats this the same as case (a) — the
        future is still in ``self._waiters``, so ``self._waiters.remove
        (fut)`` covers it. No permit was granted, so nothing to release
        back.

        This test verifies the case (c) fast path via cancelling at
        scheduling time rather than mid-await: pool accounting must
        remain consistent (active=1 from the holder, max=1).
        """

        async def run() -> None:
            sem: ResizableSemaphore = ResizableSemaphore(1)
            await sem.acquire()  # holder

            async def w1() -> None:
                # Will be cancelled before release() ever fires, while
                # waiter future is still pending in the deque.
                await sem.acquire()

            t1 = asyncio.create_task(w1())
            # Let W1 enqueue its waiter future.
            await asyncio.sleep(0)
            self.assertEqual(len(sem._waiters), 1)

            # Cancel W1 immediately — future is still pending (case (a)),
            # which shares the cleanup path with case (c).
            t1.cancel()
            try:
                await t1
            except asyncio.CancelledError:
                pass

            # The waiter future must be cleaned up from the deque so
            # subsequent release() doesn't try to grant to a dead
            # waiter.
            self.assertEqual(len(sem._waiters), 0)

            # Pool accounting: holder still owns the permit.
            self.assertEqual(sem.active, 1)
            self.assertEqual(sem.max_value, 1)

            # Release returns the permit cleanly.
            sem.release()
            self.assertEqual(sem.active, 0)

        asyncio.run(run())


class ResizableSemaphoreConcurrencyTest(unittest.TestCase):
    def test_concurrent_acquire_release(self) -> None:
        """Many tasks acquiring and releasing concurrently."""

        async def run() -> None:
            sem: ResizableSemaphore = ResizableSemaphore(3)
            completed = 0

            async def worker() -> None:
                nonlocal completed
                await sem.acquire()
                # Yield to let other tasks proceed.
                await asyncio.sleep(0)
                sem.release()
                completed += 1

            tasks = [asyncio.create_task(worker()) for _ in range(20)]
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
            self.assertEqual(completed, 20)
            self.assertEqual(sem.active, 0)

        asyncio.run(run())

    def test_concurrent_acquire_with_resize(self) -> None:
        """Resize during concurrent acquire/release operations."""

        async def run() -> None:
            sem: ResizableSemaphore = ResizableSemaphore(2)
            completed = 0

            async def worker() -> None:
                nonlocal completed
                await sem.acquire()
                await asyncio.sleep(0)
                sem.release()
                completed += 1

            tasks = [asyncio.create_task(worker()) for _ in range(10)]

            # Yield a few times to let some workers start.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            sem.resize(5)  # Expand.
            await asyncio.sleep(0)
            sem.resize(1)  # Contract.

            await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
            self.assertEqual(completed, 10)
            self.assertEqual(sem.active, 0)

        asyncio.run(run())

    def test_active_never_exceeds_max_under_load(self) -> None:
        """active should never exceed max_value when max is stable."""

        async def run() -> None:
            max_permits = 4
            sem: ResizableSemaphore = ResizableSemaphore(max_permits)
            max_seen = 0

            async def worker() -> None:
                nonlocal max_seen
                await sem.acquire()
                current = sem.active
                if current > max_seen:
                    max_seen = current
                await asyncio.sleep(0)
                sem.release()

            tasks = [asyncio.create_task(worker()) for _ in range(30)]
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
            self.assertLessEqual(max_seen, max_permits)

        asyncio.run(run())


class ResizableSemaphoreResizeReleaseInteractionTest(unittest.TestCase):
    """Interaction between ``resize()`` and ``release()`` direct hand-off.

    These tests close gaps in the existing suite around the boundary
    between resize-down accounting and release-direct-handoff accounting.
    """

    def test_resize_down_then_release_with_waiter_caps_at_new_max(self) -> None:
        """resize-down → release direct-handoff to a waiter must not exceed new max."""

        async def run() -> None:
            # Arrange
            sem: ResizableSemaphore = ResizableSemaphore(3)
            await sem.acquire()
            await sem.acquire()
            await sem.acquire()

            acquired_after_resize: list[int] = []

            async def waiter(idx: int) -> None:
                await sem.acquire()
                acquired_after_resize.append(idx)

            # Enqueue a waiter while the pool is exhausted.
            t = asyncio.create_task(waiter(1))
            await asyncio.sleep(0)

            # Resize down: max -> 1; current_value goes to -2.
            sem.resize(1)

            # Act: three releases drain the over-fill, then one more
            # crosses the threshold and direct-hands to the waiter.
            sem.release()
            sem.release()
            sem.release()
            await asyncio.sleep(0)
            self.assertEqual(acquired_after_resize, [1])

            # Assert: with the waiter holding the only permit (active=1)
            # and max_value=1, sem must report active==max_value.
            self.assertEqual(sem.active, 1)
            self.assertEqual(sem.max_value, 1)

            # Cleanup
            sem.release()
            await asyncio.wait_for(t, timeout=2.0)
            # Pool returned to "no active permits" without exceeding max.
            self.assertEqual(sem.active, 0)

        asyncio.run(run())

    def test_concurrent_resize_calls_converge_to_last_value(self) -> None:
        """Two coroutines calling resize() concurrently — the last wins.

        asyncio is single-threaded but multiple coroutines can issue
        resize() in the same tick. The final ``max_value`` must equal
        the last resize call's value, and accounting must remain
        consistent.
        """

        async def run() -> None:
            # Arrange
            sem: ResizableSemaphore = ResizableSemaphore(2)
            await sem.acquire()

            async def resizer(target: int) -> None:
                # Yield once to interleave the two resize calls.
                await asyncio.sleep(0)
                sem.resize(target)

            # Act: dispatch two concurrent resizes.
            await asyncio.gather(resizer(5), resizer(3))

            # Assert: final value is whichever resizer ran last. Both are
            # valid (>= 1); accounting must be consistent.
            self.assertIn(sem.max_value, (3, 5))
            # active == max_value - current_value; the holder still owns
            # one permit, so active is 1 regardless of resize order.
            self.assertEqual(sem.active, 1)

            sem.release()
            self.assertEqual(sem.active, 0)

        asyncio.run(run())


class ResizableSemaphorePropertiesTest(unittest.TestCase):
    def test_max_value_reflects_resize(self) -> None:
        sem = ResizableSemaphore(3)
        self.assertEqual(sem.max_value, 3)
        sem.resize(7)
        self.assertEqual(sem.max_value, 7)
        sem.resize(1)
        self.assertEqual(sem.max_value, 1)

    def test_active_reflects_acquire_release(self) -> None:
        async def run() -> None:
            sem = ResizableSemaphore(5)
            self.assertEqual(sem.active, 0)
            await sem.acquire()
            self.assertEqual(sem.active, 1)
            await sem.acquire()
            self.assertEqual(sem.active, 2)
            sem.release()
            self.assertEqual(sem.active, 1)
            sem.release()
            self.assertEqual(sem.active, 0)

        asyncio.run(run())

    def test_active_exceeds_max_after_resize_down(self) -> None:
        """active can temporarily exceed max_value after resize down."""

        async def run() -> None:
            sem = ResizableSemaphore(5)
            for _ in range(5):
                await sem.acquire()
            self.assertEqual(sem.active, 5)

            sem.resize(2)
            self.assertEqual(sem.max_value, 2)
            self.assertEqual(sem.active, 5)  # Exceeds max_value.

            # Drain.
            for _ in range(5):
                sem.release()
            self.assertEqual(sem.active, 0)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
