# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
import time

from hippo_mem.common.maintenance import BackgroundTaskManager


def test_background_task_manager_start_stop():
    counter = 0

    def tick(stop_event):
        nonlocal counter
        counter += 1

    manager = BackgroundTaskManager(tick)
    manager.start(interval=0.01)

    # Wait for at least one tick
    timeout = time.time() + 0.2
    while counter == 0 and time.time() < timeout:
        time.sleep(0.01)

    manager.stop()
    ticks_at_stop = counter

    # Ensure no further ticks after stop
    time.sleep(0.05)

    assert ticks_at_stop > 0
    assert counter == ticks_at_stop
