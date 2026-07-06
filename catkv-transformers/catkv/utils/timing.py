import time
from contextlib import contextmanager

@contextmanager
def time_scope(name: str, enabled: bool = True):
    start = time.time()
    yield
    end = time.time()
    if enabled:
        print(f"{name} took {end - start:.6f} seconds") 