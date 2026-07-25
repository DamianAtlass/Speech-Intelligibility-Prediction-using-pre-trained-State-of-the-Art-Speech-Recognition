from contextlib import contextmanager, redirect_stdout
from typing import Callable, Iterator
from time import perf_counter
import io

@contextmanager
def catch_time() -> Iterator[Callable[[], float]]:
    """
    Example:
    with catch_time() as t:
        logging_mylib.do_something()
    logger.info(f"Execution time of do_something: {t()/3600:.4f} h")
    """
    start = perf_counter()
    yield lambda: perf_counter() - start

@contextmanager
def capture_stdout(logging_function: Callable,
                   name: str):  # this can be used as a decorator *and* a context manager
    """Use to log functions which use print statements.

    Example:
    with capture_stdout(logger.info, __name__):
        some_function(1,2,3)
    """
    f = io.StringIO()

    with redirect_stdout(f):
        yield
    output = f.getvalue()
    logging_function(f"{name + ": " if name else ""}\n{output}", )