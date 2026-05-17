from contextlib import contextmanager, redirect_stdout
import io
from typing import Callable

@contextmanager
def capture_stdout(logging_function: Callable):
    """
    Log the stdout to a file.

    logging_function: Callable, something like logger.info.

    """
    f = io.StringIO()

    logging_function("Before some_function")

    with redirect_stdout(f):
        yield
    output = f.getvalue()
    logging_function(output)
    print(output)