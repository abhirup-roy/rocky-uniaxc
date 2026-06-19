import shutil
import pathlib
import functools
import inspect
from typing import Optional

try:
    import ansys.rocky.core as rocky_api
except ImportError:
    rocky_api = None


def find_rocky_exe() -> Optional[str]:
    """Attempt to locate the Rocky executable.

    Searches common installation paths and the system ``PATH`` for the Rocky
    binary.

    Returns:
        The path to the Rocky executable as a string, or ``None`` if it
        cannot be found.
    """
    possible_paths = [
        shutil.which("Rocky"),  # Check system PATH
        "/usr/local/bin/Rocky",
        "/opt/rocky/Rocky",
        str(pathlib.Path.home() / "Rocky" / "Rocky"),
    ]
    for path in possible_paths:
        if path and pathlib.Path(path).is_file():
            return path
    return None


class pyrocky_run:
    """Context manager, decorator, and class decorator for Rocky API sessions.

    Manages the lifecycle of a Rocky application instance — launching on
    entry and closing on exit.  Can be used in three ways:

    1. **As a context manager**::

           with pyrocky_run() as rocky:
               ...

    2. **As a function decorator** — the decorated function receives an
       injected ``rocky`` keyword argument::

           @pyrocky_run()
           def my_function(rocky):
               ...

    3. **As a class decorator** — the wrapped class receives a ``rocky``
       attribute that is initialised on instantiation and closed on
       garbage collection::

           @pyrocky_run()
           class MyRockyClass:
               ...

    Args:
        headless: If ``True`` (default), launch Rocky in headless mode.

    Raises:
        FileNotFoundError: If the Rocky executable cannot be located.
    """

    def __init__(self, headless: Optional[bool] = None):
        self.headless = headless
        from .. import ROCKY_EXE_PATH
        self.rocky_exe = find_rocky_exe() or ROCKY_EXE_PATH
        self.rocky = None

    def __enter__(self):
        if rocky_api is None:
            raise ImportError(
                "ansys.rocky.core is required to use pyrocky_run. "
                "Install the Ansys Rocky Python API."
            )
        if not self.rocky_exe or not pathlib.Path(self.rocky_exe).is_file():
            raise FileNotFoundError(
                "Rocky executable not found. Please set the path using set_rocky_exe_path()."
            )
        headless = self.headless
        if headless is None:
            from .. import HEADLESS
            headless = HEADLESS
        self.rocky = rocky_api.launch_rocky(
            rocky_exe=self.rocky_exe, headless=headless
        )
        return self.rocky

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.rocky:
            self.rocky.close()

    def __call__(self, obj):

        if inspect.isclass(obj):

            class WrappedClass(obj):
                def __init__(child_self, *args, **kwargs):
                    child_self._rocky_runner = type(self)(headless=self.headless)
                    child_self.rocky = child_self._rocky_runner.__enter__()
                    super().__init__(*args, **kwargs)

                def __del__(child_self):
                    if hasattr(child_self, "_rocky_runner"):
                        child_self._rocky_runner.__exit__(None, None, None)
                    if hasattr(obj, "__del__"):
                        obj.__del__(child_self)

            WrappedClass.__name__ = obj.__name__
            WrappedClass.__doc__ = obj.__doc__
            return WrappedClass

        @functools.wraps(obj)
        def inner(*args, **kwargs):
            with self as rocky:
                return obj(*args, rocky=rocky, **kwargs)

        sig = inspect.signature(obj)
        new_params = [
            param for name, param in sig.parameters.items() if name != "rocky"
        ]
        inner.__signature__ = sig.replace(parameters=new_params)  # type: ignore

        return inner
