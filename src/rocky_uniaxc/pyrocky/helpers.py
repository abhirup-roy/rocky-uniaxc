import shutil
import pathlib
import ansys.rocky.core as rocky_api
from .. import ROCKY_EXE_PATH
import functools
import inspect


def find_rocky_exe():
    """Attempt to find the Rocky executable in common locations and system PATH."""
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
    """Context manager for launching and closing the Rocky application.
    Can be used as a decorator for functions or classes that require access to the Pyrocky API.

    Examples:
        >>> @pyrocky_run()
        ... def my_function(rocky):
        ...     # Use the rocky API here
        ...     pass

        >>> @pyrocky_run()
        ... class MyRockyClass:
        ...     def __init__(self):
        ...         self.rocky = None
        ...     def do_something(self):

    """

    def __init__(self, headless=True):
        self.headless = headless
        self.rocky_exe = find_rocky_exe() or ROCKY_EXE_PATH
        self.rocky = None

    def __enter__(self):
        if not self.rocky_exe or not pathlib.Path(self.rocky_exe).is_file():
            raise FileNotFoundError(
                "Rocky executable not found. Please set the path using set_rocky_exe_path()."
            )
        self.rocky = rocky_api.launch_rocky(
            rocky_exe=self.rocky_exe, headless=self.headless
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
