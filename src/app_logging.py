"""
Module for configuring and managing logging for the application or its specific components.

This module provides utilities for setting up the logging system, trading customizability
for consistency. It defines log level, format and handlers, as well as any other necessary
helper functions strictly related to application logging.

To configure logging with the desired log level, import the `setup` function from this
module and call it with the appropriate log level, early in the application setup. It is
recommended to use a named logger for the application rather than the root logger. Multiple
modules can share the same logger.

Example:
```
    import logging
    import app_logging

    logger = logging.getLogger("logger_name")    # __name__ is commonly used
    # ...

    def main():
        app_logging.setup(logging.INFO)
        logger.info("Successfully setup logger")
```
"""
import logging


def setup(log_level: int):
    """
    Configure logging for the module, defining format and log level.

    Args:
        log_level (int): Enum corresponding to the desired log level.
    """
    logging.basicConfig(
        format="%(asctime)s    %(levelname)-8.8s[L%(lineno)3d]: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        level=log_level,
    )
