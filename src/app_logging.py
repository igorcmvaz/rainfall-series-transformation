"""
Module for configuring and managing logging for the application or its specific components.

This module provides utilities for setting up the logging system, trading customizability
for consistency. It defines log level, format and handlers, as well as any other necessary
helper functions strictly related to application logging.

To configure logging with the desired log level, import the `setup` function from this
module and call it with the appropriate quiet flag count, early in the application setup.
It is recommended to use a named logger for the application rather than the root logger.
Multiple modules can share the same logger.

Example:
```
    import logging
    import app_logging

    logger = logging.getLogger("logger_name")    # __name__ is commonly used
    # ...

    def main():
        app_logging.setup(quiet_count=1)
        logger.info("Successfully setup logger")
```
"""
import logging


def setup(quiet_count: int):
    """
    Configure logging for the module, defining format and log level according to a quiet
    flag counter.

    Args:
        quiet_count (int): Number of times the quiet flag was used for current execution.
    """
    log_level = logging.DEBUG
    if quiet_count == 1:
        log_level = logging.INFO
    elif quiet_count == 2:
        log_level = logging.WARNING
    elif quiet_count >= 3:
        log_level = logging.ERROR

    logging.basicConfig(
        format="%(asctime)s    %(levelname)-8.8s[L%(lineno)3d]: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        level=log_level,
    )
