from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """
    Get logger.

    Purpose:
        Implements get_logger for the shared infrastructure layer that exposes
            application configuration and logging utilities.
    Args:
        name (str): Input value for the name parameter.
    Returns:
        logging.Logger: Result produced by the operation.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return logging.getLogger(name)
