"""
log.py

Description:
------------
A module for logging messages to a file and/or console. This module
contains a logger named 'neural' that can be used to log messages to a
file and/or console. The logger can be configured by setting the
following constants in neural/common/constants.py:

    LOG_PATH (str):
        if not set no log files will be created. If set, the log files
        will be created at specified path.
    MAX_LOG_SIZE (int):
        maximum size of log file in bytes. If the size of log file
        exceeds this value, a backup will be created.
    LOG_BACKUP_COUNT (int):
        The number of backup log files to keep. If the log file exceeds
        MAX_LOG_SIZE, then it will be truncated and renamed to
        <LOG_PATH>.1. If <LOG_PATH>.1 exists, then it will be renamed to
        <LOG_PATH>.2, and so on. If LOG_BACKUP_COUNT is 0, then no
        backup log files will be kept.
    LOG_LEVEL (int):
        The logging level for the logger. The logger will log messages
        with a level greater than or equal to LOG_LEVEL. The available
        logging levels are: DEBUG, INFO, WARNING, ERROR, and CRITICAL.
        lower level means more messages, i.e. messages for lower levels
        that selected level are filtered out. CRITICAL > ERROR > WARNING
        > INFO > DEBUG > NOTSET

License:
--------
    MIT License. See LICENSE.md file.

Author(s):
-------
    Reza Soleymanifar, Email: Reza@Soleymanifar.com
    
Notes:
------
    If LOG_PATH is provided, creates a rotating file handler with the
    specified file name, maximum size, and backup count. If LOG_PATH is
    None, then set the file handler to a NullHandler, which will not log
    any messages to a file. 

Example:
---------
    >>> from neural.common.constants import LOG_LEVEL
    >>> from neural.common.constants import LOG_PATH
    >>> from neural.common.log import logger
    >>> LOG_LEVEL = logging.DEBUG
    >>> LOG_PATH = 'neural.log'
    >>> logger.debug('This is a debug message.')
    >>> logger.info('This is an info message.')
    >>> logger.warning('This is a warning message.')
    >>> logger.error('This is an error message.')
    >>> logger.critical('This is a critical message.')
"""

import logging
from logging.handlers import RotatingFileHandler
from neural.common.constants import (
    LOG_PATH, MAX_LOG_SIZE, LOG_BACKUP_COUNT, LOG_LEVEL
)

# =========================setup logger=============================

logger = logging.getLogger('neural')
logger.setLevel(LOG_LEVEL)

# =========================file/console handler=============================

if LOG_PATH is not None:
    file_handler = RotatingFileHandler(filename=LOG_PATH,
                                       maxBytes=MAX_LOG_SIZE,
                                       backupCount=LOG_BACKUP_COUNT)
else:
    file_handler = logging.NullHandler()
file_handler.setLevel(LOG_LEVEL)

console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# =========================logger formatter=============================

formatter = logging.Formatter('%(levelname)s - %(message)s')

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
