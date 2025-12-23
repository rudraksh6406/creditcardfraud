"""
Logging Configuration for Production System
============================================
Provides structured logging for the fraud detection system.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, 'fraud_detection.log')
ERROR_LOG_FILE = os.path.join(LOG_DIR, 'errors.log')


def setup_logger(name='fraud_detection', level=logging.INFO):
    """
    Set up a logger with file and console handlers.
    
    Parameters:
    -----------
    name : str
        Logger name
    level : int
        Logging level
        
    Returns:
    --------
    logger : logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler (rotating)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Error file handler
    error_handler = RotatingFileHandler(
        ERROR_LOG_FILE,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)
    
    return logger


# Create default logger
logger = setup_logger()

