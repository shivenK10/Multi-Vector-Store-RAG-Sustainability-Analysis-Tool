import logging
from logging.handlers import RotatingFileHandler
from typing import Literal

class Logger:
    def __init__(self, name: str, log_file_needed: bool = False, log_file: str = '', level: Literal['DEV', 'PROD'] = 'DEV'):
        """Initialize logger with console and file output"""
        
        if log_file_needed and not log_file.strip():
            raise ValueError("A file name is required when log_file_needed is set to True")
        
        self.name = name
        self.log_file_needed = log_file_needed
        self.log_file = log_file
        
        self.logger = logging.getLogger(name)
        if level.upper() == 'DEV':
            self.logger.setLevel(logging.DEBUG)
        elif level.upper() == 'PROD':
            self.logger.setLevel(logging.INFO)
        else:
            raise ValueError("The value of level must be 'DEV' or 'PROD'")
        
        self.logger.handlers.clear()
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        if self.log_file_needed:  
            file_handler = RotatingFileHandler(
                log_file, 
                maxBytes=5*1024*1024,
                backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        if self.log_file_needed:
            file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        if self.log_file_needed:
            self.logger.addHandler(file_handler)

    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message):
        """Log critical message"""
        self.logger.critical(message)

