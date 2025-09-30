import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger():
    logger = logging.getLogger()
    #Check if handler already exits, Otherwise running this multiple times, duplicate handlers get created thus leading to duplicated logs
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  #Sets the logging level for the root logger

        #Rotating File Handler at the DEBUG level
        file_path = r"logs/app.log"

        # Extract the directory from the file path
        directory = os.path.dirname(file_path)

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")
        file_handler = RotatingFileHandler(filename=file_path, maxBytes=10e6, backupCount=10)
        file_handler.setLevel(logging.INFO)  #sets the log level for the file handler
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s'))
        logger.addHandler(file_handler)

        #Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s'))
        logger.addHandler(console_handler)

        #Rotating File Handler at Error Level
        file_path2 = r"logs/error.log"
        error_handler = RotatingFileHandler(filename=file_path2, maxBytes=10e6, backupCount=10)
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s'))
        logger.addHandler(error_handler)

    return logger
