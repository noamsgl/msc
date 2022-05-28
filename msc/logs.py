import logging
import sys

from .data_utils import count_nans

def get_logger():
        logger = logging.getLogger(__name__)
        if not logger.handlers:  # for not having duplicate logs
            logger.setLevel(logging.INFO)
        
            # create a formatter that creates a single line of json with a comma at the end
            formatter = logging.Formatter(
                (
                    '{"unix_time":%(created)s, "time":"%(asctime)s", "module":"%(name)s",'
                    ' "line_no":%(lineno)s, "level":"%(levelname)s", "msg":"%(message)s"},'
                )
            )

            # create a channel for handling the logger and set its format
            ch = logging.StreamHandler(sys.stderr)
            ch.setFormatter(formatter)

            # connect the logger to the channel
            ch.setLevel(logging.INFO)
            logger.addHandler(ch)
            
        # send an example message
        logger.info('logging is working')
        return logger


def nan_report(data):
    nan_count = count_nans(data)
    return f"there are {nan_count}/{data.size} ({100 * nan_count/data.size:.0f}%) nan entries"
