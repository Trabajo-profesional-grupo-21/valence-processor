from src.processor import Processor
import logging
from src.config.config import settings

def main():
    logging_level = settings.LOGGING_LEVEL 

    initialize_log(logging_level)
    logging.debug(f"action: config | result: success | logging_level: {logging_level}")

    try:
        processor = Processor()
        processor.run()
    except Exception as e:
        logging.error(f'action: initialize_valence_processor | result: fail | error: {e}')

def initialize_log(logging_level):
    """
    Python custom logging initialization

    Current timestamp is added to be able to identify in docker
    compose logs the date when the log has arrived
    """
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging_level,
        datefmt='%Y-%m-%d %H:%M:%S',
    )


if __name__ == "__main__":
    main()
