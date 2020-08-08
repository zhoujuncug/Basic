import time
import os
import logging


def create_logger():
    root = '../outputs'
    time_str = time.strftime('%y-%m-%d-%H-%M-%S')

    # create output dir and logging file
    output_dir = os.path.join(root, time_str)
    os.makedirs(output_dir, exist_ok=True)
    final_log_file = os.path.join(output_dir, 'log.log')

    head = '%(asctime)-15s\n%(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head,
                        level=logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logger.addHandler(console)

    return logger, output_dir