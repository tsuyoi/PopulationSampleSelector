import logging
import traceback
import warnings

import matplotlib
import matplotlib.pyplot as plt

from gui import gui_main

warnings.filterwarnings('ignore')

plt.style.use('ggplot')
# matplotlib.use('TkAgg')
matplotlib.use('Agg')


def setup_logging():
    _format = f"%(levelname)s | %(asctime)s | %(filename)s::%(funcName)s | %(message)s"
    _root_handler = logging.StreamHandler()
    _root_handler.setFormatter(logging.Formatter(_format))
    _root_handler.setLevel(logging.INFO)
    _root_logger = logging.getLogger()
    _root_logger.addHandler(_root_handler)
    _root_logger.setLevel(logging.INFO)


def main():
    setup_logging()
    gui_main()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        logging.error(f"Unknown error occured")
        traceback.print_exc()
