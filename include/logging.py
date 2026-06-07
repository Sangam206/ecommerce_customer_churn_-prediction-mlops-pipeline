import logging
import os

mk_dir = "log"
os.makedirs(mk_dir, exist_ok=True)

logger = logging.getLogger("log")
logger.setLevel(logging.DEBUG)

# handler
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)

file_dir = os.path.join(mk_dir, "logs.log")
file_han = logging.FileHandler(file_dir)
file_han.setLevel(logging.DEBUG)

# formatter
format = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# attach formatter to handlers
console.setFormatter(format)
file_han.setFormatter(format)

# add handlers to logger
logger.addHandler(console)
logger.addHandler(file_han)