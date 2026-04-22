import logging
import sys
import os
import datetime
from contextlib import contextmanager


class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: '\033[94m',       # Blue
        logging.INFO: '\033[92m',        # Green
        logging.WARNING: '\033[93m',     # Yellow
        logging.ERROR: '\033[91m',       # Red
        logging.CRITICAL: '\033[1;91m',  # Bold Red
    }
    RESET = '\033[0m'

    def format(self, record):
        # Get the color for the specific log level
        color = self.COLORS.get(record.levelno, self.RESET)
        
        # Save the original format string
        format_orig = self._style._fmt
        
        # Inject the color code at the start, and the RESET code right before the message
        self._style._fmt = f"{color}[%(asctime)s | %(levelname)s]{self.RESET} %(message)s"
        
        # Format the record
        result = super().format(record)
        
        # Restore the original format string
        self._style._fmt = format_orig
        
        return result


def init_logging(log_dir):
    """
    Initializes logging to output colored prefixes to the console 
    and plain text to a timestamped file.
    """
    # 1. Create log directory
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    
    # 2. Define formats
    log_format = '[%(asctime)s | %(levelname)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 3. Configure Console Handler (WITH color prefix)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter(log_format, datefmt=date_format))
    
    # 4. Configure File Handler (WITHOUT color, plain text)
    log_file_path = os.path.join(log_dir, f"log_{timestamp}.txt")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    
    # 5. Apply handlers
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler],
        force=True  # Overwrites any existing logging configuration
    )

    # torchrun / torch.distributed.run 会注入 RANK；非 0 不写终端，避免 4 份重复 INFO
    silence_worker_console_if_not_rank0()


def silence_worker_console_if_not_rank0():
    """
    若环境变量 RANK 存在且不为 0，移除 root 上指向 stdout/stderr 的 Handler。
    文件日志保留。未设置 RANK（单进程直接 python）时视为 rank0，不改动。
    """
    if "RANK" not in os.environ:
        return
    rank = int(os.environ["RANK"])
    if rank == 0:
        return
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler) and h.stream in (sys.stdout, sys.stderr):
            root.removeHandler(h)


def set_logging_enabled(enabled: bool, level=logging.INFO):
    """
    Enables or disables logging globally.
    """
    if enabled:
        logging.disable(logging.NOTSET)
        logging.getLogger().setLevel(level)
    else:
        logging.disable(logging.CRITICAL)


@contextmanager
def disable_logging_context(highest_level=logging.CRITICAL):
    """
    A context manager to temporarily disable logging.
    
    Args:
        highest_level: All logs at or below this level will be suppressed.
                       Defaults to logging.CRITICAL (suppresses everything).
    """
    # Save the current disable level so we can restore it later
    previous_level = logging.root.manager.disable
    
    # Disable logging
    logging.disable(highest_level)
    
    try:
        yield
    finally:
        # Guarantee that logging is restored even if an exception occurs
        logging.disable(previous_level)


# --- Usage Example ---
if __name__ == "__main__":
    init_logging("logs")
    
    logging.info("This message is white, but the prefix is green!")
    logging.warning("This message is white, but the prefix is yellow!")
    logging.error("This message is white, but the prefix is red!")
