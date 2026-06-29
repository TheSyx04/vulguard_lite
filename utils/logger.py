from icecream import ic as logger
from datetime import datetime

logger.configureOutput(prefix=f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | ')