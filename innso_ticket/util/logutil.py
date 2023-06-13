import logging.handlers
import os
from logging.handlers import RotatingFileHandler

currentdir = os.path.dirname(__file__)
parentdir = os.path.abspath(os.path.join(os.path.join(currentdir, os.pardir), os.pardir))
print("log:", parentdir)
# logging初始化工作

logging.basicConfig()

log_home = os.path.join(parentdir, 'log')
if not os.path.exists(log_home):
    os.makedirs(log_home)
logger_run = logging.getLogger("run")
logger_run.setLevel(logging.INFO)

logger_interface = logging.getLogger("interface")
logger_interface.setLevel(logging.INFO)

logger_network = logging.getLogger("network")
logger_network.setLevel(logging.INFO)

# 创建interface日志
log_interface_write_path = os.path.join(log_home, 'interface.log')

# 创建run日志
log_run_write_path = os.path.join(log_home, 'run.log')

# 创建网元日志
log_network_write_path = os.path.join(log_home, 'network.log')

# 写入文件，如果文件超过1000Bytes，仅保留5个文件。
handler_run = RotatingFileHandler(log_run_write_path, maxBytes=50 * 1024 * 1024, backupCount=10, mode='w',
                                  encoding='utf-8')
handler_interface = RotatingFileHandler(log_interface_write_path, maxBytes=50 * 1024 * 1024, backupCount=10, mode='w',
                                        encoding='utf-8')
handler_network = RotatingFileHandler(log_network_write_path, maxBytes=50 * 1024 * 1024, backupCount=10, mode='w',
                                      encoding='utf-8')

interface_formatter_general = logging.Formatter('%(asctime)s|%(message)s')
handler_interface.setFormatter(interface_formatter_general)

formatter_general = logging.Formatter(
    '%(asctime)s|%(pathname)s|%(filename)s|%(funcName)s|%(lineno)s|%(levelname)s|%(message)s')
handler_run.setFormatter(formatter_general)
handler_network.setFormatter(formatter_general)

logger_run.addHandler(handler_run)
logger_interface.addHandler(handler_interface)
logger_network.addHandler(handler_network)
