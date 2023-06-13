import sys

import pkg_resources

from util import logutil

logger = logutil.logger_run


def print_versions() -> None:
    # print Python version
    logger.info(f"Python version: {sys.version}")

    # print all dependencies
    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
                                      for i in installed_packages])
    logger.info('All dependencies:\n' + '\n'.join(installed_packages_list))
