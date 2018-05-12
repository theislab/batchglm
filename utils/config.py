import os
import configparser
import errno


def getConfig(config_file="config.ini") -> configparser.ConfigParser:
    if config_file is None:
        config_file = "config.ini"
    
    config = configparser.ConfigParser()
    if os.path.exists(config_file):
        config.read(config_file)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_file)
    return config

