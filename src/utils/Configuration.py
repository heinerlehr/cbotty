# -*- coding: utf-8 -*-
"""
Reads multi-level configuration files for a particular Client. Configuration files are in yaml.
NOTE: all .yaml files in the path will be read and processed. The path is currently set as:
$CONFIGBASEDIR/Client/Cycle xxxx

We often use three levels of configuration:
- configGlobal.yaml which resides in the main project folder and holds the default values and global parameters
- config.yaml in each client folder which determines default values for that client
- config.yaml in each Cycle folder which provides values for that particular cycle

Given that the configuration files contain language dependent labels, this is also the place where
the language processor is installed.

In order to access individual parameters from the yaml parameter hierarchy the instance can be used as a function
cfg=Configuration(Client)
ParamValue1=cfg("Param1")
ParamValue2=cfg("ParamGroup","Param2")
ParamValue3=cfg("ParamGroup","ParamSubGroup","Param3")

@author: HL
"""
import os
from himl import ConfigProcessor
from pathlib import Path
from jinja2 import Template

#Local imports
from loguru import logger

from typing import overload, TypeVar, Any

T = TypeVar("T")

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class Configuration:
    ENV_BASEDIR = "CONFIG"

    _default_timezone = "Europe/Madrid"

    def __init__(self):
        self.basedir = self.get_base_dir()

        config_processor = ConfigProcessor()
        path = Path(self.basedir)

        filters = ()  # can choose to output only specific keys
        exclude_keys = ()  # can choose to remove specific keys
        output_format = "yaml"  # yaml/json

        if not path.exists():
            logger.error(f"Path {path} does not exist. Can't read configuration files.")
            raise ValueError(f"Path {path} does not exist. Can't read configuration files.")
        # logger.info(f"Reading configuration files from {path}")
        self._cfg = config_processor.process(path=str(path), filters=filters, exclude_keys=exclude_keys, output_format=output_format, print_data=False)
                    
    @overload
    def __call__(self, *params: str) -> Any: ...
    @overload
    def __call__(self, *params: str, default: T) -> T: ...

    def __call__(self, *params: str, default=None):
        try:
            cfg = self._cfg
            for param in params:
                cfg = cfg[param]
            return self.parse_tree(cfg)
        except Exception as e: # Whatever happens, we'll map it to a KeyError
            raise KeyError(f"Configuration files do not contain {params}. Original exception: {e}")

    def get(self, key: Any, alternative: Any = None, default: Any = None) -> Any:
        """
        Get a value from the configuration file. If the key is not found, the alternative is returned.
        If the alternative is not found, the default is returned.
        """
        if not isinstance(key, list):
            key = [key]
        if not alternative:
            alternative = default
        if not isinstance(alternative, list):
            alternative = [alternative]
        try:
            if alternative is None:
                return default
            return self.__call__(*key)
        except KeyError:
            try:
                return self.__call__(*alternative)
            except KeyError:
                return default
            
    def set(self, key:str|list, value):
        """
        Set a value in the configuration.
        """
        if not isinstance(key, list):
            key = [key]
        cfg = self._cfg
        for param in key[:-1]:
            if param not in cfg:
                raise KeyError(f"Configuration files do not contain {key}. Can't set value.")
            cfg = cfg[param]
        cfg[key[-1]] = value
        logger.info(f"Changed the value of configuration parameter {key} to {value}")
            
    def parse(self, key:str|list, alternative: str|list|None=None, default=None):
        """
        Parses the given key to retrieve its corresponding value from the configuration.
        If the value is a string and contains placeholders in the format `$(...)`, 
        these placeholders will be replaced with the corresponding environment variable values.
        Args:
            key (str | list): The key or list of keys to look up in the configuration.
            alternative (str | list, optional): An alternative key or list of keys to look up if the primary key is not found. Defaults to None.
            default (optional): The default value to return if the key and alternative key(s) are not found. Defaults to None.
        Returns:
            The value associated with the key, with environment variable placeholders replaced if applicable.
        """
        
        value = self.get(key, alternative, default)
        if isinstance(value, dict):
            return self.parse_tree(value)
        elif isinstance(value, str) and "$(" in value:
            value = value.replace("$(", "{{ ").replace(")", " }}")
            return Template(value).render(dict(os.environ))
        else:
            return value
        
    def parse_tree(self, tree:dict|str):
        """
        Parses the given tree to retrieve values from the configuration.
        Args:
            tree (dict): The tree structure to parse.
        Returns:
            A dictionary with the parsed values.
        """
        parsed_values = {}
        if isinstance(tree, str):
            if "$(" in tree:
                value = tree.replace("$(", "{{ ").replace(")", " }}")
                return Template(value).render(dict(os.environ))
            else:
                return tree
        elif isinstance(tree, dict):
            for key, value in tree.items():
                if isinstance(value, dict):
                    parsed_values[key] = self.parse_tree(value)
                elif isinstance(value, str) and "$(" in value:
                    value = value.replace("$(", "{{ ").replace(")", " }}")
                    parsed_values[key] = Template(value).render(dict(os.environ))
                else:
                    parsed_values[key] = value
            return parsed_values
        else:
            return tree

    def find(self, key:str, location:Any=None, default:Any=None) ->Any:
        """
        Find a value in the configuration file at the deepest level. If the key is not found, the default is returned.
        """
        
        def find_key(cfg, key, default):
            if key in cfg:
                return cfg[key]
            for k, v in cfg.items():
                if isinstance(v, dict):
                    item = find_key(v, key, default)
                    if item is not None:
                        return item
            return default
        
        if not location:
            return find_key(self._cfg, key, default)
        else:
            if location in self._cfg.keys():
                return self._cfg[location].get(key, default)
            elif (location:=location.replace(" ", "_")) in self._cfg.keys():
                return self._cfg[location].get(key, default)
            else:
                find_key(self._cfg, key, default)

    def get_base_dir(self):
        basedir = os.environ.get(self.ENV_BASEDIR)
        if basedir is None:
            logger.warning(f"No environment variable {self.ENV_BASEDIR} set. Will try current directory.")
            basedir = os.getcwd()
        return basedir
            