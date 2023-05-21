#!/usr/bin/python3

import json # for loading the parameters file in json format
import os
from dataclasses import dataclass


@dataclass(slots = True)
class OptionsLoadResult:
    options: dict        # options tree as a dictionary
    messages: str = None # messages from the options checker
    flag: int     = 0    # failure flag. 1 = failure, 0 = success

    def delete_messages(self):
        self.messages = []
        return
    
    
def load_options(file: str):
    r"""
    Load options from a JSON file and run a check on options.
    """

    with open(file, 'r') as fp:
        opts = json.load(fp)

    # TODO: options check
    messages, flag = [], 0

    return OptionsLoadResult(options = opts, messages = messages, flag = flag)



if __name__ == '__main__':

    import pprint
    file = os.path.join(os.path.split(__file__)[0], "param.json")
    f = load_options(file) 
    pprint.pprint( f )
