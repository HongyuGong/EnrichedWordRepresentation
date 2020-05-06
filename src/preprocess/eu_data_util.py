"""
eu_data_util.py
 - preprocess euphemism data
"""

import pickle
import os
import argparse
import re
import sys
from config.params import *

# create directories
for folder in [eu_folder, eu_raw_data_folder, eu_cond_data_folder, \
               eu_vocab_data_folder, eu_cooccur_folder]:
    if not os.path.isdir(folder):
        os.makedirs(folder)


if __name__=="__main__":
    print("process Euphemism datasets...")
    
