##################################################################
## Generate input files for disBatch
## Infinite MPS and the extended Hubard model in 1D 
##################################################################

import numpy as np
import sys
import csv
import os 
import shutil
import math
from io import StringIO


def generate_input_file(input_index, task_file):
    '''Generate corresponding folders and input files based on chemical potential'''
    folder_name = "Sample" + "{}".format(input_index) + "/"
    task_file.write("cd " + folder_name \
        + " &&  julia --threads=1 EHM_VUMPS.jl" + " &> holoQUADS_" \
        + "{}".format(input_index) + ".log" + "\n")
    
def main():
    sample_list = np.arange(1, 2001, 1)
    location = os.path.dirname(os.path.realpath(__file__))

    submit_file = open("T2", "a")
    for tmp in sample_list:
        generate_input_file(tmp, submit_file)
    submit_file.close()

main()