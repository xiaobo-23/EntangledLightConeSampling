##################################################################
## Generate input file for disBatch
## Run holoQUADS circuits in parallel
##################################################################

import numpy as np
import os

def generate_input_file(input_index, task_file):
    '''Generate corresponding folders and input files based on chemical potential'''
    folder_name = "Sample" + "{}".format(input_index) + "/"
    task_file.write("cd " + folder_name \
        + " &&  julia --threads=1 holoQUADS_TEBD_SDKI.jl" + " &> holoQUADS_" \
        + "{}".format(input_index) + ".log" + "\n")
    
def main():
    sample_seed=0
    # lower_bound=(sample_seed-1)*500+5001
    # upper_bound=sample_seed*500+5001
    lower_bound = 1
    upper_bound = 1001
    sample_list = np.arange(lower_bound, upper_bound, 1)
    # location = os.path.dirname(os.path.realpath(__file__))

    submit_file = open("T6_v1", "a")
    for tmp in sample_list:
        generate_input_file(tmp, submit_file)
    submit_file.close()  

    lower_bound = 1001
    upper_bound = 2001
    sample_list = np.arange(lower_bound, upper_bound, 1)
    # location = os.path.dirname(os.path.realpath(__file__))

    submit_file = open("T6_v2", "a")
    for tmp in sample_list:
        generate_input_file(tmp, submit_file)
    submit_file.close()    

main()