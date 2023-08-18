##################################################################
## Generate input file for disBatch
## Run holoQUADS circuits in parallel
##################################################################

import numpy as np

def generate_input_file(input_index, task_file):
    '''Generate corresponding folders and input files based on chemical potential'''
    folder_name = "Sample" + "{}".format(input_index) + "/"
    task_file.write("cd " + folder_name \
        + " &&  julia --threads=1 holoQUADS_Heisenberg.jl" + " &> holoQUADS_" \
        + "{}".format(input_index) + ".log" + "\n")
    
def main():
    sample_seed=0
    lower_bound=(sample_seed-1)*250+1
    upper_bound=sample_seed*250+1
    # lower_bound=(sample_seed-1)*500+5001
    # upper_bound=sample_seed*500+5001
    sample_list = np.arange(lower_bound, upper_bound, 1)

    submit_file = open("T0", "a")
    for tmp in sample_list:
        generate_input_file(tmp, submit_file)
    submit_file.close()    

main()