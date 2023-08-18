from argparse import ArgumentParser

def parse():  
    parser = ArgumentParser()
    parser.add_argument('--iterations', type=int, required=True)
    parser.add_argument('--population',type=int,required=True)
    parser.add_argument('--file',type=str)
    parser.add_argument('--initial_seed',type=float)
    parser.add_argument('--simulate_data',type =bool)
    parser.add_argument('--particles',type=int)
    args = parser.parse_args()



    return args
 

    