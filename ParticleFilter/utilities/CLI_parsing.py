from argparse import ArgumentParser

def parse():  
    parser = ArgumentParser()
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--population',type=int,required=True)
    parser.add_argument('--file',type=str)
    parser.add_argument('--initial_seed',type=float)
    parser.add_argument('--simulate_data',action='store_true')
    parser.add_argument('--particles',type=int)
    parser.add_argument('--forecast',action='store_true')
    args = parser.parse_args()



    return args
 

    