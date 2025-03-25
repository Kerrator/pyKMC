import argparse 
from pykmc import KMC, Config

def main() : 
    parser = argparse.ArgumentParser() 
    parser.add_argument("-in", "--input", type=str, required = True, help="inputs file") 
    args = parser.parse_args() 

    #Config 
    config = Config.from_file(args.input) 
    #KMC 
    kmc = KMC(config)
    kmc.run()

if __name__=='__main__' : 
    main()