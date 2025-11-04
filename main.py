#!/usr/bin/env python
import os, sys
from src import utils
from src import run
import argparse

def main():
    # load config from file to serve as default arguments
    cfg = utils.io.load_config(utils.paths.CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Run PMM experiment with given parameters. The default parameters used are those in the config file.")
    parser.add_argument("--model_name", type=str, default=None, help="The name of the physical model; e.g., 'gaussian.Gaussian1d'")
    parser.add_argument("--model_kwargs", type=utils.misc.parse_kwargs, default=None, help="Comma-separated key=value pairs for physical model, e.g. `N=128,V0=-4.0,R=2.0`")
    parser.add_argument("--pmm_name", type=str, default=None, help="The name of the PMM algorithm to run like `PMMInverse`")
    parser.add_argument("--dim", type=int, default=None, help="The dimension of the PMM algorithm")
    parser.add_argument("--num_primary", type=int, default=None, help="The number of primary matrices to use in the PMM algorithm")
    parser.add_argument("--num_secondary", type=int, default=None, help="The number of secondary matrices to use in the PMM algorithm")
    parser.add_argument("--eta", type=float, default=None, help="The learning rate for the PMM algorithm") 
    parser.add_argument("--beta1", type=float, default=None, help="The weight factor for the running average of the gradient (first moment)")
    parser.add_argument("--beta2", type=float, default=None, help="The weight factor for the running average of the gradient squared (second moment)")
    parser.add_argument("--eps", type=float, default=None, help="The safety term to prevent the denominator from exploding")
    parser.add_argument("--absmaxgrad", type=float, default=None, help="The value at which to clip the gradient")
    parser.add_argument("--l2", type=float, default=None, help="The weight for the l2 normalization term in the loss function")
    parser.add_argument("--mag", type=float, default=None, help="The factor to multiply the intial matrices by to help with training")
    parser.add_argument("--seed", type=int, default=None, help="The PRNG seed")
    parser.add_argument("--try_load", type=bool, default=None, help="Whether to attempt loading a PMM state or start a new PMM")
    parser.add_argument("--k_num_sample", type=int, default=None, help="The number of energy levels to sample at each parameter value")
    parser.add_argument("--sample_Ls", type=utils.misc.parse_Ls, default=None, help="The sample Ls array in the format `--sample_Ls 3,5` or `--sample_Ls 3,16:20;5`")
    parser.add_argument("--epochs", type=int, default=None, help="The number of PMM training cycles")
    parser.add_argument("--store_loss", type=int, default=None, help="Store the loss after this many PMM cycles")
    parser.add_argument("--k_num_predict", type=int, default=None, help="The number of energy levels to predict at each parameter value")
    parser.add_argument("--predict_Ls", type=utils.misc.parse_Ls, default=None, help="The array of parameter values at which to predict energies")
    parser.add_argument("--plot_kwargs", type=utils.misc.parse_kwargs, default=None, help="The title and labels for the energy vs parameter plots")
    parser.add_argument("--save", type=bool, default=None, help="Whether or not to save the PMM state and plots")
    parser.add_argument("--show", type=bool, default=None, help="Whether to show the plots or not")
    args = parser.parse_args()
    args_dict = vars(args)

    for key in cfg["pmm_kwargs"]:
        if args_dict.get(key) is not None:
            cfg["pmm_kwargs"][key] = args_dict[key]

    for key in args_dict:
        if key not in cfg["pmm_kwargs"] and args_dict[key] is not None:
            cfg[key] = args_dict[key]
    run.run_pmm(**cfg)


if __name__=="__main__":
    main()
