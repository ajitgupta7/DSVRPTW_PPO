import argparse
from argparse import ArgumentParser
import json


def write_config_file(args, output_file):
    with open(output_file, 'w') as f:
        json.dump(vars(args), f, indent=4)


def ParseArguments(argv=None):
    parser = ArgumentParser()
    parser = argparse.ArgumentParser(description="Reinforcement Learning for Dynamic VRP with Stochastic Requests")

    parser.add_argument("--config-file", "-f", type=str, default=None,
                        help="configuration file")
    parser.add_argument("--verbose", "-v", action='store_true', default=True,
                        help="Showing information while processing")
    parser.add_argument("--gpu", action='store_true', default=True,
                        help="Use GPU to run the model")
    parser.add_argument("--seed", type=int, default=None, help="seed to regenerate same result")

    ### Data related arguments

    parser.add_argument_group("Data Generation for DSVRPTW")
    parser.add_argument("--problem", "-p", type=str, default='DSVRPTW',
                        help="problem to solve is DSVRPTW")
    parser.add_argument("--customer-count", "-n", type=int, default=50)
    parser.add_argument("--vehicle-count", "-m", type=int, default=10,
                        help='number of vehicles for DSVRPTW')
    parser.add_argument("--vehicle-capacity", type=int, default=200,
                        help='capacities of vehicles for DSVRPTW')
    parser.add_argument("--vehicle-speed", type=int, default=1,
                        help='speed of vehicle for DSVRPTW')
    parser.add_argument("--horizon", type=int, default=480,
                        help='Working time for DSVRPTW in minutes')
    parser.add_argument("--loc-range", type=int, nargs=2, default=(0, 101))
    parser.add_argument("--dem-range", type=int, nargs=2, default=(5, 41))
    parser.add_argument("--dur-range", type=int, nargs=2, default=(10, 31))
    parser.add_argument("--tw-ratio", type=float, nargs='*', default=(0.5))
    parser.add_argument("--tw-range", type=int, nargs=2, default=(30, 91))
    parser.add_argument("--dod", type=float, nargs='*', default=(0.0, 0.25, 0.5, 0.75))
    parser.add_argument("--d-early-ratio", type=float, nargs='*', default=(0.5))

    # Environment related arguments
    parser.add_argument_group(" Environment for DSVRPTW")
    parser.add_argument("--pending-cost", type=float, default=2)
    parser.add_argument("--late-cost", type=float, default=2)
    parser.add_argument("--speed-var", type=float, default=0.1)
    parser.add_argument("--late-prob", type=float, default=0.05)
    parser.add_argument("--slow-down", type=float, default=0.5)
    parser.add_argument("--late-var", type=float, default=0.2)

    parser.add_argument_group(" Graph Attention models ")
    parser.add_argument("--model-size", type=int, default=128,
                        help=" Size of for attention models")
    parser.add_argument("--encoder-layer", type=int, default=3,
                        help='Number of Encoder Layers')
    parser.add_argument("--num-head", type=int, default=8,
                        help='Number of heads in MultiHeadAttention modules')
    parser.add_argument("--ff-size-actor", type=int, default=512,
                        help=" Size of fully connected Feed Forward Networks")
    parser.add_argument("--ff-size-critic", type=int, default=512,
                        help=" Size of fully connected Feed Forward Networks")
    parser.add_argument("--tanh-xplor", type=int, default=10)
    parser.add_argument("--edge_embedding_dim", type=int, default=64,
                        help='Edge embedding dimention for edge attributes')

    # PPO Agent Training related arguments
    parser.add_argument_group(" Training PPO Agnet ")
    parser.add_argument("--greedy", action='store_true', default=False,
                        help='weather to use greedy or smapling')
    parser.add_argument("--learning-rate", type=int, default=1e-4,
                        help='Learning rate for PPO agent')
    parser.add_argument("--ppo-epoch", type=int, default=2,
                        help='Epoch for PPO to run the sample and evaluate')
    parser.add_argument("--entropy-value", type=int, default=0.01)
    parser.add_argument("--epsilon-clip", type=int, default=0.2)
    parser.add_argument("--timestep", type=int, default=1)

    parser.add_argument("--epoch-count", "-e", type=int, default=50)
    parser.add_argument("--iter-count", "-i", type=int, default=100)
    parser.add_argument("--batch-size", "-b", type=int, default=256)
    parser.add_argument("--rate-decay", '-d', type=float, default=0.96)
    parser.add_argument("--max-grad-norm", type=float, default=2)
    parser.add_argument("--grad-norm-decay", type=float, default=None)

    ### Testing Related arguments
    parser.add_argument("--test-batch-size", type=int, default=128)

    ### Saving paramters
    parser.add_argument_group("Checkpointing")
    parser.add_argument("--output-dir", "-o", type=str, default=None)
    parser.add_argument("--checkpoint-period", "-c", type=int, default=1)
    parser.add_argument("--resume-state", type=str, default=None)

    args = parser.parse_args(argv)
    if args.config_file is not None:
        with open(args.config_file) as f:
            parser.set_defaults(**json.load(f))

    return parser.parse_args(argv)
