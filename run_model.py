import time
from problems import *
from TrainPPOAgent import *
from utils.config import *
from utils.ortool import *
from utils.Misc import *

import warnings
warnings.filterwarnings('error', category=UnicodeWarning)

ortool_available = True


def run(args):
    device = torch.device("mps" if torch.backends.mps.is_available() and args.gpu else "cpu")
    print(device)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.verbose:
        verbose_print = print
    else:
        def verbose_print(*args, **kwargs):
            pass

    ## load DVRPSR problem

    verbose_print("Uploading data for training {}".format(args.iter_count * args.batch_size), end=" ", flush=True)
    train_data = torch.load("./data/train/{}/train.pth".format(args.customer_count))
    verbose_print("Done")

    verbose_print("Uploading data for testing {}".format(args.test_batch_size), end=" ", flush=True)
    # test data is not normalized
    test_data = torch.load("./data/test/{}/test.pth".format(args.customer_count))
    verbose_print("Done")

    if ortool_available:
        reference_routes = ortool_solve(test_data)
    else:
        reference_routes = None
        reference_costs = None
        verbose_print(" No reference to calculate optimality gap", end=" ", flush=True)

    test_data.normalize()

    ## Defining Environemnt for DVRPSR
    env = {"DSVRPTW": DSVRPTW_Environment}.get(args.problem)
    env_params_train = [
                        args.vehicle_count,
                        args.vehicle_capacity,
                        train_data.vehicle_speed,
                        args.pending_cost,
                        args.late_cost,
                        args.speed_var,
                        args.late_prob,
                        args.slow_down,
                        args.late_var]

    env_test = env(test_data, None, pending_cost=args.pending_cost, late_cost=args.late_cost)

    if reference_routes is not None:
        reference_costs = eval_apriori_routes(env_test, reference_routes, 10)
        print("Reference cost on test dataset {:5.2f} +- {:5.2f}".format(reference_costs.mean(),
                                                                         reference_costs.std()))
    env_test.nodes = env_test.nodes.to(device)
    customer_feature = 7  # customer and vehicle features are fixed
    vehicle_feature = 4

    print(args.customer_count)

    trainppo = TrainPPOAgent(customer_feature, vehicle_feature, args.customer_count, args.model_size,
                             args.encoder_layer, args.num_head, args.ff_size_actor, args.ff_size_critic, args.tanh_xplor,
                             args.greedy, args.learning_rate, args.ppo_epoch, args.batch_size,
                             args.entropy_value, args.epsilon_clip, args.epoch_count, args.timestep, args.max_grad_norm)

    # Checkpoints
    verbose_print("Creating Output directry...", end=" ", flush=True)
    args.output_dir = "./output/exp10_entropy{}_{}_{}".format(args.entropy_value,
                                                              args.customer_count,
                                                              time.strftime("%y%m%d")) if args.output_dir is None else args.output_dir

    os.makedirs(args.output_dir, exist_ok=True)
    write_config_file(args, os.path.join(args.output_dir, "args.json"))
    verbose_print("Create Output dir {}".format(args.output_dir), end=" ", flush=True)

    verbose_print('start training of DVRPSR model')
    trainppo.run_train(args, train_data, env, env_params_train, device, env_test, reference_costs)


if __name__ == "__main__":
    run(ParseArguments())
