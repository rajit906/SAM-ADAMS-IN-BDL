import argparse

def get_config():
    parser = argparse.ArgumentParser()

    # Main options
    parser.add_argument("--sampler", type=str, default="sgld")
    parser.add_argument("--prior", type=str, default="NA")
    parser.add_argument("--prior_scale", type=float, default=1.0)
    parser.add_argument("--bias_prior_scale", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0) # Set to 1.0 for proper SGLD

    # SA-SGLD specific
    parser.add_argument("--alpha", type=float, default=50.0)
    parser.add_argument("--m", type=float, default=0.001)
    parser.add_argument("--M", type=float, default=1.0)
    parser.add_argument("--r", type=float, default=0.25)
    parser.add_argument("--s", type=float, default=2.0)
    parser.add_argument("--init_z", type=float, default=0.0)
    parser.add_argument("--Omega", type=int, default=int(5e4))

    # Model/training
    parser.add_argument("--hidden", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--burnin_batches", type=int, default=200*500) # 500 batch per epoch. 1000 is 2 epochs. 
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--num_runs", type=int, default=3)

    # System
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    return vars(args)