import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--data", type=str)
parser.add_argument("--inference", type=str)
parser.add_argument("--lrs", nargs="+", type=float)
parser.add_argument("--trials", type=int)
parser.add_argument("--width", type=int)
parser.add_argument("--temperature", type=float)
parser.add_argument("--sampling_decay", type=str, default="cosine")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--other_args", type=str, default=None)
parser.add_argument("--other_info", type=str, default=None)
parser.add_argument("--prior", type=str, default="gaussian")
parser.add_argument("--save_samples", type=str, default="False")
parser.add_argument("--cycles", type=int, default=5)
parser.add_argument("--burnin_batches", type=int, default=300)
parser.add_argument("--epochs_per_cycle", type=int, default=20)

args = parser.parse_args()


def run_exp(
    model,
    data,
    inference,
    lrs,
    trials,
    temperature,
    sampling_decay,
    batch_size,
    width,
    prior,
    save_samples,
    burnin_batches,
    other_args,
    cycles,
    epochs_per_cycle,
):
    if save_samples == "True":
        samples_str = "_with_samples"
    else:
        samples_str = ""
    if model == "classificationdensenet":
        parent_dir = f"{data}_{width}_{prior}_{temperature}_{batch_size}_{sampling_decay}_{cycles*epochs_per_cycle}_{burnin_batches}{samples_str}"
        bias_prior = f" bias_prior={prior} "
    else:
        parent_dir = f"{data}_{model}_{prior}_{temperature}_{batch_size}_{sampling_decay}_{cycles*epochs_per_cycle}_{burnin_batches}{samples_str}"
        bias_prior = " "
    for lr in lrs:
        for _ in range(trials):
            if other_args is not None:
                os.system(
                    f'python bnn_performance/experiments/train_bnn.py with data={data} model={model} weight_prior={prior}{bias_prior}inference={inference} lr={lr} weight_scale=1.41 sampling_decay={sampling_decay} batch_size={batch_size} width={width} temperature={temperature} save_samples={save_samples} progressbar=False log_dir=bnn_performance/results/{parent_dir}/{inference}_{lr}_{"_".join(other_args.split())} batchnorm=True uncertainty_plot=False cycles={cycles} epochs_per_cycle={epochs_per_cycle} burnin_batches={burnin_batches} {other_args}'
                )
            else:
                os.system(
                    f"python bnn_performance/experiments/train_bnn.py with data={data} model={model} weight_prior={prior}{bias_prior}inference={inference} lr={lr} weight_scale=1.41 sampling_decay={sampling_decay} batch_size={batch_size} width={width} temperature={temperature} save_samples={save_samples} progressbar=False log_dir=bnn_performance/results/{parent_dir}/{inference}_{lr} batchnorm=True uncertainty_plot=False cycles={cycles} epochs_per_cycle={epochs_per_cycle} burnin_batches={burnin_batches}"
                )


run_exp(
    model=args.model,
    data=args.data,
    inference=args.inference,
    lrs=args.lrs,
    trials=args.trials,
    temperature=args.temperature,
    sampling_decay=args.sampling_decay,
    batch_size=args.batch_size,
    width=args.width,
    prior=args.prior,
    save_samples=args.save_samples,
    other_args=args.other_args,
    cycles=args.cycles,
    burnin_batches=args.burnin_batches,
    epochs_per_cycle=args.epochs_per_cycle,
)
