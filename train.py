#!/usr/bin/env python3
"""
train.py ― launch GCBF / GCBF+ training exactly as in the original repo.

Key points
----------
* Saves the **whole argparse.Namespace** to config.yaml, so `test.py`
  can reload it and still use dot-attribute access (config.env, …).
* No parser changes: only the flags that existed upstream.
* No special handling for MPC parameters – the new environment works
  with its internal defaults, or you can override via settings.yaml.
"""

import argparse
import datetime
import os
import ipdb
import numpy as np
import wandb
import yaml

from gcbfplus.algo            import make_algo
from gcbfplus.env             import make_env
from gcbfplus.trainer.trainer import Trainer
from gcbfplus.trainer.utils   import is_connected


# --------------------------------------------------------------------------- #
#                                training loop                                #
# --------------------------------------------------------------------------- #
def train(args):
    print(f"> Running train.py {args}")

    # deterministic JAX memory usage
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # offline W&B if no internet or debug
    if not is_connected() or args.debug:
        os.environ["WANDB_MODE"] = "offline"
    if args.debug:
        os.environ["JAX_DISABLE_JIT"] = "True"

    np.random.seed(args.seed)

    # ------------------------- build environments ------------------------- #
    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        area_size=args.area_size,
    )
    env_test = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        area_size=args.area_size,
    )

    # --------------------------- build algorithm ------------------------- #
    algo = make_algo(
        algo=args.algo,
        env=env,
        node_dim   = env.node_dim,
        edge_dim   = env.edge_dim,
        state_dim  = env.state_dim,
        action_dim = env.action_dim,
        n_agents   = env.num_agents,
        gnn_layers = args.gnn_layers,
        batch_size = 256,
        buffer_size= args.buffer_size,
        horizon    = args.horizon,
        lr_actor   = args.lr_actor,
        lr_cbf     = args.lr_cbf,
        alpha      = args.alpha,
        eps        = 0.02,
        inner_epoch= 8,
        loss_action_coef = args.loss_action_coef,
        loss_unsafe_coef = args.loss_unsafe_coef,
        loss_safe_coef   = args.loss_safe_coef,
        loss_h_dot_coef  = args.loss_h_dot_coef,
        max_grad_norm    = 2.0,
        seed             = args.seed,
    )

    # ----------------------------- logging -------------------------------- #
    ts       = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    base_dir = f"{args.log_dir}/{args.env}/{args.algo}"
    os.makedirs(base_dir, exist_ok=True)
    log_dir  = f"{base_dir}/seed{args.seed}_{ts}"
    run_name = args.name or f"{args.algo}_{args.env}_{ts}"

    trainer_cfg = dict(
        run_name       = run_name,
        training_steps = args.steps,
        eval_interval  = args.eval_interval,
        eval_epi       = args.eval_epi,
        save_interval  = args.save_interval,
    )

    trainer = Trainer(
        env         = env,
        env_test    = env_test,
        algo        = algo,
        log_dir     = log_dir,
        n_env_train = args.n_env_train,
        n_env_test  = args.n_env_test,
        seed        = args.seed,
        params      = trainer_cfg,
        save_log    = not args.debug,
    )

    # save full Namespace + algo config
    wandb.config.update(vars(args))
    wandb.config.update(algo.config)
    if not args.debug:
        with open(f"{log_dir}/config.yaml", "w") as f:
            yaml.dump(args,        f)   # <- full Namespace (original behaviour)
            yaml.dump(algo.config, f)

    # start training
    trainer.train()


# --------------------------------------------------------------------------- #
#                               CLI interface                                 #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()

    # generic experiment flags
    parser.add_argument("-n",   "--num-agents", type=int,   default=8)
    parser.add_argument("--algo", type=str,     default="gcbf+")
    parser.add_argument("--env",  type=str,     default="DoubleIntegrator")
    parser.add_argument("--seed", type=int,     default=0)
    parser.add_argument("--steps",type=int,     default=1000)
    parser.add_argument("--name", type=str,     default=None)
    parser.add_argument("--debug",action="store_true", default=False)

    # environment specifics
    parser.add_argument("--obs",      type=int,   default=None)
    parser.add_argument("--n-rays",   type=int,   default=32)
    parser.add_argument("--area-size",type=float, required=True)

    # network & optimisation
    parser.add_argument("--gnn-layers", type=int,   default=1)
    parser.add_argument("--alpha",      type=float, default=1.0)
    parser.add_argument("--horizon",    type=int,   default=32)
    parser.add_argument("--lr-actor",   type=float, default=3e-5)
    parser.add_argument("--lr-cbf",     type=float, default=3e-5)

    # loss weights
    parser.add_argument("--loss-action-coef", type=float, default=1e-4)
    parser.add_argument("--loss-unsafe-coef", type=float, default=1.0)
    parser.add_argument("--loss-safe-coef",   type=float, default=1.0)
    parser.add_argument("--loss-h-dot-coef",  type=float, default=0.01)

    # replay buffer
    parser.add_argument("--buffer-size", type=int, default=512)

    # trainer runtime
    parser.add_argument("--n-env-train",   type=int, default=16)
    parser.add_argument("--n-env-test",    type=int, default=32)
    parser.add_argument("--log-dir",       type=str, default="./logs")
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--eval-epi",      type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=10)

    args = parser.parse_args()
    with ipdb.launch_ipdb_on_exception():
        train(args)


if __name__ == "__main__":
    main()
