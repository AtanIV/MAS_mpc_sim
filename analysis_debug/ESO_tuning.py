#!/usr/bin/env python3
"""
ESO tuning / replay script for v4 pipeline.

IMPORTANT:
- Logged control is FORCE.
- We convert FORCE → ACCELERATION using a = F / m.
- First-order disturbance is applied after clipping velocity.
- Plant uses Euler updates:
    x = x + v dt
    v = v + a dt
"""

import argparse
import ast
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
#                         ESO non-linear core functions                        #
# --------------------------------------------------------------------------- #

def fal(e: float, alpha: float, delta: float) -> float:
    abs_e = np.abs(e)
    if abs_e <= delta:
        return e / (delta ** (1.0 - alpha))
    else:
        return (abs_e ** alpha) * np.sign(e)

def nonlinear_eso_update(
    y_measurement: float,
    u_accel: float,
    z_prev: np.ndarray,
    h: float,
    beta01: float,
    beta02: float,
    beta03: float,
    alpha1: float,
    alpha2: float,
    delta: float,
    b0: float,
):
    """
    Discrete nonlinear ESO update.
    State: z = [pos_est, vel_est, dist_est]
    """
    e = z_prev[0] - y_measurement   # ?
    fe = fal(e, alpha1, delta)
    fe1 = fal(e, alpha2, delta)

    z1 = z_prev[0] + h * z_prev[1] - beta01 * e
    z2 = z_prev[1] + h * (z_prev[2] + b0 * u_accel) - beta02 * fe
    z3 = z_prev[2] - beta03 * fe1

    return np.array([z1, z2, z3], float), e


# --------------------------------------------------------------------------- #
#                         Beta computation (user formulas)                    #
# --------------------------------------------------------------------------- #

def compute_betas(dt: float) -> Tuple[float, float, float]:
    """
    User-selected ESO gains:
        β01 = 10
        β02 = 0.7 / (2 * sqrt(dt))
        β03 = 0.01 * (2 / (25 * dt^1.2))
    """
    # Original betas
    # beta01 = 1.0 * 0.35
    # beta02 = 1.0 / (2.0 * dt ** 0.5) * 0.4
    # beta03 = 2.0 / (25.0 * dt ** 1.2) * 0.6

    # # Beta tuning 01
    # beta01 = 1.0 * 0.1
    # beta02 = 1.0 / (2.0 * dt ** 0.5) * 0.1
    # beta03 = 2.0 / (25.0 * dt ** 1.2) * 0.01

    # beta01 = 1.0 * 0.5
    # beta02 = 1.0 / (2.0 * dt ** 0.5) * 0.2
    # beta03 = 2.0 / (25.0 * dt ** 1.2) * 0.2

    # beta01 = 1.0 * 0.3
    # beta02 = 1.0 / (2.0 * dt ** 0.5) * 0.17
    # beta03 = 2.0 / (25.0 * dt ** 1.2) * 0.15

    # beta01 = 1.0 * 0.25
    # beta02 = 1.0 / (2.0 * dt ** 0.5) * 0.17
    # beta03 = 2.0 / (25.0 * dt ** 1.2) * 0.17

    # beta01 = 1.0 * 0.26
    # beta02 = 1.0 / (2.0 * dt ** 0.5) * 0.15
    # beta03 = 2.0 / (25.0 * dt ** 1.2) * 0.18

    # 1205-1929
    beta01 = 1.0 * 0.35
    beta02 = 1.0 / (2.0 * dt ** 0.5) * 0.2
    beta03 = 2.0 / (25.0 * dt ** 1.2) * 0.23

    # 1205-1919
    # beta01 = 1.0 * 0.35
    # beta02 = 1.0 / (2.0 * dt ** 0.5) * 0.4
    # beta03 = 2.0 / (25.0 * dt ** 1.2) * 0.6

    # beta01 = 1.0 * 0.35
    # beta02 = 1.0 / (2.0 * dt ** 0.5) * 0.6
    # beta03 = 2.0 / (25.0 * dt ** 1.2) * 0.7
    return beta01, beta02, beta03


# --------------------------------------------------------------------------- #
#                         Robust control_input parsing                        #
# --------------------------------------------------------------------------- #

def extract_axis_force(a0, axis: str):
    """
    Extract applied control for axis x or y from logged MPC control_input.

    Format:
        Case A (MPC success):
            [[Fx0, Fy0], [Fx1, Fy1], [Fx2, Fy2]]
            → applied = [Fx0, Fy0]

        Case B (fallback used):
            [Fx0, Fy0]
            → applied = [Fx0, Fy0]
    """
    axis_idx = 0 if axis == "x" else 1

    # Convert strings to Python objects
    if isinstance(a0, str):
        try:
            a0 = ast.literal_eval(a0)
        except Exception:
            return np.nan

    arr = np.array(a0, dtype=float)

    # Case B: [Fx, Fy]
    if arr.ndim == 1:
        if arr.size >= 2:
            return float(arr[axis_idx])
        return float(arr[0])

    # Case A: MPC horizon → (H,2)
    if arr.ndim == 2:
        # applied control is FIRST element
        first = arr[0]
        if first.size >= 2:
            return float(first[axis_idx])
        return float(first[0])

    # Anything weird → flatten and fallback to first two numbers
    flat = arr.reshape(-1)
    if flat.size >= 2:
        return float(flat[axis_idx])
    return float(flat[0])



# --------------------------------------------------------------------------- #
#                            Data loading / dataset                           #
# --------------------------------------------------------------------------- #

@dataclass
class ESOTuningDataset:
    """Dataset for offline ESO replay."""

    true_pos: np.ndarray
    true_vel: np.ndarray
    true_dist: np.ndarray

    logged_eso_pos: np.ndarray
    logged_eso_vel: np.ndarray
    logged_eso_dist: np.ndarray

    control_force: np.ndarray
    control_accel: np.ndarray

    logged_comp_accel: np.ndarray

    steps: np.ndarray
    time: np.ndarray

    dt_plant: float
    dt_eso: float
    vel_limit: float
    mass: float

    @staticmethod
    def from_logs(
        log_dir: str,
        agent_id: int,
        episode: int,
        axis: str = "x",
        dt_plant: float = 0.03,
        dt_eso: float = 0.01,
        vel_limit: float = 0.5,
        mass: float = 0.1,
    ):
        # --- Load ESO file --------------------------------------------------
        df_eso = pd.read_csv(f"{log_dir}/eso_data.csv")
        for col, val in [("agent_id", agent_id), ("episode", episode), ("axis", axis)]:
            if col in df_eso.columns:
                df_eso = df_eso[df_eso[col] == val]

        df_eso = df_eso.sort_values("step")
        if df_eso.empty:
            raise ValueError("No matching ESO rows found.")

        # --- Load agent control logs ----------------------------------------
        df_agent = pd.read_csv(f"{log_dir}/agent_step_logs/agent_{agent_id:02d}.csv")
        df_agent = df_agent.sort_values("step")

        # --- Load MPC status logs (contains compensated actions) ------------
        df_mpc = pd.read_csv(f"{log_dir}/mpc_status.csv")
        # filter for this episode/agent if columns exist
        for col, val in [("agent_id", agent_id), ("episode", episode)]:
            if col in df_mpc.columns:
                df_mpc = df_mpc[df_mpc[col] == val]

        df_mpc = df_mpc.sort_values("step")


        # --- Merge on step -----------------------------------
        df = (
            df_eso
            .merge(df_agent, on="step", how="inner")
            .merge(df_mpc, on="step", how="left")
            .sort_values("step")
        )

        steps = df["step"].to_numpy(int)
        time = steps * dt_plant

        # Ground truth states
        true_pos = df["true_pos"].to_numpy(float)
        true_vel = df["true_vel"].to_numpy(float)
        true_dist = df["true_dist"].to_numpy(float)

        # Logged ESO
        logged_eso_pos = df["eso_pos"].to_numpy(float)
        logged_eso_vel = df["eso_vel"].to_numpy(float)
        logged_eso_dist = df["eso_dist"].to_numpy(float)

        # Extract forces robustly
        control_force = df["control_input"].apply(
            lambda x: extract_axis_force(x, axis)
        ).to_numpy(float)
        control_accel = control_force / mass

        # Logged compensated control (from MPC logger)
        # mpc_status.csv columns: comp_x, comp_y (already float)
        comp_col = "compensated_x" if axis == "x" else "compensated_y"
        if comp_col in df.columns:
            logged_comp_accel = df[comp_col].to_numpy(float)
        else:
            # fallback if missing
            logged_comp_accel = np.zeros_like(control_force)


        return ESOTuningDataset(
            true_pos=true_pos,
            true_vel=true_vel,
            true_dist=true_dist,
            logged_eso_pos=logged_eso_pos,
            logged_eso_vel=logged_eso_vel,
            logged_eso_dist=logged_eso_dist,
            control_force=control_force,
            control_accel=control_accel,
            logged_comp_accel=logged_comp_accel,
            steps=steps,
            time=time,
            dt_plant=dt_plant,
            dt_eso=dt_eso,
            vel_limit=vel_limit,
            mass=mass,
        )


# --------------------------------------------------------------------------- #
#                      Plant + ESO replay simulator (1D)                      #
# --------------------------------------------------------------------------- #

@dataclass
class ESOTuningSimulator:

    dataset: ESOTuningDataset
    alpha1: float = 0.5
    alpha2: float = 0.25
    # alpha2: float = 0.5
    b0: float = 1.0
    use_disturbances: bool = True

    sim_pos: np.ndarray = None
    sim_vel: np.ndarray = None
    sim_u_raw: np.ndarray = None
    sim_u_comp: np.ndarray = None
    sim_eso_pos: np.ndarray = None
    sim_eso_vel: np.ndarray = None
    sim_eso_dist: np.ndarray = None
    eso_errors: np.ndarray = None

    beta01: float = None
    beta02: float = None
    beta03: float = None

    def simulate_plant_step(self, x, v, u_accel, d):
        dt = self.dataset.dt_plant
        vel_lim = self.dataset.vel_limit

        # Euler step
        x_new = x + v * dt
        v_new = v + u_accel * dt

        # Clip velocity (env limit)
        # v_new = np.clip(v_new, -vel_lim, vel_lim)

        # Disturbance added AFTER clipping
        if self.use_disturbances:
            v_new = v_new + d * dt

        return x_new, v_new

    def run(self):
        data = self.dataset
        n = len(data.true_pos)

        dt_plant = data.dt_plant
        dt_eso = data.dt_eso

        self.beta01, self.beta02, self.beta03 = compute_betas(dt_plant)

        eso_per_plant = int(round(dt_plant / dt_eso))
        if eso_per_plant < 1:
            eso_per_plant = 1

        sim_pos = np.zeros(n)
        sim_vel = np.zeros(n)
        sim_u_raw = np.zeros(n)
        sim_u_comp = np.zeros(n)

        sim_eso_pos = np.zeros(n)
        sim_eso_vel = np.zeros(n)
        sim_eso_dist = np.zeros(n)
        eso_errors = np.zeros(n)

        # ---------- init from truth at step 0 ----------
        sim_pos[0] = data.true_pos[0]
        sim_vel[0] = data.true_vel[0]
        z = np.array([sim_pos[0], sim_vel[0], 0.0], float)

        sim_eso_pos[0], sim_eso_vel[0], sim_eso_dist[0] = z

        u_prev_for_eso = 0.0

        for k in range(0, n - 1):

            # ---------------- ESO inner loop ----------------
            y_meas = data.true_pos[k]

            # Uses previous applied accel (u_{k-1}); for k=0 it's 0
            e_last = 0.0
            for _ in range(eso_per_plant):
                z, e_last = nonlinear_eso_update(
                    y_measurement=y_meas,
                    u_accel=u_prev_for_eso,
                    z_prev=z,
                    h=dt_eso,
                    beta01=self.beta01,
                    beta02=self.beta02,
                    beta03=self.beta03,
                    alpha1=self.alpha1,
                    alpha2=self.alpha2,
                    delta=dt_eso,
                    b0=self.b0,
                )
                # z[1] = np.clip(z[1], -data.vel_limit, data.vel_limit)

            # Store ESO estimates aligned to step k (start-of-step estimate)
            sim_eso_pos[k], sim_eso_vel[k], sim_eso_dist[k] = z
            eso_errors[k] = e_last

            # ---------------- CURRENT plant step control ----------------
            u_raw = data.control_accel[k]  # accel applied over [k -> k+1)

            # Current ESO disturbance estimate (accel units)
            d_hat = z[2]

            # Compensate CURRENT control
            u_comp = u_raw - (d_hat / self.b0)
            # u_comp = u_raw

            sim_u_raw[k] = u_raw
            sim_u_comp[k] = u_comp

            # True disturbance for plant sim
            d_true = data.true_dist[k] if self.use_disturbances else 0.0

            # ---------------- PLANT update ----------------
            x_new, v_new = self.simulate_plant_step(
                sim_pos[k], sim_vel[k], u_comp, d_true
            )
            sim_pos[k + 1], sim_vel[k + 1] = x_new, v_new

            # Update prev input for ESO for next step
            u_prev_for_eso = u_comp

        sim_eso_pos[-1], sim_eso_vel[-1], sim_eso_dist[-1] = z
        eso_errors[-1] = eso_errors[-2]

        self.sim_pos = sim_pos
        self.sim_vel = sim_vel
        self.sim_u_raw = sim_u_raw
        self.sim_u_comp = sim_u_comp
        self.sim_eso_pos = sim_eso_pos
        self.sim_eso_vel = sim_eso_vel
        self.sim_eso_dist = sim_eso_dist
        self.eso_errors = eso_errors


# --------------------------------------------------------------------------- #
#                         ANALYSIS & PLOTTING                       #
# --------------------------------------------------------------------------- #

def compute_rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def analyze_results(dataset, sim):
    res = {}
    res["plant_pos_rmse"] = compute_rmse(sim.sim_pos, dataset.true_pos)
    res["plant_vel_rmse"] = compute_rmse(sim.sim_vel, dataset.true_vel)
    res["eso_pos_rmse_logged"] = compute_rmse(sim.sim_eso_pos, dataset.logged_eso_pos)
    res["eso_vel_rmse_logged"] = compute_rmse(sim.sim_eso_vel, dataset.logged_eso_vel)
    res["eso_dist_rmse_logged"] = compute_rmse(sim.sim_eso_dist, dataset.logged_eso_dist)
    res["eso_pos_rmse_true"] = compute_rmse(sim.sim_eso_pos, dataset.true_pos)
    res["eso_vel_rmse_true"] = compute_rmse(sim.sim_eso_vel, dataset.true_vel)
    res["eso_dist_rmse_true"] = compute_rmse(sim.sim_eso_dist, dataset.true_dist)
    return res


def plot_comparison(dataset, sim, metrics, axis, agent_id, episode, save_path=None):

    t = dataset.time
    beta01, beta02, beta03 = sim.beta01, sim.beta02, sim.beta03

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"ESO Tuning Replay (agent {agent_id}, episode {episode}, axis {axis})\n"
        f"β01={beta01:.3g}, β02={beta02:.3g}, β03={beta03:.3g}",
        fontsize=14,
        fontweight="bold",
    )

    # Position
    ax = axes[0, 0]
    ax.plot(t, dataset.true_pos, "k-", lw=2, label="True position")
    # ax.plot(t, sim.sim_pos, "r--", lw=2, label="Sim plant position")
    ax.plot(t, dataset.logged_eso_pos, "C0-.", lw=2, label="Logged ESO pos")
    # ax.plot(t, sim.sim_eso_pos, color="orange", ls=":", lw=2, label="Sim ESO pos")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"Position (plant RMSE={metrics['plant_pos_rmse']:.4f}, "
        f"ESO→true RMSE={metrics['eso_pos_rmse_true']:.4f})"
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position")
    ax.legend()

    # Velocity
    ax = axes[0, 1]
    ax.plot(t, dataset.true_vel, "k-", lw=2, label="True velocity")
    # ax.plot(t, sim.sim_vel, "r--", lw=2, label="Sim plant velocity")
    ax.plot(t, dataset.logged_eso_vel, "C0-.", lw=2, label="Logged ESO vel")
    # ax.plot(t, sim.sim_eso_vel, color="orange", ls=":", lw=2, label="Sim ESO vel")
    ax.axhline(dataset.vel_limit, color="gray", ls="--", alpha=0.5)
    ax.axhline(-dataset.vel_limit, color="gray", ls="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"Velocity (plant RMSE={metrics['plant_vel_rmse']:.4f}, "
        f"ESO→true RMSE={metrics['eso_vel_rmse_true']:.4f})"
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity")
    ax.legend()

    # Disturbance
    ax = axes[1, 0]
    ax.plot(t, dataset.true_dist, "k-", lw=2, label="True disturbance")
    ax.plot(t, dataset.logged_eso_dist, "C0-.", lw=2, label="Logged ESO dist")
    # ax.plot(t, sim.sim_eso_dist, color="orange", ls=":", lw=2, label="Sim ESO dist")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        "Disturbance "
        f"(ESO→true RMSE={metrics['eso_dist_rmse_true']:.4f}, "
        f"ESO→logged RMSE={metrics['eso_dist_rmse_logged']:.4f})"
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Disturbance (accel)")
    ax.legend()

    # Control
    ax = axes[1, 1]

    # Raw logged control (already in dataset)
    ax.plot(t, dataset.control_accel, "C3--", lw=2, label="Raw control accel (log)")

    # Logged compensated control from mpc_status.csv
    ax.plot(t, dataset.logged_comp_accel, "C2-", lw=2, label="Compensated accel (log)")

    # Offline compensated control from replay ESO
    # ax.plot(t, sim.sim_u_comp, color="orange", ls=":", lw=2, label="Compensated accel (offline)")

    ax.grid(True, alpha=0.3)
    ax.set_title("Control Input (Raw vs Logged Comp vs Offline Comp)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Acceleration")
    ax.legend()


    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot → {save_path}")

    plt.show()


# --------------------------------------------------------------------------- #
#                                   CLI                                       #
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="ESO tuning / replay for v4 pipeline.")
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--agent-id", type=int, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--axis", type=str, choices=["x", "y"], default="x")

    parser.add_argument("--mass", type=float, default=0.1)
    parser.add_argument("--dt-plant", type=float, default=0.03)
    parser.add_argument("--dt-eso", type=float, default=0.01)
    parser.add_argument("--vel-limit", type=float, default=0.5)

    parser.add_argument("--no-disturbances", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--save-plot", type=str, default=None)

    return parser.parse_args()



# --------------------------------------------------------------------------- #
#                                     MAIN                                    #
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    dataset = ESOTuningDataset.from_logs(
        log_dir=args.log_dir,
        agent_id=args.agent_id,
        episode=args.episode,
        axis=args.axis,
        dt_plant=args.dt_plant,
        dt_eso=args.dt_eso,
        vel_limit=args.vel_limit,
        mass=args.mass,
    )

    sim = ESOTuningSimulator(
        dataset=dataset,
        use_disturbances=not args.no_disturbances,
    )
    sim.run()

    metrics = analyze_results(dataset, sim)

    print("\n--- RMSE diagnostics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
    print("-------------------------\n")

    if not args.no_plot:
        plot_comparison(
            dataset,
            sim,
            metrics,
            axis=args.axis,
            agent_id=args.agent_id,
            episode=args.episode,
            save_path=args.save_plot,
        )


if __name__ == "__main__":
    main()
