# src/utils/config.py
import sys
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.simulation.dgp0 import Tier0Config


def load_tier0_config(path: str) -> tuple[Tier0Config, dict]:
    """
    Load YAML config and return:
    - Tier0Config object
    - raw dictionary (for other values like n_markets)
    """
    with open(Path(path), "r") as f:
        cfg_dict = yaml.safe_load(f)

    sim = cfg_dict["simulation"]
    cost = cfg_dict["cost_process"]
    noise = cfg_dict["price_noise"]
    regime = cfg_dict["regime_parameters"]
    markov = cfg_dict["markov"]

    tier_cfg = Tier0Config(
        T=sim["T"],
        burn_in=sim["burn_in"],

        rho_c_low=cost["rho_c_low"],
        rho_c_high=cost["rho_c_high"],
        sigma_c_low=cost["sigma_c_low"],
        sigma_c_high=cost["sigma_c_high"],
        jump_prob_low=cost["jump_prob_low"],
        jump_prob_high=cost["jump_prob_high"],
        sigma_J_low=cost["sigma_J_low"],
        sigma_J_high=cost["sigma_J_high"],

        sigma_p_low=noise["sigma_p_low"],
        sigma_p_high=noise["sigma_p_high"],

        beta_C=tuple(regime["beta_C"]),
        beta_T=tuple(regime["beta_T"]),
        beta_K=tuple(regime["beta_K"]),

        kappa_C=tuple(regime["kappa_C"]),
        kappa_T=tuple(regime["kappa_T"]),
        kappa_K=tuple(regime["kappa_K"]),

        stay_C=markov["stay_C"],
        stay_T=markov["stay_T"],
        stay_K=markov["stay_K"],
    )

    return tier_cfg, cfg_dict