from __future__ import annotations

import numpy as np

from .graph_utils import adjacency_from_radius


def _dbm_to_mw(value_dbm: np.ndarray | float) -> np.ndarray | float:
    return 10.0 ** (np.asarray(value_dbm) / 10.0)


def _mw_to_dbm(value_mw: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(np.asarray(value_mw), 1e-30))


def _rayleigh_power_gain(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    amplitude = rng.rayleigh(scale=1.0 / np.sqrt(2.0), size=shape)
    return np.maximum(amplitude * amplitude, 1e-12)


def _nakagami_power_gain(rng: np.random.Generator, shape: tuple[int, ...], m: float) -> np.ndarray:
    # Unit-mean Nakagami-m power gain. m=1 is Rayleigh fading.
    return np.maximum(rng.gamma(shape=max(float(m), 1e-6), scale=1.0 / max(float(m), 1e-6), size=shape), 1e-12)


def received_power_dbm(
    distances: np.ndarray,
    rng: np.random.Generator,
    radio_cfg: dict,
) -> np.ndarray:
    """Per-direction received power with path loss, shadowing, fading, and chain asymmetry."""
    n_nodes = distances.shape[0]
    d0 = float(radio_cfg.get("reference_distance_m", 1.0))
    d = np.maximum(distances, d0)
    path_loss_exp = float(radio_cfg.get("path_loss_exponent", 2.7))
    pl_d0_db = float(radio_cfg.get("path_loss_at_d0_db", 40.0))
    shadow_sigma = float(radio_cfg.get("shadowing_sigma_db", 4.0))
    tx_power_dbm = float(radio_cfg.get("tx_power_dbm", 20.0))
    tx_gain_db = float(radio_cfg.get("tx_antenna_gain_dbi", 5.0))
    rx_gain_db = float(radio_cfg.get("rx_antenna_gain_dbi", 5.0))
    system_loss_db = float(radio_cfg.get("system_loss_db", 2.0))
    link_margin_db = float(radio_cfg.get("link_margin_db", 10.0))
    chain_sigma = float(radio_cfg.get("chain_asymmetry_sigma_db", 1.0))
    fading = str(radio_cfg.get("fading", "rayleigh")).lower()
    nakagami_m = float(radio_cfg.get("nakagami_m", 1.0))

    shadowing = rng.normal(0.0, shadow_sigma, size=(n_nodes, n_nodes))
    tx_chain = rng.normal(0.0, chain_sigma, size=(n_nodes, 1))
    rx_chain = rng.normal(0.0, chain_sigma, size=(1, n_nodes))
    path_loss = pl_d0_db + 10.0 * path_loss_exp * np.log10(d / d0) + shadowing
    if fading == "nakagami":
        fading_gain = _nakagami_power_gain(rng, (n_nodes, n_nodes), nakagami_m)
    else:
        fading_gain = _rayleigh_power_gain(rng, (n_nodes, n_nodes))
    fading_db = 10.0 * np.log10(fading_gain)
    rx_power = (
        tx_power_dbm
        + tx_gain_db
        + rx_gain_db
        + tx_chain
        + rx_chain
        - path_loss
        - system_loss_db
        - link_margin_db
        + fading_db
    )
    np.fill_diagonal(rx_power, np.inf)
    return rx_power.astype(np.float32)


def physical_power_adjacency(
    distances: np.ndarray,
    radius: float,
    rng: np.random.Generator,
    radio_cfg: dict,
) -> np.ndarray:
    rx_power = received_power_dbm(distances, rng, radio_cfg)
    sensitivity = float(radio_cfg.get("receiver_sensitivity_dbm", -88.0))
    directed = rx_power >= sensitivity
    if bool(radio_cfg.get("apply_radius_cap", True)):
        directed &= distances <= radius
    np.fill_diagonal(directed, False)
    undirected = directed & directed.T
    return undirected.astype(np.float32)


def _physical_adjacency_from_power(rx_power: np.ndarray, distances: np.ndarray, radius: float, radio_cfg: dict) -> np.ndarray:
    sensitivity = float(radio_cfg.get("receiver_sensitivity_dbm", -88.0))
    directed = rx_power >= sensitivity
    if bool(radio_cfg.get("apply_radius_cap", True)):
        directed &= distances <= radius
    np.fill_diagonal(directed, False)
    return (directed & directed.T).astype(np.float32)


def sinr_adjacency(
    distances: np.ndarray,
    radius: float,
    rng: np.random.Generator,
    radio_cfg: dict,
) -> np.ndarray:
    rx_power_dbm = received_power_dbm(distances, rng, radio_cfg)
    n_nodes = distances.shape[0]
    p_tx = float(radio_cfg.get("tx_probability", 0.35))
    tx_active = rng.random(n_nodes) < p_tx
    if not np.any(tx_active):
        tx_active[int(rng.integers(0, n_nodes))] = True

    rx_power_mw = _dbm_to_mw(rx_power_dbm)
    noise_floor_dbm = float(radio_cfg.get("noise_floor_dbm", -101.0))
    noise_mw = float(_dbm_to_mw(noise_floor_dbm))
    sinr_threshold_db = float(radio_cfg.get("sinr_threshold_db", 8.0))
    directed = np.zeros((n_nodes, n_nodes), dtype=bool)
    active_idx = np.flatnonzero(tx_active)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            interference_sources = active_idx[active_idx != i]
            interference = float(rx_power_mw[interference_sources, j].sum()) if interference_sources.size else 0.0
            sinr_db = float(_mw_to_dbm(rx_power_mw[i, j] / max(noise_mw + interference, 1e-30)))
            directed[i, j] = sinr_db >= sinr_threshold_db
    if bool(radio_cfg.get("apply_radius_cap", True)):
        directed &= distances <= radius
    np.fill_diagonal(directed, False)
    return (directed & directed.T).astype(np.float32)


def build_link_adjacency(
    distances: np.ndarray,
    radius: float,
    rng: np.random.Generator,
    sim_cfg: dict,
) -> np.ndarray:
    link_model = str(sim_cfg.get("link_model", "radius")).lower()
    radio_cfg = sim_cfg.get("physical_layer", {})
    if link_model in {"radius", "proximity", "distance"}:
        return adjacency_from_radius(distances, radius)
    if link_model in {"physical", "path_loss", "power"}:
        return physical_power_adjacency(distances, radius, rng, radio_cfg)
    if link_model in {"sinr", "interference"}:
        return sinr_adjacency(distances, radius, rng, radio_cfg)
    raise ValueError(f"Unknown link_model: {link_model}")


def build_fixed_adaptive_adjacencies(
    distances: np.ndarray,
    radius_fixed: float,
    radius_adaptive: float,
    rng: np.random.Generator,
    sim_cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    link_model = str(sim_cfg.get("link_model", "radius")).lower()
    radio_cfg = sim_cfg.get("physical_layer", {})
    if link_model in {"radius", "proximity", "distance"}:
        return adjacency_from_radius(distances, radius_fixed), adjacency_from_radius(distances, radius_adaptive)
    if link_model in {"physical", "path_loss", "power"}:
        rx_power = received_power_dbm(distances, rng, radio_cfg)
        return (
            _physical_adjacency_from_power(rx_power, distances, radius_fixed, radio_cfg),
            _physical_adjacency_from_power(rx_power, distances, radius_adaptive, radio_cfg),
        )
    if link_model in {"sinr", "interference"}:
        seed = int(rng.integers(0, 2**63 - 1))
        return (
            sinr_adjacency(distances, radius_fixed, np.random.default_rng(seed), radio_cfg),
            sinr_adjacency(distances, radius_adaptive, np.random.default_rng(seed), radio_cfg),
        )
    raise ValueError(f"Unknown link_model: {link_model}")
