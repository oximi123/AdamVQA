
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, Literal, Iterable
import math
import optuna

# -------------------------------
# Token ⇄ Latency model
# -------------------------------

@dataclass

    a: float = 0.0
    b: float = 0.0

    def tokens_to_ttft(self, tokens: float) -> float:
        return self.a * float(tokens) + self.b

    def max_tokens_under_slo(self, slo_seconds: float) -> float:
        if self.a <= 0:
            # Fall back to a generous max if not fitted yet.
            return float("inf")
        return max(0.0, (slo_seconds - self.b) / self.a)


# -------------------------------
# Monotonic knob spec
# -------------------------------

MonotonicDir = Literal["increasing", "decreasing"]

@dataclass
class KnobSpec:
    modality: str
    name: str
    direction: MonotonicDir


class LatencyAwareSpacePruner(optuna.pruners.BasePruner):

    def __init__(
        self,
        slo_seconds: float,
        token_latency_model: TokenLatencyModel,
        monotone_knobs: Iterable[KnobSpec] = (),
        study_bounds_key: str = "latency_pruner_bounds",
        safety_margin: float = 0.0,
    ) -> None:
        super().__init__()
        self.slo = float(slo_seconds)
        self.tl = token_latency_model
        self.study_bounds_key = study_bounds_key
        self.safety_margin = float(safety_margin)
        self._mono: Dict[Tuple[str, str], MonotonicDir] = {
            (k.modality, k.name): k.direction for k in monotone_knobs
        }


    def max_tokens(self) -> float:
        return self.tl.max_tokens_under_slo(self.slo - self.safety_margin)

    def update_bounds_from_violation(
        self, study: "optuna.study.Study", config: Dict[str, Any]
    ) -> None:
        if not self._mono:
            return
        bounds = self._get_bounds(study)

        for (mod, knob), direction in self._mono.items():
            if mod not in config or knob not in config[mod]:
                continue
            key = f"{mod}.{knob}"
            val = config[mod][knob]

            if direction == "increasing":
                prev = bounds["increasing"].get(key, None)
                new = val if prev is None else min(prev, val)
                bounds["increasing"][key] = new
            else:  # decreasing
                prev = bounds["decreasing"].get(key, None)
                new = val if prev is None else max(prev, val)
                bounds["decreasing"][key] = new

        study.set_system_attr(self.study_bounds_key, bounds)

    @staticmethod
    def config_respects_bounds(
        config: Dict[str, Any],
        bounds: Dict[str, Dict[str, float]]
    ) -> bool:
        for key, max_allowed in bounds.get("increasing", {}).items():
            mod, knob = key.split(".", 1)
            if mod in config and knob in config[mod]:
                if config[mod][knob] > max_allowed:
                    return False
        for key, min_allowed in bounds.get("decreasing", {}).items():
            mod, knob = key.split(".", 1)
            if mod in config and knob in config[mod]:
                if config[mod][knob] < min_allowed:
                    return False
        return True

    def get_pruning_bounds(self, study: "optuna.study.Study") -> Dict[str, Dict[str, float]]:
        return self._get_bounds(study)

    # ------------- Optuna hook -------------

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        if not trial.intermediate_values:
            return False

        last_step = max(trial.intermediate_values.keys())
        ttft = trial.intermediate_values[last_step]
        if ttft is None or math.isnan(ttft):
            return False

        if ttft > (self.slo - self.safety_margin):
            config = trial.user_attrs.get("config", None)
            if isinstance(config, dict):
                try:
                    self.update_bounds_from_violation(study, config)
                except Exception:
                    pass
            return True

        return False


    def _get_bounds(self, study: "optuna.study.Study") -> Dict[str, Dict[str, float]]:
        b = study.system_attrs.get(self.study_bounds_key, None)
        if isinstance(b, dict) and "increasing" in b and "decreasing" in b:
            return b
        init = {"increasing": {}, "decreasing": {}}
        return init


def precap_vt_bounds(
    study: "optuna.study.Study",
    pruner: "LatencyAwareSpacePruner",
    vt_grid: Dict[str, Iterable[float]],
    base_cfg: Dict[str, Any],
    token_estimator: "callable[[Dict[str, Any]], float]" = None,
    vt_modality_name: str = "VT",
) -> Dict[str, Dict[str, float]]:
    keys = list(vt_grid.keys())
    values_list = [list(vt_grid[k]) for k in keys]
    if not keys:
        return pruner.get_pruning_bounds(study)

    feasible_points = []
    def cartesian(idx: int, cur: Dict[str, Any]):
        if idx == len(keys):
            cfg = {k: (v.copy() if isinstance(v, dict) else v) for k, v in base_cfg.items()}
            cfg.setdefault(vt_modality_name, {})
            cfg[vt_modality_name].update(cur)
            if token_estimator is not None:
                tokens = float(token_estimator(cfg))
            else:
                tokens = 1.0
                for k, v in cur.items():
                    try:
                        fv = float(v)
                        tokens *= fv if fv > 0 else 1.0
                    except Exception:
                        tokens *= 1.0
            ttft = pruner.tl.tokens_to_ttft(tokens)
            if ttft <= pruner.slo - pruner.safety_margin + 1e-12:
                feasible_points.append(cur.copy())
            return
        for v in values_list[idx]:
            nxt = dict(cur)
            nxt[keys[idx]] = v
            cartesian(idx+1, nxt)

    cartesian(0, {})

    bounds = pruner.get_pruning_bounds(study)
    inc = bounds.get("increasing", {})
    dec = bounds.get("decreasing", {})

    if not feasible_points:
        study.set_system_attr(pruner.study_bounds_key, bounds)
        return bounds

    for (mod, knob), direction in pruner._mono.items():
        if mod != vt_modality_name or knob not in keys:
            continue
        if direction == "increasing":
            max_feasible = max(fp[knob] for fp in feasible_points if knob in fp)
            inc[f"{mod}.{knob}"] = max_feasible if f"{mod}.{knob}" not in inc else min(inc[f"{mod}.{knob}"], max_feasible)
        else:
            min_feasible = min(fp[knob] for fp in feasible_points if knob in fp)
            dec[f"{mod}.{knob}"] = min_feasible if f"{mod}.{knob}" not in dec else max(dec[f"{mod}.{knob}"], min_feasible)

    bounds["increasing"] = inc
    bounds["decreasing"] = dec
    study.set_system_attr(pruner.study_bounds_key, bounds)
    return bounds


def profile_monotonic_knobs(
    estimator_tokens: "callable[[Dict[str, Any]], float]",
    base_cfg: Dict[str, Any],
    knob_space: Dict[Tuple[str,str], Tuple[Any, Any]],
) -> Iterable[KnobSpec]:

    specs = []
    for (mod, knob), (lo, hi) in knob_space.items():
        cfg_lo = {k: (v.copy() if isinstance(v, dict) else v) for k,v in base_cfg.items()}
        cfg_hi = {k: (v.copy() if isinstance(v, dict) else v) for k,v in base_cfg.items()}
        cfg_lo.setdefault(mod, {}); cfg_hi.setdefault(mod, {})
        cfg_lo[mod][knob] = lo
        cfg_hi[mod][knob] = hi
        try:
            t_lo = float(estimator_tokens(cfg_lo))
            t_hi = float(estimator_tokens(cfg_hi))
        except Exception:
            continue
        if math.isfinite(t_lo) and math.isfinite(t_hi) and abs(t_hi - t_lo) > 1e-9:
            direction = "increasing" if t_hi > t_lo else "decreasing"
            specs.append(KnobSpec(mod, knob, direction))  # type: ignore[arg-type]
    return specs
