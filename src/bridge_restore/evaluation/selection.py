"""Independent best states, selected from a single validation report."""
import math
import re


def validate_criteria(criteria, metrics):
    if not isinstance(criteria, dict) or not criteria:
        raise ValueError("criteria must be a nonempty mapping")
    for name, rule in criteria.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError("Criterion names must be safe directory names")
        if not isinstance(rule, dict) or set(rule) != {"metric", "mode", "test"}:
            raise ValueError("Each criterion requires exactly metric, mode, test")
        if rule["metric"] not in metrics or rule["mode"] not in ("min", "max") or type(rule["test"]) is not bool:
            raise ValueError("Invalid selection criterion")


class BestSelection:
    def __init__(self, criteria):
        self.criteria = criteria
        self.best = {}

    def update(self, summary, step, epoch):
        values = {name: summary["restoration"][rule["metric"]]["mean"]
                  for name, rule in self.criteria.items()}
        if any(not math.isfinite(value) for value in values.values()):
            raise FloatingPointError("Nonfinite selection metric")
        improved = []
        for name, rule in self.criteria.items():
            value, old = values[name], self.best.get(name)
            better = old is None or (value < old["value"] if rule["mode"] == "min" else value > old["value"])
            if better:
                self.best[name] = {"value": value, "step": step, "epoch": epoch}
                improved.append(name)
        return improved
