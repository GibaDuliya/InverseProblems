from dataclasses import dataclass, field
from pathlib import Path
import json
import math

def safe(value):
    if isinstance(value, float) and not math.isfinite(value): return str(value)
    if isinstance(value, dict): return {k: safe(v) for k,v in value.items()}
    if isinstance(value, list): return [safe(v) for v in value]
    return value

@dataclass
class EvaluationReport:
    summary: dict
    per_image: list
    protocol: dict
    visual_examples: dict = field(default_factory=dict)
    def save(self, directory):
        from .visualization import save_visual_examples
        caption = f"step={self.protocol.get('step', self.protocol.get('checkpoint_step', '?'))} weights={self.protocol.get('config', {}).get('weights', '?')}"
        save_visual_examples(self.visual_examples, Path(directory) / "images", caption)
        path = Path(directory); path.mkdir(parents=True, exist_ok=True)
        for name, value in (("summary", self.summary), ("protocol", self.protocol)):
            (path / (name+".json")).write_text(json.dumps(safe(value), indent=2, allow_nan=False))
        (path / "per_image.jsonl").write_text("".join(json.dumps(safe(row), allow_nan=False)+"\n" for row in self.per_image))
