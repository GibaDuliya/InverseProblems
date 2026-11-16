"""Fixed visual panels from existing reconstructions; never rerun sampling."""
import json
from pathlib import Path
from PIL import Image, ImageDraw
import torch


def load_visual_samples(path, split):
    if path is None:
        return []
    value = json.loads(Path(path).read_text())
    ids = value.get('sample_ids')
    if value.get('split') != split:
        raise ValueError(f'Visual sample split mismatch: expected {split}')
    if not isinstance(ids, list) or not all(isinstance(key, str) and key for key in ids):
        raise ValueError('Visual sample_ids must be a list of nonempty strings')
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate visual sample IDs')
    return ids


def save_visual_examples(examples, directory, caption):
    """examples is ordered sample_id -> (clean, observation, reconstruction)."""
    path = Path(directory)
    if not examples:
        return
    path.mkdir(parents=True, exist_ok=True)
    entries = list(examples.items())
    for start in range(0, len(entries), 8):
        rows = entries[start:start+8]
        h, w = rows[0][1][0].shape[-2:]
        canvas = Image.new('RGB', (3*w, 32+len(rows)*(h+20)), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((3, 0), caption, fill='black')
        for j, label in enumerate(['clean', 'observation', 'restoration']):
            draw.text((j*w+3, 16), label, fill='black')
        for i, (key, images) in enumerate(rows):
            top = 32+i*(h+20)
            draw.text((3, top), key, fill='black')
            for j, tensor in enumerate(images):
                array = ((tensor.float().clamp(-1,1)+1)*127.5).round().to(torch.uint8).permute(1,2,0).numpy()
                canvas.paste(Image.fromarray(array), (j*w, top+20))
        canvas.save(path/f'panel_{start//8:02d}.png')
