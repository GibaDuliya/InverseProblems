"""Persistent source-image partitions, optionally disjoint by identity."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os
import random
import tempfile


@dataclass
class SplitManifest:
    train_ids: list
    val_ids: list
    test_ids: list
    group_by_sample: dict
    dataset_fingerprint: str
    creation_parameters: dict

    @property
    def checksum(self):
        return _hash(asdict(self))


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _metadata(dataset):
    ids, groups = list(dataset.sample_ids), list(dataset.group_ids)
    if len(ids) != len(dataset) or len(ids) != len(groups) or len(set(ids)) != len(ids):
        raise ValueError("Dataset IDs must be unique and metadata lengths must match")
    if not ids or any(not isinstance(key, str) or not key for key in ids):
        raise ValueError("Dataset must have nonempty string IDs")
    if any(g is not None and (not isinstance(g, str) or not g) for g in groups):
        raise ValueError("group_ids must contain nonempty strings or None")
    return dict(zip(ids, groups))


def load_manifest(path):
    return SplitManifest(**json.loads(Path(path).read_text()))


def save_manifest(manifest, path):
    """Atomic create, never overwrite. Concurrent identical creators are harmless."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".split-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(asdict(manifest), f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if load_manifest(path).checksum != manifest.checksum:
                raise ValueError(f"Refusing to replace different split: {path}")
    finally:
        os.unlink(temporary)


def validate_manifest(manifest, dataset, group_by="identity"):
    metadata = _metadata(dataset)
    all_ids = manifest.train_ids + manifest.val_ids + manifest.test_ids
    if not all((manifest.train_ids, manifest.val_ids, manifest.test_ids)):
        raise ValueError("All three splits must be nonempty")
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(metadata):
        raise ValueError("Split must partition all source IDs exactly once")
    if manifest.group_by_sample != metadata or manifest.dataset_fingerprint != _hash(metadata):
        raise ValueError("Dataset ID/group metadata changed since split creation")
    if group_by == "identity":
        if any(g is None for g in metadata.values()):
            raise ValueError("Identity split requires group_id for every sample")
        groups = [{metadata[key] for key in ids} for ids in
                  (manifest.train_ids, manifest.val_ids, manifest.test_ids)]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("Identity leakage between splits")


def create_group_split(dataset, fractions, seed, group_by="identity"):
    metadata = _metadata(dataset)
    buckets = {}
    for key in sorted(metadata):
        group = metadata[key] if group_by == "identity" else key
        if group is None:
            raise ValueError("Missing identity annotation")
        buckets.setdefault(group, []).append(key)
    groups = sorted(buckets)
    if len(groups) < 3:
        raise ValueError("At least three independent groups are needed")
    random.Random(seed).shuffle(groups)
    n = len(groups)
    first = max(1, min(n - 2, round(n * fractions[0])))
    second = max(first + 1, min(n - 1, round(n * (fractions[0] + fractions[1]))))
    parts = [sorted(key for group in subset for key in buckets[group])
             for subset in (groups[:first], groups[first:second], groups[second:])]
    return SplitManifest(*parts, metadata, _hash(metadata),
                         {"method": "random_groups", "seed": seed,
                          "fractions": list(fractions), "group_by": group_by})


def import_official_split(dataset):
    metadata = _metadata(dataset)
    assignments = getattr(dataset, "official_split_assignments", None)
    if assignments is None or set(assignments) != set(metadata):
        raise ValueError("Dataset must provide official assignments for all IDs")
    if set(assignments.values()) - {"train", "val", "test"}:
        raise ValueError("Official split names must be train, val, test")
    parts = [sorted(key for key, part in assignments.items() if part == name)
             for name in ("train", "val", "test")]
    return SplitManifest(*parts, metadata, _hash(metadata), {"method": "official"})


def load_or_create_split(dataset, config):
    # Official assignments are authoritative; no generated cache can override them.
    # Their grouping guarantee belongs to the dataset adapter/source.
    if config.mode == "official":
        manifest = import_official_split(dataset)
        validate_manifest(manifest, dataset, group_by="sample")
        return manifest
    path = Path(config.manifest_path)
    loaded = path.exists()
    if loaded:
        manifest = load_manifest(path)
    elif config.mode == "load":
        raise FileNotFoundError(path)
    elif config.mode == "create_if_missing":
        manifest = create_group_split(dataset, config.fractions, config.seed, config.group_by)
    else:
        raise ValueError(f"Unknown split mode: {config.mode}")
    validate_manifest(manifest, dataset, config.group_by)
    if not loaded:
        save_manifest(manifest, path)
    return manifest
