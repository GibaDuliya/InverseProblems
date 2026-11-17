from dataclasses import replace
import pytest
from bridge_restore.config import SplitConfig
from bridge_restore.data.splits import (create_group_split, validate_manifest, save_manifest,
                                       load_manifest, load_or_create_split, import_official_split)


def test_group_partition_and_persistence(clean, tmp_path):
    manifest = create_group_split(clean, [.5, .25, .25], 12)
    validate_manifest(manifest, clean)
    assert manifest == create_group_split(clean, [.5, .25, .25], 12)
    path = tmp_path / "split.json"
    save_manifest(manifest, path)
    save_manifest(manifest, path)
    assert load_manifest(path) == manifest
    # Existing file is authoritative even if a later config proposes another seed.
    config = SplitConfig(mode="create_if_missing", manifest_path=str(path), seed=99)
    assert load_or_create_split(clean, config) == manifest
    other = replace(manifest, creation_parameters={"different": True})
    with pytest.raises(ValueError, match="Refusing"):
        save_manifest(other, path)


def test_reject_identity_leak_and_changed_metadata(clean):
    # Explicit identity validation remains available for custom partitions.
    clean.official_split_assignments = {key: ("train", "val", "test")[i % 3]
                                       for i, key in enumerate(clean.sample_ids)}
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest(import_official_split(clean), clean)
    manifest = create_group_split(clean, [.5, .25, .25], 12)
    clean.group_ids[0] = "changed"
    with pytest.raises(ValueError, match="changed"):
        validate_manifest(manifest, clean)


def test_missing_identity_is_not_silently_replaced(clean):
    clean.group_ids[0] = None
    with pytest.raises(ValueError, match="Missing identity"):
        create_group_split(clean, [.5, .25, .25], 12)


def test_official_split_is_used_directly_without_identity_or_cache(clean, tmp_path):
    clean.group_ids = [None] * len(clean)
    clean.official_split_assignments = {
        key: ("train" if i < 6 else "val" if i < 8 else "test")
        for i, key in enumerate(clean.sample_ids)}
    # An unrelated cache must not override the dataset's official partition.
    path = tmp_path / "unused.json"
    path.write_text("not a manifest")
    config = SplitConfig(mode="official", manifest_path=str(path))
    manifest = load_or_create_split(clean, config)
    assert manifest.train_ids == clean.sample_ids[:6]
    assert manifest.val_ids == clean.sample_ids[6:8]
    assert manifest.test_ids == sorted(clean.sample_ids[8:])
    assert path.read_text() == "not a manifest"
    assert load_or_create_split(clean, SplitConfig()).checksum == manifest.checksum
