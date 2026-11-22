from copy import deepcopy
from dataclasses import asdict
import pytest
import torch
from bridge_restore.bridge import VEBridge
from bridge_restore.models import build_model
from bridge_restore.training import Trainer
from bridge_restore.config import config_from_dict
from bridge_restore.sampling import SDESampler
from bridge_restore.data.build import build_loaders


@pytest.mark.parametrize("mode", ["bm", "abm"])
@pytest.mark.parametrize("initial_steps", [1, 3])
def test_training_resume_matches_uninterrupted(tiny_config, loaders, tmp_path, mode, initial_steps):
    tiny_config.model.mode = mode
    bridge = VEBridge(**asdict(tiny_config.bridge))
    torch.manual_seed(7)
    model = build_model(tiny_config.model, bridge, 8)
    trainer = Trainer(model, bridge, loaders[0], loaders[1], tiny_config, "split", {"quality": 10})
    before = {k: v.clone() for k, v in model.state_dict().items()}
    for _ in range(initial_steps):
        trainer.train_step()
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    checkpoint = tmp_path / "state.pt"
    trainer.save(checkpoint)
    rng = torch.get_rng_state().clone()
    first_val = trainer.validate()
    assert torch.equal(rng, torch.get_rng_state())
    assert first_val == trainer.validate()
    trainer.train_step()
    expected = deepcopy(trainer.model.state_dict())
    model2 = build_model(tiny_config.model, bridge, 8)
    resumed = Trainer(model2, bridge, loaders[0], loaders[1], tiny_config, "split", {"quality": 10})
    resumed.resume(checkpoint)
    resumed.train_step()
    for key, value in expected.items():
        torch.testing.assert_close(value, model2.state_dict()[key], rtol=0, atol=0)
    assert resumed.step == initial_steps + 1
    resumed.ema.copy_to(model2)
    batch = next(iter(loaders[1]))
    result = SDESampler(bridge, num_steps=4).sample(model2, batch["xT"])
    assert torch.isfinite(result.reconstruction).all()
    assert result.reconstruction.shape == batch["xT"].shape


def test_accumulation_handles_unequal_batch_sizes(tiny_config):
    class Pairs(torch.utils.data.Dataset):
        def __len__(self): return 3
        def __getitem__(self, i):
            return {"x0": torch.full((1,), float(i)), "xT": torch.zeros(1)}
    class DeterministicLoss:
        def training_loss(self, model, x0, xT, generator=None):
            return (model(xT) - x0).square().flatten(1).mean(1)
    dataset = Pairs()
    initial = torch.nn.Linear(1, 1)
    results = []
    for batch_size, accum in ((3, 1), (2, 2)):
        config = deepcopy(tiny_config)
        config.training.batch_size = batch_size
        config.training.grad_accum_steps = accum
        loaders = build_loaders(dataset, dataset, dataset, batch_size, 0, 42)
        model = deepcopy(initial)
        trainer = Trainer(model, DeterministicLoss(), loaders[0], loaders[1], config, "split", {})
        trainer.train_step()
        results.append(model.state_dict())
    for key in results[0]:
        torch.testing.assert_close(results[0][key], results[1][key])


def test_config_rejects_typos_and_invalid_times():
    config_from_dict({})
    with pytest.raises(ValueError, match="Unknown"):
        config_from_dict({"model": {"mod": "bm"}})
    with pytest.raises(ValueError, match="scales"):
        config_from_dict({"bridge": {"sigma_min": 2.}})


@pytest.mark.parametrize("invalid", ["loss", "gradient"])
def test_nonfinite_update_does_not_modify_weights_or_ema(tiny_config, loaders, invalid):
    model = torch.nn.Linear(3, 3)
    class Loss:
        def training_loss(self, model, x0, xT, generator=None):
            output = model(x0.mean((2, 3)))
            loss = output.square().mean(1)
            return loss * float("nan") if invalid == "loss" else loss
    if invalid == "gradient":
        model.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    trainer = Trainer(model, Loss(), *loaders[:2], tiny_config, "split", {})
    before = deepcopy(model.state_dict())
    ema = deepcopy(trainer.ema.averaged_parameters)
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        trainer.train_step()
    assert trainer.step == 0 and trainer.samples_seen == 0
    assert not trainer.optimizer.state
    for name in before:
        torch.testing.assert_close(model.state_dict()[name], before[name], rtol=0, atol=0)
        torch.testing.assert_close(trainer.ema.averaged_parameters[name], ema[name], rtol=0, atol=0)
