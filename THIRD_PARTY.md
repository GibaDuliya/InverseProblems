# Third-party source provenance

`src/bridge_restore/models/adm.py` and the neural-network utilities in
`models/nn.py` are adapted from [DDBM](https://github.com/alexzhou907/DDBM),
commit `1a928343d198ffd48e26962b8af790b49b94fbba`, files `ddbm/unet.py`,
`ddbm/nn.py` and the convolution precision-conversion helpers in `ddbm/fp16_util.py`.
The DDBM source itself credits OpenAI guided-diffusion and related implementations.
Original attribution comments are retained where applicable.

Changes in this project:

- Rename the U-Net to `ADMUNet`; retain its residual, attention and conditioning structure.
- Use legacy PyTorch attention; remove optional FlashAttention and unused attention variants.
- Replace the custom autograd checkpoint with PyTorch non-reentrant checkpointing.
- Disable implicit attention-only checkpointing; explicit U-Net checkpointing remains configurable.
- Keep convolution conversion helpers local; do not import DDBM training infrastructure.
- Use native autocast in the new trainer rather than the upstream FP16 training utilities.

The upstream snapshot has no top-level license file. This project does not assign
a new license to the borrowed code; provenance here is not a license grant.

Bridge formula reference: [Denoising Diffusion Bridge Models](https://arxiv.org/abs/2309.16948),
Linqi Zhou, Aaron Lou, Samar Khanna, Stefano Ermon. The new VE implementation uses
the paired Gaussian marginal and x0 preconditioning, with a pure reverse-SDE sampler.
Training/data/checkpoint code here is independent of the cloned `DDBM/` directory.
