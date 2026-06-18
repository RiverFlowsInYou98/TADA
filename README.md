# Time-Averaged Drift Approximations are Inconsistent for Inference in Drift Diffusion Models

This repository contains code and examples from the paper:  
[*Time-Averaged Drift Approximations are Inconsistent for Inference in Drift Diffusion Models*](https://arxiv.org/abs/2512.10250).

## Installation

This project depends on the [efpt](https://github.com/RiverFlowsInYou98/efficient-fpt) package for efficient first-passage time computations. 

## Repository Structure

### `ddm_with_one_sided_boundary/`

DDM with a one-sided absorbing boundary. Demonstrates that TADA converges to a biased limit ($\tilde\mu \neq \mu$) while MLE recovers the true parameter.

- `tada_gen_data.py` — Simulate first-passage time data.
- `tada_mle.ipynb` — Compare TADA MLE vs exact MLE.

### `ddm_with_double_sided_boundary/`

DDM with double-sided (symmetric) boundaries and alternating drift stages.

- `tada_mle.ipynb` — Compare TADA MLE vs exact MLE for multi-stage models.

### `addm/inconsistency/`

Parameter recovery experiments for the attentional DDM (aDDM), sweeping one parameter at a time while fixing the others. Each subfolder contains:

- `addm_gen_data.ipynb` — Generate simulated aDDM data.
- `param_recovery.ipynb` — Run MLE and TADA parameter recovery across sample sizes.

Subfolders: `varying_a/`, `varying_eta/`, `varying_kappa/`, `varying_x0/`. 

In each experiment, three nuisance parameters are fixed, and the relationship between the varied parameter and the asymptotic bias of the recovery methods is studied.

Shared utilities are in `addm/shared.py`.

### `addm/hypothesis_testing/`

Demonstrates that TADA produces incorrect hypothesis testing results when comparing two groups/subjects.

- `population_with_different_rating_distributions/` — Two subjects with different stimulus rating distributions. TADA's covariate-dependent bias leads to inflated Type I error in one-sided tests.
- `population_with_different_boundaries/` — Two subjects with different boundary heights. TADA's nuisance-parameter-dependent bias leads to inflated Type II error.

## License

[MIT License](https://opensource.org/licenses/MIT)

## Citation
If this repository is useful for your research, please cite the associated paper:

```bibtex
@article{liu2025time,
  title={Time-Averaged Drift Approximations are Inconsistent for Inference in Drift Diffusion Models},
  author={Liu, Sicheng and Fengler, Alexander and Frank, Michael J and Harrison, Matthew T},
  journal={arXiv preprint arXiv:2512.10250},
  year={2025}
}
```
The likelihood computation methods in this repository are developed and implemented in [efpt](https://github.com/RiverFlowsInYou98/efficient-fpt). If you use those components, please also cite:

```bibtex
@article{liu2026efficient,
  title={Efficient inference in first passage time models},
  author={Liu, Sicheng and Fengler, Alexander and Frank, Michael J. and Harrison, Matthew T.},
  journal={Statistics and Computing},
  volume={36},
  number={3},
  pages={101},
  year={2026},
  doi={10.1007/s11222-026-10854-4}
}
```
