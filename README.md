# Time-Averaged Drift Approximations are Inconsistent for Inference in Drift Diffusion Models

This repository contains the code and examples accompanying the paper  
*Time-Averaged Drift Approximations Are Inconsistent for Inference in Drift Diffusion Models*:

- [arXiv preprint](https://arxiv.org/abs/2512.10250)
- [Published article](https://www.sciencedirect.com/science/article/pii/S0022249626000350)

## Installation

This project depends on the [efpt](https://github.com/RiverFlowsInYou98/efficient-fpt) package for efficient first-passage time computations.

## Repository Structure

### `ddm_with_one_sided_boundary/`

DDM with a one-sided absorbing boundary. Demonstrates that TADA converges to a biased limit ($\tilde\mu \neq \mu$) while MLE recovers the true parameter.

- `tada_gen_data.py` — Simulate first-passage time data.
- `tada_mle.ipynb` — Compare TADA-based "MLE" vs generic MLE.

### `ddm_with_double_sided_boundary/`

DDM with double-sided (symmetric) boundaries and alternating drift stages.

- `tada_mle.ipynb` — Compare TADA-based "MLE" vs generic MLE for multi-stage models.

### `addm/inconsistency/`

Parameter recovery experiments for the attentional DDM (aDDM), sweeping one parameter at a time while fixing the others. Each subfolder contains:

- `addm_gen_data.ipynb` — Generate simulated aDDM data.
- `param_recovery.ipynb` — Run MLE and TADA parameter recovery across sample sizes.

Subfolders: `varying_a/`, `varying_eta/`, `varying_kappa/`, `varying_x0/`. 

In each experiment, three nuisance parameters are fixed, and the relationship between the varied parameter and the asymptotic bias of the recovery methods is studied. The associated paper reports `varying_eta/` results only.

Shared utilities are in `addm/shared.py`.

### `addm/hypothesis_testing/`

Demonstrates that TADA produces incorrect hypothesis testing results when comparing two groups/subjects.

- `population_with_different_rating_distributions/` — Two subjects with different stimulus rating distributions. TADA's covariate-dependent bias leads to inflated Type I error in one-sided tests.
- `population_with_different_boundaries/` — Two subjects with different boundary heights. TADA's nuisance-parameter-dependent bias leads to inflated Type II error.

## License

This project is licensed under the [MIT License](LICENSE).

## Citation
If this repository is useful for your research, please cite the associated paper:

```bibtex
@article{liu2026time,
  title={Time-averaged drift approximations are inconsistent for inference in drift diffusion models},
  author={Liu, Sicheng and Fengler, Alexander and Frank, Michael J and Harrison, Matthew T},
  journal={Journal of Mathematical Psychology},
  volume={130},
  pages={103004},
  year={2026},
  publisher={Elsevier}
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
