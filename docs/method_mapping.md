# Paper-to-code mapping

| Paper definition | Public implementation |
|---|---|
| L2-normalized feature, Equation (1) | `group_ood_gmmd.gmmd.l2_rows` |
| Low-rank-plus-isotropic covariance, Equations (2)--(3) | `fit_gmmd_reference`, `_mahalanobis` |
| GMMD-Central Gaussian mean law, Equation (4) | `score_gmmd_central` |
| Basic GMMD image score, Equation (5) | `score_gmmd_image` |
| Excess-direction GMMD image score, Equation (6) | `fit_gmmd_mean`, `score_gmmd_mean` |
| Arithmetic group score mean, Equation (7) | `arithmetic_group_means` |
| Experiment-1 group split | `group_ood_gmmd.grouping.experiment1_split` |
| Experiment-2 paired partitions | `group_ood_gmmd.grouping.experiment2_groups` |

The shared reference uses one component per predicted ID class. Both public
methods use `R=128`; GMMD-Mean uses `nu=5`. GMMD-Central receives the mean of
individually normalized features and does not normalize that mean again.
GMMD-Mean applies the basic score when the centered classifier row rank does
not exceed `R`; otherwise it separates the excess directions before computing
the image score. Group information enters GMMD-Mean only through the final
arithmetic mean.
