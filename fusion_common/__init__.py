"""Shared utilities for the three fusion branches.

Sibling fusion directories (``loose_fusion``, ``cross_attention``) all need:

    - the same paired vision-+-IMU dataset,
    - the same ``smooth_l1 + geodesic`` pose loss,
    - the same per-sequence trajectory dead-reckoning + ATE/RTE metrics.

Putting these here keeps them in one place.

The fusion branches add the project root to ``sys.path`` and import as::

    from fusion_common.dataset import PairedDataset
    from fusion_common.loss import pose_loss
"""
