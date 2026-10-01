"""SiT attention alignment with selectable CKA/relational-L1 H-diversity."""

from .loss import (
    SILoss,
    cosine_token_relation_matrix,
    row_relational_l1_distance,
    row_relational_l1_diversity_loss,
)

__all__ = [
    "SILoss",
    "cosine_token_relation_matrix",
    "row_relational_l1_distance",
    "row_relational_l1_diversity_loss",
]
