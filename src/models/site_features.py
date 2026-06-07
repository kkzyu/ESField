"""Site feature vocabularies for compatibility potential training."""

SITE_TYPE_VOCAB: tuple[str, ...] = (
    "unknown",
    "high_energy_water",
    "stable_water",
    "hydrophobic_cavity",
)


def site_type_to_index(site_type: str) -> int:
    try:
        return SITE_TYPE_VOCAB.index(site_type)
    except ValueError:
        return 0

