"""Structured site map schema for ESField MVP."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

SCHEMA_VERSION = "1.0"

SiteType = Literal["high_energy_water", "stable_water", "hydrophobic_cavity"]
SITE_TYPES: frozenset[str] = frozenset(
    {"high_energy_water", "stable_water", "hydrophobic_cavity"}
)


class SchemaValidationError(ValueError):
    """Raised when a site map payload violates the ESField schema."""


@dataclass(frozen=True)
class Site:
    """One offline physical opportunity site in the original pocket frame."""

    site_id: int
    site_type: SiteType
    center: tuple[float, float, float]
    radius: float
    score: float
    confidence: float
    source: str
    features: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        site_id = _as_int(self.site_id, "site_id")
        site_type = _as_site_type(self.site_type, "site_type")
        center = _as_vec3(self.center, "center")
        radius = _as_float(self.radius, "radius")
        score = _as_float(self.score, "score")
        confidence = _as_float(self.confidence, "confidence")
        source = _as_non_empty_string(self.source, "source")
        features = _as_json_mapping(self.features, "features")

        _validate_positive(radius, "radius")
        _validate_probability(confidence, "confidence")

        object.__setattr__(self, "site_id", site_id)
        object.__setattr__(self, "site_type", site_type)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "features", features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "site_type": self.site_type,
            "center": list(self.center),
            "radius": self.radius,
            "score": self.score,
            "confidence": self.confidence,
            "source": self.source,
            "features": self.features,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Site":
        _require_keys(
            data,
            {
                "site_id",
                "site_type",
                "center",
                "radius",
                "score",
                "confidence",
                "source",
                "features",
            },
            "site",
        )
        return cls(
            site_id=_as_int(data["site_id"], "site.site_id"),
            site_type=_as_site_type(data["site_type"], "site.site_type"),
            center=_as_vec3(data["center"], "site.center"),
            radius=_as_float(data["radius"], "site.radius"),
            score=_as_float(data["score"], "site.score"),
            confidence=_as_float(data["confidence"], "site.confidence"),
            source=_as_non_empty_string(data["source"], "site.source"),
            features=_as_json_mapping(data["features"], "site.features"),
        )


@dataclass(frozen=True)
class SiteMap:
    """Site map attached to one protein-ligand pocket."""

    protein_id: str
    ligand_id: str
    pocket_center: tuple[float, float, float]
    coordinate_frame: str
    sites: tuple[Site, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = _as_non_empty_string(self.schema_version, "schema_version")
        protein_id = _as_non_empty_string(self.protein_id, "protein_id")
        ligand_id = _as_non_empty_string(self.ligand_id, "ligand_id")
        pocket_center = _as_vec3(self.pocket_center, "pocket_center")
        coordinate_frame = _as_non_empty_string(self.coordinate_frame, "coordinate_frame")
        sites = _as_site_sequence(self.sites, "sites")

        _validate_schema_version(schema_version)
        _validate_site_ids(sites)

        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "protein_id", protein_id)
        object.__setattr__(self, "ligand_id", ligand_id)
        object.__setattr__(self, "pocket_center", pocket_center)
        object.__setattr__(self, "coordinate_frame", coordinate_frame)
        object.__setattr__(self, "sites", sites)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protein_id": self.protein_id,
            "ligand_id": self.ligand_id,
            "pocket_center": list(self.pocket_center),
            "coordinate_frame": self.coordinate_frame,
            "sites": [site.to_dict() for site in self.sites],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def write_json(self, path: str | Path, *, indent: int | None = 2) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SiteMap":
        _require_keys(
            data,
            {
                "schema_version",
                "protein_id",
                "ligand_id",
                "pocket_center",
                "coordinate_frame",
                "sites",
            },
            "site_map",
        )
        sites_raw = data["sites"]
        if not isinstance(sites_raw, Sequence) or isinstance(sites_raw, (str, bytes)):
            raise SchemaValidationError("site_map.sites must be a sequence of site objects")
        return cls(
            schema_version=_as_non_empty_string(
                data["schema_version"], "site_map.schema_version"
            ),
            protein_id=_as_non_empty_string(data["protein_id"], "site_map.protein_id"),
            ligand_id=_as_non_empty_string(data["ligand_id"], "site_map.ligand_id"),
            pocket_center=_as_vec3(data["pocket_center"], "site_map.pocket_center"),
            coordinate_frame=_as_non_empty_string(
                data["coordinate_frame"], "site_map.coordinate_frame"
            ),
            sites=tuple(Site.from_dict(_as_mapping(site, "site")) for site in sites_raw),
        )

    @classmethod
    def from_json(cls, text: str) -> "SiteMap":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(f"invalid site map JSON: {exc}") from exc
        return cls.from_dict(_as_mapping(payload, "site_map"))

    @classmethod
    def read_json(cls, path: str | Path) -> "SiteMap":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def _require_keys(data: Mapping[str, Any], keys: set[str], name: str) -> None:
    missing = sorted(keys.difference(data.keys()))
    if missing:
        raise SchemaValidationError(f"{name} missing required keys: {missing}")


def _as_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{field_name} must be a JSON object")
    return value


def _as_json_mapping(value: Any, field_name: str) -> dict[str, Any]:
    mapping = dict(_as_mapping(value, field_name))
    _validate_json_mapping(mapping, field_name)
    return mapping


def _as_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{field_name} must be a non-empty string")
    return value


def _as_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaValidationError(f"{field_name} must be an integer")
    return value


def _as_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{field_name} must be a number")
    return float(value)


def _as_vec3(value: Any, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SchemaValidationError(f"{field_name} must be a 3-element sequence")
    if len(value) != 3:
        raise SchemaValidationError(f"{field_name} must contain exactly 3 numbers")
    return tuple(_as_float(item, f"{field_name}[{idx}]") for idx, item in enumerate(value))  # type: ignore[return-value]


def _as_site_type(value: Any, field_name: str) -> SiteType:
    if not isinstance(value, str):
        raise SchemaValidationError(f"{field_name} must be a string")
    _validate_site_type(value, field_name)
    return value  # type: ignore[return-value]


def _as_site_sequence(value: Any, field_name: str) -> tuple[Site, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SchemaValidationError(f"{field_name} must be a sequence of Site objects")
    sites = tuple(value)
    for idx, site in enumerate(sites):
        if not isinstance(site, Site):
            raise SchemaValidationError(f"{field_name}[{idx}] must be a Site object")
    return sites


def _validate_schema_version(schema_version: str) -> None:
    if schema_version != SCHEMA_VERSION:
        raise SchemaValidationError(
            f"unsupported schema_version {schema_version!r}; expected {SCHEMA_VERSION!r}"
        )


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{field_name} must be a non-empty string")


def _validate_site_type(value: str, field_name: str) -> None:
    if value not in SITE_TYPES:
        allowed = ", ".join(sorted(SITE_TYPES))
        raise SchemaValidationError(f"{field_name} must be one of: {allowed}")


def _validate_vec3(value: Iterable[float], field_name: str) -> None:
    values = tuple(value)
    if len(values) != 3:
        raise SchemaValidationError(f"{field_name} must contain exactly 3 numbers")
    for idx, item in enumerate(values):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise SchemaValidationError(f"{field_name}[{idx}] must be a number")


def _validate_positive(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise SchemaValidationError(f"{field_name} must be a positive number")


def _validate_probability(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{field_name} must be a number in [0, 1]")
    if value < 0 or value > 1:
        raise SchemaValidationError(f"{field_name} must be in [0, 1]")


def _validate_json_mapping(value: Mapping[str, Any], field_name: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"{field_name} must be JSON serializable") from exc


def _validate_site_ids(sites: Sequence[Site]) -> None:
    site_ids = [site.site_id for site in sites]
    if len(site_ids) != len(set(site_ids)):
        raise SchemaValidationError("site_map.sites must have unique site_id values")
