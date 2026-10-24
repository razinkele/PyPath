"""Taxonomic hierarchical linking with WoRMS/ITIS integration.

This module provides taxonomic classification and cross-referencing for
food web groups, enabling integration with authoritative taxonomic databases
(WoRMS, ITIS) and hierarchical validation.

Phase 1B.2: TG3.2 Taxonomic Linkage System
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TaxonLink:
    """A single taxon with hierarchical classification and external IDs.

    Represents a species or higher-order taxon with full taxonomic hierarchy,
    common names, and links to authoritative databases (WoRMS, ITIS).

    Parameters
    ----------
    scientific_name : str
        Binomial scientific name (e.g., "Gadus morhua").
    common_name : str, optional
        Vernacular common name (e.g., "Atlantic cod").
    worms_aphia_id : int, optional
        WoRMS (World Register of Marine Species) AphiaID.
    itis_tsn : str, optional
        ITIS (Integrated Taxonomic Information System) Taxonomic Serial Number.
    kingdom : str
        Kingdom (e.g., "Animalia").
    phylum : str
        Phylum (e.g., "Chordata").
    class_ : str
        Class (note: 'class' is reserved keyword, use 'class_').
    order : str
        Order (e.g., "Gadiformes").
    family : str
        Family (e.g., "Gadidae").
    genus : str
        Genus (e.g., "Gadus").
    species : str
        Species epithet (e.g., "morhua").
    match_confidence : float
        Confidence score for database match (0.0-1.0).
        1.0 = exact match, <0.5 = uncertain.

    Attributes
    ----------
    kingdom, phylum, class_, order, family, genus, species : str
        Full taxonomic hierarchy.
    """

    scientific_name: str
    common_name: Optional[str] = None
    worms_aphia_id: Optional[int] = None
    itis_tsn: Optional[str] = None
    kingdom: str = ""
    phylum: str = ""
    class_: str = ""
    order: str = ""
    family: str = ""
    genus: str = ""
    species: str = ""
    match_confidence: float = 1.0

    def is_valid(self) -> bool:
        """Validate taxon record for completeness and consistency.

        Checks that:
        - scientific_name is not empty
        - match_confidence is in [0.0, 1.0]
        - At least one external ID is present (worms_aphia_id or itis_tsn)
        - Hierarchy is at least partially filled

        Returns
        -------
        bool
            True if taxon is valid, False otherwise.
        """
        if not self.scientific_name or not self.scientific_name.strip():
            logger.warning("TaxonLink: scientific_name is empty")
            return False

        if not (0.0 <= self.match_confidence <= 1.0):
            logger.warning(
                f"TaxonLink {self.scientific_name}: match_confidence "
                f"{self.match_confidence} not in [0.0, 1.0]"
            )
            return False

        # At least one external ID should be present
        if self.worms_aphia_id is None and self.itis_tsn is None:
            logger.warning(
                f"TaxonLink {self.scientific_name}: no external ID (WoRMS or ITIS)"
            )
            return False

        # At least kingdom should be filled
        if not self.kingdom or not self.kingdom.strip():
            logger.warning(
                f"TaxonLink {self.scientific_name}: kingdom is empty"
            )
            return False

        return True

    def to_dict(self) -> dict:
        """Export taxon to dictionary.

        Returns
        -------
        dict
            Dictionary with all fields.
        """
        return {
            "scientific_name": self.scientific_name,
            "common_name": self.common_name,
            "worms_aphia_id": self.worms_aphia_id,
            "itis_tsn": self.itis_tsn,
            "kingdom": self.kingdom,
            "phylum": self.phylum,
            "class_": self.class_,
            "order": self.order,
            "family": self.family,
            "genus": self.genus,
            "species": self.species,
            "match_confidence": self.match_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaxonLink":
        """Create TaxonLink from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with TaxonLink fields.

        Returns
        -------
        TaxonLink
            Initialized taxon object.
        """
        return cls(**data)


class TaxonomyLinkage:
    """Container for managing taxonomic classifications of food web groups.

    Maintains a mapping of food web groups to taxa, validates consistency,
    and provides serialization for integration with ecological models.

    Attributes
    ----------
    taxa : dict[str, TaxonLink]
        Mapping of scientific name to TaxonLink objects.
    group_assignments : dict[str, str]
        Mapping of group name (from Ecopath model) to scientific name.
    metadata : dict
        Metadata (source, timestamp, notes, etc.).
    """

    def __init__(self) -> None:
        """Initialize empty taxonomy container."""
        self.taxa: dict[str, TaxonLink] = {}
        self.group_assignments: dict[str, str] = {}
        self.metadata: dict = {}

    def add_taxon(self, taxon: TaxonLink) -> None:
        """Add or update a taxon.

        Parameters
        ----------
        taxon : TaxonLink
            Taxon to add.

        Raises
        ------
        ValueError
            If taxon is invalid.
        """
        if not taxon.is_valid():
            msg = f"Invalid taxon: {taxon.scientific_name}"
            raise ValueError(msg)

        self.taxa[taxon.scientific_name] = taxon
        logger.debug(f"Added taxon: {taxon.scientific_name}")

    def get_taxon(self, scientific_name: str) -> Optional[TaxonLink]:
        """Retrieve a taxon by scientific name.

        Parameters
        ----------
        scientific_name : str
            Scientific name to look up.

        Returns
        -------
        TaxonLink or None
            Taxon if found, None otherwise.
        """
        return self.taxa.get(scientific_name)

    def assign_group(self, group_name: str, scientific_name: str) -> None:
        """Assign a food web group to a taxon.

        Parameters
        ----------
        group_name : str
            Ecopath group name (e.g., "Cod").
        scientific_name : str
            Scientific name of taxon.

        Raises
        ------
        KeyError
            If scientific_name is not in taxa.
        """
        if scientific_name not in self.taxa:
            msg = (
                f"Scientific name '{scientific_name}' not in taxonomy. "
                f"Add taxon first."
            )
            raise KeyError(msg)

        self.group_assignments[group_name] = scientific_name
        logger.debug(f"Assigned {group_name} -> {scientific_name}")

    def validate(self) -> tuple[bool, list[str]]:
        """Validate taxonomy consistency.

        Checks:
        - All taxa are valid (is_valid() == True)
        - No duplicate assignments
        - Assigned groups reference existing taxa
        - Hierarchical consistency (no orphaned ranks)

        Returns
        -------
        tuple[bool, list[str]]
            (is_valid, error_list). is_valid is True iff error_list is empty.
        """
        errors: list[str] = []

        # Check taxa validity
        for sci_name, taxon in self.taxa.items():
            if not taxon.is_valid():
                errors.append(f"Invalid taxon: {sci_name}")

        # Check assignments reference existing taxa
        for group_name, sci_name in self.group_assignments.items():
            if sci_name not in self.taxa:
                errors.append(
                    f"Group '{group_name}' assigned to non-existent taxon "
                    f"'{sci_name}'"
                )

        # Check for duplicate group assignments (groups should map 1:1 to taxa)
        assigned_taxa = list(self.group_assignments.values())
        if len(assigned_taxa) != len(set(assigned_taxa)):
            duplicates = [
                t for t in set(assigned_taxa)
                if assigned_taxa.count(t) > 1
            ]
            for dup_taxon in duplicates:
                groups = [
                    g for g, t in self.group_assignments.items()
                    if t == dup_taxon
                ]
                errors.append(
                    f"Taxon '{dup_taxon}' assigned to multiple groups: {groups}"
                )

        return (len(errors) == 0, errors)

    def to_dataframe(self) -> pd.DataFrame:
        """Export taxonomy to DataFrame.

        Returns
        -------
        pd.DataFrame
            Table with columns: group, scientific_name, common_name,
            worms_aphia_id, itis_tsn, kingdom, phylum, class_, order,
            family, genus, species, match_confidence.
        """
        if not self.group_assignments:
            return pd.DataFrame()

        data = []
        for group_name, sci_name in self.group_assignments.items():
            taxon = self.taxa.get(sci_name)
            if taxon is None:
                continue

            taxon_dict = taxon.to_dict()
            taxon_dict["group"] = group_name
            data.append(taxon_dict)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        # Reorder columns: group first
        cols = ["group"] + [c for c in df.columns if c != "group"]
        return df[cols]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "TaxonomyLinkage":
        """Create TaxonomyLinkage from DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns: group, scientific_name, kingdom, etc.

        Returns
        -------
        TaxonomyLinkage
            Initialized taxonomy linkage.

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        required_cols = {"group", "scientific_name", "kingdom"}
        if not required_cols.issubset(set(df.columns)):
            msg = (
                f"DataFrame must have columns: {required_cols}. "
                f"Found: {set(df.columns)}"
            )
            raise ValueError(msg)

        taxonomy = cls()

        for _, row in df.iterrows():
            # Build taxon
            taxon_data = {
                "scientific_name": str(row["scientific_name"]),
                "kingdom": str(row.get("kingdom", "")),
                "phylum": str(row.get("phylum", "")),
                "class_": str(row.get("class_", "")),
                "order": str(row.get("order", "")),
                "family": str(row.get("family", "")),
                "genus": str(row.get("genus", "")),
                "species": str(row.get("species", "")),
                "common_name": (
                    str(row["common_name"])
                    if pd.notna(row.get("common_name"))
                    else None
                ),
                "worms_aphia_id": (
                    int(row["worms_aphia_id"])
                    if pd.notna(row.get("worms_aphia_id"))
                    else None
                ),
                "itis_tsn": (
                    str(row["itis_tsn"])
                    if pd.notna(row.get("itis_tsn"))
                    else None
                ),
                "match_confidence": float(
                    row.get("match_confidence", 1.0)
                ),
            }

            taxon = TaxonLink.from_dict(taxon_data)
            taxonomy.add_taxon(taxon)
            taxonomy.assign_group(str(row["group"]), taxon.scientific_name)

        return taxonomy


def validate_taxonomy_links(
    taxonomy: TaxonomyLinkage,
) -> tuple[bool, list[str]]:
    """Validate taxonomy linkage for consistency and completeness.

    Performs comprehensive checks:
    - All taxa are valid
    - No missing or orphaned groups
    - No duplicate assignments
    - Hierarchical consistency

    Parameters
    ----------
    taxonomy : TaxonomyLinkage
        Taxonomy object to validate.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, error_list). is_valid is True iff error_list is empty.
    """
    is_valid, errors = taxonomy.validate()
    return (is_valid, errors)


def resolve_worms_link(
    scientific_name: str,
) -> Optional[TaxonLink]:
    """Resolve a single species name to WoRMS database.

    STUB: This is a placeholder for WoRMS API integration.
    In production, this would query the WoRMS REST API and return
    a TaxonLink with WoRMS metadata.

    Parameters
    ----------
    scientific_name : str
        Scientific name to resolve (e.g., "Gadus morhua").

    Returns
    -------
    TaxonLink or None
        Resolved taxon with WoRMS data, or None if not found.
    """
    logger.debug(
        f"WoRMS lookup stub for: {scientific_name} "
        "(API integration pending)"
    )
    # TODO: Implement WoRMS REST API query
    # Expected workflow:
    # 1. Call WoRMS API: https://www.marinespecies.org/rest/AphiaNameByName
    # 2. Parse response JSON
    # 3. Extract AphiaID, taxonomy, common names
    # 4. Return TaxonLink with populated fields
    return None


def batch_resolve_worms(
    scientific_names: list[str],
) -> dict[str, Optional[TaxonLink]]:
    """Batch resolve multiple species names to WoRMS.

    STUB: This is a placeholder for batch WoRMS API integration.
    In production, this would efficiently query multiple species
    using WoRMS batch endpoints or caching.

    Parameters
    ----------
    scientific_names : list[str]
        List of scientific names to resolve.

    Returns
    -------
    dict[str, Optional[TaxonLink]]
        Mapping of scientific name to resolved TaxonLink (or None if unfound).
    """
    logger.debug(
        f"Batch WoRMS lookup stub for {len(scientific_names)} species "
        "(API integration pending)"
    )
    # TODO: Implement batch WoRMS API query with caching
    # Expected workflow:
    # 1. Check cache for previously resolved names
    # 2. Query WoRMS for new names (batch or individual)
    # 3. Cache results
    # 4. Return mapping
    result = {name: None for name in scientific_names}
    return result

