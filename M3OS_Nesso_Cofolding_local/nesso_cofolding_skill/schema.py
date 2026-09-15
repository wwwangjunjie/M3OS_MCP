import json
from pathlib import Path
from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


IUPAC_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYBJOUXZ")


class Input(BaseModel):
    model_config = ConfigDict(
        title="nesso-cofolding-local input",
        extra="forbid",
        populate_by_name=True,
    )

    job_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("job_name", "task_name"),
        description="Task name used in result filenames.",
    )
    protein_fasta_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("protein_fasta_path", "fasta_path", "protein_fasta"),
        description="Protein FASTA path. Provide exactly one of this field and protein_sequence.",
    )
    protein_sequence: str | None = Field(
        default=None,
        description=(
            "Raw protein sequence. Whitespace is removed and the sequence is "
            "written to a FASTA file before inference. Provide exactly one of "
            "this field and protein_fasta_path."
        ),
    )
    protein_id: str | None = Field(
        default=None,
        description="Optional FASTA record ID used with protein_sequence.",
    )
    ligand_csv_path: str = Field(
        ...,
        validation_alias=AliasChoices("ligand_csv_path", "input_table", "ligand_table_path"),
        description="CSV containing ligand SMILES and an optional compound ID column.",
    )
    smiles_column: str | None = Field(default=None, description="SMILES column; auto-detected when omitted.")
    ligand_id_column: str | None = Field(default=None, description="Compound ID column; auto-detected when omitted.")
    max_ligands: int | None = Field(default=None, ge=1, description="Use only the first N rows for an explicit smoke test.")

    prepare_only: bool = Field(default=False, description="Generate YAML inputs without model inference.")

    @field_validator("protein_sequence")
    @classmethod
    def normalize_protein_sequence(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = "".join(value.split()).upper()
        if not normalized:
            raise ValueError("protein_sequence must not be empty")
        invalid = sorted(set(normalized) - IUPAC_AMINO_ACIDS)
        if invalid:
            raise ValueError(
                "protein_sequence contains invalid amino-acid characters: "
                + ", ".join(invalid)
            )
        return normalized

    @model_validator(mode="after")
    def require_one_protein_input(self) -> "Input":
        supplied = sum(
            value is not None
            for value in (self.protein_fasta_path, self.protein_sequence)
        )
        if supplied != 1:
            raise ValueError(
                "provide exactly one of protein_fasta_path and protein_sequence"
            )
        return self


class ResultRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    rank: int
    ligand_id: str
    smiles: str
    nesso_record_id: str
    nesso_affinity_pred_value: float | None = None
    nesso_affinity_probability_binary: float | None = None
    nesso_entropy_pl: float | None = None
    nesso_entropy_crop_pl: float | None = None
    status: str
    confidence_warning: str = ""
    error: str = ""


class Output(BaseModel):
    model_config = ConfigDict(title="nesso-cofolding-local output", extra="forbid")

    success: bool
    job_name: str
    output_dir: str
    result_csv: str
    summary_json: str
    input_manifest_csv: str = ""
    n_input_rows: int = 0
    n_success: int = 0
    n_failed: int = 0
    n_low_confidence: int = 0
    best_ligand_id: str | None = None
    best_nesso_affinity_pred_value: float | None = None
    best_nesso_affinity_probability_binary: float | None = None
    wall_seconds: float | None = None
    gpu_ids: list[str] = Field(default_factory=list)
    results: list[ResultRow] = Field(default_factory=list)
    command: list[str] = Field(default_factory=list)
    error_code: str = ""
    error_message: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    (here / "input.schema.json").write_text(
        json.dumps(Input.model_json_schema(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (here / "output.schema.json").write_text(
        json.dumps(Output.model_json_schema(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
