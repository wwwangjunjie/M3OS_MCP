#!/usr/bin/env python3
"""Resident TamGen inference runtime."""

from __future__ import annotations

from collections import defaultdict
import logging
import os
from pathlib import Path
import random
import time
from typing import Any
import warnings

import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from fairseq import checkpoint_utils, options, tasks, utils


LOGGER = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings(
    "ignore",
    message=r"An output with one or more elements was resized.*",
    category=UserWarning,
    module=r"fairseq\..*",
)
ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "crossdock_pdb_A10" / "checkpoint_best.pt"
DEFAULT_GPT_CHECKPOINT = ROOT / "gpt_model" / "checkpoint_best.pt"


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def _validate_runtime_files(checkpoint: Path, gpt_checkpoint: Path) -> None:
    missing = [str(path) for path in (checkpoint, gpt_checkpoint) if not path.is_file()]
    dictionary = gpt_checkpoint.parent / "dict.txt"
    if not dictionary.is_file():
        missing.append(str(dictionary))
    if missing:
        raise FileNotFoundError("TamGen model files are missing: " + ", ".join(missing))


def _canonical_candidate(smiles: str) -> tuple[str, Chem.Mol] | None:
    value = str(smiles or "").replace("[generation]", "").replace(" ", "").strip()
    if not value or "*" in value or "." in value:
        return None
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        return None
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        return None
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    return canonical, mol


class TamGenRuntime:
    """Load TamGen once and reuse it across serialized MCP requests."""

    def __init__(self) -> None:
        self.checkpoint = _env_path("TAMGEN_CHECKPOINT", DEFAULT_CHECKPOINT)
        self.gpt_checkpoint = _env_path("TAMGEN_GPT_CHECKPOINT", DEFAULT_GPT_CHECKPOINT)
        _validate_runtime_files(self.checkpoint, self.gpt_checkpoint)

        requested_device = os.environ.get("TAMGEN_DEVICE", "cuda:0").strip()
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("TAMGEN_DEVICE requests CUDA, but CUDA is unavailable")
        self.device = torch.device(requested_device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        template_data = SOURCE_DIR / "TamGen_Demo_Data"
        input_args = [
            str(template_data),
            "-s", "tg",
            "-t", "m1",
            "--task", "translation_coord",
            "--path", str(self.checkpoint),
            "--gen-subset", "test",
            "--beam", "20",
            "--nbest", "20",
            "--max-tokens", os.environ.get("TAMGEN_MAX_TOKENS", "4096"),
            "--seed", "1",
            "--sample-beta", "1.0",
            "--use-src-coord",
            "--gen-vae",
        ]
        if os.environ.get("TAMGEN_FP16", "0").strip().lower() in {"1", "true", "yes"}:
            input_args.append("--fp16")
        parser = options.get_generation_parser()
        self.args = options.parse_args_and_arch(parser, input_args)
        self.args.pretrained_gpt_checkpoint = str(self.gpt_checkpoint)
        utils.import_user_module(self.args)
        self.task = tasks.setup_task(self.args)

        overrides = {
            "sample_beta": 1.0,
            "gen_coord_noise": False,
            "gen_rot": False,
            "gen_vae": True,
            "pretrained_gpt_checkpoint": str(self.gpt_checkpoint),
        }
        started = time.perf_counter()
        self.models, _model_args = checkpoint_utils.load_model_ensemble(
            [str(self.checkpoint)],
            arg_overrides=overrides,
            task=self.task,
        )
        for model in self.models:
            model.make_generation_fast_(beamable_mm_beam_size=None, need_attn=False)
            if self.args.fp16:
                model.half()
            model.to(self.device)
            model.eval()
        self.max_position = utils.resolve_max_positions(
            self.task.max_positions(),
            *[model.max_positions() for model in self.models],
        )
        self.loaded_at = time.time()
        self.load_seconds = time.perf_counter() - started
        LOGGER.info(
            "TamGen loaded checkpoint=%s device=%s load_seconds=%.3f",
            self.checkpoint,
            self.device,
            self.load_seconds,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "loaded": True,
            "checkpoint": str(self.checkpoint),
            "gpt_checkpoint": str(self.gpt_checkpoint),
            "device": str(self.device),
            "load_seconds": self.load_seconds,
            "loaded_at": self.loaded_at,
        }

    def _configure_context(
        self,
        dataset_path: Path,
        subset: str,
        *,
        conditional: bool,
        beam_size: int,
        sample_beta: float,
    ):
        self.args.data = str(dataset_path)
        self.task.args.data = str(dataset_path)
        self.args.gen_subset = subset
        self.args.gen_vae = conditional
        self.task.args.gen_vae = conditional
        self.args.beam = beam_size
        self.args.nbest = beam_size
        self.args.sample_beta = sample_beta
        for model in self.models:
            encoder = getattr(model, "encoder", None)
            if encoder is not None and hasattr(encoder, "gen_vae"):
                encoder.gen_vae = conditional
            if encoder is not None and hasattr(encoder, "sample_beta"):
                encoder.sample_beta = sample_beta

        self.task.load_dataset(subset)
        generator = self.task.build_generator(self.args)
        return generator

    def _new_iterator(self, subset: str):
        return self.task.get_batch_iterator(
            dataset=self.task.dataset(subset),
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.max_sentences,
            max_positions=self.max_position,
            ignore_invalid_inputs=self.args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=self.args.required_batch_size_multiple,
            num_shards=1,
            shard_id=0,
            num_workers=0,
        ).next_epoch_itr(shuffle=False)

    @staticmethod
    def _similarity_fn(seed_smiles: str):
        if not seed_smiles:
            return lambda _mol: None
        seed = Chem.MolFromSmiles(seed_smiles)
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        seed_fp = generator.GetFingerprint(seed)
        return lambda mol: float(DataStructs.TanimotoSimilarity(seed_fp, generator.GetFingerprint(mol)))

    @torch.inference_mode()
    def generate(
        self,
        context: dict[str, Any],
        *,
        num_candidates: int,
        beam_size: int,
        sample_beta: float,
        max_random_seeds: int,
        random_seed: int,
        min_seed_similarity: float,
        require_scaffold_match: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not 1 <= num_candidates <= 10000:
            raise ValueError("num_candidates must be between 1 and 10000")
        if not 1 <= beam_size <= 100:
            raise ValueError("beam_size must be between 1 and 100")
        if not 0 <= sample_beta <= 5:
            raise ValueError("sample_beta must be between 0 and 5")
        if not 1 <= max_random_seeds <= 1000:
            raise ValueError("max_random_seeds must be between 1 and 1000")
        if not 0 <= min_seed_similarity <= 1:
            raise ValueError("min_seed_similarity must be between 0 and 1")

        dataset_path = Path(context["dataset_path"]).resolve()
        subset = str(context.get("subset", "test"))
        conditional = bool(context.get("conditional", True))
        seed_smiles = str(context.get("seed_smiles", ""))
        scaffold_smiles = str(context.get("scaffold_smiles", ""))
        scaffold = Chem.MolFromSmiles(scaffold_smiles) if scaffold_smiles else None
        if require_scaffold_match and scaffold is None:
            raise ValueError(
                "require_scaffold_match needs a scaffold-conditioned TamGen context"
            )
        similarity = self._similarity_fn(seed_smiles)
        generator = self._configure_context(
            dataset_path,
            subset,
            conditional=conditional,
            beam_size=beam_size,
            sample_beta=sample_beta,
        )

        scores: dict[str, list[float]] = defaultdict(list)
        similarities: dict[str, float | None] = {}
        attempts_used = 0
        tgt_tokens_present = False
        tgt_nonpadding_tokens = 0
        started = time.perf_counter()
        for offset in range(max_random_seeds):
            attempts_used = offset + 1
            seed = int(random_seed) + offset
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(seed)
            iterator = self._new_iterator(subset)
            for sample in iterator:
                sample = utils.move_to_cuda(sample) if self.device.type == "cuda" else sample
                if "net_input" not in sample:
                    continue
                tgt_tokens = sample["net_input"].get("tgt_tokens")
                if tgt_tokens is not None:
                    nonpadding = int(
                        tgt_tokens.ne(self.task.target_dictionary.pad()).sum().item()
                    )
                    tgt_tokens_present = tgt_tokens_present or nonpadding > 0
                    tgt_nonpadding_tokens += nonpadding
                hypotheses = self.task.inference_step(generator, self.models, sample, None)
                for item_hypotheses in hypotheses:
                    for hypothesis in item_hypotheses[:beam_size]:
                        _tokens, text, _alignment = utils.post_process_prediction(
                            hypo_tokens=hypothesis["tokens"].int().cpu(),
                            src_str="",
                            alignment=(
                                hypothesis["alignment"].int().cpu()
                                if hypothesis.get("alignment") is not None
                                else None
                            ),
                            align_dict=None,
                            tgt_dict=self.task.target_dictionary,
                            remove_bpe=self.args.remove_bpe,
                        )
                        parsed = _canonical_candidate(text)
                        if parsed is None:
                            continue
                        canonical, mol = parsed
                        if (
                            require_scaffold_match
                            and scaffold is not None
                            and not mol.HasSubstructMatch(scaffold, useChirality=False)
                        ):
                            continue
                        candidate_similarity = similarity(mol)
                        if (
                            candidate_similarity is not None
                            and candidate_similarity < min_seed_similarity
                        ):
                            continue
                        scores[canonical].append(float(hypothesis["score"]))
                        similarities[canonical] = candidate_similarity
            if len(scores) >= num_candidates:
                break

        ranked = sorted(
            scores,
            key=lambda smiles: (float(np.mean(scores[smiles])), max(scores[smiles])),
            reverse=True,
        )[:num_candidates]
        records = [
            {
                "SMILES": smiles,
                "TamGen_generation_score": float(np.mean(scores[smiles])),
                "TamGen_seed_similarity": similarities[smiles],
                "TamGen_generation_count": len(scores[smiles]),
                "TamGen_context_id": context["context_id"],
                "TamGen_mode": "conditional" if conditional else "de_novo",
                "Input_SMILES": seed_smiles,
                "TamGen_scaffold_smiles": scaffold_smiles,
                "TamGen_scaffold_match": bool(
                    scaffold is not None
                    and Chem.MolFromSmiles(smiles).HasSubstructMatch(
                        scaffold, useChirality=False
                    )
                ),
            }
            for smiles in ranked
        ]
        timing = {
            "generation_seconds": time.perf_counter() - started,
            "beam_size": beam_size,
            "sample_beta": sample_beta,
            "random_seeds_used": attempts_used,
            "unique_valid_count": len(scores),
            "requested_count": num_candidates,
            "returned_count": len(records),
            "use_conditional": conditional,
            "gen_vae": bool(
                conditional
                and all(
                    getattr(model.encoder, "gen_vae", False)
                    for model in self.models
                )
            ),
            "tgt_tokens_present": tgt_tokens_present,
            "tgt_nonpadding_tokens_seen": tgt_nonpadding_tokens,
            "conditional_path_active": bool(
                conditional
                and tgt_tokens_present
                and all(
                    getattr(model.encoder, "gen_vae", False)
                    for model in self.models
                )
            ),
            "condition_type": str(context.get("condition_type", "seed")),
            "conditioning_smiles": str(
                context.get("conditioning_smiles", seed_smiles)
            ),
            "require_scaffold_match": bool(require_scaffold_match),
        }
        if conditional and not timing["conditional_path_active"]:
            raise RuntimeError(
                "TamGen conditional generation was requested but the model did not "
                "receive non-empty target tokens with gen_vae enabled"
            )
        return records, timing
