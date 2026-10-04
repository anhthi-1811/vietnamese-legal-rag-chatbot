import os
import json
import time
import logging
from typing import Any, Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

ALLOWED_CORRECTNESS_LABELS = {
    "correct",
    "partial",
    "incorrect",
}

ALLOWED_COMPLETENESS_LABELS = {
    "complete",
    "partial",
    "incomplete",
}

ALLOWED_GROUNDEDNESS_LABELS = {
    "supported",
    "partially_supported",
    "unsupported",
}

ALLOWED_CITATION_SUPPORT_LABELS = {
    "supported",
    "partially_supported",
    "unsupported",
}

SCORE_TO_LABEL = {
    "answer_correctness": {0: "incorrect", 1: "partial", 2: "correct"},
    "answer_completeness": {0: "incomplete", 1: "partial", 2: "complete"},
    "groundedness": {0: "unsupported", 1: "partially_supported", 2: "supported"},
    "citation_support": {0: "unsupported", 1: "partially_supported", 2: "supported"},
}

ALLOWED_LABELS_BY_DIM = {
    "answer_correctness": ALLOWED_CORRECTNESS_LABELS,
    "answer_completeness": ALLOWED_COMPLETENESS_LABELS,
    "groundedness": ALLOWED_GROUNDEDNESS_LABELS,
    "citation_support": ALLOWED_CITATION_SUPPORT_LABELS,
}

# Safe within-dimension aliases only (never map across dimensions)
SAFE_LABEL_ALIASES = {
    "answer_correctness": {
        "fully_correct": "correct",
        "partially_correct": "partial",
    },
    "answer_completeness": {
        "fully_complete": "complete",
        "partially_complete": "partial",
    },
    "groundedness": {
        "fully_supported": "supported",
    },
    "citation_support": {
        "fully_supported": "supported",
    },
}

ALLOWED_ERROR_TAGS = {
    "retrieval_miss",
    "equivalent_non_gold_evidence",
    "partial_multi_gold_retrieval",
    "unnecessary_abstention",
    "valid_abstention",
    "generator_omission",
    "generator_overanswer",
    "generator_wrong_scope",
    "unsupported_claim",
    "semantic_citation_issue",
    "benchmark_label_noise",
    "incomplete_information",
    "conflicting_evidence",
    "temporal_legal_conflict",
    "legal_inference_error",
    "evidence_selection_error",
}

ABSTENTION_ONLY_TAGS = {"valid_abstention", "unnecessary_abstention"}


def normalize_and_clean_judge_data(
    parsed_data: Dict[str, Any], answer_status: str
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Applies safe within-dimension alias normalization and deterministic abstention rules.
    Returns (cleaned_data, warnings).
    """
    warnings = []
    data = dict(parsed_data)

    # 1. Safe label alias normalization per dimension
    for dim, alias_map in SAFE_LABEL_ALIASES.items():
        dim_obj = data.get(dim)
        if isinstance(dim_obj, dict):
            raw_label = str(dim_obj.get("label", "")).strip().lower()
            if raw_label in alias_map:
                mapped = alias_map[raw_label]
                warnings.append(f"Normalized {dim}.label '{raw_label}' -> '{mapped}'")
                dim_obj["label"] = mapped
            else:
                dim_obj["label"] = raw_label

            # Coerce score to int if possible
            try:
                dim_obj["score"] = int(dim_obj.get("score"))
            except (ValueError, TypeError):
                pass

    # 2. Extract and normalize raw error tags
    raw_tags = data.get("error_tags", [])
    if isinstance(raw_tags, (list, tuple, np.ndarray)):
        tag_list = [str(t).strip().lower() for t in raw_tags if str(t).strip()]
    else:
        tag_list = []

    # 3. Deterministic abstention enforcement
    abstention = data.get("abstention")
    if not isinstance(abstention, dict):
        abstention = {"applicable": False, "appropriate": False, "reason": ""}

    if answer_status != "insufficient_evidence":
        if abstention.get("applicable") is True:
            warnings.append("Forced abstention.applicable=False for answered query.")
        abstention["applicable"] = False
        abstention["appropriate"] = False
        abstention["reason"] = ""

        # Strip forbidden abstention tags from answered queries
        cleaned_tags = []
        for t in tag_list:
            if t in ABSTENTION_ONLY_TAGS:
                warnings.append(f"Removed forbidden tag '{t}' from answered query.")
            else:
                cleaned_tags.append(t)
        tag_list = cleaned_tags
    else:
        abstention["applicable"] = True
        approp = bool(abstention.get("appropriate", False))
        abstention["appropriate"] = approp

        # Remove LLM-supplied abstention tags and derive deterministically
        tag_list = [t for t in tag_list if t not in ABSTENTION_ONLY_TAGS]
        derived_tag = "valid_abstention" if approp else "unnecessary_abstention"
        tag_list.append(derived_tag)

    # Preserve unique order of tags
    seen = set()
    dedup_tags = []
    for t in tag_list:
        if t not in seen:
            seen.add(t)
            dedup_tags.append(t)

    data["abstention"] = abstention
    data["error_tags"] = dedup_tags
    return data, warnings


def validate_judge_data(data: Dict[str, Any], answer_status: str) -> List[str]:
    """
    Strictly validates normalized judge output.
    Returns a list of validation error strings (empty if valid).
    """
    errors = []

    # Validate the 4 dimensions
    for dim, allowed_labels in ALLOWED_LABELS_BY_DIM.items():
        dim_obj = data.get(dim)
        if not isinstance(dim_obj, dict):
            errors.append(f"Field '{dim}' is missing or not an object.")
            continue

        score = dim_obj.get("score")
        label = dim_obj.get("label")

        if score not in (0, 1, 2) or isinstance(score, bool):
            errors.append(
                f"Invalid {dim}.score='{score}'. Allowed scores: 0, 1, 2."
            )
        if label not in allowed_labels:
            errors.append(
                f"Invalid {dim}.label='{label}'. Allowed labels for {dim}: {sorted(list(allowed_labels))}."
            )
        if score in (0, 1, 2) and not isinstance(score, bool) and label in allowed_labels:
            expected_label = SCORE_TO_LABEL[dim][score]
            if label != expected_label:
                errors.append(
                    f"Score-label mismatch in {dim}: score={score} requires label='{expected_label}', but got label='{label}'."
                )

    # Validate abstention consistency
    abstention = data.get("abstention", {})
    tags = set(data.get("error_tags", []))

    if answer_status != "insufficient_evidence":
        if abstention.get("applicable") is not False:
            errors.append("For answer_status='answered', abstention.applicable must be false.")
        if tags.intersection(ABSTENTION_ONLY_TAGS):
            errors.append(
                f"Answered query cannot contain abstention tags: {tags.intersection(ABSTENTION_ONLY_TAGS)}."
            )
    else:
        if abstention.get("applicable") is not True:
            errors.append("For answer_status='insufficient_evidence', abstention.applicable must be true.")
        if not isinstance(abstention.get("appropriate"), bool):
            errors.append("abstention.appropriate must be a boolean.")
        else:
            if abstention["appropriate"] is True:
                if "valid_abstention" not in tags or "unnecessary_abstention" in tags:
                    errors.append("Abstention tags inconsistent with abstention.appropriate=True.")
            else:
                if "unnecessary_abstention" not in tags or "valid_abstention" in tags:
                    errors.append("Abstention tags inconsistent with abstention.appropriate=False.")

    # Validate error tags against controlled vocabulary
    unknown_tags = [t for t in data.get("error_tags", []) if t not in ALLOWED_ERROR_TAGS]
    if unknown_tags:
        errors.append(
            f"Unknown error_tags {unknown_tags}. Allowed error_tags: {sorted(list(ALLOWED_ERROR_TAGS))}."
        )

    return errors


class SemanticRAGJudge:
    """
    Strict schema-validated LLM-as-a-judge module for Vietnamese Legal RAG evaluation.
    """
    JUDGE_PROMPT = """You are an expert Vietnamese legal scholar and AI evaluation judge. Your task is to evaluate an answer generated by a Vietnamese Legal RAG system.

    CRITICAL BENCHMARK EVALUATION 
CRITICAL SOURCE SEPARATION PROTOCOL (MANDATORY):
1. CORRECTNESS & COMPLETENESS -> Evaluate against REFERENCE/GOLD EVIDENCE.
2. GROUNDEDNESS -> Evaluate STRICTLY against RETRIEVED CONTEXT ONLY.
   - Ask ONLY: "Are the claims in the answer supported by ANY source inside RETRIEVED CONTEXT?"
   - NEVER penalize groundedness using REFERENCE/GOLD EVIDENCE. If retrieved sources conflict with gold or with each other, still mark groundedness="supported" if the answer's claims appear in retrieved context (and tag `conflicting_evidence` or `temporal_legal_conflict`).
   - If the answer states that the provided sources do not contain X, verify ONLY whether X is in RETRIEVED CONTEXT (ignore GOLD_CID if GOLD_CID is not in RETRIEVED CONTEXT).
3. CITATION SUPPORT -> Evaluate STRICTLY against cited sources in RETRIEVED CONTEXT ONLY.
4. ABSTENTION APPROPRIATENESS -> Evaluate STRICTLY against RETRIEVED CONTEXT ONLY.
   - NEVER use REFERENCE/GOLD EVIDENCE to determine whether the generator had sufficient evidence to answer.
   - For abstention evaluation, PRETEND THAT GOLD/REFERENCE EVIDENCE DOES NOT EXIST.
   - Do NOT require the generator to extrapolate from general rules to a specific named entity/fund/organization or infer a numeric rate from "state-funded" if the exact requested figure/entity is absent from RETRIEVED CONTEXT. In such cases, abstention is VALID (`abstention_appropriate=true`, tag `valid_abstention`).

RULES (CLOSED-WORLD ASSUMPTION):
    1. You are evaluating THIS BENCHMARK, NOT verifying current real-world Vietnamese law.
    2. For `answer_correctness` and `answer_completeness`: use ONLY the supplied REFERENCE EVIDENCE.
    3. For `groundedness` and `citation_support`: use ONLY the RETRIEVED EVIDENCE and cited source texts.
    4. NEVER lower a score because you believe a supplied source is outdated, superseded, legally inaccurate, or inconsistent with your internal legal knowledge. If the retrieved source explicitly supports a generated claim, that claim IS GROUNDED for this evaluation even if you believe the source may be outdated or superseded.
    5. NEVER use phrases such as "theo pháp luật hiện hành", "pháp luật hiện nay", "current law", "outdated", "superseded", or "quy định cũ" unless that explicit temporal statement is literally contained inside the supplied text.

    EXPLICIT RUBRIC & STRICT SCORE-LABEL MAPPING:

    1. `answer_correctness` (Allowed labels ONLY: "correct", "partial", "incorrect"):
    - score 2, label "correct": The generated answer correctly resolves the legal question according to the provided reference evidence.
    - score 1, label "partial": The core answer is mostly correct according to the reference evidence, but contains a minor factual/legal qualification issue, incomplete scope, or partially inaccurate detail. (NEVER use "incomplete" here; use "partial").
    - score 0, label "incorrect": The essential answer is wrong, answers the wrong subject/scope, or conflicts with the reference evidence.

    2. `answer_completeness` (Allowed labels ONLY: "complete", "partial", "incomplete"):
    - score 2, label "complete": All essential information required by the question is included based on the reference evidence.
    - score 1, label "partial": The core answer is present, but one or more relevant conditions, exceptions, steps, or secondary details from the reference evidence are missing.
    - score 0, label "incomplete": Essential information needed to resolve the question is missing. (NEVER use "incorrect" here; use "incomplete").

    3. `groundedness` (Allowed labels ONLY: "supported", "partially_supported", "unsupported"):
    - score 2, label "supported": All material factual/legal claims in the answer are supported by the retrieved context (regardless of external law validity).
    - score 1, label "partially_supported": The main answer is supported, but one or more claims go beyond the retrieved evidence.
    - score 0, label "unsupported": The essential answer is not supported by retrieved evidence.

    4. `citation_support` (Allowed labels ONLY: "supported", "partially_supported", "unsupported"):
    - score 2, label "supported": Every cited [CID=xxx] directly supports the specific claim it is attached to.
    - score 1, label "partially_supported": Some cited claims are supported, but at least one citation only partially supports the associated claim.
    - score 0, label "unsupported": The cited source does not support the claim it is attached to.

    5. `abstention`:
    - If Answer Status is "answered": set `"applicable": false, "appropriate": false, "reason": ""`.
    - If Answer Status is "insufficient_evidence": set `"applicable": true`. Evaluate `appropriate` STRICTLY against RETRIEVED EVIDENCE ONLY:
        + `"appropriate": true` if RETRIEVED EVIDENCE truly lacks information to answer safely.
        + `"appropriate": false` if RETRIEVED EVIDENCE clearly contains the information.

    6. `error_tags`:
    Choose 0 or more tags ONLY from:
    ["retrieval_miss", "equivalent_non_gold_evidence", "partial_multi_gold_retrieval", "unnecessary_abstention", "valid_abstention", "generator_omission", "generator_overanswer", "generator_wrong_scope", "unsupported_claim", "semantic_citation_issue", "benchmark_label_noise", "incomplete_information", "conflicting_evidence", "temporal_legal_conflict", "legal_inference_error", "evidence_selection_error"]

    RETURN STRICTLY A VALID JSON OBJECT:
    {
    "answer_correctness": {"score": 2, "label": "correct", "reason": "Explanation"},
    "answer_completeness": {"score": 2, "label": "complete", "reason": "Explanation"},
    "groundedness": {"score": 2, "label": "supported", "reason": "Explanation"},
    "citation_support": {"score": 2, "label": "supported", "reason": "Explanation"},
    "abstention": {"applicable": false, "appropriate": false, "reason": ""},
    "error_tags": [] 
    }
    """
    def __init__(self, model_name: str = "gemini-3.1-flash-lite", api_key: str = None, api_retries: int = 3):
        self.model_name = model_name
        self.api_retries = api_retries
        self.client = None

        try:
            from google import genai
            from google.genai import types
            key = api_key or os.environ.get("GEMINI_API_KEY")
            if key:
                self.client = genai.Client(api_key=key)
            self.gen_config = types.GenerateContentConfig(
                system_instruction=self.JUDGE_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
            )
        except ImportError:
            logger.error("google-genai package is not installed.")

    def _call_llm(self, prompt: str) -> Tuple[str, float]:
        """Calls Gemini with exponential backoff for transient/rate-limit errors."""
        max_retries = max(self.api_retries, 4)
        for attempt in range(1, max_retries + 1):
            try:
                t0 = time.time()
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self.gen_config,
                )
                latency = (time.time() - t0) * 1000
                return response.text, latency
            except Exception as e:
                err_str = str(e)
                if attempt == max_retries:
                    raise
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    time.sleep(62)
                else:
                    time.sleep(5 * (2 ** (attempt - 1)))
        raise RuntimeError("Exhausted API retries")

    @staticmethod
    def _parse_json(raw_text: str) -> Dict[str, Any]:
        cleaned = raw_text.strip().strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
        return json.loads(cleaned)

    def evaluate(
        self,
        question: str,
        answer: str,
        answer_status: str,
        reference_context: str,
        retrieved_context: str,
    ) -> Dict[str, Any]:
        """
        Evaluates a single RAG sample with strict post-validation and 1 correction retry if invalid.
        """
        base_prompt = (
            f"--- QUERY INFORMATION ---\n"
            f"Question: {question}\n"
            f"Answer Status: {answer_status}\n"
            f"Generated Answer: {answer}\n\n"
            f"--- REFERENCE EVIDENCE (For Correctness & Completeness ONLY) ---\n"
            f"{reference_context}\n\n"
            f"--- RETRIEVED EVIDENCE (For Groundedness, Citation Support & Abstention ONLY) ---\n"
            f"{retrieved_context}\n"
        )

        raw_responses = []
        total_latency = 0.0

        for validation_attempt in range(2):  # Initial call + at most 1 correction retry
            prompt_to_send = base_prompt if validation_attempt == 0 else correction_prompt
            try:
                raw_text, latency = self._call_llm(prompt_to_send)
                total_latency += latency
                raw_responses.append(raw_text)
            except Exception as api_err:
                return {
                    "judge_status": "error",
                    "judge_error": f"API error on attempt {validation_attempt + 1}: {api_err}",
                    "judge_raw_response": "\n---RETRY---\n".join(raw_responses),
                    "judge_latency_ms": total_latency,
                    "answer_correctness": {"score": 0, "label": "incorrect", "reason": ""},
                    "answer_completeness": {"score": 0, "label": "incomplete", "reason": ""},
                    "groundedness": {"score": 0, "label": "unsupported", "reason": ""},
                    "citation_support": {"score": 0, "label": "unsupported", "reason": ""},
                    "abstention": {"applicable": answer_status == "insufficient_evidence", "appropriate": False, "reason": ""},
                    "error_tags": [],
                }

            try:
                parsed = self._parse_json(raw_text)
                cleaned, warnings = normalize_and_clean_judge_data(parsed, answer_status)
                val_errors = validate_judge_data(cleaned, answer_status)
            except Exception as parse_err:
                cleaned = {}
                warnings = []
                val_errors = [f"JSON parse failure: {parse_err}"]

            if not val_errors:
                cleaned["judge_raw_response"] = "\n---RETRY---\n".join(raw_responses)
                cleaned["judge_latency_ms"] = total_latency
                cleaned["judge_status"] = "ok"
                cleaned["judge_error"] = ""
                cleaned["validation_warnings"] = warnings
                return cleaned

            # Prepare correction prompt for the single retry
            if validation_attempt == 0:
                err_bullet = "\n".join(f"- {e}" for e in val_errors)
                correction_prompt = (
                    f"{base_prompt}\n\n"
                    f"=== SCHEMA VALIDATION ERROR IN YOUR PREVIOUS JSON ===\n"
                    f"Your previous JSON response was:\n{raw_text}\n\n"
                    f"It failed strict validation for the following reason(s):\n{err_bullet}\n\n"
                    f"STRICT ALLOWED MAPPINGS:\n"
                    f"- answer_correctness: score 2 -> 'correct', score 1 -> 'partial', score 0 -> 'incorrect'\n"
                    f"- answer_completeness: score 2 -> 'complete', score 1 -> 'partial', score 0 -> 'incomplete'\n"
                    f"- groundedness: score 2 -> 'supported', score 1 -> 'partially_supported', score 0 -> 'unsupported'\n"
                    f"- citation_support: score 2 -> 'supported', score 1 -> 'partially_supported', score 0 -> 'unsupported'\n"
                    f"- Allowed error_tags: {sorted(list(ALLOWED_ERROR_TAGS))}\n\n"
                    f"Return ONLY the corrected valid JSON object."
                )

        # Second response is still invalid -> fail explicitly without accepting invalid output
        cleaned["judge_raw_response"] = "\n---RETRY---\n".join(raw_responses)
        cleaned["judge_latency_ms"] = total_latency
        cleaned["judge_status"] = "error"
        cleaned["judge_error"] = "Validation failed after retry: " + "; ".join(val_errors)
        return cleaned
