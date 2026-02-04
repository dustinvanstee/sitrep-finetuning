"""
LLM-as-a-Judge evaluation for INDOPACOM SITREPs.

Uses a configurable LLM to evaluate generated SITREPs against reference outputs
and scoring rubrics. Uses OpenAI-compatible API (works with NVIDIA build.nvidia.com).

Usage:
    from src.evaluation.llm_judge import LLMJudge, JudgeConfig
    
    config = JudgeConfig(
        provider="nvidia",
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
    )
    judge = LLMJudge(config)
    
    result = judge.evaluate(
        generated_sitrep="...",
        reference_sitrep="...",
        input_intel="..."
    )
"""

import os
import json
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum

from openai import OpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class JudgeProvider(str, Enum):
    """Supported LLM providers for judging."""
    NVIDIA = "nvidia"           # NVIDIA Build API (integrate.api.nvidia.com)
    ASTRA = "astra"             # NVIDIA Astra Inference API (inference-api.nvidia.com)
    OPENAI = "openai"           # OpenAI API
    LOCAL_NIM = "local-nim"     # Local NIM endpoint


@dataclass
class JudgeConfig:
    """Configuration for the LLM judge (OpenAI-compatible API)."""
    provider: str = "nvidia"
    model: str = "moonshotai/kimi-k2-thinking"  # Kimi K2 has excellent JSON output
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.3  # Lower temp for consistent JSON output
    max_tokens: int = 4096
    use_thinking: bool = False  # Kimi handles thinking internally, returns clean JSON
    
    def __post_init__(self):
        # Set defaults based on provider
        if self.provider == "nvidia":
            # Use LLM_JUDGE_BASE_URL from env, fallback to integrate.api.nvidia.com
            if not self.base_url:
                self.base_url = os.environ.get(
                    "LLM_JUDGE_BASE_URL", 
                    "https://integrate.api.nvidia.com/v1"
                )
            if not self.api_key:
                self.api_key = os.environ.get("NVIDIA_API_KEY")
        elif self.provider == "astra":
            # NVIDIA Astra Inference API (GPT models, etc.)
            if not self.base_url:
                self.base_url = "https://inference-api.nvidia.com/v1"
            if not self.api_key:
                self.api_key = os.environ.get("ASTRA_API_KEY")
        elif self.provider == "openai":
            if not self.api_key:
                self.api_key = os.environ.get("OPENAI_API_KEY")
            # OpenAI doesn't need base_url override
        elif self.provider == "local-nim":
            if not self.base_url:
                self.base_url = os.environ.get("NIM_LOCAL_BASE_URL", "http://localhost:8000/v1")
            self.api_key = self.api_key or "not-needed"


# =============================================================================
# EVALUATION RUBRIC & STRUCTURED OUTPUT
# =============================================================================

class DimensionScore(BaseModel):
    """Score for a single evaluation dimension."""
    dimension: str = Field(description="Name of the evaluation dimension")
    score: int = Field(description="Score from 1-5", ge=1, le=5)
    reasoning: str = Field(description="Brief explanation for the score")


class JudgeEvaluation(BaseModel):
    """Structured evaluation output from the LLM judge."""
    # Overall assessment
    overall_score: int = Field(description="Overall score from 1-100", ge=1, le=100)
    overall_reasoning: str = Field(description="Summary of evaluation rationale")
    
    # Dimension scores (each 1-5)
    section_completeness: DimensionScore = Field(
        description="Are all 8 required sections present and properly structured?"
    )
    factual_accuracy: DimensionScore = Field(
        description="Does the SITREP accurately reflect the input intelligence?"
    )
    analytical_quality: DimensionScore = Field(
        description="Quality of assessments, recommendations, and analyst judgments"
    )
    formatting_consistency: DimensionScore = Field(
        description="Proper military format, headers, bullet structure"
    )
    actionability: DimensionScore = Field(
        description="Are risks, watch items, and commander's guidance actionable?"
    )
    
    # Comparison to reference (if provided)
    similarity_to_reference: Optional[int] = Field(
        default=None,
        description="How similar to reference output (1-5), null if no reference",
        ge=1, le=5
    )
    
    # Specific issues found
    issues: List[str] = Field(
        default_factory=list,
        description="List of specific issues or errors found"
    )
    
    # Strengths noted
    strengths: List[str] = Field(
        default_factory=list,
        description="List of notable strengths"
    )


EVALUATION_RUBRIC = """
You are an expert military intelligence analyst evaluating SITREP (Situation Report) quality.

## Evaluation Criteria

Score each dimension from 1-5:
- 5: Excellent - Exceeds expectations, professional quality
- 4: Good - Meets all requirements with minor issues
- 3: Acceptable - Meets basic requirements but has notable gaps
- 2: Below Standard - Missing key elements or significant errors
- 1: Unacceptable - Fails to meet basic requirements

### 1. Section Completeness (1-5)
All 8 sections must be present and properly labeled:
1. BLUF (Bottom Line Up Front)
2. Friendly Forces
3. Adversary Activity
4. Operations
5. Logistics
6. Intelligence Assessment
7. Risks / Watch Items
8. Commander's Comments

### 2. Factual Accuracy (1-5)
- Information from input intel reports accurately reflected
- Vessel/aircraft counts, locations, unit names consistent
- No hallucinated facts or contradictions

### 3. Analytical Quality (1-5)
- BLUF provides clear, prioritized bottom line
- Assessments are logical and supported by evidence
- Intelligence gaps identified
- Analyst judgment provides actionable recommendation

### 4. Formatting Consistency (1-5)
- Proper SITREP header (Unit, DTG, Classification, Period)
- Numbered sections with separators (---)
- Consistent bullet point formatting
- Confidence levels stated where appropriate

### 5. Actionability (1-5)
- Risks clearly identified with potential impacts
- Watch items specific and monitorable
- Commander's comments provide clear guidance
- Recommendations are specific and executable

## Overall Score (1-100)
Weighted combination:
- Section Completeness: 25%
- Factual Accuracy: 25%
- Analytical Quality: 20%
- Formatting Consistency: 15%
- Actionability: 15%
"""


# =============================================================================
# LLM JUDGE CLASS
# =============================================================================

class LLMJudge:
    """LLM-based judge for SITREP evaluation using OpenAI-compatible API."""
    
    def __init__(self, config: JudgeConfig):
        """
        Initialize the LLM judge.
        
        Args:
            config: Judge configuration
        """
        self.config = config
        
        # Initialize OpenAI client
        client_kwargs = {"api_key": config.api_key or "dummy-key"}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        
        self.client = OpenAI(**client_kwargs)
        logger.info(f"Initialized LLM Judge: {config.model} @ {config.base_url or 'default'}")
    
    def _call_llm(self, prompt: str, use_json: bool = True) -> str:
        """
        Call the configured LLM via OpenAI-compatible API.
        
        Args:
            prompt: The prompt to send
            use_json: Whether to request JSON output (hint in system prompt)
            
        Returns:
            LLM response text
        """
        messages = [{"role": "user", "content": prompt}]
        
        # Build request kwargs
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": 0.95,
        }
        
        # Check if this model needs streaming (Kimi K2 returns None in non-streaming)
        needs_streaming = "kimi" in self.config.model.lower()
        
        try:
            if needs_streaming:
                # Use streaming for models like Kimi K2
                kwargs["stream"] = True
                response = self.client.chat.completions.create(**kwargs)
                
                content_parts = []
                for chunk in response:
                    if not getattr(chunk, "choices", None):
                        continue
                    # Get main content (skip reasoning_content for thinking models)
                    chunk_content = chunk.choices[0].delta.content
                    if chunk_content:
                        content_parts.append(chunk_content)
                
                content = "".join(content_parts)
                if not content:
                    logger.warning("LLM streaming returned no content")
                    return "{}"
                return content
            else:
                # Non-streaming for other models
                response = self.client.chat.completions.create(**kwargs)
                
                content = response.choices[0].message.content
                if content is None:
                    logger.warning("LLM returned None content")
                    return "{}"
                return content
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def evaluate(
        self,
        generated_sitrep: str,
        input_intel: Optional[str] = None,
        reference_sitrep: Optional[str] = None,
    ) -> JudgeEvaluation:
        """
        Evaluate a generated SITREP.
        
        Args:
            generated_sitrep: The SITREP to evaluate
            input_intel: Original intelligence inputs (for factual accuracy check)
            reference_sitrep: Reference/ground truth SITREP (optional)
            
        Returns:
            JudgeEvaluation with detailed scores
        """
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(
            generated_sitrep=generated_sitrep,
            input_intel=input_intel,
            reference_sitrep=reference_sitrep
        )
        
        # Call LLM
        response = self._call_llm(prompt, use_json=True)
        
        # === INSTRUMENTATION: Log raw response ===
        response_len = len(response) if response else 0
        logger.info(f"[INSTRUMENT] Raw response length: {response_len} chars")
        if response_len > 0:
            # Show first 200 and last 200 chars
            preview_start = response[:200].replace('\n', '\\n')
            preview_end = response[-200:].replace('\n', '\\n') if response_len > 200 else ""
            logger.info(f"[INSTRUMENT] Response START: {preview_start}")
            if preview_end:
                logger.info(f"[INSTRUMENT] Response END: ...{preview_end}")
        else:
            logger.warning(f"[INSTRUMENT] Response is EMPTY!")
        
        # Parse response
        try:
            # Try to extract JSON from response (model might include extra text)
            json_str = response
            
            # Look for JSON object in response
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                logger.info(f"[INSTRUMENT] Extracted JSON from position {start} to {end} ({end-start} chars)")
            else:
                logger.warning(f"[INSTRUMENT] No JSON braces found in response!")
            
            data = json.loads(json_str)
            logger.info(f"[INSTRUMENT] JSON parsed successfully, overall_score={data.get('overall_score')}")
            return JudgeEvaluation(**data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse judge response as JSON: {e}")
            logger.error(f"[INSTRUMENT] JSON parse failed at position {e.pos}, showing context:")
            # Show context around the error
            if json_str and e.pos:
                context_start = max(0, e.pos - 50)
                context_end = min(len(json_str), e.pos + 50)
                logger.error(f"[INSTRUMENT] Context: ...{json_str[context_start:context_end]}...")
            # Save full response to file for debugging
            self._save_failed_response(response, "json_decode_error")
            # Try to extract a simple score from text response
            return self._parse_text_response(response)
        except Exception as e:
            logger.error(f"Failed to create JudgeEvaluation: {e}")
            self._save_failed_response(response, "validation_error")
            return self._create_fallback_evaluation(str(e))
    
    def _parse_text_response(self, response: str) -> JudgeEvaluation:
        """Attempt to parse a non-JSON text response from the judge."""
        import re
        
        # Try to find score patterns in text
        score_match = re.search(r'overall[:\s]+(\d+)', response, re.IGNORECASE)
        overall_score = int(score_match.group(1)) if score_match else 50
        overall_score = max(1, min(100, overall_score))  # Clamp to valid range
        
        # Default dimension score based on overall
        default_dim_score = max(1, min(5, overall_score // 20))
        
        def make_dim(name: str) -> DimensionScore:
            pattern = rf'{name}[:\s]+(\d)'
            match = re.search(pattern, response, re.IGNORECASE)
            score = int(match.group(1)) if match else default_dim_score
            return DimensionScore(
                dimension=name,
                score=max(1, min(5, score)),
                reasoning=f"Extracted from text response"
            )
        
        return JudgeEvaluation(
            overall_score=overall_score,
            overall_reasoning=response[:500] if response else "Text response parsed",
            section_completeness=make_dim("Section Completeness"),
            factual_accuracy=make_dim("Factual Accuracy"),
            analytical_quality=make_dim("Analytical Quality"),
            formatting_consistency=make_dim("Formatting"),
            actionability=make_dim("Actionability"),
            issues=["Response was not in expected JSON format"],
            strengths=[]
        )
    
    def _build_evaluation_prompt(
        self,
        generated_sitrep: str,
        input_intel: Optional[str],
        reference_sitrep: Optional[str]
    ) -> str:
        """Build the evaluation prompt."""
        
        prompt_parts = [
            EVALUATION_RUBRIC,
            "\n## SITREP to Evaluate\n",
            "```",
            generated_sitrep[:8000],  # Truncate if very long
            "```\n"
        ]
        
        if input_intel:
            prompt_parts.extend([
                "\n## Original Intelligence Inputs\n",
                "Use this to verify factual accuracy:\n",
                "```",
                input_intel[:6000],  # Truncate
                "```\n"
            ])
        
        if reference_sitrep:
            prompt_parts.extend([
                "\n## Reference SITREP (Ground Truth)\n",
                "Compare the evaluated SITREP against this reference:\n",
                "```",
                reference_sitrep[:6000],  # Truncate
                "```\n"
            ])
        
        prompt_parts.extend([
            "\n## Instructions\n",
            "Evaluate the SITREP according to the rubric above.\n",
            "Respond with a JSON object matching this schema:\n",
            "```json",
            json.dumps({
                "overall_score": "integer 1-100",
                "overall_reasoning": "string",
                "section_completeness": {
                    "dimension": "Section Completeness",
                    "score": "integer 1-5",
                    "reasoning": "string"
                },
                "factual_accuracy": {
                    "dimension": "Factual Accuracy", 
                    "score": "integer 1-5",
                    "reasoning": "string"
                },
                "analytical_quality": {
                    "dimension": "Analytical Quality",
                    "score": "integer 1-5", 
                    "reasoning": "string"
                },
                "formatting_consistency": {
                    "dimension": "Formatting Consistency",
                    "score": "integer 1-5",
                    "reasoning": "string"
                },
                "actionability": {
                    "dimension": "Actionability",
                    "score": "integer 1-5",
                    "reasoning": "string"
                },
                "similarity_to_reference": "integer 1-5 or null if no reference",
                "issues": ["list of specific issues found"],
                "strengths": ["list of notable strengths"]
            }, indent=2),
            "```\n",
            "\nProvide your evaluation as valid JSON:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _save_failed_response(self, response: str, error_type: str):
        """Save failed response to file for debugging."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_judge_failed_{error_type}_{timestamp}.txt"
        filepath = os.path.join("outputs/evaluation_results_v2/llm_judge", filename)
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(f"=== FAILED RESPONSE ({error_type}) ===\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Response length: {len(response)} chars\n")
                f.write(f"{'='*50}\n\n")
                f.write(response)
            logger.info(f"[INSTRUMENT] Saved failed response to: {filepath}")
        except Exception as e:
            logger.error(f"[INSTRUMENT] Failed to save response file: {e}")
    
    def _create_fallback_evaluation(self, error_msg: str) -> JudgeEvaluation:
        """Create a fallback evaluation when parsing fails."""
        default_dim = DimensionScore(
            dimension="Unknown",
            score=1,
            reasoning=f"Evaluation failed: {error_msg}"
        )
        return JudgeEvaluation(
            overall_score=1,  # Minimum valid score
            overall_reasoning=f"Evaluation parsing failed: {error_msg}",
            section_completeness=DimensionScore(
                dimension="Section Completeness", score=1, reasoning="Parse failed"
            ),
            factual_accuracy=DimensionScore(
                dimension="Factual Accuracy", score=1, reasoning="Parse failed"
            ),
            analytical_quality=DimensionScore(
                dimension="Analytical Quality", score=1, reasoning="Parse failed"
            ),
            formatting_consistency=DimensionScore(
                dimension="Formatting Consistency", score=1, reasoning="Parse failed"
            ),
            actionability=DimensionScore(
                dimension="Actionability", score=1, reasoning="Parse failed"
            ),
            issues=[f"Evaluation failed: {error_msg}"]
        )
    
    def evaluate_batch(
        self,
        generated_sitreps: List[str],
        input_intels: Optional[List[str]] = None,
        reference_sitreps: Optional[List[str]] = None,
    ) -> List[JudgeEvaluation]:
        """
        Evaluate a batch of SITREPs.
        
        Args:
            generated_sitreps: List of SITREPs to evaluate
            input_intels: Corresponding intelligence inputs (optional)
            reference_sitreps: Corresponding reference SITREPs (optional)
            
        Returns:
            List of JudgeEvaluation results
        """
        results = []
        n = len(generated_sitreps)
        
        for i, sitrep in enumerate(generated_sitreps):
            logger.info(f"Evaluating SITREP {i+1}/{n}")
            
            input_intel = input_intels[i] if input_intels else None
            reference = reference_sitreps[i] if reference_sitreps else None
            
            try:
                result = self.evaluate(
                    generated_sitrep=sitrep,
                    input_intel=input_intel,
                    reference_sitrep=reference
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate SITREP {i+1}: {e}")
                results.append(self._create_fallback_evaluation(str(e)))
        
        return results


def compute_judge_statistics(evaluations: List[JudgeEvaluation]) -> Dict[str, Any]:
    """
    Compute aggregate statistics from judge evaluations.
    
    Args:
        evaluations: List of JudgeEvaluation results
        
    Returns:
        Dictionary with aggregate statistics
    """
    if not evaluations:
        return {"num_evaluated": 0}
    
    valid_evals = [e for e in evaluations if e.overall_score > 0]
    
    if not valid_evals:
        return {
            "num_evaluated": len(evaluations),
            "num_valid": 0,
            "error": "All evaluations failed"
        }
    
    def avg(scores: List[int]) -> float:
        return sum(scores) / len(scores) if scores else 0
    
    stats = {
        "num_evaluated": len(evaluations),
        "num_valid": len(valid_evals),
        "overall": {
            "mean": avg([e.overall_score for e in valid_evals]),
            "min": min(e.overall_score for e in valid_evals),
            "max": max(e.overall_score for e in valid_evals),
        },
        "section_completeness": {
            "mean": avg([e.section_completeness.score for e in valid_evals]),
        },
        "factual_accuracy": {
            "mean": avg([e.factual_accuracy.score for e in valid_evals]),
        },
        "analytical_quality": {
            "mean": avg([e.analytical_quality.score for e in valid_evals]),
        },
        "formatting_consistency": {
            "mean": avg([e.formatting_consistency.score for e in valid_evals]),
        },
        "actionability": {
            "mean": avg([e.actionability.score for e in valid_evals]),
        },
    }
    
    # Similarity if references were provided
    similarity_scores = [
        e.similarity_to_reference for e in valid_evals 
        if e.similarity_to_reference is not None
    ]
    if similarity_scores:
        stats["similarity_to_reference"] = {
            "mean": avg(similarity_scores),
            "min": min(similarity_scores),
            "max": max(similarity_scores),
        }
    
    # Common issues and strengths
    all_issues = []
    all_strengths = []
    for e in valid_evals:
        all_issues.extend(e.issues)
        all_strengths.extend(e.strengths)
    
    # Count frequency
    from collections import Counter
    stats["common_issues"] = dict(Counter(all_issues).most_common(10))
    stats["common_strengths"] = dict(Counter(all_strengths).most_common(10))
    
    return stats


def compare_models_with_judge(
    judge: LLMJudge,
    baseline_sitreps: List[str],
    finetuned_sitreps: List[str],
    input_intels: Optional[List[str]] = None,
    reference_sitreps: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compare baseline and fine-tuned models using LLM judge.
    
    Args:
        judge: Configured LLMJudge instance
        baseline_sitreps: Outputs from baseline model
        finetuned_sitreps: Outputs from fine-tuned model
        input_intels: Original inputs (for factual accuracy)
        reference_sitreps: Ground truth references
        
    Returns:
        Comparison statistics
    """
    logger.info("Evaluating baseline model outputs...")
    baseline_evals = judge.evaluate_batch(
        baseline_sitreps, input_intels, reference_sitreps
    )
    baseline_stats = compute_judge_statistics(baseline_evals)
    
    logger.info("Evaluating fine-tuned model outputs...")
    finetuned_evals = judge.evaluate_batch(
        finetuned_sitreps, input_intels, reference_sitreps
    )
    finetuned_stats = compute_judge_statistics(finetuned_evals)
    
    # Compute improvements
    improvement = {}
    if baseline_stats.get("overall") and finetuned_stats.get("overall"):
        baseline_mean = baseline_stats["overall"]["mean"]
        finetuned_mean = finetuned_stats["overall"]["mean"]
        improvement["overall_improvement"] = finetuned_mean - baseline_mean
        improvement["overall_improvement_pct"] = (
            (finetuned_mean - baseline_mean) / baseline_mean * 100
            if baseline_mean > 0 else 0
        )
        
        # Per-dimension improvements
        for dim in ["section_completeness", "factual_accuracy", "analytical_quality",
                    "formatting_consistency", "actionability"]:
            if dim in baseline_stats and dim in finetuned_stats:
                improvement[f"{dim}_improvement"] = (
                    finetuned_stats[dim]["mean"] - baseline_stats[dim]["mean"]
                )
    
    return {
        "baseline": baseline_stats,
        "finetuned": finetuned_stats,
        "improvement": improvement,
        "baseline_evaluations": [e.model_dump() for e in baseline_evals],
        "finetuned_evaluations": [e.model_dump() for e in finetuned_evals],
    }
