"""
Formatting quality metrics for INDOPACOM SITREP evaluation (V2).
Evaluates the 8-section military SITREP format from V2 data generation.

Sections evaluated:
1. BLUF (Bottom Line Up Front)
2. Friendly Forces
3. Adversary Activity
4. Operations
5. Logistics
6. Intelligence Assessment
7. Risks / Watch Items
8. Commander's Comments
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SectionScore:
    """Score for a specific section or metric."""
    name: str
    score: float
    max_score: float
    present: bool
    details: str = ""

    @property
    def percentage(self) -> float:
        """Get score as percentage."""
        return (self.score / self.max_score * 100) if self.max_score > 0 else 0


@dataclass
class FormattingScoreV2:
    """Complete formatting score for an INDOPACOM SITREP."""
    total_score: float
    max_score: float = 100.0

    # Category scores (weights: 50 + 25 + 25 = 100)
    section_completeness: float = 0.0     # 50 points - all 8 sections present
    content_quality: float = 0.0          # 25 points - bullets, assessments, structure
    formatting_quality: float = 0.0       # 25 points - markdown, headers, separators

    # Detailed section scores
    sections: List[SectionScore] = field(default_factory=list)

    @property
    def percentage(self) -> float:
        """Get total score as percentage."""
        return (self.total_score / self.max_score * 100)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'total_score': self.total_score,
            'max_score': self.max_score,
            'percentage': self.percentage,
            'section_completeness': self.section_completeness,
            'content_quality': self.content_quality,
            'formatting_quality': self.formatting_quality,
            'sections': [
                {
                    'name': s.name,
                    'score': s.score,
                    'max_score': s.max_score,
                    'percentage': s.percentage,
                    'present': s.present,
                    'details': s.details
                }
                for s in self.sections
            ]
        }


# Expected 8 sections in military SITREP format
EXPECTED_SECTIONS = [
    ("BLUF", r'1\.\s*BLUF|##?\s*1\.\s*BLUF|BLUF\s*\(Bottom Line'),
    ("Friendly Forces", r'2\.\s*FRIENDLY FORCES|##?\s*2\.\s*FRIENDLY'),
    ("Adversary Activity", r'3\.\s*ADVERSARY ACTIVITY|##?\s*3\.\s*ADVERSARY'),
    ("Operations", r'4\.\s*OPERATIONS|##?\s*4\.\s*OPERATIONS'),
    ("Logistics", r'5\.\s*LOGISTICS|##?\s*5\.\s*LOGISTICS'),
    ("Intelligence Assessment", r'6\.\s*INTELLIGENCE ASSESSMENT|##?\s*6\.\s*INTELLIGENCE'),
    ("Risks / Watch Items", r'7\.\s*RISKS|##?\s*7\.\s*RISKS|WATCH ITEMS'),
    ("Commander's Comments", r"8\.\s*COMMANDER'?S?\s*COMMENTS|##?\s*8\.\s*COMMANDER"),
]


class SitRepFormattingEvaluatorV2:
    """Evaluate INDOPACOM SITREP formatting quality."""

    def __init__(self, weights: dict = None):
        """
        Initialize evaluator.

        Args:
            weights: Custom weights for scoring categories
        """
        self.weights = weights or {
            'section_completeness': 50,  # All 8 sections present
            'content_quality': 25,       # Bullets, assessments, details
            'formatting_quality': 25     # Markdown, separators, structure
        }

    def evaluate(self, sitrep_text: str) -> FormattingScoreV2:
        """
        Evaluate a SITREP's formatting quality.

        Args:
            sitrep_text: Text of the SITREP

        Returns:
            FormattingScoreV2 with detailed metrics
        """
        sections = []

        # Section Completeness (50 points)
        section_scores = self._evaluate_section_completeness(sitrep_text)
        sections.extend(section_scores)
        section_completeness = sum(s.score for s in section_scores)

        # Content Quality (25 points)
        content_scores = self._evaluate_content_quality(sitrep_text)
        sections.extend(content_scores)
        content_quality = sum(s.score for s in content_scores)

        # Formatting Quality (25 points)
        formatting_scores = self._evaluate_formatting_quality(sitrep_text)
        sections.extend(formatting_scores)
        formatting_quality = sum(s.score for s in formatting_scores)

        total_score = (
            section_completeness +
            content_quality +
            formatting_quality
        )

        return FormattingScoreV2(
            total_score=total_score,
            section_completeness=section_completeness,
            content_quality=content_quality,
            formatting_quality=formatting_quality,
            sections=sections
        )

    def _evaluate_section_completeness(self, text: str) -> List[SectionScore]:
        """
        Evaluate presence of all 8 required sections (50 points total).
        Each section worth 6.25 points.
        """
        scores = []
        points_per_section = 6.25

        for section_name, pattern in EXPECTED_SECTIONS:
            has_section = bool(re.search(pattern, text, re.IGNORECASE | re.MULTILINE))
            scores.append(SectionScore(
                name=f"has_{section_name.lower().replace(' ', '_').replace('/', '_')}",
                score=points_per_section if has_section else 0,
                max_score=points_per_section,
                present=has_section,
                details=f"Section '{section_name}' {'found' if has_section else 'missing'}"
            ))

        return scores

    def _evaluate_content_quality(self, text: str) -> List[SectionScore]:
        """
        Evaluate content quality within sections (25 points total).
        """
        scores = []

        # BLUF section quality (8 points)
        bluf_score = self._evaluate_bluf_quality(text)
        scores.append(bluf_score)

        # Assessment statements present (6 points)
        assessment_count = len(re.findall(r'Assessment:', text, re.IGNORECASE))
        has_assessments = assessment_count >= 2
        scores.append(SectionScore(
            name="has_assessment_statements",
            score=6 if has_assessments else min(assessment_count * 2, 6),
            max_score=6,
            present=has_assessments,
            details=f"Found {assessment_count} 'Assessment:' statements (need 2+)"
        ))

        # Bullet points in sections (6 points)
        bullet_count = len(re.findall(r'^[-•]\s+', text, re.MULTILINE))
        has_bullets = bullet_count >= 10
        bullet_score = min(bullet_count / 10 * 6, 6)
        scores.append(SectionScore(
            name="has_bullet_points",
            score=bullet_score,
            max_score=6,
            present=has_bullets,
            details=f"Found {bullet_count} bullet points (target: 10+)"
        ))

        # Confidence/probability assessment (5 points)
        has_confidence = bool(re.search(r'Confidence:\s*(HIGH|MEDIUM|LOW)', text, re.IGNORECASE))
        has_probability = bool(re.search(r'(Probability|assessment).*\d+%', text, re.IGNORECASE))
        conf_prob_score = 0
        if has_confidence:
            conf_prob_score += 2.5
        if has_probability:
            conf_prob_score += 2.5
        scores.append(SectionScore(
            name="has_confidence_probability",
            score=conf_prob_score,
            max_score=5,
            present=has_confidence or has_probability,
            details=f"Confidence: {'✓' if has_confidence else '✗'}, Probability: {'✓' if has_probability else '✗'}"
        ))

        return scores

    def _evaluate_bluf_quality(self, text: str) -> SectionScore:
        """Evaluate BLUF (Bottom Line Up Front) section quality (8 points)."""
        # Extract BLUF section
        bluf_match = re.search(
            r'1\.\s*BLUF.*?\n(.+?)(?=\n[-—]{3,}|\n2\.\s*FRIENDLY|\Z)',
            text,
            re.DOTALL | re.IGNORECASE
        )

        if not bluf_match:
            return SectionScore(
                name="bluf_quality",
                score=0,
                max_score=8,
                present=False,
                details="BLUF section not found"
            )

        bluf_text = bluf_match.group(1)
        score = 0
        details = []

        # Has at least 2 bullet points (3 points)
        bullet_count = len(re.findall(r'^[-•]\s+', bluf_text, re.MULTILINE))
        if bullet_count >= 2:
            score += 3
            details.append(f"{bullet_count} bullets ✓")
        else:
            details.append(f"{bullet_count} bullets (need 2+)")

        # Has Confidence level (2.5 points)
        if re.search(r'Confidence:\s*(HIGH|MEDIUM|LOW)', bluf_text, re.IGNORECASE):
            score += 2.5
            details.append("Confidence ✓")
        else:
            details.append("Confidence ✗")

        # Has probability/assessment (2.5 points)
        if re.search(r'(Probability|assessment).*\d+%', bluf_text, re.IGNORECASE):
            score += 2.5
            details.append("Probability ✓")
        else:
            details.append("Probability ✗")

        return SectionScore(
            name="bluf_quality",
            score=score,
            max_score=8,
            present=True,
            details=", ".join(details)
        )

    def _evaluate_formatting_quality(self, text: str) -> List[SectionScore]:
        """
        Evaluate markdown formatting quality (25 points total).
        """
        scores = []

        # Has SITREP header with metadata (7 points)
        has_sitrep_header = bool(re.search(r'SITUATIONAL REPORT|SITREP', text, re.IGNORECASE))
        has_unit = bool(re.search(r'Unit:', text))
        has_dtg = bool(re.search(r'DTG:', text))
        has_classification = bool(re.search(r'Classification:', text, re.IGNORECASE))
        has_period = bool(re.search(r'Reporting Period:', text, re.IGNORECASE))

        header_score = 0
        header_details = []
        if has_sitrep_header:
            header_score += 2
            header_details.append("SITREP header ✓")
        if has_unit:
            header_score += 1.25
            header_details.append("Unit ✓")
        if has_dtg:
            header_score += 1.25
            header_details.append("DTG ✓")
        if has_classification:
            header_score += 1.25
            header_details.append("Classification ✓")
        if has_period:
            header_score += 1.25
            header_details.append("Period ✓")

        scores.append(SectionScore(
            name="header_formatting",
            score=header_score,
            max_score=7,
            present=has_sitrep_header,
            details=", ".join(header_details) if header_details else "Missing header elements"
        ))

        # Has section separators (6 points)
        separator_count = len(re.findall(r'^[-—]{3,}$', text, re.MULTILINE))
        separator_score = min(separator_count / 8 * 6, 6)  # Target: 8 separators
        scores.append(SectionScore(
            name="section_separators",
            score=separator_score,
            max_score=6,
            present=separator_count >= 4,
            details=f"Found {separator_count} separators (target: 8)"
        ))

        # Has numbered sections (6 points)
        numbered_sections = len(re.findall(r'^[#]?\s*[1-8]\.\s+[A-Z]', text, re.MULTILINE))
        numbered_score = min(numbered_sections / 8 * 6, 6)
        scores.append(SectionScore(
            name="numbered_sections",
            score=numbered_score,
            max_score=6,
            present=numbered_sections >= 6,
            details=f"Found {numbered_sections} numbered sections (target: 8)"
        ))

        # Proper line spacing and structure (6 points)
        # Check for reasonable paragraph breaks
        paragraphs = text.split('\n\n')
        has_good_structure = len(paragraphs) >= 10
        # Check for consistent bullet formatting
        consistent_bullets = len(re.findall(r'^- ', text, re.MULTILINE)) > len(re.findall(r'^• ', text, re.MULTILINE))

        structure_score = 0
        structure_details = []
        if has_good_structure:
            structure_score += 3
            structure_details.append("Good structure ✓")
        else:
            structure_details.append(f"Structure: {len(paragraphs)} paragraphs")
        if consistent_bullets:
            structure_score += 3
            structure_details.append("Consistent bullets ✓")
        else:
            structure_details.append("Mixed bullet styles")

        scores.append(SectionScore(
            name="document_structure",
            score=structure_score,
            max_score=6,
            present=has_good_structure,
            details=", ".join(structure_details)
        ))

        return scores


def evaluate_batch(sitreps: List[str]) -> Tuple[List[FormattingScoreV2], dict]:
    """
    Evaluate a batch of SITREPs.

    Args:
        sitreps: List of SITREP texts

    Returns:
        Tuple of (list of scores, aggregate statistics)
    """
    evaluator = SitRepFormattingEvaluatorV2()
    scores = [evaluator.evaluate(text) for text in sitreps]

    if not scores:
        return [], {'num_sitreps': 0}

    # Calculate aggregate statistics
    stats = {
        'num_sitreps': len(scores),
        'mean_total': sum(s.total_score for s in scores) / len(scores),
        'mean_section_completeness': sum(s.section_completeness for s in scores) / len(scores),
        'mean_content_quality': sum(s.content_quality for s in scores) / len(scores),
        'mean_formatting_quality': sum(s.formatting_quality for s in scores) / len(scores),
        'min_total': min(s.total_score for s in scores),
        'max_total': max(s.total_score for s in scores),
        'std_total': (sum((s.total_score - sum(s.total_score for s in scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5,
    }

    return scores, stats


def compare_models(
    baseline_sitreps: List[str],
    finetuned_sitreps: List[str]
) -> dict:
    """
    Compare formatting quality between baseline and fine-tuned models.

    Args:
        baseline_sitreps: SITREPs from baseline model
        finetuned_sitreps: SITREPs from fine-tuned model

    Returns:
        Dictionary with comparison statistics
    """
    baseline_scores, baseline_stats = evaluate_batch(baseline_sitreps)
    finetuned_scores, finetuned_stats = evaluate_batch(finetuned_sitreps)

    improvement = {
        'total_score_improvement': finetuned_stats['mean_total'] - baseline_stats['mean_total'],
        'total_score_improvement_pct': (
            (finetuned_stats['mean_total'] - baseline_stats['mean_total']) /
            baseline_stats['mean_total'] * 100
        ) if baseline_stats['mean_total'] > 0 else 0,
        'section_completeness_improvement': (
            finetuned_stats['mean_section_completeness'] -
            baseline_stats['mean_section_completeness']
        ),
        'content_quality_improvement': (
            finetuned_stats['mean_content_quality'] -
            baseline_stats['mean_content_quality']
        ),
        'formatting_quality_improvement': (
            finetuned_stats['mean_formatting_quality'] -
            baseline_stats['mean_formatting_quality']
        ),
    }

    return {
        'baseline': baseline_stats,
        'finetuned': finetuned_stats,
        'improvement': improvement
    }
