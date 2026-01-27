"""
Formatting quality metrics for sit rep evaluation.
STRICT evaluation focusing on precise formatting patterns.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Tuple
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
class FormattingScore:
    """Complete formatting score for a sit rep."""
    total_score: float
    max_score: float = 100.0

    # Category scores
    section_completeness: float = 0.0
    tldr_quality: float = 0.0
    markdown_formatting: float = 0.0
    structure_quality: float = 0.0

    # Detailed section scores
    sections: List[SectionScore] = None

    def __post_init__(self):
        if self.sections is None:
            self.sections = []

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
            'tldr_quality': self.tldr_quality,
            'markdown_formatting': self.markdown_formatting,
            'structure_quality': self.structure_quality,
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


class SitRepFormattingEvaluator:
    """Evaluate sit rep formatting quality with STRICT criteria."""

    def __init__(self, weights: dict = None):
        """
        Initialize evaluator.

        Args:
            weights: Custom weights for scoring categories
        """
        self.weights = weights or {
            'section_completeness': 40,
            'tldr_quality': 20,
            'markdown_formatting': 20,
            'structure_quality': 20
        }

    def evaluate(self, sitrep_text: str) -> FormattingScore:
        """
        Evaluate a sit rep's formatting quality.

        Args:
            sitrep_text: Markdown text of the sit rep

        Returns:
            FormattingScore with detailed metrics
        """
        sections = []

        # Section Completeness (40 points)
        section_scores = self._evaluate_section_completeness(sitrep_text)
        sections.extend(section_scores)
        section_completeness = sum(s.score for s in section_scores)

        # TLDR Quality (20 points)
        tldr_scores = self._evaluate_tldr_quality(sitrep_text)
        sections.extend(tldr_scores)
        tldr_quality = sum(s.score for s in tldr_scores)

        # Markdown Formatting (20 points)
        markdown_scores = self._evaluate_markdown_formatting(sitrep_text)
        sections.extend(markdown_scores)
        markdown_formatting = sum(s.score for s in markdown_scores)

        # Structure Quality (20 points)
        structure_scores = self._evaluate_structure_quality(sitrep_text)
        sections.extend(structure_scores)
        structure_quality = sum(s.score for s in structure_scores)

        total_score = (
            section_completeness +
            tldr_quality +
            markdown_formatting +
            structure_quality
        )

        return FormattingScore(
            total_score=total_score,
            section_completeness=section_completeness,
            tldr_quality=tldr_quality,
            markdown_formatting=markdown_formatting,
            structure_quality=structure_quality,
            sections=sections
        )

    def _evaluate_section_completeness(self, text: str) -> List[SectionScore]:
        """Evaluate presence of required sections with EXACT formatting (40 points total)."""
        scores = []

        # Has title with exact format: # [Scenario] - Day [N] Logistics Deployment Analysis (5 points)
        title_pattern = r'^#\s+.+\s+-\s+Day\s+\d+\s+Logistics Deployment Analysis'
        has_title = bool(re.search(title_pattern, text, re.MULTILINE))
        scores.append(SectionScore(
            name="has_exact_title_format",
            score=5 if has_title else 0,
            max_score=5,
            present=has_title,
            details="Exact title format present" if has_title else "Missing exact title format"
        ))

        # TLDR must be exactly "## TLDR" (not case variations) (10 points)
        has_tldr = bool(re.search(r'^##\s+TLDR\s*$', text, re.MULTILINE))
        scores.append(SectionScore(
            name="has_exact_tldr_header",
            score=10 if has_tldr else 0,
            max_score=10,
            present=has_tldr,
            details="Exact TLDR header format" if has_tldr else "Missing exact TLDR format"
        ))

        # Exact section headers (5 points each)
        exact_sections = [
            ("Executive Summary", r'^##\s+Executive Summary\s*$'),
            ("Location Analysis", r'^##\s+Location Analysis\s*$'),
            ("Comparative Analysis", r'^##\s+Comparative Analysis\s*$'),
            ("Temporal Trends", r'^##\s+Temporal Trends\s*$'),
            ("Recommendations", r'^##\s+Recommendations\s*$'),
            ("Action Items", r'^##\s+Action Items\s*$'),
        ]

        for section_name, pattern in exact_sections:
            has_section = bool(re.search(pattern, text, re.MULTILINE))
            scores.append(SectionScore(
                name=f"has_exact_{section_name.lower().replace(' ', '_')}",
                score=2.5 if has_section else 0,
                max_score=2.5,
                present=has_section,
                details=f"Exact '{section_name}' header" if has_section else f"Missing/incorrect '{section_name}'"
            ))

        return scores

    def _evaluate_tldr_quality(self, text: str) -> List[SectionScore]:
        """Evaluate TLDR section with STRICT format requirements (20 points total)."""
        scores = []

        # Extract TLDR section
        tldr_match = re.search(
            r'##\s+TLDR\s*\n(.+?)(?=\n##|\Z)',
            text,
            re.DOTALL
        )

        tldr_text = tldr_match.group(1) if tldr_match else ""

        # Must have EXACT format: **Recommended Deployment Site:** on its own line (7 points)
        has_exact_rec = bool(re.search(
            r'^\*\*Recommended Deployment Site:\*\*',
            tldr_text,
            re.MULTILINE
        ))
        scores.append(SectionScore(
            name="exact_recommendation_format",
            score=7 if has_exact_rec else 0,
            max_score=7,
            present=has_exact_rec,
            details="Exact recommendation format" if has_exact_rec else "Missing exact '**Recommended Deployment Site:**' format"
        ))

        # Must have EXACT format: **Key Insight:** on its own line (7 points)
        has_exact_insight = bool(re.search(
            r'^\*\*Key Insight:\*\*',
            tldr_text,
            re.MULTILINE
        ))
        scores.append(SectionScore(
            name="exact_insight_format",
            score=7 if has_exact_insight else 0,
            max_score=7,
            present=has_exact_insight,
            details="Exact insight format" if has_exact_insight else "Missing exact '**Key Insight:**' format"
        ))

        # Must have "**Rationale:**" followed by bulleted list with (See: ...) citations (6 points)
        has_rationale_section = bool(re.search(r'^\*\*Rationale:\*\*\s*$', tldr_text, re.MULTILINE))
        see_citation_count = len(re.findall(r'\(See:', tldr_text))
        has_see_citations = see_citation_count >= 2
        has_proper_rationale = has_rationale_section and has_see_citations

        scores.append(SectionScore(
            name="exact_rationale_with_citations",
            score=6 if has_proper_rationale else 0,
            max_score=6,
            present=has_proper_rationale,
            details=f"Rationale section: {has_rationale_section}, (See:...) citations: {see_citation_count}" if tldr_text else "No TLDR content"
        ))

        return scores

    def _evaluate_markdown_formatting(self, text: str) -> List[SectionScore]:
        """Evaluate STRICT markdown formatting patterns (20 points total)."""
        scores = []

        # Must have comparison table with proper structure (5 points)
        # Look for table with header row, separator row, and data rows
        table_pattern = r'\|.+\|.+\n\|[-:]+\|.+\n(\|.+\|.+\n)+'
        has_proper_table = bool(re.search(table_pattern, text))
        scores.append(SectionScore(
            name="proper_comparison_table",
            score=5 if has_proper_table else 0,
            max_score=5,
            present=has_proper_table,
            details="Proper table structure with headers and separators" if has_proper_table else "Missing proper table structure"
        ))

        # Must have "### Primary Recommendation:" subsection (5 points)
        has_primary_rec = bool(re.search(r'^###\s+Primary Recommendation:', text, re.MULTILINE))
        scores.append(SectionScore(
            name="primary_recommendation_subsection",
            score=5 if has_primary_rec else 0,
            max_score=5,
            present=has_primary_rec,
            details="Has '### Primary Recommendation:' subsection" if has_primary_rec else "Missing primary recommendation subsection"
        ))

        # Must have at least 3 location H3 subsections under Location Analysis (5 points)
        location_section = re.search(r'##\s+Location Analysis\s*\n(.+?)(?=\n##|\Z)', text, re.DOTALL)
        if location_section:
            location_h3_count = len(re.findall(r'^###\s+', location_section.group(1), re.MULTILINE))
            has_location_subsections = location_h3_count >= 3
        else:
            has_location_subsections = False
            location_h3_count = 0

        scores.append(SectionScore(
            name="location_subsections",
            score=5 if has_location_subsections else 0,
            max_score=5,
            present=has_location_subsections,
            details=f"{location_h3_count} location subsections (need 3+)" if location_section else "No Location Analysis section"
        ))

        # Must have proper checkbox format: "- [ ]" followed by category (5 points)
        checkbox_pattern = r'- \[ \]\s+(Immediate|Short-term|Planning):'
        checkbox_matches = re.findall(checkbox_pattern, text)
        has_categorized_checkboxes = len(checkbox_matches) >= 2
        scores.append(SectionScore(
            name="categorized_action_items",
            score=5 if has_categorized_checkboxes else 0,
            max_score=5,
            present=has_categorized_checkboxes,
            details=f"Found {len(checkbox_matches)} categorized action items" if checkbox_matches else "Missing categorized action items (Immediate/Short-term/Planning)"
        ))

        return scores

    def _evaluate_structure_quality(self, text: str) -> List[SectionScore]:
        """Evaluate STRICT document structure (20 points total)."""
        scores = []

        # Must have exactly 7 main sections (H2) in correct order (10 points)
        expected_sections = [
            "TLDR",
            "Executive Summary",
            "Location Analysis",
            "Comparative Analysis",
            "Temporal Trends",
            "Recommendations",
            "Action Items"
        ]

        h2_headers = re.findall(r'^##\s+(.+)$', text, re.MULTILINE)
        h2_headers_clean = [h.strip() for h in h2_headers]

        # Check if we have exactly these 7 sections in order
        has_correct_sections = len(h2_headers_clean) >= 7 and all(
            exp in h2_headers_clean for exp in expected_sections
        )

        scores.append(SectionScore(
            name="seven_required_sections",
            score=10 if has_correct_sections else 0,
            max_score=10,
            present=has_correct_sections,
            details=f"Found {len(h2_headers_clean)} H2 sections (need 7 exact)" if h2_headers_clean else "No H2 sections found"
        ))

        # Recommendations must have "**Strengths:**" and "**Risks and Mitigations:**" subsections (5 points)
        rec_section = re.search(r'##\s+Recommendations\s*\n(.+?)(?=\n##|\Z)', text, re.DOTALL)
        if rec_section:
            rec_text = rec_section.group(1)
            has_strengths = bool(re.search(r'\*\*Strengths:\*\*', rec_text))
            has_risks = bool(re.search(r'\*\*Risks and Mitigations:\*\*', rec_text))
            has_proper_rec_structure = has_strengths and has_risks
        else:
            has_proper_rec_structure = False

        scores.append(SectionScore(
            name="recommendation_structure",
            score=5 if has_proper_rec_structure else 0,
            max_score=5,
            present=has_proper_rec_structure,
            details="Has Strengths and Risks subsections" if has_proper_rec_structure else "Missing Strengths or Risks subsections in Recommendations"
        ))

        # Must have "### Alternative Options" subsection (5 points)
        has_alternatives = bool(re.search(r'^###\s+Alternative Options', text, re.MULTILINE))
        scores.append(SectionScore(
            name="alternative_options_subsection",
            score=5 if has_alternatives else 0,
            max_score=5,
            present=has_alternatives,
            details="Has Alternative Options subsection" if has_alternatives else "Missing '### Alternative Options' subsection"
        ))

        return scores


def evaluate_batch(sitreps: List[str]) -> Tuple[List[FormattingScore], dict]:
    """
    Evaluate a batch of sit reps.

    Args:
        sitreps: List of sit rep texts

    Returns:
        Tuple of (list of scores, aggregate statistics)
    """
    evaluator = SitRepFormattingEvaluator()
    scores = [evaluator.evaluate(text) for text in sitreps]

    # Calculate aggregate statistics
    stats = {
        'num_sitreps': len(scores),
        'mean_total': sum(s.total_score for s in scores) / len(scores) if scores else 0,
        'mean_section_completeness': sum(s.section_completeness for s in scores) / len(scores) if scores else 0,
        'mean_tldr_quality': sum(s.tldr_quality for s in scores) / len(scores) if scores else 0,
        'mean_markdown_formatting': sum(s.markdown_formatting for s in scores) / len(scores) if scores else 0,
        'mean_structure_quality': sum(s.structure_quality for s in scores) / len(scores) if scores else 0,
        'min_total': min(s.total_score for s in scores) if scores else 0,
        'max_total': max(s.total_score for s in scores) if scores else 0,
    }

    return scores, stats


def compare_models(
    baseline_sitreps: List[str],
    finetuned_sitreps: List[str]
) -> dict:
    """
    Compare formatting quality between baseline and fine-tuned models.

    Args:
        baseline_sitreps: Sit reps from baseline model
        finetuned_sitreps: Sit reps from fine-tuned model

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
        'tldr_quality_improvement': (
            finetuned_stats['mean_tldr_quality'] -
            baseline_stats['mean_tldr_quality']
        ),
        'markdown_formatting_improvement': (
            finetuned_stats['mean_markdown_formatting'] -
            baseline_stats['mean_markdown_formatting']
        ),
        'structure_quality_improvement': (
            finetuned_stats['mean_structure_quality'] -
            baseline_stats['mean_structure_quality']
        ),
    }

    return {
        'baseline': baseline_stats,
        'finetuned': finetuned_stats,
        'improvement': improvement
    }
