from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import UUID4, BaseModel, Field
from pyparsing import Literal
from typing_extensions import TypedDict

from certiorari.utils import BaseEntity


class CaseDocumentAnalysis(BaseModel):
    key_facts: List[Dict]
    dates: List[Dict]
    legal_issues: List[Dict]
    evidence_points: List[Dict]
    relationships: List[Dict]


class CaseDocument(BaseEntity):
    uploaded_at: datetime = Field(default_factory=datetime.now)
    tags: List[str]
    content: str
    recipients: List[str]
    author: str
    doc_type: str


class CaseDeposition(CaseDocument):
    document_type: Literal["deposition"]
    deponent: str
    testimony_summary: Dict
    exhibits_referenced: List[str]
    key_admissions: List[str]
    objections: List[Dict]


class CaseExhibit(CaseDocument):
    document_type: Literal["exhibit"]
    exhibit_number: str
    description: str
    source: str
    authentication_info: Dict
    related_testimony: List[str]


class CaseFiling(CaseDocument):
    document_type: Literal["filing"]
    filing_type: str
    case_number: str
    court: str
    relief_requested: str
    legal_arguments: List[Dict]
    exhibits_attached: List[str]


class CaseCorrespondence(CaseDocument):
    document_type: Literal["correspondence"]
    subject: str
    communication_type: str  # email, letter, memo
    related_matters: List[str]
    action_items: List[str]
    confidentiality_status: str


class CaseTimelineEvent(BaseEntity):
    date: datetime
    description: str


class CaseEventRelationship(TypedDict):
    previous_event_id: UUID4
    subsequent_event_id: UUID4
    relationship: str


class CaseTimelineGap(TypedDict):
    start_date: datetime
    end_date: datetime
    description: str


class CaseTimelineConflict(TypedDict):
    event_id: UUID4
    description: str


class CaseTimeline(BaseModel):
    timeline: List[CaseTimelineEvent]
    event_relationships: List[CaseEventRelationship]
    gaps: List[CaseTimelineGap]
    conflicts: List[CaseTimelineConflict]


class CaseMotion(CaseFiling):
    document_type: Literal["motion"]
    table_of_authorities: List[Dict]
    evidence_citations: List[Dict]
    counter_arguments: List[Dict]
    weakness_analysis: Dict


class CasePrecedent(BaseEntity):
    case_id: UUID4
    relevance_score: float
    key_holdings: List[str]
    distinguishing_factors: List[str]
    application_analysis: str


class CaseStrategy(BaseModel):
    strengths: List[Dict]
    weaknesses: List[Dict]
    evidence_gaps: List[Dict]
    recommended_actions: List[Dict]
    risk_analysis: Dict
    timeline_issues: List[Dict]


class CaseDetails(BaseModel):
    case_name: str
    jurisdiction: str
    case_type: str
    parties: List[str]


class CaseDocumentMetadata(BaseModel):
    doc_type: str  # deposition|exhibit|filing|correspondence
    date: str
    author: str
    recipients: List[str]


class ProcessedDocument(BaseModel):
    key_facts: List[Dict]
    dates: List[Dict]
    legal_issues: List[Dict]
    evidence_points: List[Dict]
    relationships: List[Dict]


class Timeline(BaseModel):
    timeline: List[Dict]  # Chronological events
    causation_chains: List[Dict]  # Related event sequences
    gaps: List[Dict]  # Missing evidence periods
    conflicts: List[Dict]  # Contradictory evidence


class MotionParams(BaseModel):
    motion_type: str
    legal_basis: str
    key_arguments: List[str]
    evidence_ids: List[str]
    jurisdiction: str


class MotionResult(BaseModel):
    motion_text: str
    table_of_authorities: List[Dict]
    evidence_citations: List[Dict]
    counter_arguments: List[Dict]
    weakness_analysis: Dict


class PrecedentQuery(BaseModel):
    legal_issue: str
    jurisdiction: str
    favorable: bool  # Search for supporting/opposing
    key_facts: List[str]


class PrecedentResult(BaseModel):
    case_name: str
    citation: str
    relevance_score: float
    key_holdings: List[str]
    distinguishing_factors: List[str]
    application_analysis: str


class StrategyAnalysis(BaseModel):
    strengths: List[Dict]
    weaknesses: List[Dict]
    evidence_gaps: List[Dict]
    recommended_actions: List[Dict]
    risk_analysis: Dict
    timeline_issues: List[Dict]


class DiscoveryParams(BaseModel):
    discovery_type: str  # interrogatories|production|admission
    legal_issues: List[str]
    evidence_gaps: List[str]


class DiscoveryRequest(BaseModel):
    request_text: str
    legal_basis: str
    target_evidence: str
    strategic_purpose: str


class OppositionDetails(BaseModel):
    opposing_counsel: str
    past_cases: List[str]
    # Additional fields...


class OppositionAnalysis(BaseModel):
    strategies: List[str]
    weaknesses: List[str]
    # Additional fields...


class CaseOutcomePrediction(BaseModel):
    likelihood_of_success: float
    potential_awards: Optional[float]
    key_factors: List[str]
    risk_factors: List[str]


class CommunicationParams(BaseModel):
    subject: str
    message: str
    preferred_method: str  # email, letter, phone call
    urgency_level: str  # high, medium, low


class ClientCommunication(BaseModel):
    communication_text: str
    next_steps: List[str]
    scheduled_meetings: List[datetime]


class ResearchSummary(BaseModel):
    summary_points: List[str]
    important_cases: List[str]
    statutory_references: List[str]


class DocumentComparison(BaseModel):
    identical_sections: List[str]
    differing_sections: List[str]
    analysis: str


class Schedule(BaseModel):
    events: List[Dict[str, str]]  # date, event description
    reminders: List[Dict[str, str]]  # date, reminder text
    deadlines: List[Dict[str, str]]  # date, deadline description


class RiskAssessment(BaseModel):
    identified_risks: List[str]
    probability_levels: List[str]  # high, medium, low
    impact_assessment: List[str]
    mitigation_plan: List[str]
