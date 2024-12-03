from typing import List
import uuid
from datetime import datetime
from typing import Dict
from jinja2 import Environment, FileSystemLoader

from certiorari.app import app
from certiorari.schema import (
    CaseDetails,
    CaseDocumentMetadata,
    CaseStrategy,
    CaseTimeline,
    DiscoveryParams,
    DiscoveryRequest,
    MotionParams,
    MotionResult,
    PrecedentQuery,
    PrecedentResult,
    ProcessedDocument,
    StrategyAnalysis,
    Timeline,
    OppositionDetails,
    OppositionAnalysis,
    CaseOutcomePrediction,
    CommunicationParams,
    ClientCommunication,
    ResearchSummary,
    DocumentComparison,
    Schedule,
    RiskAssessment,
    CaseDocument,
    CaseTimelineEvent,
)
from certiorari.utils import BaseEntity
from truffle import LLM, VectorStore

# Initialize LLM and VectorStore instances
llm = LLM()
vector_store = VectorStore()

# In-memory data stores (simulating a database)
case_store: dict[str, CaseDetails] = {}
document_store: dict[str, CaseDocument] = {}
timeline_event_store: dict[str, List[CaseTimelineEvent]] = {}

# Set up Jinja2 environment
env = Environment(loader=FileSystemLoader('templates'))


@app.tool
def initialize_case(case_details: CaseDetails) -> str:
    """
    Creates new case workspace.

    Args:
        case_details: CaseDetails containing case name, jurisdiction, type, and parties

    Returns:
        case_id: Unique identifier for the case
    """
    # Validate required fields from schema
    if not all([case_details.case_name, case_details.jurisdiction, case_details.case_type]):
        raise ValueError("Case details missing required fields: 'case_name', 'jurisdiction', or 'case_type'")

    # Initialize case details and store it
    case = CaseDetails(
        case_name=case_details.case_name,
        jurisdiction=case_details.jurisdiction,
        case_type=case_details.case_type,
        parties=case_details.parties,
    )
    case_store[str(case.id)] = case  # Store the case using its UUID as the key

    # Return the case ID as a string
    return str(case.id)


@app.tool
def process_document(
    doc_content: str, metadata: CaseDocumentMetadata
) -> ProcessedDocument:
    """
    Analyzes legal document and extracts key information.

    Args:
        doc_content: Document text
        metadata: CaseDocumentMetadata

    Returns:
        ProcessedDocument
    """
    # Step 1: Preprocess the document
    preprocessed_content = preprocess_document(doc_content)

    # Step 2: Extract entities and relationships using the LLM
    # Load and render the Jinja template
    template = env.get_template('process_document_extraction_prompt.j2')
    extraction_prompt = template.render(preprocessed_content=preprocessed_content)

    extraction_response = llm.chat([{"role": "user", "content": extraction_prompt}])

    # Step 3: Parse the response into ProcessedDocument
    try:
        processed_document = llm.structured_output(ProcessedDocument)
    except Exception as e:
        raise ValueError(f"Error parsing LLM response into ProcessedDocument: {e}")

    # Step 4: Store embeddings for semantic search
    doc_id = str(uuid.uuid4())
    embeddings = llm.embed_text(preprocessed_content)
    vector_store.add_text(
        text=preprocessed_content,
        metadata={"doc_id": doc_id, **metadata.model_dump()},
        embeddings=embeddings,
    )

    # Step 5: Save the document to the document store
    document_store[doc_id] = CaseDocument(
        id=uuid.UUID(doc_id),
        content=doc_content,
        **metadata.model_dump(),
        uploaded_at=datetime.now(),
        tags=[],
    )

    return processed_document


def preprocess_document(doc_content: str) -> str:
    """
    Preprocesses the document content for analysis.

    Args:
        doc_content: Document text

    Returns:
        Preprocessed text
    """
    # Implement preprocessing steps (e.g., removing headers, footers, OCR artifacts)
    # For this example, we'll assume the content is clean
    return doc_content


@app.tool
def construct_timeline(case_id: str) -> Timeline:
    """
    Builds a comprehensive case timeline.

    Args:
        case_id: Case identifier

    Returns:
        Timeline
    """
    # Retrieve all relevant documents for the case
    related_docs = fetch_case_documents(case_id)

    # Initialize an empty list for events
    events = []

    # Extract events from documents iteratively
    for doc in related_docs:
        events.extend(extract_events_from_document(doc))

    if not events:
        raise ValueError("No events found in the case documents.")

    # Sort events chronologically
    events.sort(key=lambda x: x.date)

    # Detect causation chains using the LLM
    causation_chains = detect_causation_chains(events)

    # Identify gaps and conflicts in the timeline
    gaps = identify_timeline_gaps(events)
    conflicts = identify_timeline_conflicts(events)

    # Save events to the timeline store
    timeline_event_store[case_id] = events

    return Timeline(
        timeline=[event.model_dump() for event in events],
        causation_chains=causation_chains,
        gaps=gaps,
        conflicts=conflicts,
    )


def fetch_case_documents(case_id: str) -> List[CaseDocument]:
    """
    Fetches all documents related to a case.

    Args:
        case_id: Case identifier

    Returns:
        List of CaseDocument
    """
    # Retrieve documents associated with the case
    # For simplicity, return all documents in this example
    return list(document_store.values())


def extract_events_from_document(doc: CaseDocument) -> List[CaseTimelineEvent]:
    """
    Extracts events from a single document.

    Args:
        doc: CaseDocument

    Returns:
        List of CaseTimelineEvent
    """
    # Load and render the Jinja template
    template = env.get_template('extract_events_from_document_prompt.j2')
    event_prompt = template.render(doc_content=doc.content)

    event_response = llm.chat([{"role": "user", "content": event_prompt}])

    # Parse response into List[CaseTimelineEvent]
    try:
        events = llm.structured_output(List[CaseTimelineEvent])
    except Exception as e:
        raise ValueError(f"Error parsing events from LLM response: {e}")

    return events


def detect_causation_chains(events: List[CaseTimelineEvent]) -> List[Dict]:
    """
    Detects causation chains among events.

    Args:
        events: List of CaseTimelineEvent

    Returns:
        List of causation chains
    """
    # Prepare events for analysis
    events_text = '\n'.join([f"{event.date}: {event.description}" for event in events])

    # Load and render the Jinja template
    template = env.get_template('detect_causation_chains_prompt.j2')
    causation_prompt = template.render(events_text=events_text)

    causation_response = llm.chat([{"role": "user", "content": causation_prompt}])

    try:
        causation_chains = llm.structured_output(List[Dict])
    except Exception as e:
        raise ValueError(f"Error parsing causation chains from LLM response: {e}")

    return causation_chains


def identify_timeline_gaps(events: List[CaseTimelineEvent]) -> List[Dict]:
    """
    Identifies gaps in the timeline.

    Args:
        events: List of CaseTimelineEvent

    Returns:
        List of gaps
    """
    # Implement gap identification logic
    # Placeholder implementation
    return []


def identify_timeline_conflicts(events: List[CaseTimelineEvent]) -> List[Dict]:
    """
    Identifies conflicts in the timeline.

    Args:
        events: List of CaseTimelineEvent

    Returns:
        List of conflicts
    """
    # Implement conflict detection logic
    # Placeholder implementation
    return []


@app.tool
def generate_motion(params: MotionParams) -> MotionResult:
    """
    Creates a motion draft with citations.

    Args:
        params: MotionParams

    Returns:
        MotionResult
    """
    # Step 1: Research relevant precedents
    precedents = find_precedents(
        PrecedentQuery(
            legal_issue=params.legal_basis,
            jurisdiction=params.jurisdiction,
            favorable=True,
            key_facts=params.key_arguments,
        )
    )

    # Step 2: Generate motion outline using LLM with structured output
    # Load and render the Jinja template for the outline prompt
    template = env.get_template('generate_motion_outline_prompt.j2')
    outline_prompt = template.render(
        motion_type=params.motion_type,
        jurisdiction=params.jurisdiction,
        legal_basis=params.legal_basis,
        key_arguments=params.key_arguments,
        precedents=precedents
    )

    outline_response = llm.chat([{"role": "user", "content": outline_prompt}])
    try:
        motion_outline = llm.structured_output(Dict)
    except Exception as e:
        raise ValueError(f"Error parsing motion outline: {e}")

    # Step 3: Draft the motion using the outline
    # Load and render the Jinja template for the draft prompt
    template = env.get_template('generate_motion_draft_prompt.j2')
    draft_prompt = template.render(
        motion_type=params.motion_type,
        jurisdiction=params.jurisdiction,
        motion_outline=motion_outline
    )

    motion_text = llm.chat([{"role": "user", "content": draft_prompt}])

    # Step 4: Extract table of authorities and evidence citations
    authorities, evidence_citations = extract_citations(motion_text)

    # Step 5: Analyze weaknesses and counter-arguments
    analysis_prompt = f"""
    Analyze the drafted motion for potential weaknesses and counter-arguments.

    Motion Text:
    {motion_text}

    Provide the following in JSON format according to the MotionResult schema:
    {{
        "counter_arguments": [...],
        "weakness_analysis": "..."
    }}
    """
    analysis_response = llm.chat([{"role": "user", "content": analysis_prompt}])
    try:
        analysis = llm.structured_output(Dict)
    except Exception as e:
        raise ValueError(f"Error parsing motion analysis: {e}")

    return MotionResult(
        motion_text=motion_text,
        table_of_authorities=authorities,
        evidence_citations=evidence_citations,
        counter_arguments=analysis.get("counter_arguments", []),
        weakness_analysis=analysis.get("weakness_analysis", ""),
    )


def parse_outline(outline_response: str) -> str:
    """
    Parses the outline response into a structured format.

    Args:
        outline_response: Response from the LLM

    Returns:
        Structured outline
    """
    # Implement parsing logic
    # For simplicity, return the response as-is in this example
    return outline_response


def extract_citations(motion_text: str) -> (List[Dict], List[Dict]):
    """
    Extracts the table of authorities and evidence citations from the motion text.

    Args:
        motion_text: The drafted motion text

    Returns:
        Tuple of table_of_authorities and evidence_citations
    """
    citation_prompt = f"""
    From the following motion text, extract:

    - Table of Authorities (list of all legal cases cited)
    - Evidence Citations (list of all evidence referenced)

    Motion Text:
    {motion_text}

    Provide the information in a structured JSON format.
    """
    citation_response = llm.chat([{"role": "user", "content": citation_prompt}])
    citations = llm.structured_output(
        {"authorities": List[Dict], "evidences": List[Dict]}
    )
    return citations["authorities"], citations["evidences"]


@app.tool
def find_precedents(query: PrecedentQuery) -> List[PrecedentResult]:
    """
    Searches relevant case law.

    Args:
        query: PrecedentQuery

    Returns:
        List[PrecedentResult]
    """
    # Formulate search query
    search_text = f"{query.legal_issue} {' '.join(query.key_facts)}"

    # Retrieve relevant documents from the vector store
    results = vector_store.get_text(
        query=search_text,
        k=10,
        threshold=0.7,
        filter=lambda text, metadata: metadata.get("jurisdiction") == query.jurisdiction,
    )

    precedent_results = []
    for result_text, metadata in results:
        # Load and render the Jinja template
        template = env.get_template('find_precedents_analysis_prompt.j2')
        analysis_prompt = template.render(case_text=result_text)

        analysis_response = llm.chat([{"role": "user", "content": analysis_prompt}])

        # Parse the result into PrecedentResult
        try:
            precedent = llm.structured_output(PrecedentResult)
            precedent_results.append(precedent)
        except Exception as e:
            # Log error and skip the result if parsing fails
            continue

    # Rank precedents by relevance score
    precedent_results.sort(key=lambda x: x.relevance_score, reverse=True)

    return precedent_results


@app.tool
def analyze_strategy(case_id: str) -> StrategyAnalysis:
    """
    Evaluates case strategy holistically.

    Args:
        case_id: Case identifier

    Returns:
        StrategyAnalysis
    """
    # Gather case details and related documents
    case_details = case_store.get(case_id)
    if not case_details:
        raise ValueError(f"No case found with ID: {case_id}")
    documents = fetch_case_documents(case_id)

    # Extract key points from documents using LLM
    key_points = []
    for doc in documents:
        points = extract_key_points(doc)
        key_points.append(points)

    # Prepare data for analysis
    analysis_data = {
        "case_details": case_details.model_dump(),
        "key_points": key_points,
    }

    # Load and render the Jinja template
    template = env.get_template('analyze_strategy_prompt.j2')
    analysis_prompt = template.render(analysis_data=analysis_data)

    analysis_response = llm.chat([{"role": "user", "content": analysis_prompt}])

    # Parse the result into StrategyAnalysis model
    try:
        strategy_analysis = llm.structured_output(StrategyAnalysis)
    except Exception as e:
        raise ValueError(f"Error parsing StrategyAnalysis from LLM response: {e}")

    return strategy_analysis


def extract_key_points(doc: CaseDocument) -> Dict:
    """
    Extracts key points from a document.

    Args:
        doc: CaseDocument

    Returns:
        Dictionary of key points
    """
    # Load and render the Jinja template
    template = env.get_template('extract_key_points_prompt.j2')
    key_points_prompt = template.render(doc_content=doc.content)

    key_points_response = llm.chat([{"role": "user", "content": key_points_prompt}])

    try:
        key_points = llm.structured_output(Dict)
    except Exception as e:
        raise ValueError(f"Error parsing key points from LLM response: {e}")

    return key_points


@app.tool
def generate_discovery(params: DiscoveryParams) -> List[DiscoveryRequest]:
    """
    Creates discovery requests.

    Args:
        params: DiscoveryParams

    Returns:
        List[DiscoveryRequest]
    """
    # Step 1: Generate potential discovery items
    template = env.get_template('generate_discovery_items_prompt.j2')
    discovery_items_prompt = template.render(
        discovery_type=params.discovery_type,
        jurisdiction=params.jurisdiction,
        legal_issues=params.legal_issues,
        evidence_gaps=params.evidence_gaps
    )

    discovery_items_response = llm.chat([{"role": "user", "content": discovery_items_prompt}])
    try:
        discovery_items = llm.structured_output(List[str])
    except Exception as e:
        raise ValueError(f"Error parsing discovery items: {e}")

    # Step 2: Create detailed DiscoveryRequest for each item
    discovery_requests = []
    for item in discovery_items:
        template = env.get_template('generate_discovery_request_prompt.j2')
        request_prompt = template.render(
            discovery_type=params.discovery_type,
            item=item,
            jurisdiction=params.jurisdiction
        )

        request_response = llm.chat([{"role": "user", "content": request_prompt}])
        try:
            discovery_request = llm.structured_output(DiscoveryRequest)
            discovery_requests.append(discovery_request)
        except Exception as e:
            continue  # Skip invalid responses

    return discovery_requests


@app.tool
def analyze_opposition(opposition_details: OppositionDetails) -> OppositionAnalysis:
    """
    Analyzes opposing counsel's past cases and strategies.

    Args:
        opposition_details: OppositionDetails

    Returns:
        OppositionAnalysis
    """
    # Load and render the Jinja template
    template = env.get_template('analyze_opposition_prompt.j2')
    analysis_prompt = template.render(
        opposing_counsel=opposition_details.opposing_counsel,
        past_cases=opposition_details.past_cases
    )

    analysis_result = llm.chat([{"role": "user", "content": analysis_prompt}])

    # Parse the result into OppositionAnalysis model
    try:
        opposition_analysis = llm.structured_output(OppositionAnalysis)
    except Exception as e:
        raise ValueError(f"Error parsing OppositionAnalysis: {e}")

    return opposition_analysis


@app.tool
def predict_case_outcome(case_id: str) -> CaseOutcomePrediction:
    """
    Predicts the potential outcome of the case.

    Args:
        case_id: Case identifier

    Returns:
        CaseOutcomePrediction
    """
    # Gather case data
    case_details = case_store[case_id]
    documents = fetch_case_documents(case_id)

    # Load and render the Jinja template
    template = env.get_template('predict_case_outcome_prompt.j2')
    prediction_prompt = template.render(
        case_details=case_details,
        documents=documents
    )

    prediction_result = llm.chat([{"role": "user", "content": prediction_prompt}])

    # Parse the result into CaseOutcomePrediction model
    try:
        outcome_prediction = llm.structured_output(CaseOutcomePrediction)
    except Exception as e:
        raise ValueError(f"Error parsing CaseOutcomePrediction: {e}")

    return outcome_prediction


@app.tool
def generate_client_communication(
    client_id: str, communication_params: CommunicationParams
) -> ClientCommunication:
    """
    Generates communication drafts for clients.

    Args:
        client_id: Client identifier
        communication_params: CommunicationParams

    Returns:
        ClientCommunication
    """
    # Fetch client details (simulated)
    client_details = {"name": "Client Name"}  # Placeholder

    # Load and render the communication prompt template
    template = env.get_template('generate_client_communication_prompt.j2')
    communication_prompt = template.render(
        client_name=client_details['name'],
        communication_params=communication_params
    )

    communication_text = llm.chat([{"role": "user", "content": communication_prompt}])

    # Load and render the next steps prompt template
    template = env.get_template('generate_next_steps_prompt.j2')
    next_steps_prompt = template.render(communication_text=communication_text)

    next_steps_result = llm.chat([{"role": "user", "content": next_steps_prompt}])

    # Parse results into ClientCommunication model
    try:
        client_communication = ClientCommunication(
            communication_text=communication_text,
            next_steps=[],  # Parsing from next_steps_result
            scheduled_meetings=[],  # Parsing from next_steps_result
        )
    except Exception as e:
        raise ValueError(f"Error parsing ClientCommunication: {e}")

    return client_communication


@app.tool
def summarize_legal_research(doc_content: str) -> ResearchSummary:
    """
    Summarizes legal documents into key points.

    Args:
        doc_content: Document text

    Returns:
        ResearchSummary
    """
    # Load and render the Jinja template
    template = env.get_template('summarize_legal_research_prompt.j2')
    summary_prompt = template.render(doc_content=doc_content)

    summary_result = llm.chat([{"role": "user", "content": summary_prompt}])

    # Parse the result into ResearchSummary model
    try:
        research_summary = llm.structured_output(ResearchSummary)
    except Exception as e:
        raise ValueError(f"Error parsing ResearchSummary: {e}")

    return research_summary


@app.tool
def compare_documents(doc_id_1: str, doc_id_2: str) -> DocumentComparison:
    """
    Compares two documents to find differences and similarities.

    Args:
        doc_id_1: First document identifier
        doc_id_2: Second document identifier

    Returns:
        DocumentComparison
    """
    # Retrieve documents
    doc1 = document_store[doc_id_1]
    doc2 = document_store[doc_id_2]

    # Load and render the Jinja template
    template = env.get_template('compare_documents_prompt.j2')
    compare_prompt = template.render(
        doc1_content=doc1.content,
        doc2_content=doc2.content
    )

    comparison_result = llm.chat([{"role": "user", "content": compare_prompt}])

    # Parse the result into DocumentComparison model
    try:
        document_comparison = llm.structured_output(DocumentComparison)
    except Exception as e:
        raise ValueError(f"Error parsing DocumentComparison: {e}")

    return document_comparison


@app.tool
def schedule_case_events(case_id: str) -> Schedule:
    """
    Manages scheduling of important case events.

    Args:
        case_id: Case identifier

    Returns:
        Schedule
    """
    # Retrieve timeline events
    events = timeline_event_store.get(case_id, [])

    # Load and render the Jinja template
    template = env.get_template('schedule_case_events_prompt.j2')
    schedule_prompt = template.render(
        events=events
    )

    schedule_result = llm.chat([{"role": "user", "content": schedule_prompt}])

    # Parse the result into Schedule model
    try:
        schedule = llm.structured_output(Schedule)
    except Exception as e:
        raise ValueError(f"Error parsing Schedule: {e}")

    return schedule


@app.tool
def assess_risks(case_id: str) -> RiskAssessment:
    """
    Assesses risks related to the case.

    Args:
        case_id: Case identifier

    Returns:
        RiskAssessment
    """
    # Gather case data
    case_details = case_store[case_id]

    # Load and render the Jinja template
    template = env.get_template('assess_risks_prompt.j2')
    risk_prompt = template.render(case_details=case_details)

    risk_result = llm.chat([{"role": "user", "content": risk_prompt}])

    # Parse the result into RiskAssessment model
    try:
        risk_assessment = llm.structured_output(RiskAssessment)
    except Exception as e:
        raise ValueError(f"Error parsing RiskAssessment: {e}")

    return risk_assessment
