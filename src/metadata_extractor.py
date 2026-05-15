"""
Metadata Extraction Module for Legal Documents
Extracts case-specific metadata from Pearson Specter Litt legal documents
"""
import re
from typing import Dict, Optional, List


class LegalMetadataExtractor:
    """Extracts metadata from legal case documents"""

    def __init__(self):
        # Common patterns in legal documents
        self.case_name_patterns = [
            r'([A-Z][A-Za-z\s&]+)\s+[Vv][Ss]?\.?\s+([A-Z][A-Za-z\s&]+)',
            r'([A-Z][A-Za-z\s]+)\s+[Aa][Nn][Dd]\s+[Oo][Tt][Hh][Ee][Rr][Ss]',
        ]

        self.citation_patterns = [
            r'(\d+)\s+SCOB\s+(\d+)',
            r'(\d+)\s+BLD\s+(\d+)',
            r'(\d+)\s+DLR\s+(\d+)',
        ]

        self.court_patterns = [
            r'(Supreme Court|High Court Division|Appellate Division)',
            r'(Civil|Criminal|Constitutional|Commercial)\s+(Appeal|Petition|Revision)',
        ]

        self.judge_patterns = [
            r'(?:Justice|J\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:Hon\'?ble|Honourable)\s+(?:Mr\.|Ms\.)?\s*Justice\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]

    def extract_case_name(self, text: str) -> Optional[str]:
        """Extract case name (parties involved)"""
        for pattern in self.case_name_patterns:
            match = re.search(pattern, text[:2000])  # Search in first 2000 chars
            if match:
                if len(match.groups()) == 2:
                    return f"{match.group(1).strip()} vs {match.group(2).strip()}"
                else:
                    return match.group(0).strip()
        return None

    def extract_citations(self, text: str) -> List[str]:
        """Extract legal citations"""
        citations = []
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, text[:3000])
            for match in matches:
                citations.append(match.group(0))
        return list(set(citations))  # Remove duplicates

    def extract_court_info(self, text: str) -> Optional[str]:
        """Extract court information"""
        for pattern in self.court_patterns:
            match = re.search(pattern, text[:2000], re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    def extract_judges(self, text: str) -> List[str]:
        """Extract judge names"""
        judges = []
        for pattern in self.judge_patterns:
            matches = re.finditer(pattern, text[:3000])
            for match in matches:
                judge_name = match.group(1).strip()
                if len(judge_name) > 3:  # Avoid initials
                    judges.append(judge_name)
        return list(set(judges))[:5]  # Max 5 judges, remove duplicates

    def extract_case_number(self, text: str) -> Optional[str]:
        """Extract case/appeal number"""
        patterns = [
            r'(?:Civil|Criminal|Constitutional)\s+(?:Appeal|Petition|Revision)\s+No\.?\s*(\d+\s+of\s+\d+)',
            r'Case\s+No\.?\s*(\d+\s+of\s+\d+)',
            r'Appeal\s+No\.?\s*(\d+\s+of\s+\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:2000], re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def extract_judgment_date(self, text: str) -> Optional[str]:
        """Extract judgment date"""
        date_patterns = [
            r'(?:Judgment|Judgement|Decided|Date).*?(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text[:2000], re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def extract_subject_matter(self, text: str) -> List[str]:
        """Extract legal subject matter/topics"""
        # Common legal topics
        topics = []
        legal_topics = [
            'Constitution', 'Contract', 'Property', 'Criminal', 'Civil',
            'Service', 'Land', 'Tax', 'Administrative', 'Writ',
            'Fundamental Rights', 'Tort', 'Family', 'Succession',
            'Evidence', 'Procedure', 'Arbitration', 'Company',
            'Banking', 'Insurance', 'Labour', 'Employment'
        ]

        text_lower = text[:3000].lower()
        for topic in legal_topics:
            if topic.lower() in text_lower:
                topics.append(topic)

        return topics[:5]  # Max 5 topics

    def extract_all_metadata(self, text: str, filename: str) -> Dict:
        """Extract all available metadata from legal document"""
        metadata = {
            'filename': filename,
            'case_name': self.extract_case_name(text),
            'citations': self.extract_citations(text),
            'court': self.extract_court_info(text),
            'judges': self.extract_judges(text),
            'case_number': self.extract_case_number(text),
            'judgment_date': self.extract_judgment_date(text),
            'subject_matter': self.extract_subject_matter(text),
        }

        return metadata

    def format_metadata_for_display(self, metadata: Dict) -> str:
        """Format metadata into a readable string"""
        parts = []

        if metadata.get('case_name'):
            parts.append(f"Case: {metadata['case_name']}")

        if metadata.get('case_number'):
            parts.append(f"Case No: {metadata['case_number']}")

        if metadata.get('court'):
            parts.append(f"Court: {metadata['court']}")

        if metadata.get('judges'):
            parts.append(f"Judges: {', '.join(metadata['judges'])}")

        if metadata.get('judgment_date'):
            parts.append(f"Date: {metadata['judgment_date']}")

        if metadata.get('citations'):
            parts.append(f"Citations: {', '.join(metadata['citations'])}")

        if metadata.get('subject_matter'):
            parts.append(f"Topics: {', '.join(metadata['subject_matter'])}")

        return " | ".join(parts) if parts else "No metadata extracted"
