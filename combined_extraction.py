"""
Robust reference extraction from academic PDFs.
Designed for arXiv-style CS papers but works with most academic formats.
"""

import pdfx
import requests
import logging
import os
import re
import fitz  # PyMuPDF
import time
from typing import List, Dict, Optional, Set


# Rate limiting for API calls
LAST_API_CALL = 0
API_DELAY = 1.0  # seconds between API calls (increased to avoid throttling)
MAX_RETRIES = 2


def rate_limit():
    """Simple rate limiting for API calls."""
    global LAST_API_CALL
    elapsed = time.time() - LAST_API_CALL
    if elapsed < API_DELAY:
        time.sleep(API_DELAY - elapsed)
    LAST_API_CALL = time.time()


def api_call_with_retry(func, *args, retries=MAX_RETRIES, **kwargs):
    """Execute an API call with retry logic."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            rate_limit()
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < retries:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                logging.debug("API call failed, retrying in %ds: %s", wait_time, str(e))
                time.sleep(wait_time)
    logging.debug("API call failed after %d retries: %s", retries, str(last_error))
    return None


def extract_references_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract ALL references from a PDF using multiple strategies.
    Returns a list of reference info dicts.
    """
    references = []
    seen_ids = set()
    
    try:
        # Extract full text using PyMuPDF
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        doc.close()
        
        logging.info("Extracted %d characters of text from PDF", len(full_text))
        
        # Strategy 1: Extract arXiv IDs from entire document
        arxiv_refs = extract_arxiv_ids(full_text, seen_ids)
        references.extend(arxiv_refs)
        logging.info("Found %d arXiv references", len(arxiv_refs))
        
        # Strategy 2: Extract DOIs from entire document
        doi_refs = extract_dois(full_text, seen_ids)
        references.extend(doi_refs)
        logging.info("Found %d DOI references", len(doi_refs))
        
        # Strategy 3: Parse the references section
        ref_section = extract_references_section(full_text)
        if ref_section:
            logging.info("Found references section (%d chars)", len(ref_section))
            
            # Parse individual reference entries
            ref_entries = parse_all_reference_entries(ref_section)
            logging.info("Parsed %d reference entries", len(ref_entries))
            
            for entry in ref_entries:
                raw_text = entry.get('raw_text', '')
                
                # Skip if we already have this reference
                entry_id = entry.get('title', '') or raw_text[:100]
                if entry_id in seen_ids:
                    continue
                seen_ids.add(entry_id)
                
                # Check for arXiv ID in this entry - try multiple patterns
                arxiv_patterns = [
                    r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)',
                    r'arXiv[:\s]*(\d{4}\.\d{4,5}(?:v\d+)?)',
                ]
                arxiv_id = None
                for pattern in arxiv_patterns:
                    arxiv_match = re.search(pattern, raw_text, re.I)
                    if arxiv_match:
                        arxiv_id = arxiv_match.group(1)
                        break
                
                if arxiv_id and arxiv_id not in seen_ids:
                    seen_ids.add(arxiv_id)
                    references.append({
                        'type': 'arxiv',
                        'id': arxiv_id,
                        'query': f'arXiv:{arxiv_id}',
                        'raw_text': raw_text
                    })
                    continue
                
                # Check for DOI in this entry
                doi_match = re.search(r'(10\.\d{4,}/[^\s\]>\)\"\']+)', raw_text)
                if doi_match:
                    doi = clean_doi(doi_match.group(1))
                    if doi not in seen_ids:
                        seen_ids.add(doi)
                        references.append({
                            'type': 'doi',
                            'id': doi,
                            'query': doi,
                            'raw_text': raw_text
                        })
                        continue
                
                # For entries without arXiv or DOI, extract what we can and use title lookup
                # This handles many CS conference papers that don't have DOIs/arXiv IDs in the reference
                query = entry.get('title') or extract_search_query(raw_text)
                
                # Skip entries that are clearly not valid references
                if not query or len(query) < 15:
                    continue
                if query.lower().startswith('url ') or query.lower().startswith('http'):
                    continue
                if re.match(r'^[A-Z]\s+[A-Z]', query):  # Skip appendix headers like "A Pareto-Optimality"
                    continue
                    
                if len(raw_text) > 40:
                    references.append({
                        'type': 'title',
                        'id': entry.get('title') or raw_text[:150],
                        'query': query,
                        'raw_text': raw_text,
                        'authors': entry.get('authors', ''),
                        'year': entry.get('year')
                    })
        
        # Strategy 4: Try pdfx as backup (suppress its verbose logging)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Temporarily raise pdfx logging level
                pdfx_logger = logging.getLogger('pdfx')
                pdfminer_logger = logging.getLogger('pdfminer')
                old_pdfx_level = pdfx_logger.level
                old_pdfminer_level = pdfminer_logger.level
                pdfx_logger.setLevel(logging.ERROR)
                pdfminer_logger.setLevel(logging.ERROR)
                
                pdf = pdfx.PDFx(pdf_path)
                pdfx_refs = pdf.get_references()
                
                # Restore logging levels
                pdfx_logger.setLevel(old_pdfx_level)
                pdfminer_logger.setLevel(old_pdfminer_level)
            if isinstance(pdfx_refs, dict):
                for arxiv_id in pdfx_refs.get('arxiv', []):
                    if arxiv_id not in seen_ids:
                        seen_ids.add(arxiv_id)
                        references.append({
                            'type': 'arxiv',
                            'id': arxiv_id,
                            'query': f'arXiv:{arxiv_id}'
                        })
                for doi in pdfx_refs.get('doi', []):
                    if doi not in seen_ids:
                        seen_ids.add(doi)
                        references.append({
                            'type': 'doi',
                            'id': doi,
                            'query': doi
                        })
        except Exception as e:
            logging.debug("pdfx extraction failed: %s", str(e))
        
        logging.info("Total references extracted: %d", len(references))
        return references
        
    except Exception as e:
        logging.error("Error extracting references from PDF: %s", str(e))
        return []


def extract_arxiv_ids(text: str, seen: Set[str]) -> List[Dict]:
    """Extract all arXiv IDs from text."""
    refs = []
    
    # Normalize text first to fix line breaks
    text = normalize_pdf_text(text)
    
    # Multiple patterns for arXiv IDs - order matters, more specific first
    patterns = [
        r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)',  # arxiv.org URLs (most reliable)
        r'arXiv[:\s]*(\d{4}\.\d{4,5}(?:v\d+)?)',  # arXiv:2301.12345
        r'(\d{4}\.\d{4,5}(?:v\d+)?)\s*\[(?:cs|stat|math|physics|q-bio|q-fin|eess|econ)',  # 2301.12345 [cs.LG]
        r'(?<=URL\s)https?://[^\s]*?(\d{4}\.\d{4,5})',  # URL containing arXiv ID
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            arxiv_id = match.strip()
            # Remove version suffix for deduplication check, but keep it in the ID
            base_id = re.sub(r'v\d+$', '', arxiv_id)
            if arxiv_id and base_id not in seen and is_valid_arxiv_id(arxiv_id):
                seen.add(base_id)
                refs.append({
                    'type': 'arxiv',
                    'id': arxiv_id,
                    'query': f'arXiv:{arxiv_id}'
                })
    
    return refs


def is_valid_arxiv_id(arxiv_id: str) -> bool:
    """Check if string looks like a valid arXiv ID."""
    # Must be in format YYMM.NNNNN (4 or 5 digits after period)
    match = re.match(r'^(\d{4})\.(\d{4,5})(v\d+)?$', arxiv_id)
    if not match:
        return False
    yymm = match.group(1)
    # Year should be reasonable (07-30 for 2007-2030)
    year = int(yymm[:2])
    month = int(yymm[2:])
    return 7 <= year <= 35 and 1 <= month <= 12


def extract_dois(text: str, seen: Set[str]) -> List[Dict]:
    """Extract all DOIs from text."""
    refs = []
    
    # DOI patterns
    patterns = [
        r'doi[:\s]*(?:https?://(?:dx\.)?doi\.org/)?(10\.\d{4,}/[^\s\]>\)\"\']+)',
        r'https?://(?:dx\.)?doi\.org/(10\.\d{4,}/[^\s\]>\)\"\']+)',
        r'(?:^|[\s\(])(10\.\d{4,}/[^\s\]>\)\"\']+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            doi = clean_doi(match)
            if doi and doi not in seen and is_valid_doi(doi):
                seen.add(doi)
                refs.append({
                    'type': 'doi',
                    'id': doi,
                    'query': doi
                })
    
    return refs


def clean_doi(doi: str) -> str:
    """Clean up a DOI string."""
    doi = doi.strip()
    # Remove trailing punctuation and common suffixes
    doi = re.sub(r'[.,;:\)\]]+$', '', doi)
    doi = re.sub(r'\.URL$', '', doi, flags=re.IGNORECASE)
    doi = re.sub(r'\.pdf$', '', doi, flags=re.IGNORECASE)
    # Remove HTML entities
    doi = doi.replace('&amp;', '&')
    return doi


def is_valid_doi(doi: str) -> bool:
    """Check if string looks like a valid DOI."""
    return bool(re.match(r'^10\.\d{4,}/\S+$', doi))


def normalize_pdf_text(text: str) -> str:
    """
    Normalize PDF text to fix common extraction issues.
    Rejoins URLs and arXiv IDs that were split across lines.
    """
    # Fix URLs split across lines (arxiv.org URLs are common in CS papers)
    text = re.sub(r'(https?://[\w.]+)\s*\n\s*([\w./\-]+)', r'\1\2', text)
    
    # Fix arxiv.org/abs/ split with arXiv ID on next line  
    text = re.sub(r'(arxiv\.org/(?:abs|pdf)/)\s*\n?\s*(\d{4})', r'\1\2', text, flags=re.IGNORECASE)
    
    # Fix arXiv IDs split across lines like "2503.\n01307" (after abs/)
    text = re.sub(r'(arxiv\.org/(?:abs|pdf)/\d{4})\.\s*\n?\s*(\d{4,5})', r'\1.\2', text, flags=re.IGNORECASE)
    
    # Fix arXiv IDs split across lines like "arXiv:\n2301.12345"
    text = re.sub(r'arXiv[:\s]*\n\s*(\d)', r'arXiv:\1', text, flags=re.IGNORECASE)
    
    # Fix arXiv ID with split decimal like "arXiv:2503.\n01307"  
    text = re.sub(r'(arXiv[:\s]*\d{4})\.\s*\n?\s*(\d{4,5})', r'\1.\2', text, flags=re.IGNORECASE)
    
    # Fix DOIs split across lines
    text = re.sub(r'(10\.\d{4,}/[^\s]*)\s*\n\s*([^\s\]>\)\"\']+)', r'\1\2', text)
    
    return text


def extract_references_section(text: str) -> Optional[str]:
    """Extract the References/Bibliography section from paper text."""
    
    # Normalize text first
    text = normalize_pdf_text(text)
    
    # Headers that indicate start of references
    start_patterns = [
        r'\n\s*References\s*\n',
        r'\n\s*REFERENCES\s*\n',
        r'\n\s*Bibliography\s*\n',
        r'\n\s*BIBLIOGRAPHY\s*\n',
        r'\n\s*Literature Cited\s*\n',
        r'\n\s*Works Cited\s*\n',
        r'\n\s*R\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S\s*\n',  # Spaced out
        r'\nReferences\s*$',  # At end of line
    ]
    
    best_match = None
    best_pos = -1
    
    for pattern in start_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            # Take the LAST match (references usually at end)
            for m in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                if m.end() > best_pos:
                    best_pos = m.end()
                    best_match = m
    
    if not best_match:
        # Fallback: look for [1] pattern indicating numbered references
        numbered_match = re.search(r'\n\s*\[1\]\s+[A-Z]', text)
        if numbered_match:
            best_pos = numbered_match.start()
    
    if best_pos == -1:
        return None
    
    # Extract from start of references to end of document (or next major section)
    ref_text = text[best_pos:]
    
    # Try to find end of references section
    end_patterns = [
        r'\n\s*Appendix',
        r'\n\s*APPENDIX',
        r'\n\s*Supplementary',
        r'\n\s*SUPPLEMENTARY',
        r'\n\s*Acknowledgment',
        r'\n\s*ACKNOWLEDGMENT',
        r'\n\s*A\.\s+[A-Z][a-z]+',  # Appendix A. Title
    ]
    
    for pattern in end_patterns:
        end_match = re.search(pattern, ref_text, re.IGNORECASE)
        if end_match:
            ref_text = ref_text[:end_match.start()]
            break
    
    return ref_text


def parse_all_reference_entries(ref_text: str) -> List[Dict]:
    """Parse reference text into individual entries."""
    entries = []
    
    # Normalize text to fix line-break issues
    ref_text = normalize_pdf_text(ref_text)
    
    # Try multiple splitting strategies
    
    # Strategy 1: Split by [number] pattern
    numbered_refs = re.split(r'\n\s*\[(\d+)\]\s*', ref_text)
    if len(numbered_refs) > 3:  # Found numbered references
        for i in range(1, len(numbered_refs), 2):
            if i + 1 < len(numbered_refs):
                ref_num = numbered_refs[i]
                ref_content = numbered_refs[i + 1].strip()
                if len(ref_content) > 20:
                    entry = parse_single_reference(ref_content)
                    entry['ref_num'] = ref_num
                    entries.append(entry)
        if entries:
            return entries
    
    # Strategy 2: Split by number. pattern (1., 2., etc.)
    dot_numbered = re.split(r'\n\s*(\d+)\.\s+', ref_text)
    if len(dot_numbered) > 3:
        for i in range(1, len(dot_numbered), 2):
            if i + 1 < len(dot_numbered):
                ref_content = dot_numbered[i + 1].strip()
                if len(ref_content) > 20:
                    entry = parse_single_reference(ref_content)
                    entry['ref_num'] = dot_numbered[i]
                    entries.append(entry)
        if entries:
            return entries
    
    # Strategy 3: Split by blank lines
    paragraphs = re.split(r'\n\s*\n', ref_text)
    for para in paragraphs:
        para = para.strip()
        if len(para) > 30:  # Long enough to be a reference
            # Check if it looks like a reference (starts with author name or number)
            if re.match(r'^(?:\[?\d+\]?\.?\s*)?[A-Z][a-z]+', para):
                entry = parse_single_reference(para)
                entries.append(entry)
    
    if entries:
        return entries
    
    # Strategy 4: Line by line for dense references
    lines = ref_text.split('\n')
    current_ref = []
    for line in lines:
        line = line.strip()
        if not line:
            if current_ref:
                ref_content = ' '.join(current_ref)
                if len(ref_content) > 30:
                    entry = parse_single_reference(ref_content)
                    entries.append(entry)
                current_ref = []
        elif re.match(r'^(?:\[?\d+\]?\.?\s*)?[A-Z]', line):
            # New reference starts
            if current_ref:
                ref_content = ' '.join(current_ref)
                if len(ref_content) > 30:
                    entry = parse_single_reference(ref_content)
                    entries.append(entry)
            current_ref = [line]
        else:
            current_ref.append(line)
    
    if current_ref:
        ref_content = ' '.join(current_ref)
        if len(ref_content) > 30:
            entry = parse_single_reference(ref_content)
            entries.append(entry)
    
    return entries


def parse_single_reference(ref_text: str) -> Dict:
    """Parse a single reference entry to extract title, authors, year."""
    ref_text = ' '.join(ref_text.split())  # Normalize whitespace
    
    entry = {
        'raw_text': ref_text[:1000],  # Keep first 1000 chars
        'title': None,
        'authors': None,
        'year': None
    }
    
    # Extract year
    year_match = re.search(r'\b(19|20)\d{2}\b', ref_text)
    if year_match:
        entry['year'] = int(year_match.group())
    
    # Extract title using multiple strategies
    title = extract_title_from_reference(ref_text)
    if title:
        entry['title'] = title
    
    # Extract authors (names before the year or title)
    authors = extract_authors_from_reference(ref_text)
    if authors:
        entry['authors'] = authors
    
    return entry


def extract_title_from_reference(ref_text: str) -> Optional[str]:
    """Extract paper title from a reference entry."""
    
    # Strategy 1: Title in quotes
    quote_patterns = [
        r'["\u201c]([^"\u201d]{15,300})["\u201d]',
        r"'([^']{15,300})'(?:\s*,|\s*\.|\s*In\b)",
    ]
    for pattern in quote_patterns:
        match = re.search(pattern, ref_text)
        if match:
            return clean_title(match.group(1))
    
    # Strategy 2: Title in italics (marked with _ or * in some formats)
    italic_match = re.search(r'[_\*]([^_\*]{15,300})[_\*]', ref_text)
    if italic_match:
        return clean_title(italic_match.group(1))
    
    # Strategy 3: Title after year and period/comma
    after_year = re.search(
        r'(?:19|20)\d{2}[a-z]?\)?[\.,]\s*([A-Z][^\.]{15,200}?)(?:\.|\sIn\s|\sarXiv|\sProceedings|\sJournal|\sConference|\sIEEE|\sACM|\sNeurIPS|\sICML|\sICLR|\sCVPR)',
        ref_text
    )
    if after_year:
        return clean_title(after_year.group(1))
    
    # Strategy 4: After author list pattern
    # Look for: Names. Title. or Names, Title,
    author_title = re.search(
        r'(?:[A-Z][a-z]+(?:,?\s+(?:and\s+)?[A-Z]\.?\s*)+[,\.])\s*([A-Z][^\.]{15,200}?)(?:\.|,\s*(?:In|Proceedings|Journal|arXiv))',
        ref_text
    )
    if author_title:
        return clean_title(author_title.group(1))
    
    # Strategy 5: Look for common conference/journal patterns and get text before
    venue_match = re.search(
        r'([A-Z][^\.]{15,200}?)(?:\.|\s+)(?:In\s+)?(?:Proceedings|Proc\.|Conference|Journal|Trans\.|NeurIPS|ICML|ICLR|CVPR|ICCV|ECCV|ACL|EMNLP|NAACL|AAAI|IJCAI)',
        ref_text,
        re.IGNORECASE
    )
    if venue_match:
        # Make sure this is after author names
        potential_title = venue_match.group(1)
        # Check it's not just author names
        if not re.match(r'^[A-Z][a-z]+(?:\s+[A-Z]\.?)+$', potential_title):
            return clean_title(potential_title)
    
    return None


def clean_title(title: str) -> str:
    """Clean up extracted title."""
    title = title.strip()
    # Remove leading/trailing punctuation
    title = re.sub(r'^[\.\,\:\;\s]+|[\.\,\:\;\s]+$', '', title)
    # Remove multiple spaces
    title = ' '.join(title.split())
    # Minimum reasonable title length
    if len(title) < 10:
        return None
    return title


def extract_authors_from_reference(ref_text: str) -> Optional[str]:
    """Extract author names from reference."""
    # Look for pattern: Name, Name, and Name.
    # Or: Name et al.
    author_match = re.match(
        r'^(?:\[?\d+\]?\.?\s*)?([A-Z][a-z]+(?:(?:,?\s*(?:and\s+)?[A-Z](?:\.|[a-z]+)\s*)+)?(?:\s+et\s+al\.?)?)',
        ref_text
    )
    if author_match:
        authors = author_match.group(1).strip()
        if len(authors) > 2:
            return authors
    return None


def extract_search_query(ref_text: str) -> str:
    """
    Extract a search query from a reference text.
    Removes URLs, numbers, and tries to get the most relevant part.
    """
    # Remove URLs
    text = re.sub(r'https?://\S+', '', ref_text)
    # Remove common noise
    text = re.sub(r'URL\s*', '', text)
    text = re.sub(r'\b(Proceedings|Conference|Journal|Trans\.|arXiv|preprint)\b.*$', '', text, flags=re.I)
    # Clean up
    text = ' '.join(text.split())
    # Limit length
    if len(text) > 200:
        text = text[:200]
    return text.strip()


def lookup_paper_semantic_scholar(query: str, query_type: str = 'title') -> Optional[Dict]:
    """Look up a paper using Semantic Scholar API."""
    rate_limit()
    
    base_url = "https://api.semanticscholar.org/graph/v1/paper"
    fields = 'paperId,title,authors,abstract,year,externalIds,url,venue'
    
    try:
        if query_type == 'arxiv':
            # Clean arXiv ID
            arxiv_id = query.replace('arXiv:', '').strip()
            url = f"{base_url}/arXiv:{arxiv_id}"
            params = {'fields': fields}
            response = requests.get(url, params=params, timeout=15)
            
        elif query_type == 'doi':
            url = f"{base_url}/{query}"
            params = {'fields': fields}
            response = requests.get(url, params=params, timeout=15)
            
        else:
            # Search by title
            search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                'query': query[:200],  # Limit query length
                'limit': 3,
                'fields': fields
            }
            response = requests.get(search_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                papers = data.get('data', [])
                if papers:
                    # Find best match by title similarity
                    best_match = find_best_title_match(query, papers)
                    if best_match:
                        return format_semantic_scholar_result(best_match)
            return None
        
        if response.status_code == 200:
            paper = response.json()
            return format_semantic_scholar_result(paper)
        elif response.status_code == 404:
            logging.debug("Paper not found in Semantic Scholar: %s", query[:50])
        else:
            logging.debug("Semantic Scholar API returned %d for %s", response.status_code, query[:50])
            
    except requests.exceptions.Timeout:
        logging.debug("Semantic Scholar API timeout for %s", query[:50])
    except Exception as e:
        logging.debug("Semantic Scholar lookup failed: %s", str(e))
    
    return None


def find_best_title_match(query: str, papers: List[Dict]) -> Optional[Dict]:
    """Find the best matching paper by title similarity."""
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    best_paper = None
    best_score = 0
    
    for paper in papers:
        title = paper.get('title', '').lower()
        title_words = set(title.split())
        
        # Jaccard similarity
        if query_words and title_words:
            intersection = len(query_words & title_words)
            union = len(query_words | title_words)
            score = intersection / union if union > 0 else 0
            
            if score > best_score and score > 0.3:  # Minimum 30% similarity
                best_score = score
                best_paper = paper
    
    return best_paper


def format_semantic_scholar_result(paper: Dict) -> Optional[Dict]:
    """Format Semantic Scholar API result."""
    if not paper:
        return None
    
    authors = []
    for author in paper.get('authors', []):
        if author.get('name'):
            authors.append(author['name'])
    
    external_ids = paper.get('externalIds', {}) or {}
    
    return {
        'title': paper.get('title', ''),
        'authors': ', '.join(authors),
        'abstract': paper.get('abstract', '') or '',
        'year': paper.get('year'),
        'doi': external_ids.get('DOI'),
        'arxiv_id': external_ids.get('ArXiv'),
        'url': paper.get('url', ''),
        'venue': paper.get('venue', ''),
        'semantic_scholar_id': paper.get('paperId')
    }


def lookup_paper_openalex(query: str, query_type: str = 'title') -> Optional[Dict]:
    """Look up a paper using OpenAlex API (free, no auth required)."""
    rate_limit()
    
    try:
        if query_type == 'doi':
            url = f"https://api.openalex.org/works/https://doi.org/{query}"
        elif query_type == 'arxiv':
            arxiv_id = query.replace('arXiv:', '').strip()
            url = f"https://api.openalex.org/works?filter=ids.arxiv:{arxiv_id}"
        else:
            # Search by title
            url = "https://api.openalex.org/works"
            params = {
                'search': query[:200],
                'per-page': 3
            }
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                if results:
                    return format_openalex_result(results[0])
            return None
        
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if query_type == 'arxiv' and 'results' in data:
                results = data.get('results', [])
                if results:
                    return format_openalex_result(results[0])
            else:
                return format_openalex_result(data)
                
    except Exception as e:
        logging.debug("OpenAlex lookup failed: %s", str(e))
    
    return None


def format_openalex_result(work: Dict) -> Optional[Dict]:
    """Format OpenAlex API result."""
    if not work:
        return None
    
    authors = []
    for authorship in work.get('authorships', []):
        author = authorship.get('author', {})
        if author.get('display_name'):
            authors.append(author['display_name'])
    
    ids = work.get('ids', {})
    doi = ids.get('doi', '').replace('https://doi.org/', '') if ids.get('doi') else None
    
    # Get abstract from inverted index
    abstract = ''
    abstract_index = work.get('abstract_inverted_index', {})
    if abstract_index:
        # Reconstruct abstract from inverted index
        word_positions = []
        for word, positions in abstract_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort()
        abstract = ' '.join(word for _, word in word_positions)
    
    return {
        'title': work.get('title', ''),
        'authors': ', '.join(authors[:10]),  # Limit to 10 authors
        'abstract': abstract[:2000],  # Limit abstract length
        'year': work.get('publication_year'),
        'doi': doi,
        'arxiv_id': None,  # OpenAlex doesn't directly provide arXiv ID
        'url': work.get('id', ''),
        'venue': work.get('primary_location', {}).get('source', {}).get('display_name', '') if work.get('primary_location') else ''
    }


def get_open_access_pdf_url(identifier: str, email: str = None) -> Optional[str]:
    """Get open access PDF URL from Unpaywall or arXiv."""
    if not identifier:
        return None
    
    # If it looks like an arXiv ID, return arXiv PDF URL directly
    if re.match(r'^\d{4}\.\d{4,5}', identifier):
        return f"https://arxiv.org/pdf/{identifier}.pdf"
    
    # For DOIs, try Unpaywall
    if identifier.startswith('10.'):
        if email is None:
            email = os.getenv("UNPAYWALL_EMAIL", "your-email@example.com")
        
        rate_limit()
        api_url = f"https://api.unpaywall.org/v2/{identifier}?email={email}"
        try:
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                oa_location = data.get("best_oa_location", {})
                if oa_location:
                    return oa_location.get("url_for_pdf") or oa_location.get("url")
        except Exception as e:
            logging.debug("Unpaywall lookup failed for %s: %s", identifier, str(e))
    
    return None


def fetch_paper_metadata(ref_info: Dict, email: str = None) -> Optional[Dict]:
    """
    Fetch paper metadata using multiple services.
    Tries Semantic Scholar first, then OpenAlex, then Crossref.
    """
    if isinstance(ref_info, str):
        ref_info = {'type': 'doi', 'id': ref_info, 'query': ref_info}
    
    ref_type = ref_info.get('type', 'title')
    query = ref_info.get('query', ref_info.get('id', ''))
    
    if not query:
        return None
    
    # Try Semantic Scholar first (best for CS papers)
    result = lookup_paper_semantic_scholar(query, ref_type)
    if result and result.get('title'):
        logging.debug("Found via Semantic Scholar: %s", result['title'][:50])
        return result
    
    # Try OpenAlex (comprehensive, free)
    result = lookup_paper_openalex(query, ref_type)
    if result and result.get('title'):
        logging.debug("Found via OpenAlex: %s", result['title'][:50])
        return result
    
    # For DOIs, try Crossref as last resort
    if ref_type == 'doi':
        result = fetch_from_crossref(query, email)
        if result and result.get('title'):
            logging.debug("Found via Crossref: %s", result['title'][:50])
            return result
    
    # If we have raw text info, return that as fallback
    if ref_info.get('raw_text'):
        return {
            'title': ref_info.get('title') or ref_info.get('raw_text', '')[:200],
            'authors': ref_info.get('authors', ''),
            'abstract': '',
            'year': ref_info.get('year'),
            'doi': ref_info.get('id') if ref_type == 'doi' else None,
            'arxiv_id': ref_info.get('id') if ref_type == 'arxiv' else None,
            'url': '',
            'source': 'extracted'
        }
    
    return None


def fetch_from_crossref(doi: str, email: str = None) -> Optional[Dict]:
    """Fetch paper metadata from Crossref API."""
    if email is None:
        email = os.getenv("UNPAYWALL_EMAIL", "your-email@example.com")
    
    rate_limit()
    api_url = f"https://api.crossref.org/works/{doi}"
    headers = {"User-Agent": f"ResearchAssistant/1.0 (mailto:{email})"}
    
    try:
        response = requests.get(api_url, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            message = data.get("message", {})
            
            title_list = message.get("title", [])
            title = title_list[0] if title_list else ''
            
            authors_list = message.get("author", [])
            authors = ", ".join([
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in authors_list
            ])
            
            abstract = message.get("abstract", "") or ""
            if abstract:
                abstract = re.sub(r'<[^>]+>', '', abstract)
            
            # Get year
            year = None
            published = message.get('published', {}) or message.get('published-print', {}) or message.get('published-online', {})
            if published and 'date-parts' in published:
                date_parts = published['date-parts']
                if date_parts and date_parts[0]:
                    year = date_parts[0][0]
            
            return {
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'doi': doi,
                'year': year,
                'arxiv_id': None,
                'url': f"https://doi.org/{doi}",
                'venue': message.get('container-title', [''])[0] if message.get('container-title') else '',
                'publisher': message.get("publisher", "")
            }
    except Exception as e:
        logging.debug("Crossref lookup failed for DOI %s: %s", doi, str(e))
    
    return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        print("Usage: python combined_extraction.py <pdf_path>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    
    print(f"Extracting references from: {pdf_path}")
    references = extract_references_from_pdf(pdf_path)
    
    print(f"\nFound {len(references)} references:\n")
    for i, ref in enumerate(references[:20], 1):  # Show first 20
        print(f"[{i}] Type: {ref['type']}")
        print(f"    ID: {ref['id'][:80]}...")
        
        metadata = fetch_paper_metadata(ref)
        if metadata:
            print(f"    Title: {metadata.get('title', 'N/A')[:80]}")
            print(f"    Authors: {metadata.get('authors', 'N/A')[:60]}")
            print(f"    Year: {metadata.get('year', 'N/A')}")
        print()
    
    if len(references) > 20:
        print(f"... and {len(references) - 20} more references")
