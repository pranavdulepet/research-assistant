import pdfx
import sqlite3
import logging

def extract_references_from_pdf(pdf_path):
    pdf = pdfx.PDFx(pdf_path)
    references = pdf.get_references()  # Returns a dictionary containing DOIs and citations.
    # For example, references may look like:
    # { '10.1145/3442188': 'Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency', ... }
    return references

# Example usage:
pdf_path = "path/to/your/paper.pdf"
refs = extract_references_from_pdf(pdf_path)
print("Extracted references:")
for doi, citation in refs.items():
    print(f"{doi} | {citation}")

def store_cited_paper(metadata, full_text, upload_id, db_path="cited_papers.db"):
    """Store cited paper metadata and full text into the local SQLite database with its associated upload_id."""
    doi = metadata.get("DOI")
    title = metadata.get("title", [""])[0] if metadata.get("title") else ""
    publisher = metadata.get("publisher", "")
    if metadata.get("author"):
        authors = ", ".join([f"{a.get('given', '').strip()} {a.get('family', '').strip()}" for a in metadata.get("author", [])])
    else:
        authors = ""
    abstract = metadata.get("abstract", "")
    url = ""
    # Check if the "link" key exists, is a list, and is non-empty before accessing it.
    if "link" in metadata and isinstance(metadata["link"], list) and len(metadata["link"]) > 0:
        first_link = metadata["link"][0]
        url = first_link.get("URL", "")
    else:
        url = metadata.get("URL", "")
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
        c.execute('''
            INSERT OR REPLACE INTO cited_papers (doi, upload_id, title, publisher, authors, abstract, full_text, url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (doi, upload_id, title, publisher, authors, abstract, full_text, url))
        conn.commit()
        logging.info("Stored cited paper with DOI: %s for upload_id: %s", doi, upload_id)
    except Exception as e:
        logging.error("Error inserting into DB for DOI %s and upload_id %s: %s", doi, upload_id, str(e))
    finally:
        conn.close()
