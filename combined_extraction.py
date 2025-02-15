import pdfx
import requests

def extract_references_from_pdf(pdf_path):
    pdf = pdfx.PDFx(pdf_path)
    return pdf.get_references()

def get_open_access_pdf_url(doi, email="your-email@example.com"):
    api_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            oa_location = data.get("best_oa_location", {})
            return oa_location.get("url_for_pdf") or oa_location.get("url")
        else:
            print(f"Unpaywall API for DOI {doi} returned HTTP {response.status_code}")
    except Exception as e:
        print(f"Error querying Unpaywall for DOI {doi}: {e}")
    return None

if __name__ == "__main__":
    pdf_path = "path/to/your/paper.pdf"
    references = extract_references_from_pdf(pdf_path)
    
    print("Extracted References:")
    for doi, citation in references.items():
        print(f"{doi} | {citation}")
        pdf_url = get_open_access_pdf_url(doi)
        if pdf_url:
            print(f"  --> Full text available: {pdf_url}")
        else:
            print(f"  --> Full text NOT available")
