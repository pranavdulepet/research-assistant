import requests

def get_open_access_pdf_url(doi, email="your-email@example.com"):
    """
    Query Unpaywall API for a given DOI to get the OA PDF URL, if available.
    Make sure to supply your email address as requested by the API.
    """
    api_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            oa_location = data.get("best_oa_location")
            if oa_location:
                return oa_location.get("url_for_pdf") or oa_location.get("url")
        else:
            print(f"Unpaywall API for DOI {doi} returned HTTP {response.status_code}")
    except Exception as e:
        print(f"Error querying Unpaywall for DOI {doi}: {e}")

    return None

# Example usage:
doi_example = "10.1145/3442188"
oa_pdf_url = get_open_access_pdf_url(doi_example)
if oa_pdf_url:
    print(f"Full text available at: {oa_pdf_url}")
else:
    print("No full text available for this DOI.")
