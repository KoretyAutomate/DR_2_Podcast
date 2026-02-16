
import asyncio
import os
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

async def test_pubmed_auth():
    print(f"Testing PubMed API with Key: {PUBMED_API_KEY[:5]}..." if PUBMED_API_KEY else "No API Key found")
    
    if not PUBMED_API_KEY:
        print("Skipping auth test as no API key is present.")
        return

    query = "cognitive performance"
    max_results = 1
    
    try:
        async with httpx.AsyncClient(timeout=15) as http:
            # Step 1: esearch with API key
            params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "api_key": PUBMED_API_KEY
            }
            
            print(f"Sending esearch request to {PUBMED_BASE_URL}/esearch.fcgi")
            resp = await http.get(
                f"{PUBMED_BASE_URL}/esearch.fcgi",
                params=params
            )
            print(f"Response Status: {resp.status_code}")
            resp.raise_for_status()
            
            data = resp.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            print(f"Found IDs: {id_list}")
            
            if not id_list:
                print("No results found, but auth seems okay.")
                return

            # Step 2: efetch with API key
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml",
                "api_key": PUBMED_API_KEY
            }
            
            print(f"Sending efetch request to {PUBMED_BASE_URL}/efetch.fcgi")
            resp = await http.get(
                f"{PUBMED_BASE_URL}/efetch.fcgi",
                params=fetch_params
            )
            print(f"Response Status: {resp.status_code}")
            resp.raise_for_status()
            print("Successfully fetched article details.")
            
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_pubmed_auth())
