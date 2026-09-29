import requests


def reconstruct_abstract(inverted_index):
    """
    OpenAlex stores abstracts as an inverted index:
        {"The": [0], "quick": [1], "fox": [2, 5], ...}
    This rebuilds the original text from that structure.
    """
    if not inverted_index:
        return None

    # Find the highest word position to know how long the abstract is
    max_pos = max(pos for positions in inverted_index.values() for pos in positions)
    words = [""] * (max_pos + 1)

    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word

    return " ".join(words)


def get_work_by_doi(doi, api_key):
    """
    Fetch a work's metadata from OpenAlex given a DOI.

    Parameters
    ----------
    doi : str
        The DOI, with or without the "https://doi.org/" prefix.
    api_key : str
        Your OpenAlex API key (free — get one at
        https://openalex.org/settings/api). Required as of Feb 13, 2026;
        without it you're capped at 100 credits/day before getting 409s.

    Returns
    -------
    dict with keys: title, doi, abstract, publication_year, venue,
                     authors, cited_by_count, open_access_url
        or None if the DOI isn't found.
    """
    # Normalize the DOI (OpenAlex accepts the bare DOI in the URL path)
    doi = doi.strip()
    if doi.startswith("https://doi.org/"):
        doi = doi.replace("https://doi.org/", "")
    elif doi.startswith("doi.org/"):
        doi = doi.replace("doi.org/", "")

    url = f"https://api.openalex.org/works/https://doi.org/{doi}"

    params = {"api_key": api_key}

    response = requests.get(url, params=params, timeout=15)

    if response.status_code == 404:
        return None
    response.raise_for_status()

    data = response.json()

    abstract = reconstruct_abstract(data.get("abstract_inverted_index"))

    authors = [
        authorship["author"]["display_name"]
        for authorship in data.get("authorships", [])
    ]

    venue = None
    primary_location = data.get("primary_location") or {}
    source = primary_location.get("source") or {}
    if source:
        venue = source.get("display_name")

    oa_url = (data.get("open_access") or {}).get("oa_url")

    return {
        "title": data.get("title"),
        "doi": data.get("doi"),
        "abstract": abstract,
        "publication_year": data.get("publication_year"),
        "venue": venue,
        "authors": authors,
        "cited_by_count": data.get("cited_by_count"),
        "open_access_url": oa_url,
    }
