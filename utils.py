import requests
import streamlit as st
from io import BytesIO
import pandas as pd



def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buf.seek(0)
    return buf.read()

EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"



# ---- Caching: fetch all works for an author (paginated) ----
@st.cache_data(show_spinner=False)
def fetch_author_works(author_id_url: str, max_results: int = 2000):
    works = []
    cursor = "*"
    per_page = 200
    base = f"https://api.openalex.org/works?filter=authorships.author.id:{author_id_url}&per_page={per_page}"
    while len(works) < max_results and cursor:
        r = requests.get(f"{base}&cursor={cursor}").json()
        works.extend(r.get("results", []))
        cursor = r.get("meta", {}).get("next_cursor")
        if not r.get("results"):
            break
    return works[:max_results]


# ---- Build a coauthor table from a list of OpenAlex work dicts ----
def build_coauthor_table(works: list, focus_author_ids: set):
    """
    Returns:
      - co_df: rows = coauthor (id/name); metrics = #joint papers with focus set, total citations of those papers, first/last year, etc.
      - edges_df: (focus_author_id, coauthor_id, work_id) rows (useful if you want to draw a network later)
    Only papers containing at least one focus author are considered; co-authors are the *other* authors on that paper.
    """
    co_rows = []
    edge_rows = []

    for w in works:
        wid = w.get("id")
        year = pd.to_datetime(w.get("publication_date", None), errors="coerce").year
        citations = w.get("cited_by_count", 0) or 0
        auths = w.get("authorships", []) or []
        paper_author_ids = {a["author"]["id"] for a in auths if a.get("author", {}).get("id")}
        # consider only if paper has at least one focus author
        if not paper_author_ids.intersection(focus_author_ids):
            continue

        # for each focus author on this paper, record edges to the other coauthors
        for a in auths:
            this_id = a.get("author", {}).get("id")
            this_name = a.get("author", {}).get("display_name", "N/A")
            if not this_id:
                continue

            # if this author is a coauthor (not necessarily in focus set)
            for b in auths:
                co_id = b.get("author", {}).get("id")
                co_name = b.get("author", {}).get("display_name", "N/A")
                if not co_id or co_id == this_id:
                    continue
                # record only pairs where THIS is focus and other is coauthor
                if this_id in focus_author_ids:
                    co_rows.append({
                        "Coauthor_ID": co_id,
                        "Coauthor_Name": co_name,
                        "Work_ID": wid,
                        "Year": year,
                        "Paper_Citations": citations,
                    })
                    edge_rows.append({
                        "Focus_Author_ID": this_id,
                        "Coauthor_ID": co_id,
                        "Work_ID": wid,
                    })

    if not co_rows:
        return pd.DataFrame(), pd.DataFrame()

    co_df = pd.DataFrame(co_rows)
    # aggregate per coauthor
    agg = (co_df.groupby(["Coauthor_ID", "Coauthor_Name"], as_index=False)
           .agg(
               Joint_Papers=("Work_ID", "nunique"),
               Joint_Citations=("Paper_Citations", "sum"),
               First_Year=("Year", "min"),
               Last_Year=("Year", "max"),
           )
           .sort_values(["Joint_Papers", "Joint_Citations"], ascending=False))
    edges_df = pd.DataFrame(edge_rows)
    return agg, edges_df


def get_concept_ids(keywords):
    """
    Given a list or comma-separated string of keywords, query OpenAlex
    for matching concepts and return their IDs (without the full URL prefix).
    """
    if isinstance(keywords, str):
        keywords = [kw.strip() for kw in keywords.split(",") if kw.strip()]

    ids = []
    for keyword in keywords:
        try:
            r = requests.get(f"https://api.openalex.org/concepts?search={keyword}").json()
            if r.get("results"):
                concept_id = r["results"][0]["id"].split("/")[-1]
                ids.append(concept_id)
        except Exception as e:
            st.warning(f"Error retrieving concept ID for '{keyword}': {e}")
    return ids


def build_filter_string(
    abstract_keywords=None,
    abstract_keywords_optional=None,
    concept_keywords=None,
    concept_keywords_optional=None,
    country_codes=None,
    from_date=None,
    to_date=None,
    institutions=None
):
    """
    Build an OpenAlex API filter string based on provided parameters.
    """
    filters = []

    # Abstract keywords (must all match)
    if abstract_keywords:
        filters += [f"title_and_abstract.search:{kw.strip()}" for kw in abstract_keywords.split(",") if kw.strip()]

    # Abstract keywords (any match)
    if abstract_keywords_optional:
        keywords = "|".join([kw.strip() for kw in abstract_keywords_optional.split(",") if kw.strip()])
        filters.append(f"title_and_abstract.search:{keywords}")

    # Concept keywords (must all match)
    if concept_keywords:
        ids = get_concept_ids(concept_keywords)
        filters += [f"concepts.id:{cid}" for cid in ids]

    # Concept keywords (any match)
    if concept_keywords_optional:
        ids = get_concept_ids(concept_keywords_optional)
        if ids:
            filters.append(f"concepts.id:{'|'.join(ids)}")

    # Country filter
    if country_codes:
        codes = "|".join([x.strip() for x in country_codes.split(",") if x.strip()])
        filters.append(f"institutions.country_code:{codes}")

    # Institution filter
    if institutions:
        insts = "|".join([x.strip() for x in institutions.split(",") if x.strip()])
        filters.append(f"institutions.id:{insts}")

    # Date range
    if from_date:
        filters.append(f"from_publication_date:{from_date}")
    if to_date:
        filters.append(f"to_publication_date:{to_date}")

    return ",".join(filters)


def parse_abstract(indexed_dict):
    """
    Convert OpenAlex indexed_abstract dictionary format into a normal string.
    """
    if not indexed_dict or not isinstance(indexed_dict, dict):
        return ""
    word_list = sorted(
        [(int(k), v) for v, ks in indexed_dict.items() for k in ks],
        key=lambda x: x[0]
    )
    return " ".join([word for _, word in word_list])


def safe_get_nested(d, keys, default="N/A"):
    """
    Safely retrieve nested dictionary values using a list of keys.
    Returns `default` if any key is missing or value is falsy.
    """
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, {})
        else:
            return default
    return d if d else default

def query_openalex(filter_string, max_results=100):
    """
    Query the OpenAlex works endpoint with a given filter string.
    Returns:
        papers (list): paper metadata dictionaries (max_results limited)
        total_count (int): total number of matching works in OpenAlex
    """
    papers = []
    cursor = "*"
    per_page = 200  # OpenAlex maximum per page
    total_count = None

    while len(papers) < max_results:
        remaining = max_results - len(papers)
        page_size = min(per_page, remaining)

        request_url = (
            f"https://api.openalex.org/works?filter={filter_string}"
            f"&per_page={page_size}&cursor={cursor}"
        )
        try:
            r = requests.get(request_url).json()
        except Exception as e:
            st.error(f"Error querying OpenAlex: {e}")
            break

        if total_count is None:
            total_count = r.get("meta", {}).get("count", 0)

        new_results = [
            {
                "Title": result["display_name"],
                "Authors": "; ".join([a["author"]["display_name"] for a in result["authorships"]]),
                "AuthorTuples": [
                    (
                        a["author"]["display_name"],
                        a["author"]["id"],
                        a["author"].get("orcid")  # ORCID if present
                    )
                    for a in result["authorships"]
                ],
                # NEW: author-affiliation mapping (first listed institution per author on this paper)
                "AuthorAffiliations": [
                    {
                        "author_name": a["author"]["display_name"],
                        "author_id": a["author"]["id"],
                        "orcid": a["author"].get("orcid"),
                        "affiliation_name": (a.get("institutions") or [{}])[0].get("display_name", "N/A"),
                        "affiliation_id": (a.get("institutions") or [{}])[0].get("id")
                    }
                    for a in result["authorships"]
                ],
                "Publication Date": result.get("publication_date"),
                "Citations": result.get("cited_by_count"),
                "Journal": safe_get_nested(result, ["primary_location", "source", "display_name"]),
                "Link": safe_get_nested(result, ["primary_location", "landing_page_url"]),
                "Work ID": result["id"],
                "InstitutionTuples": [
                    (inst.get("display_name"), inst.get("id"))
                    for a in result["authorships"]
                    for inst in a.get("institutions", [])
                ],
                "CountryTuples": [
                    (inst.get("country_code", "").upper())
                    for a in result["authorships"]
                    for inst in a.get("institutions", [])
                    if inst.get("country_code")
                ],
                "concepts": result.get("concepts", []),
            }
            for result in r.get("results", [])
        ]


        papers.extend(new_results)

        cursor = r.get("meta", {}).get("next_cursor")
        if not cursor or not new_results:
            break  # no more data

    return papers[:max_results], total_count
