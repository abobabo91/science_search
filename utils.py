import requests

def get_concept_ids(keywords):
    ids = []
    for keyword in keywords:
        r = requests.get(f"https://api.openalex.org/concepts?search={keyword.strip()}").json()
        if r["results"]:
            concept_id = r["results"][0]["id"].split("/")[-1]
            ids.append(concept_id)
    return ids

def build_filter_string(
    abstract_keywords,
    abstract_keywords_optional,
    concept_keywords,
    concept_keywords_optional,
    country_codes,
    from_date,
    to_date,
    institutions
):
    filters = []

    if abstract_keywords:
        filters += [f"abstract.search:{kw.strip()}" for kw in abstract_keywords.split(",")]

    if abstract_keywords_optional:
        keywords = "|".join([kw.strip() for kw in abstract_keywords_optional.split(",")])
        filters.append(f"abstract.search:{keywords}")

    if concept_keywords:
        ids = get_concept_ids(concept_keywords.split(","))
        filters += [f"concepts.id:{cid}" for cid in ids]

    if concept_keywords_optional:
        ids = get_concept_ids(concept_keywords_optional.split(","))
        if ids:
            filters.append(f"concepts.id:{'|'.join(ids)}")

    if country_codes:
        filters.append(f"institutions.country_code:{'|'.join([x.strip() for x in country_codes.split(',')])}")

    if institutions:
        filters.append(f"institutions.id:{'|'.join([x.strip() for x in institutions.split(',')])}")

    if from_date:
        filters.append(f"from_publication_date:{from_date}")
    if to_date:
        filters.append(f"to_publication_date:{to_date}")
        
    return ",".join(filters)

def parse_abstract(indexed_dict):
    if not indexed_dict or not isinstance(indexed_dict, dict):
        return ""
    word_list = sorted([(int(k), v) for v, ks in indexed_dict.items() for k in ks], key=lambda x: x[0])
    return " ".join([word for _, word in word_list])

def safe_get_nested(d, keys, default="N/A"):
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, {})
        else:
            return default
    return d if d else default

def query_openalex(filter_string, max_results=100):
    papers = []
    cursor = "*"
    per_page = 200  # OpenAlex maximum

    while len(papers) < max_results:
        remaining = max_results - len(papers)
        page_size = min(per_page, remaining)

        try:
            r = requests.get(
                f"https://api.openalex.org/works?filter={filter_string}&sort=cited_by_count:desc&per_page={page_size}&cursor={cursor}"
            ).json()

            new_results = [
                {
                    "Title": result["display_name"],
                    "Authors": "; ".join([a["author"]["display_name"] for a in result["authorships"]]),
                    "AuthorTuples": [(a["author"]["display_name"], a["author"]["id"]) for a in result["authorships"]],
                    "Publication Date": result.get("publication_date"),
                    "Citations": result.get("cited_by_count"),
                    "Journal": safe_get_nested(result, ["primary_location", "source", "display_name"]),
                    "Link": result.get("primary_location", {}).get("landing_page_url", "N/A"),
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
                }
                for result in r["results"]
            ]

            papers.extend(new_results)

            cursor = r["meta"].get("next_cursor")
            if not cursor or not new_results:
                break  # no more data

        except:
            break

    return papers[:max_results]
