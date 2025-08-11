import streamlit as st
import pandas as pd
import datetime
import numpy as np
import pycountry
import plotly.express as px
from collections import Counter
from utils import *
import requests

# --- Session defaults ---
for k, v in {
    "paper_results": None,
    "paper_total_found": None,
    "paper_df": None,
    "paper_filter_str": None,
    "researcher_results": None,   # list of dicts (one per searched name)
    "researcher_last_names": "",
    "paper_author_stats": None,   
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Research Finder",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "paper_keywords" not in st.session_state:
    st.session_state.paper_keywords = "myocardial infarction, Heart Failure"
if "paper_keywords_opt" not in st.session_state:
    st.session_state.paper_keywords_opt = "post-myocardial infarction heart failure, post-mi heart failure, Infarction-induced Heart Failure, heart failure after myocardial infarction, heart failure after mi"
if "paper_concept_kw" not in st.session_state:
    st.session_state.paper_concept_kw = ""
if "paper_concept_kw_opt" not in st.session_state:
    st.session_state.paper_concept_kw_opt = ""
if "paper_countries" not in st.session_state:
    st.session_state.paper_countries = ""
if "paper_institutions" not in st.session_state:
    st.session_state.paper_institutions = ""
if "paper_from_date" not in st.session_state:
    default_from_date = datetime.date.today().replace(year=datetime.date.today().year - 10)
    st.session_state.paper_from_date = default_from_date
if "paper_to_date" not in st.session_state:
    st.session_state.paper_to_date = datetime.date.today()


# ===== SIDEBAR NAVIGATION =====
page = st.sidebar.radio("📑 Select page", ["Paper Search", "Researcher Lookup", "Research Trajectory"])

# =========================================================
# 📄 PAPER SEARCH PAGE
# =========================================================
if page == "Paper Search":
    st.title("🔍 Research Finder")

    # --- Sidebar inputs only ---
    with st.sidebar:
        run = st.button("Search 🔍")
        st.header("Search Filters")
    
        abstract_kw = st.text_input(
            "Keywords:",
            key="paper_keywords",  # binds directly to session_state
            help="Use comma to separate words/phrases"
        )
        abstract_kw_opt = st.text_input(
            "Keywords (any of these):",
            key="paper_keywords_opt"
        )
    
        concept_kw = st.text_input(
            "Concepts (exact match):",
            key="paper_concept_kw"
        )
        concept_kw_opt = st.text_input(
            "Concepts (any of these):",
            key="paper_concept_kw_opt"
        )
    
        from_date = st.date_input(
            "Published from:",
            key="paper_from_date",
            min_value=datetime.date(1000, 1, 1),
            max_value=datetime.date(2100, 1, 1),
            format="YYYY-MM-DD"
        )
        to_date = st.date_input(
            "Published until:",
            key="paper_to_date",
            min_value=datetime.date(1000, 1, 1),
            max_value=datetime.date(2100, 1, 1),
            format="YYYY-MM-DD"
        )
    
        num_results = st.number_input(
            "How many papers do you want?",
            min_value=1, max_value=10000, value=1000, step=50
        )
        countries = st.text_input(
            "Country codes (e.g. us, es, ca):",
            key="paper_countries"
        )
        institutions = st.text_input(
            "Institution IDs (OpenAlex format):",
            key="paper_institutions"
        )
    

        if run:
            with st.spinner("Querying OpenAlex..."):
                filter_str = build_filter_string(
                    st.session_state.paper_keywords,
                    st.session_state.paper_keywords_opt,
                    st.session_state.paper_concept_kw,
                    st.session_state.paper_concept_kw_opt,
                    st.session_state.paper_countries,
                    str(st.session_state.paper_from_date),
                    str(st.session_state.paper_to_date),
                    st.session_state.paper_institutions
                )
                results, total_found = query_openalex(filter_str, max_results=num_results)


            # Save to session
            st.session_state.paper_results = results
            st.session_state.paper_total_found = total_found
            st.session_state.paper_filter_str = filter_str
            st.session_state.paper_df = None  # will rebuild below

    # --- Render from session (outside the sidebar) ---
    results = st.session_state.paper_results
    total_found = st.session_state.paper_total_found

    if not results:
        st.info("Run a search to see results.")
        st.stop()

    # Build df (or reuse)
    if st.session_state.paper_df is None:
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "AuthorTuples"} for r in results])
        df["Publication Date"] = pd.to_datetime(df["Publication Date"], errors="coerce")
        today = pd.Timestamp.today()
        df["Paper Age"] = (today - df["Publication Date"]).dt.total_seconds() / (365.25 * 24 * 3600)
        df["Paper Age"] = df["Paper Age"].clip(lower=0.1, upper=5)
        df["Normalized Citations"] = (df["Citations"] / df["Paper Age"]).round()
        df = df[~df["Title"].str.contains("guidelines", case=False, na=False)]
        df["Year"] = df["Publication Date"].dt.year.astype("Int64").astype(str)
        st.session_state.paper_df = df
    else:
        df = st.session_state.paper_df

    st.subheader(f"📚 Showing {len(results)} of {total_found} total papers found — sorted by citations")
    st.dataframe(df[["Title", "Authors", "Year", "Citations", "Normalized Citations", "Journal", "Link", "Work ID"]])
    st.download_button(
        "⬇️ Download Excel",
        to_excel_bytes(df, "results"),
        "openalex_results.xlsx",
        EXCEL_MIME
    )
        
        
    # ----- Top Authors (with affiliation + normalized citations) -----
    if "Authors" in df.columns and "Citations" in df.columns:
        st.markdown("### 👥 Top Authors by Papers and Citations")
    
        author_records = []
        for r in results:
            # author_id -> (name, affiliation)
            aff_map = {}
            for aa in r.get("AuthorAffiliations", []):
                aff_map[aa["author_id"]] = (
                    (aa.get("author_name") or "").strip(),
                    aa.get("affiliation_name") or "N/A"
                )
    
            authors = r.get("AuthorTuples", [])
            num_authors = len(authors) or 1
            citations = r.get("Citations", 0) or 0
    
            for name, aid, orcid in authors:
                disp_name = (name or "").strip()
                nm, aff = aff_map.get(aid, (disp_name, "N/A"))
                author_records.append({
                    "Author": nm,
                    "Author ID": aid,
                    "AuthorID_short": aid.split("/")[-1] if isinstance(aid, str) else None,
                    "ORCID": orcid,
                    "Affiliation": aff,
                    "Contribution": 1 / num_authors,
                    "Citations": citations,
                    "Citations_Fraction": citations / num_authors
                })
    
        authors_df = pd.DataFrame(author_records)
        authors_df["OpenAlex URL"] = authors_df["AuthorID_short"].apply(
            lambda sid: f"https://openalex.org/{sid}" if sid else "N/A"
        )
    
        author_stats = (
            authors_df.groupby("AuthorID_short", as_index=False)
            .agg(
                Author=("Author", "first"),
                Affiliation=("Affiliation", lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A"),
                ORCID=("ORCID", "first"),
                Paper_Count=("Author", "count"),
                Normalized_Paper_Count=("Contribution", "sum"),
                Total_Citations=("Citations", "sum"),
                Normalized_Citations=("Citations_Fraction", "sum"),
                OpenAlex_URL=("OpenAlex URL", "first")
            )
        )
    
        author_stats["Normalized_Citations"] = author_stats["Normalized_Citations"].astype(int)
        author_stats["Impact"] = ((author_stats["Paper_Count"] - 0.8) * author_stats["Normalized_Citations"]).astype(int)
        author_stats = author_stats.sort_values(by=["Impact"], ascending=False).reset_index(drop=True)
    
        # Always keep the latest base table available for the other pages
        st.session_state.paper_author_stats = author_stats.copy()
    
        # --- UI for Breakthrough Potential ---
        base_cols = ["Author", "Affiliation", "Paper_Count", "Normalized_Paper_Count",
                     "Total_Citations", "Normalized_Citations", "Impact", "OpenAlex_URL"]
        extra_cols = ["Recent", "Baseline", "Ratio", "BreakthroughPotential"]
    

        top_k = st.number_input(
            "Top K to evaluate",
            min_value=5,
            max_value=int(min(2000, len(author_stats))),
            value=min(50, len(author_stats)),
            step=10,
            help="Only the top K by Impact will get Breakthrough metrics."
        )
        calc_bpot = st.button("⚡ (Re)Calculate Breakthrough Potential for Top K")
    
        # Pre-create extra columns (NaN) so base display works cleanly
        for c in extra_cols:
            if c not in author_stats.columns:
                author_stats[c] = np.nan
    
        if calc_bpot:
            # Clear caches so we fetch fresh data
            st.cache_data.clear()
    
            @st.cache_data(show_spinner=False)
            def _build_trajectory_for_author(author_id_url: str, max_results: int = 400):
                works = fetch_author_works(author_id_url, max_results=max_results)
                rows = []
                today = pd.Timestamp.today()
                for w in works:
                    pub_date = pd.to_datetime(w.get("publication_date", None), errors="coerce")
                    if pd.isna(pub_date):
                        continue
                    year = int(pub_date.year)
                    cits = w.get("cited_by_count", 0) or 0
                    age_years = max((today - pub_date).days / 365.25, 0.1)
                    norm = cits / age_years
                    rows.append({"Year": year, "Citations": cits, "NormalizedCitations": norm})
                if not rows:
                    return pd.DataFrame(columns=["Year", "Publications", "Citations", "NormalizedCitations"])
                dfw = pd.DataFrame(rows)
                out = (dfw.groupby("Year", as_index=False)
                         .agg(
                             Publications=("Citations", "size"),
                             Citations=("Citations", "sum"),
                             NormalizedCitations=("NormalizedCitations", "sum"),
                         )
                       .sort_values("Year"))
                yr_min, yr_max = int(out["Year"].min()), int(out["Year"].max())
                out = (out.set_index("Year")
                         .reindex(range(yr_min, yr_max + 1))
                         .fillna(0.0)
                         .reset_index()
                         .rename(columns={"index": "Year"}))
                return out
    
            def _trim_early_zero_run(df_):
                if df_.empty or "Publications" not in df_.columns:
                    return df_
                out = df_.reset_index(drop=True).copy()
                pubs = out["Publications"].astype(int).tolist()
                zero_run, last_run_end_idx = 0, None
                for i, v in enumerate(pubs):
                    if v == 0:
                        zero_run += 1
                        if zero_run >= 5:
                            last_run_end_idx = i
                    else:
                        zero_run = 0
                if last_run_end_idx is not None and any(x > 0 for x in pubs[last_run_end_idx + 1:]):
                    start_year = int(out.loc[last_run_end_idx, "Year"]) + 1
                    out = out[out["Year"] >= start_year].reset_index(drop=True)
                return out
    
            def _smooth_5y_centered(df_):
                sm = df_.copy()
                for c in ["Publications", "Citations", "NormalizedCitations"]:
                    sm[f"{c}_Smoothed"] = sm[c].rolling(5, min_periods=1).mean()
                return sm
    
            def _compute_ratio_metric(df_, col="NormalizedCitations_Smoothed"):
                if df_.empty or col not in df_.columns:
                    return (np.nan, np.nan, np.nan, None)
                d = df_.dropna(subset=["Year", col]).sort_values("Year")
                if d.empty:
                    return (np.nan, np.nan, np.nan, None)
                y_max = int(d["Year"].max())
                last3 = d[d["Year"].between(y_max - 2, y_max)][col].sum()
                base_win = d[d["Year"].between(y_max - 13, y_max - 10)]
                base3 = base_win[col].sum() if not base_win.empty else d.head(3)[col].sum()
                if base3 == 0:
                    return (1.0, last3, base3, y_max)
                ratio = last3 / base3
                return (ratio, last3, base3, y_max)
    
            # Work only on the top K
            topN = author_stats.head(int(top_k)).copy()
            st.markdown(f"### ⚡ Computing breakthrough potential for Top {len(topN)} authors")
            prog = st.progress(0.0)
    
            # Fill only the top K rows with computed values; others remain NaN
            rows_update = []
            total = len(topN)
            for n_done, (i, row) in enumerate(topN.iterrows(), start=1):
                aid_short = row["AuthorID_short"]
                a_url = row.get("OpenAlex_URL") or f"https://openalex.org/{aid_short}"
                try:
                    t = _build_trajectory_for_author(a_url, max_results=2000)
                    if t.empty:
                        ratio, last3, base3 = (np.nan, np.nan, np.nan)
                        bpot = np.nan
                    else:
                        t = _trim_early_zero_run(t)
                        t_sm = _smooth_5y_centered(t)
                        ratio, last3, base3, _ = _compute_ratio_metric(t_sm)
                        bpot_raw = last3 * (ratio ** 0.5) if pd.notna(ratio) and pd.notna(last3) else np.nan
                        bpot = np.log(bpot_raw) if pd.notna(bpot_raw) and bpot_raw > 0 else np.nan
                except Exception:
                    ratio, last3, base3, bpot = (np.nan, np.nan, np.nan, np.nan)
    
                author_stats.loc[i, "Recent"] = last3
                author_stats.loc[i, "Baseline"] = base3
                author_stats.loc[i, "Ratio"] = ratio
                author_stats.loc[i, "BreakthroughPotential"] = bpot
    
                prog.progress(n_done / max(total, 1))
    
            # Save enhanced table for other pages to use
            st.session_state.paper_author_stats = author_stats.copy()
    
        # ---- Display (base only if not computed; base+extra if computed) ----
        has_bpot = author_stats["BreakthroughPotential"].notna().any() if "BreakthroughPotential" in author_stats.columns else False
        cols_to_show = base_cols + (extra_cols if has_bpot else [])
        st.dataframe(author_stats[cols_to_show].head(250).reset_index(drop=True))
    
        st.download_button(
            f"⬇️ Download full author list (Excel) ({len(author_stats)} Authors)",
            to_excel_bytes(author_stats[cols_to_show], "authors"),
            "top_authors.xlsx",
            EXCEL_MIME
        )
    
            


    # ----- Top Institutions (unique per paper, with normalized citations) -----
    if "InstitutionTuples" in results[0]:
        st.markdown("### 🏛️ Top Institutions by Appearances and Citations")

        inst_records = []
        for r in results:
            raw = r.get("InstitutionTuples", [])
            uniq_insts = {((name.strip() if name else "N/A"), inst_id) for name, inst_id in raw if inst_id}
            k = len(uniq_insts) if uniq_insts else 1
            for name, inst_id in (uniq_insts or {("N/A", None)}):
                inst_records.append({
                    "Institution": name,
                    "Institution ID": inst_id,
                    "Citations": r["Citations"],
                    "Contribution": 1 / k,
                    "Citations_Fraction": (r["Citations"] or 0) / k
                })

        inst_df = pd.DataFrame(inst_records)
        inst_df["InstitutionID_short"] = inst_df["Institution ID"].apply(
            lambda x: x.split("/")[-1] if isinstance(x, str) else None
        )
        inst_df["OpenAlex URL"] = inst_df["InstitutionID_short"].apply(
            lambda sid: f"https://openalex.org/{sid}" if sid else "N/A"
        )

        key = inst_df["InstitutionID_short"].fillna(inst_df["Institution"])
        inst_df["_Key"] = key

        name_mode = (inst_df.groupby(["_Key", "Institution"])
                            .size()
                            .reset_index(name="n")
                            .sort_values(["_Key", "n"], ascending=[True, False])
                            .drop_duplicates("_Key")
                            .set_index("_Key")["Institution"])

        inst_stats = (
            inst_df.groupby("_Key", as_index=False)
            .agg(
                Paper_Count=("Contribution", "size"),
                Normalized_Paper_Count=("Contribution", "sum"),
                Total_Citations=("Citations", "sum"),
                Normalized_Citations=("Citations_Fraction", "sum"),
                InstitutionID_short=("InstitutionID_short", "first"),
                OpenAlex_URL=("OpenAlex URL", "first"),
            )
        )
        inst_stats["Institution"] = inst_stats["_Key"].map(name_mode)
        # ✅ Impact (same formula as authors)
        inst_stats = inst_stats[inst_stats["Institution"]!='N/A']
        inst_stats["Normalized_Citations"] = inst_stats["Normalized_Citations"].astype(int)
        inst_stats["Impact"] = ((inst_stats["Normalized_Paper_Count"] - 0.0) * inst_stats["Normalized_Citations"]).astype(int)
        inst_stats = inst_stats.sort_values(by=["Impact"], ascending=False)
        
        # show + download
        show_cols_inst = ["Institution", "Paper_Count", "Normalized_Paper_Count",
                          "Total_Citations", "Normalized_Citations", "Impact", "OpenAlex_URL"]
        st.dataframe(inst_stats[show_cols_inst].head(20).reset_index(drop=True))
        st.download_button(
            f"⬇️ Download full institution list (Excel) ({len(inst_stats)} Institutions)",
            to_excel_bytes(inst_stats, "institutions"),
            "top_institutions.xlsx",
            EXCEL_MIME
        )


    # ----- Top Countries (unique per paper, with normalized citations) -----
    if "CountryTuples" in results[0]:
        st.markdown("### 🌍 Top Countries by Paper Count and Citations")

        country_records = []
        for r in results:
            raw = r.get("CountryTuples", [])
            uniq_countries = sorted({c for c in raw if c})
            k = len(uniq_countries) if uniq_countries else 1
            for c in (uniq_countries or ["N/A"]):
                country_records.append({
                    "Country": c,
                    "Citations": r["Citations"],
                    "Contribution": 1 / k,
                    "Citations_Fraction": (r["Citations"] or 0) / k
                })

        country_df = pd.DataFrame(country_records)
        country_stats = (
            country_df.groupby("Country", as_index=False)
            .agg(
                Paper_Count=("Contribution", "size"),
                Normalized_Paper_Count=("Contribution", "sum"),
                Total_Citations=("Citations", "sum"),
                Normalized_Citations=("Citations_Fraction", "sum")
            )
        )

        def iso2_to_iso3(code):
            try:
                return pycountry.countries.get(alpha_2=code.upper()).alpha_3
            except Exception:
                return None

        country_stats["ISO-3"] = country_stats["Country"].apply(iso2_to_iso3)
        # ✅ Impact (same formula as authors)
        country_stats["Normalized_Citations"] = country_stats["Normalized_Citations"].astype(int)
        country_stats["Impact"] = ((country_stats["Normalized_Paper_Count"] - 0.0) * country_stats["Normalized_Citations"]).astype(int)
        country_stats = country_stats.sort_values(by=["Impact"], ascending=False)
        
        # show + download
        show_cols_ctry = ["Country", "Paper_Count", "Normalized_Paper_Count",
                          "Total_Citations", "Normalized_Citations", "Impact", "ISO-3"]
        st.dataframe(country_stats[show_cols_ctry].head(20).reset_index(drop=True))

        st.download_button(
            "⬇️ Download full country list (Excel)",
            to_excel_bytes(country_stats, "countries"),
            "top_countries.xlsx",
            EXCEL_MIME
        )


        country_stats["Log Paper Count"] = np.log10(country_stats["Paper_Count"].replace(0, 1))
        fig = px.choropleth(
            country_stats,
            locations="ISO-3",
            color="Log Paper Count",
            hover_name="Country",
            hover_data={
                "Paper_Count": True,
                "Normalized_Paper_Count": True,
                "Total_Citations": True,
                "Normalized_Citations": True,
                "Log Paper Count": False,
                "ISO-3": False
            },
            color_continuous_scale="Viridis",
            title="Top Countries by Paper Count (unique per paper)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ----- Publications per Year -----
    if "Year" in df.columns and not df["Year"].isnull().all():
        st.markdown("### 🕒 Publications per year in the results")
        st.bar_chart(df["Year"].value_counts().sort_index())

    # ----- Top Concepts Over Time -----
    st.markdown("### 📈 Top 10 Most Frequent Concepts Over Time")
    concept_records = []
    for r in results:
        pub_year = pd.to_datetime(r.get("Publication Date", ""), errors="coerce").year
        for concept in r.get("concepts", []):
            if concept.get("display_name") and not pd.isna(pub_year):
                concept_records.append({"Year": pub_year, "Concept": concept["display_name"]})

    if concept_records:
        concept_df = pd.DataFrame(concept_records)
        total_per_year = df["Year"].value_counts().rename_axis("Year").reset_index(name="TotalPapers")
        total_per_year["Year"] = total_per_year["Year"].astype(int)

        concept_year_counts = (
            concept_df.groupby(["Year", "Concept"])
            .size()
            .reset_index(name="ConceptCount")
            .merge(total_per_year, on="Year", how="left")
        )
        concept_year_counts["Share"] = concept_year_counts["ConceptCount"] / concept_year_counts["TotalPapers"]

        top_concepts = concept_df["Concept"].value_counts().nlargest(10).index
        trend_df = (concept_year_counts[concept_year_counts["Concept"].isin(top_concepts)]
                    .pivot(index="Year", columns="Concept", values="Share")
                    .fillna(0))
        st.line_chart(trend_df)
    else:
        st.info("No concept data available to plot trends.")
  
    
  
    
  
    
  
    
  
    
# =========================================================
# 👤 RESEARCHER LOOKUP PAGE (with Top Author rank tracking)
# =========================================================
if page == "Researcher Lookup":
    import re

    st.title("🔍 Researcher Lookup")

    # ---------- helpers ----------
    @st.cache_data(show_spinner=False)
    def fetch_author_works(author_id_url: str, max_results: int = 2000):
        works, cursor, per_page = [], "*", 200
        base = f"https://api.openalex.org/works?filter=authorships.author.id:{author_id_url}&per_page={per_page}"
        while len(works) < max_results and cursor:
            r = requests.get(f"{base}&cursor={cursor}").json()
            works.extend(r.get("results", []))
            cursor = r.get("meta", {}).get("next_cursor")
            if not r.get("results"):
                break
        return works[:max_results]

    def _mode_or_first(s: pd.Series):
        try:
            m = s.mode()
            return m.iloc[0] if not m.empty else (s.dropna().iloc[0] if s.dropna().size else None)
        except Exception:
            return s.dropna().iloc[0] if s.dropna().size else None

    def _uniq_join(series: pd.Series, sep="; "):
        vals = []
        for x in series.dropna().astype(str):
            vals.extend([t.strip() for t in x.split(";") if t.strip()])
        return sep.join(sorted(set(vals))) if vals else ""

    # ---------- inputs ----------
    names_input = st.text_area(
        "Enter researcher names, IDs, or OpenAlex URLs (comma separated):",
        value=st.session_state.get("researcher_last_names", "") or ""
    )
    num_results_per_author = st.number_input(
        "Max works per author (cap per author):",
        min_value=50, max_value=5000, value=1000, step=50
    )

    clicked = st.button("Find Co-authors", key="researcher_search_btn")

    if clicked:
        st.session_state.researcher_last_names = names_input

        # Resolve authors
        raw_items = [n.strip() for n in names_input.split(",") if n.strip()]
        input_authors = []
        for raw in raw_items:
            m = re.search(r'(?:openalex\.org/)?(A\d+)', raw, flags=re.IGNORECASE)
            if m:
                aid_short = m.group(1)
                try:
                    a = requests.get(f"https://api.openalex.org/authors/{aid_short}").json()
                    if a and a.get("id"):
                        input_authors.append({
                            "name": raw,
                            "display_name": a.get("display_name"),
                            "author_id_url": a.get("id"),
                            "author_id_short": (a.get("id") or "").split("/")[-1],
                        })
                    else:
                        input_authors.append({"name": raw, "error": f"Author not found for ID {aid_short}"})
                except Exception as e:
                    input_authors.append({"name": raw, "error": f"Error fetching ID {aid_short}: {e}"})
            else:
                try:
                    resp = requests.get(f"https://api.openalex.org/authors?search={raw}&per_page=1").json()
                    if resp.get("results"):
                        a = resp["results"][0]
                        input_authors.append({
                            "name": raw,
                            "display_name": a.get("display_name"),
                            "author_id_url": a.get("id"),
                            "author_id_short": (a.get("id") or "").split("/")[-1],
                        })
                    else:
                        input_authors.append({"name": raw, "error": "No author found"})
                except Exception as e:
                    input_authors.append({"name": raw, "error": str(e)})

        # Top Authors table from Paper Search
        top_df = st.session_state.get("paper_author_stats")
        if top_df is None or top_df.empty:
            st.warning("Top Authors table not found. Run a Paper Search first.")
        else:
            # Ensure rank info exists
            top_df = top_df.copy()
            if "AuthorID_short" not in top_df.columns or top_df["AuthorID_short"].isna().all():
                if "OpenAlex_URL" in top_df.columns:
                    top_df["AuthorID_short"] = top_df["OpenAlex_URL"].str.rstrip("/").str.split("/").str[-1]
            top_df = top_df.dropna(subset=["AuthorID_short"])
            top_df["TopAuthor_Rank"] = range(1, len(top_df) + 1)  # rank from Paper Search order
            top_index = top_df.set_index("AuthorID_short")

            # Find co-authors
            per_author_matches = []
            for a in input_authors:
                a_url = a.get("author_id_url")
                a_name = a.get("display_name") or a.get("name")
                if not a_url:
                    per_author_matches.append({
                        "input_name": a_name,
                        "matches_df": pd.DataFrame(),
                        "error": a.get("error", "No author ID")
                    })
                    continue

                works = fetch_author_works(a_url, max_results=num_results_per_author)

                co_map = {}
                for w in works:
                    auths = w.get("authorships") or []
                    if not any((au.get("author", {}) or {}).get("id") == a_url for au in auths):
                        continue
                    wid = w.get("id")
                    title = w.get("display_name", "")
                    for au in auths:
                        co_id = (au.get("author", {}) or {}).get("id")
                        if not co_id or co_id == a_url:
                            continue
                        entry = co_map.setdefault(co_id, {"work_ids": set(), "titles": set()})
                        if wid:
                            entry["work_ids"].add(wid)
                        if title:
                            entry["titles"].add(title)

                rows = []
                for co_id_url, entry in co_map.items():
                    co_short = co_id_url.split("/")[-1]
                    if co_short not in top_index.index:
                        continue
                    matched = top_index.loc[[co_short]].copy()
                    keep_cols = [c for c in [
                        "Author", "Affiliation", "Paper_Count", "Normalized_Paper_Count",
                        "Total_Citations", "Normalized_Citations", "Impact", "OpenAlex_URL", "TopAuthor_Rank", "BreakthroughPotential" 
                    ] if c in matched.columns]
                    rec = matched.reset_index()[["AuthorID_short"] + keep_cols].iloc[0].to_dict()
                    rec.update({
                        "Common_Papers_Count": len(entry["work_ids"]),
                        "Common_Paper_Titles": "; ".join(sorted(entry["titles"]))
                    })
                    rows.append(rec)

                matches_df = pd.DataFrame(rows)
                if not matches_df.empty:
                    matches_df = matches_df.sort_values(by=["Impact", "Common_Papers_Count"], ascending=False)
                per_author_matches.append({
                    "input_name": a_name,
                    "matches_df": matches_df,
                    "error": None
                })

            # Merge into combined table
            combined_rows = []
            for item in per_author_matches:
                if not item["matches_df"].empty:
                    tmp = item["matches_df"].copy()
                    tmp["Matched_With"] = item["input_name"]
                    combined_rows.append(tmp)
            combined_df_long = pd.concat(combined_rows, ignore_index=True) if combined_rows else pd.DataFrame()

            if not combined_df_long.empty:
                agg_map = {
                    "Author": _mode_or_first,
                    "Affiliation": _mode_or_first,
                    "Impact": "max",
                    "Paper_Count": "max",
                    "Normalized_Paper_Count": "max",
                    "Total_Citations": "max",
                    "Normalized_Citations": "max",
                    "Common_Papers_Count": "sum",
                    "Common_Paper_Titles": _uniq_join,
                    "Matched_With": lambda s: "; ".join(sorted(set(s))),
                    "OpenAlex_URL": "first",
                    "TopAuthor_Rank": "min", # lowest rank number (best position)
                    "BreakthroughPotential": "max",   # <- add this
                }
                combined_df = (
                    combined_df_long.groupby("AuthorID_short", as_index=False)
                    .agg(agg_map)
                    .sort_values(["Impact", "Common_Papers_Count"], ascending=False)
                )
            else:
                combined_df = pd.DataFrame()

            # Top Authors not co-authors
            matched_ids = set(combined_df["AuthorID_short"]) if not combined_df.empty else set()
            gap_df = top_df[~top_df["AuthorID_short"].isin(matched_ids)].copy()

    # ---------- Render ----------
    if "combined_df" in locals() and not combined_df.empty:
        st.subheader("All matches combined (with Top Author Rank)")
        st.dataframe(combined_df[["TopAuthor_Rank", "Author", "Matched_With","Impact", "BreakthroughPotential", "Common_Papers_Count", "Common_Paper_Titles"]])
    if "gap_df" in locals() and not gap_df.empty:
        st.subheader("Top Authors not among co-authors (with Rank)")
        st.dataframe(gap_df[["TopAuthor_Rank", "Author", "Impact", "BreakthroughPotential"]])
  
    
    
  
    
  # =========================================================
# 📈 RESEARCH TRAJECTORY PAGE (fixed 5y smoothing, unified normalization)
# =========================================================
if page == "Research Trajectory":
    import re
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go

    st.title("📈 Research Trajectory")

    # ---------- Single source of truth for the pipeline ----------
    TRAJ_VERSION = "v3-normexp0.5-trim5y"  # bump to invalidate cache after logic changes

    def resolve_author(input_text: str):
        """
        Accepts: name, A123..., or full OpenAlex URL.
        Returns: dict with display_name, author_id_url, author_id_short OR dict with 'error'.
        """
        input_text = (input_text or "").strip()
        if not input_text:
            return {"error": "Please enter a researcher name or OpenAlex Author ID/URL."}

        m = re.search(r'(?:openalex\.org/)?(A\d+)', input_text, flags=re.IGNORECASE)
        try:
            if m:
                aid_short = m.group(1)
                a = requests.get(f"https://api.openalex.org/authors/{aid_short}").json()
                if a and a.get("id"):
                    return {
                        "display_name": a.get("display_name"),
                        "author_id_url": a.get("id"),
                        "author_id_short": (a.get("id") or "").split("/")[-1],
                    }
                return {"error": f"Author not found for ID {aid_short}"}
            # name search
            resp = requests.get(f"https://api.openalex.org/authors?search={input_text}&per_page=1").json()
            if resp.get("results"):
                a = resp["results"][0]
                return {
                    "display_name": a.get("display_name"),
                    "author_id_url": a.get("id"),
                    "author_id_short": (a.get("id") or "").split("/")[-1],
                }
            return {"error": f"No author found for name '{input_text}'"}
        except Exception as e:
            return {"error": f"Error resolving author: {e}"}

    @st.cache_data(show_spinner=False)
    def build_trajectory(author_id_url: str, max_results: int = 2000, norm_exp: float = 0.5, _v: str = TRAJ_VERSION):
        """
        Pull works for the author and compute per-year:
          - Publications (count)
          - Citations (sum)
          - NormalizedCitations (sum of citations / age_years**norm_exp), with age floor 0.1 yrs
        Returns dataframe with continuous Year range.
        """
        works = fetch_author_works(author_id_url, max_results=max_results)

        rows = []
        today = pd.Timestamp.today()
        for w in works:
            pub_date = pd.to_datetime(w.get("publication_date", None), errors="coerce")
            if pd.isna(pub_date):
                continue
            year = int(pub_date.year)
            cits = w.get("cited_by_count", 0) or 0
            age_years = max((today - pub_date).days / 365.25, 0.1)
            norm = cits / (age_years)
            rows.append({"Year": year, "Citations": cits, "NormalizedCitations": norm})

        if not rows:
            return pd.DataFrame(columns=["Year", "Publications", "Citations", "NormalizedCitations"])

        df = pd.DataFrame(rows)
        out = (df.groupby("Year", as_index=False)
                 .agg(
                     Publications=("Citations", "size"),
                     Citations=("Citations", "sum"),
                     NormalizedCitations=("NormalizedCitations", "sum"),
                 )
               .sort_values("Year"))

        # continuous range
        yr_min, yr_max = int(out["Year"].min()), int(out["Year"].max())
        out = (out.set_index("Year")
                 .reindex(range(yr_min, yr_max + 1))
                 .fillna(0.0)
                 .reset_index()
                 .rename(columns={"index": "Year"}))
        return out

    def trim_early_zero_run(df: pd.DataFrame) -> pd.DataFrame:
        """If there is a run of ≥5 consecutive zero-publication years and nonzero after, drop years before that run."""
        if df.empty or "Publications" not in df.columns:
            return df
        out = df.reset_index(drop=True).copy()
        pubs = out["Publications"].astype(int).tolist()
        zero_run, last_run_end_idx = 0, None
        for i, v in enumerate(pubs):
            if v == 0:
                zero_run += 1
                if zero_run >= 5:
                    last_run_end_idx = i
            else:
                zero_run = 0
        if last_run_end_idx is not None and any(x > 0 for x in pubs[last_run_end_idx + 1:]):
            start_year = int(out.loc[last_run_end_idx, "Year"]) + 1
            out = out[out["Year"] >= start_year].reset_index(drop=True)
        return out

    def smooth_5y_centered(df: pd.DataFrame) -> pd.DataFrame:
        sm = df.copy()
        for c in ["Publications", "Citations", "NormalizedCitations"]:
            sm[f"{c}_Smoothed"] = sm[c].rolling(5, min_periods=1).mean()
        return sm

    def compute_ratio_metric(df: pd.DataFrame, col="NormalizedCitations_Smoothed"):
        """
        Ratio = SUM(last 3 years) / SUM(years 10–13 before the most recent year).
        If 10–13 window is empty, use the FIRST 3 years of the series as baseline.
        Returns (ratio, last3_sum, baseline_sum, max_year).
        """
        if df.empty or col not in df.columns:
            return (np.nan, np.nan, np.nan, None)

        d = df.dropna(subset=["Year", col]).sort_values("Year")
        if d.empty:
            return (np.nan, np.nan, np.nan, None)

        y_max = int(d["Year"].max())
        last3 = d[d["Year"].between(y_max - 2, y_max)][col].sum()

        base_win = d[d["Year"].between(y_max - 13, y_max - 10)]
        if base_win.empty:
            base3 = d.head(3)[col].sum()
        else:
            base3 = base_win[col].sum()

        # FIX: if baseline is zero, force ratio = 1
        if base3 == 0:
            return (1.0, last3, base3, y_max)

        ratio = last3 / base3
        return (ratio, last3, base3, y_max)


    # ---------- UI ----------
    col_in, col_btn = st.columns([4, 1])
    with col_in:
        person_input = st.text_input(
            "Name or OpenAlex Author ID/URL:",
            help="Examples: 'Jane Doe', 'A1969205036', or 'https://openalex.org/A1969205036'"
        )
    with col_btn:
        run_traj = st.button("Build trajectory")

    max_works = st.number_input(
        "Max works to fetch:",
        min_value=100, max_value=5000, value=2000, step=100
    )

    if run_traj:
        with st.spinner("Resolving author and computing trajectory..."):
            author = resolve_author(person_input)
            if author.get("error"):
                st.error(author["error"])
                st.stop()

            traj = build_trajectory(author["author_id_url"], max_results=max_works)
            traj = trim_early_zero_run(traj)
            smoothed = smooth_5y_centered(traj)

        st.subheader(f"Author: {author['display_name']} ({author['author_id_short']})")
        st.markdown(f"[OpenAlex Profile]({author['author_id_url']})")

        if smoothed.empty:
            st.info("No dated works found for this author.")
            st.stop()

        # ---------- Table + Download ----------
        st.dataframe(traj)
        st.download_button(
            "⬇️ Download trajectory (Excel)",
            to_excel_bytes(traj, "trajectory"),
            f"{author['author_id_short']}_trajectory.xlsx",
            EXCEL_MIME
        )

        # ---------- Charts (bars/points raw + smoothed line) ----------
        st.markdown("### 🧾 Publications per Year")
        fig_pub = px.line(smoothed, x="Year", y="Publications_Smoothed", markers=True,
                          title="Publications per Year (5-year centered rolling)")
        fig_pub.add_bar(x=traj["Year"], y=traj["Publications"], name="Raw (bars)", opacity=0.4)
        st.plotly_chart(fig_pub, use_container_width=True)

        st.markdown("### 🔗 Citations per Year")
        fig_cit = px.line(smoothed, x="Year", y="Citations_Smoothed", markers=True,
                          title="Citations per Year (5-year centered rolling)")
        fig_cit.add_scatter(x=traj["Year"], y=traj["Citations"], mode="markers",
                            name="Raw (points)", opacity=0.5)
        st.plotly_chart(fig_cit, use_container_width=True)

        st.markdown("### ⚖️ Normalized Citations per Year")
        fig_norm = px.line(smoothed, x="Year", y="NormalizedCitations_Smoothed", markers=True,
                           title="Normalized Citations per Year (5-year centered rolling)")
        fig_norm.add_scatter(x=traj["Year"], y=traj["NormalizedCitations"], mode="markers",
                             name="Raw (points)", opacity=0.5)
        st.plotly_chart(fig_norm, use_container_width=True)

        # ---------- Ratio metric for the queried author ----------
        # FIX: use the already-computed 'smoothed' df for the queried author
        q_ratio, q_last3, q_base3, _ = compute_ratio_metric(smoothed, col="NormalizedCitations_Smoothed")
        q_breakthrough = q_last3 * (q_ratio ** 0.5) if pd.notna(q_ratio) and pd.notna(q_last3) else np.nan
        q_breakthrough = np.log(q_breakthrough)
        st.markdown(
            f"**Lab Ratio:** {q_ratio:.3g}  "
            f"(Recent output: {q_last3:.2f}, Baseline output: {q_base3:.2f})"
        )
        st.markdown(
            f"**Breakthrough potential:** {q_breakthrough:.2f}"
           
        )


