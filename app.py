import streamlit as st
import pandas as pd
from utils import build_filter_string, query_openalex
import datetime

st.set_page_config(page_title="Research Finder", layout="wide", initial_sidebar_state="expanded")
st.title("🔍 Research Finder")

from st_paywall import add_auth

# Basic usage with defaults
#add_auth()

with st.sidebar:
    run = st.button("Search 🔍")
    
    st.header("Search Filters (Use comma to separate words/phrases)")

    abstract_kw = st.text_input("Keywords:", value='DNA')
    abstract_kw_opt = st.text_input("Keywords (any of these):")

    concept_kw = st.text_input("Concepts (exact match):")
    concept_kw_opt = st.text_input("Concepts (any of these):")

    # Published date range
    min_date = datetime.date(1000, 1, 1)
    max_date = datetime.date(2100, 1, 1)
    default_from_date = datetime.date.today().replace(year=datetime.date.today().year - 10)
    from_date = st.date_input("Published from:", value=default_from_date, min_value=min_date, max_value=max_date, format="YYYY-MM-DD")
    to_date = st.date_input("Published until:", min_value=min_date, max_value=max_date, format="YYYY-MM-DD")
    
    num_results = st.number_input("How many papers do you want?", min_value=1, max_value=2000, value=100, step=50)
    

    countries = st.text_input("Country codes (e.g. us, es, ca):")
    institutions = st.text_input("Institution IDs (OpenAlex format):")
    
    
if run:
    with st.spinner("Querying OpenAlex..."):
        from_str = str(from_date) if from_date else ""
        to_str = str(to_date) if to_date else ""
        
        filter_str = build_filter_string(
            abstract_kw, abstract_kw_opt,
            concept_kw, concept_kw_opt, countries, str(from_date), str(to_date), institutions
        )

        results = query_openalex(filter_str, max_results=num_results)
    
    if results:
        from datetime import datetime

        df = pd.DataFrame([{k: v for k, v in r.items() if k != "AuthorTuples"} for r in results])
        
        # Convert publication date
        df["Publication Date"] = pd.to_datetime(df["Publication Date"], errors="coerce")
        
        # Calculate paper age in fractional years
        today = pd.Timestamp.today()
        df["Paper Age"] = (today - df["Publication Date"]).dt.total_seconds() / (365.25 * 24 * 3600)
        df["Paper Age"] = df["Paper Age"].clip(lower=0.1, upper=5)  # avoid div by zero
        
        # Normalized citations
        df["Normalized Citations"] = (df["Citations"] / df["Paper Age"]).round()
        
        # Optional: convert Year for other plots
        df["Year"] = df["Publication Date"].dt.year.astype("Int64").astype(str)
        
        st.subheader(f"📚 {len(df)} results found (sorted by citations)")

        
        st.dataframe(df[["Title", "Authors", "Year", "Citations", "Normalized Citations", 'Journal', 'Link', 'Work ID']])

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "openalex_results.csv", "text/csv")
        # Simple histogram of publication years
        # Combined author stats: paper count and total citations
                                                                                                                                        
        if "Authors" in df.columns and "Citations" in df.columns:
            st.markdown("### 👥 Top Authors by Papers and Citations")
        
            all_records = []
            for result in results:
                authors = result.get("AuthorTuples", [])
                num_authors = len(authors)
                for name, aid in authors:
                    all_records.append({
                        "Author": name.strip(),
                        "Author ID": aid,
                        "Citations": result["Citations"],
                        "Contribution": 1 / num_authors if num_authors else 1
                    })
            
            authors_df = pd.DataFrame(all_records)
            
            authors_df["OpenAlex URL"] = authors_df["Author ID"].apply(
                lambda x: f"https://openalex.org/{x.split('/')[-1]}" if isinstance(x, str) else "N/A"
            )
            
            author_counts = (
                authors_df["Author"]
                .value_counts()
                .rename_axis("Author")
                .reset_index(name="Paper Count")
            )
            
            author_contrib = (
                authors_df.groupby("Author", as_index=False)["Contribution"]
                .sum()
                .rename(columns={"Contribution": "Normalized Paper Count"})
            )
            
            author_citations = (
                authors_df.groupby("Author", as_index=False)["Citations"]
                .sum()
                .rename(columns={"Citations": "Total Citations"})
            )
            
            author_urls = (
                authors_df[["Author", "OpenAlex URL"]]
                .drop_duplicates(subset="Author")
            )
            
            author_stats = author_counts.merge(author_contrib, on="Author")
            author_stats = author_stats.merge(author_citations, on="Author")
            author_stats = author_stats.merge(author_urls, on="Author", how="left")
            author_stats = author_stats.sort_values(by=["Paper Count", "Normalized Paper Count", "Total Citations"], ascending=False)
            
            st.dataframe(author_stats.head(20).reset_index(drop=True))
            
            author_csv = author_stats.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download full author list as CSV",
                data=author_csv,
                file_name="top_authors.csv",
                mime="text/csv"
            )


        if "InstitutionTuples" in results[0]:
            st.markdown("### 🏛️ Top Institutions by Appearances and Citations")

            inst_records = []
            for result in results:
                institutions = result.get("InstitutionTuples", [])
                num_insts = len(institutions)
                for name, inst_id in institutions:
                    inst_records.append({
                        "Institution": name.strip() if name else "N/A",
                        "Institution ID": inst_id,
                        "Citations": result["Citations"],
                        "Contribution": 1 / num_insts if num_insts else 1
                    })

            inst_df = pd.DataFrame(inst_records)

            inst_df["OpenAlex URL"] = inst_df["Institution ID"].apply(
                lambda x: f"https://openalex.org/{x.split('/')[-1]}" if isinstance(x, str) else "N/A"
            )

            inst_counts = (
                inst_df["Institution"]
                .value_counts()
                .rename_axis("Institution")
                .reset_index(name="Paper Count")
            )

            inst_contrib = (
                inst_df.groupby("Institution", as_index=False)["Contribution"]
                .sum()
                .rename(columns={"Contribution": "Normalized Paper Count"})
            )

            inst_citations = (
                inst_df.groupby("Institution", as_index=False)["Citations"]
                .sum()
                .rename(columns={"Citations": "Total Citations"})
            )

            inst_urls = (
                inst_df[["Institution", "OpenAlex URL"]]
                .drop_duplicates(subset="Institution")
            )

            inst_stats = inst_counts.merge(inst_contrib, on="Institution")
            inst_stats = inst_stats.merge(inst_citations, on="Institution")
            inst_stats = inst_stats.merge(inst_urls, on="Institution", how="left")
            inst_stats = inst_stats.sort_values(by=["Paper Count", "Total Citations"], ascending=False)

            st.dataframe(inst_stats.head(20).reset_index(drop=True))

            inst_csv = inst_stats.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download full institution list as CSV",
                data=inst_csv,
                file_name="top_institutions.csv",
                mime="text/csv"
            )

        if "CountryTuples" in results[0]:
            st.markdown("### 🌍 Top Countries by Paper Count and Citations")

            country_records = []
            for result in results:
                countries = result.get("CountryTuples", [])
                num_countries = len(countries)
                for country in countries:
                    country_records.append({
                        "Country": country,
                        "Citations": result["Citations"],
                        "Contribution": 1 / num_countries if num_countries else 1
                    })

            country_df = pd.DataFrame(country_records)

            country_counts = (
                country_df["Country"]
                .value_counts()
                .rename_axis("Country")
                .reset_index(name="Paper Count")
            )

            country_contrib = (
                country_df.groupby("Country", as_index=False)["Contribution"]
                .sum()
                .rename(columns={"Contribution": "Normalized Paper Count"})
            )

            country_citations = (
                country_df.groupby("Country", as_index=False)["Citations"]
                .sum()
                .rename(columns={"Citations": "Total Citations"})
            )

            country_stats = country_counts.merge(country_contrib, on="Country")
            country_stats = country_stats.merge(country_citations, on="Country")
            country_stats = country_stats.sort_values(by=["Paper Count", "Total Citations"], ascending=False)

            st.dataframe(country_stats.head(20).reset_index(drop=True))

            country_csv = country_stats.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download full country list as CSV",
                data=country_csv,
                file_name="top_countries.csv",
                mime="text/csv"
            )
    
            import pycountry
            
            def iso2_to_iso3(code):
                try:
                    return pycountry.countries.get(alpha_2=code.upper()).alpha_3
                except:
                    return None
            
            country_stats["ISO-3"] = country_stats["Country"].apply(iso2_to_iso3)
            
            import plotly.express as px
            import numpy as np
            country_stats["Log Paper Count"] = country_stats["Paper Count"].replace(0, 1)
            country_stats["Log Paper Count"] = np.log10(country_stats["Log Paper Count"])
            fig = px.choropleth(
                country_stats,
                locations="ISO-3",
                locationmode="ISO-3",  # if ISO-3 codes; use "ISO-2" for ISO alpha-2
                color="Log Paper Count",
                hover_name="Country",
                hover_data={
                    "Paper Count": True,         # show this
                    "Log Paper Count": False,    # hide this
                    "ISO-3": False               # hide this if you don’t want it
                },
                color_continuous_scale="Viridis",
                title="Top Countries by Paper Count"
            )
    
            st.plotly_chart(fig, use_container_width=True)
        
        
        if "Year" in df.columns and not df["Year"].isnull().all():
            st.markdown("### 🕒 Publications per year in the results")
            hist = df["Year"].value_counts().sort_index()
            st.bar_chart(hist)
    else:
        st.warning("No results found. Try adjusting your filters.")
