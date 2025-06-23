import streamlit as st
import pandas as pd
from utils import build_filter_string, query_openalex
import datetime

st.set_page_config(page_title="Research Finder", layout="wide")
st.title("🔍 Research Finder")

with st.sidebar:
    run = st.button("Search 🔍")
    
    st.header("Search Filters (Use comma to separate words/phrases)")

    abstract_kw = st.text_input("Must contain:", value='DNA')
    abstract_kw_opt = st.text_input("May contain at least one of:")

    concept_kw = st.text_input("Concepts (exact match):")
    concept_kw_opt = st.text_input("Concepts (any of these):")

    countries = st.text_input("Country codes (e.g. us, es, ca):")
    
    institutions = st.text_input("Institution IDs (OpenAlex format):")

    # Published date range
    default_from_date = datetime.date.today().replace(year=datetime.date.today().year - 3)
    from_date = st.date_input("Published from:", value=default_from_date, format="YYYY-MM-DD")
    to_date = st.date_input("Published until:", format="YYYY-MM-DD")
    
    num_results = st.number_input("How many papers do you want?", min_value=1, max_value=2000, value=100, step=50)
    
    

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
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "AuthorTuples"} for r in results])
        df["Year"] = df["Year"].astype("Int64").astype(str)
        st.subheader(f"📚 {len(df)} results found (sorted by citations)")

        with st.expander("🔬 Show Full Results"):
            st.dataframe(df)

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

               




        if "Year" in df.columns and not df["Year"].isnull().all():
            st.markdown("### 🕒 Publications per year in the results")
            hist = df["Year"].value_counts().sort_index()
            st.bar_chart(hist)
    else:
        st.warning("No results found. Try adjusting your filters.")
