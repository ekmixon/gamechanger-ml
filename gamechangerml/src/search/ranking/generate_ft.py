# from gamechangerml.src.search.ranking import matamo as mt
from gamechangerml.src.search.ranking import search_data as meta
from gamechangerml.src.search.ranking import rank
from gamechangerml import REPO_PATH
import datetime
import pandas as pd
from tqdm import tqdm
import argparse
import logging
import os

"""
Usage:
    example:
    python -m gamechangerml.src.search.ranking.generate_ft -c test/small_corpus/ -dd 80 --prod gamechangerml/src/search/ranking/generated_files/prod_test_data.csv

optional arguements:
    --corpus, -c Corpus directory
    --days -dd days since today to get data
"""


logger = logging.getLogger("gamechanger")

corpus_dir = "test/corpus_new"
prod_data_file = os.path.join(
    REPO_PATH,
    "gamechangerml/src/search/ranking/generated_files/prod_test_data.csv"
)


def generate_pop_docs(pop_kw_df: pd.DataFrame, corpus_df: pd.DataFrame) -> pd.DataFrame:
    """ generate popular documents based on keyword searches
        Args:
            pop_kw_df: dataframe of keywords and counts
            corpus_df: dataframe of corpus unique ID with text
        Returns:
            dataframe

    """

    docList = []
    for row_kw in tqdm(pop_kw_df.itertuples()):
        for row_corp in corpus_df.itertuples():
            if len(row_corp.keywords):
                if row_kw.keywords in row_corp.keywords[0]:
                    docList.append(
                        {"id": row_corp.id, "keywords": row_kw.keywords})
    docDf = pd.DataFrame(docList, columns=["id", "keywords"])
    docCounts = docDf.groupby("id").count().sort_values(
        "keywords", ascending=False)
    docCounts.rename(columns={"keywords": "kw_in_doc_score"}, inplace=True)

    return docCounts


def generate_ft_doc(corpus_dir: str, days: int = 80, prod_data: str = prod_data_file):
    """ generate feature document
        Args:
            corpus_dir: corpus directory
            days: how many days to retrieve data
        Returns:

    """
    today = datetime.datetime.now()
    out_dir = os.path.join(
        REPO_PATH,
        "gamechangerml/src/search/ranking/generated_files"
    )
    r = rank.Rank()
    day_delta = 80
    d = datetime.timedelta(days=day_delta)
    from_date = today - d

    # TELEMETRY
    # tele_df = mt.get_telemetry(day_delta)
    # kw_doc_pairs = mt.parse_onDocOpen(tele_df)
    # kw_doc_df = pd.DataFrame(kw_doc_pairs)

    # SEARCH LOGS
    # resp = meta.get_searchLogs(str(from_date.date()))

    # until we get connection to prod data
    resp = pd.read_csv(prod_data)
    popular_keywords = meta.get_top_keywords(resp)
    meta_df = meta.scored_logs(resp)

    # CORPUS
    corp_df = r._getCorpusData(corpus_dir)
    pr_df = r.get_pr_docs(corpus_dir)
    corp_df = corp_df.merge(pr_df)

    docCounts = generate_pop_docs(popular_keywords, corp_df)
    corp_df = corp_df.merge(docCounts, on="id", how="outer")

    corp_df["kw_in_doc_score"] = corp_df["kw_in_doc_score"].fillna(0.00001)
    corp_df["kw_in_doc_score"] = (corp_df['kw_in_doc_score'] - corp_df['kw_in_doc_score'].min()) / (
        corp_df['kw_in_doc_score'].max() - corp_df['kw_in_doc_score'].min())
    corp_df.kw_in_doc_score.loc[corp_df.kw_in_doc_score == 0] = 0.00001
    corp_df.to_csv(os.path.join(out_dir, "corpus_meta.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Features CSV")
    parser.add_argument(
        "--corpus", "-c", dest="corpus_dir", help="corpus directory, full path"
    )
    parser.add_argument(
        "--days", "-dd", dest="day_delta", default=80, help="days of data to grab since todays date"
    )
    # Until we can pull data from postgres from production automatically (currently avail in dev)
    parser.add_argument(
        "--prod",
        "-p",
        dest="prod_data",
        default=os.path.join(
            REPO_PATH,
            "gamechangerml/src/search/ranking/generated_files/prod_test_data.csv"
        ),
        help="production data historical search logs csv ",
    )

    # parser.add_argument(
    #    "--outfile", "-o", dest="outfile", help="generated outfile name"
    # )

    args = parser.parse_args()
    corpus_dir = args.corpus_dir
    days = args.day_delta
    prod_data = args.prod_data
    # outfilename = args.outfile
    generate_ft_doc(corpus_dir=corpus_dir, days=days, prod_data=prod_data)
