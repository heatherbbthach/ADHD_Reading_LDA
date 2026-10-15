# Research Data

The original Reddit dataset (6,295 records) analyzed in the published study is not included in this public repository to protect user privacy.

The data-collection script (`src/01_collect_reddit_data.py`) retrieves Reddit posts and comments using PRAW and saves the resulting dataset to `data/reddit_adhd_data.csv`.

The dataset contains six fields: `ID`, `Parent ID`, `Post Title`, `Text`, `Post URL`, and `Type`.

Reddit search results may change over time, so running the data-collection script will not necessarily recreate the dataset used in the published study.

Raw Reddit posts, comments, and identifiers should not be committed to this public repository.
