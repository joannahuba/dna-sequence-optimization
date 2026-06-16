import pandas as pd


def truncate_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Truncates specified sequences in the DataFrame based on iteration thresholds.

    Rows for 'seq_2_broken' are retained only if iteration <= 16.
    Rows for 'seq_3_broken' are retained only if iteration <= 20.

    :param df: The input DataFrame containing sequence and iteration data.
    :type df: pandas.DataFrame
    :return: A filtered DataFrame with truncated sequences.
    :rtype: pandas.DataFrame
    """
    # Configuration of filtering logic
    ## Define retention thresholds for sequences
    thresholds = {
        'seq_2_broken': 16,
        'seq_3_broken': 20
    }

    # Data processing pipeline
    ## Initialize mask for exclusion
    rows_to_drop = pd.Series([False] * len(df))

    ## Calculate rows violating iteration constraints
    for seq_name, limit in thresholds.items():
        ### Apply conditional mask to identify rows exceeding limits
        mask = (df['sequence_name'] == seq_name) & (df['iteration'] > limit)
        rows_to_drop = rows_to_drop | mask

    filtered_df = df[~rows_to_drop].copy()

    return filtered_df

# Execution
## Apply truncation to target DataFrame
df = truncate_sequences(df)