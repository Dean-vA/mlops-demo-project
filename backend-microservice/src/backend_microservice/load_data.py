import logging
import os

import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data_from_uri(uri: str) -> pd.DataFrame:
    """
    Load data from a given URI. The URI can be a local file path or an S3 URL.

    Args:
        uri (str): The URI of the data source.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    logging.info(f"Loading data from {uri}...")
    # Try if the URI is a directory
    if os.path.isdir(uri):
        logging.info("Found the following folders in the directory:", os.listdir(uri))
        # Log directory contents for debugging
        if os.path.exists(uri):
            logger.info(f"Found the following items in {uri}:")
            for root, dirs, files in os.walk(uri):
                logger.info(f"  Root: {root}")
                if dirs:
                    logger.info(f"  Dirs: {dirs}")
                if files:
                    logger.info(f"  Files: {files}")

    if not os.path.exists(uri):
        raise FileNotFoundError(f"File not found: {uri}")

    if not uri.endswith(".csv"):
        raise ValueError(f"Expected CSV file, got: {uri}")

    csv_file = uri  # Assuming uri is a local file path to the CSV

    try:
        # Load the CSV data
        df = pd.read_csv(csv_file)
        logger.info(f"Successfully loaded {len(df)} rows from {csv_file}")
        logger.info(f"Columns: {list(df.columns)}")

        # Validate expected columns
        expected_columns = ["chunk_no", "transcript", "summary"]
        missing_columns = [col for col in expected_columns if col not in df.columns]

        if missing_columns:
            logger.warning(f"Missing expected columns: {missing_columns}")
            logger.info(f"Available columns: {list(df.columns)}")

        # Show sample data
        logger.info("Sample data preview:")
        logger.info(f"\n{df.head(2)}")

        return df

    except Exception as e:
        logger.error(f"Error loading CSV file {csv_file}: {e}")
        raise


def prepare_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame for training by cleaning and standardizing columns.

    Args:
        df (pd.DataFrame): Raw data from CSV

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for training
    """

    logger.info("Preparing training DataFrame...")

    # Create a copy to avoid modifying original
    training_df = df.copy()

    # Clean and standardize data
    training_df["input_text"] = training_df["transcript"].astype(str)
    training_df["target_summary"] = training_df["summary"].astype(str)

    # Remove rows with empty transcripts or summaries
    initial_len = len(training_df)
    training_df = training_df[(training_df["input_text"].str.len() > 10) & (training_df["target_summary"].str.len() > 10)].reset_index(drop=True)

    if len(training_df) < initial_len:
        logger.warning(f"Removed {initial_len - len(training_df)} rows with insufficient content")

        logger.info(f"Prepared DataFrame with {len(training_df)} training examples")

    # Log sample data
    if len(training_df) > 0:
        sample = training_df.iloc[0]
        logger.info("Sample training example:")
        logger.info(f"  Chunk ID: {sample['chunk_no']}")
        logger.info(f"  Input length: {len(sample['input_text'])} chars")
        logger.info(f"  Summary length: {len(sample['target_summary'])} chars")
        logger.info(f"  Input preview: {sample['input_text'][:200]}...")
        logger.info(f"  Summary preview: {sample['target_summary'][:200]}...")

    return training_df
