import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split


def read_transcript_file(filepath):
    """Read transcript file and extract clean text"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Remove timestamp patterns like [14.4s - 15.4s]
        content = re.sub(r"\[\d+\.?\d*s?\s*-\s*\d+\.?\d*s?\]", "", content)

        # Remove session metadata lines
        content = re.sub(r"D&D Session Chunk \d+.*?minutes.*?\n", "", content)
        content = re.sub(r"Duration:.*?\n", "", content)
        content = re.sub(r"Files:.*?\n", "", content)
        content = re.sub(r"Segments:\s*\d+.*?\n", "", content)
        content = re.sub(r"={50,}.*?\n", "", content)

        # Clean up extra whitespace
        content = re.sub(r"\n\s*\n", "\n", content)
        content = content.strip()

        return content
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""


def read_summary_file(filepath):
    """Read summary file and extract clean text"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Remove metadata lines
        lines = content.split("\n")
        summary_lines = []
        skip_metadata = True

        for line in lines:
            # Skip metadata section until we hit the separator
            if "=" in line and len(line) > 20:
                skip_metadata = False
                continue
            if skip_metadata:
                continue

            # Skip empty lines at the beginning
            if line.strip():
                summary_lines.append(line.strip())

        return " ".join(summary_lines)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""


def create_training_data(data_dir):
    """Create training data from transcript and summary files"""

    # Get paths
    transcript_dir = os.path.join(data_dir, "combined_transcripts_20min")
    summary_dir = os.path.join(data_dir, "summaries")

    if not os.path.exists(transcript_dir):
        print(f"Transcript directory not found: {transcript_dir}")
        return None

    if not os.path.exists(summary_dir):
        print(f"Summary directory not found: {summary_dir}")
        return None

    # Collect data
    training_data = []

    # Get all chunk files
    transcript_files = [f for f in os.listdir(transcript_dir) if f.startswith("chunk_") and f.endswith(".txt")]

    for transcript_file in sorted(transcript_files):
        # Extract chunk number
        chunk_match = re.search(r"chunk_(\d+)", transcript_file)
        if not chunk_match:
            continue

        chunk_no = int(chunk_match.group(1))

        # Find corresponding summary file
        summary_file = f"chunk_{chunk_no:02d}_summary.txt"
        summary_path = os.path.join(summary_dir, summary_file)
        transcript_path = os.path.join(transcript_dir, transcript_file)

        if not os.path.exists(summary_path):
            print(f"Warning: Summary file not found for chunk {chunk_no}")
            continue

        # Read files
        transcript = read_transcript_file(transcript_path)
        summary = read_summary_file(summary_path)

        if transcript and summary:
            training_data.append({"chunk_no": chunk_no, "transcript": transcript, "summary": summary})
            print(f"Processed chunk {chunk_no}")

    return training_data


def split_and_save_data(training_data, output_dir="training_data"):
    """Split data into train/val/test sets and save as CSV files"""

    if not training_data:
        print("No training data to split")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert to DataFrame
    df = pd.DataFrame(training_data)

    # Save full dataset
    full_path = os.path.join(output_dir, "training_data.csv")
    df.to_csv(full_path, index=False, encoding="utf-8")
    print(f"Saved full dataset: {full_path} ({len(df)} samples)")

    # Split into train/val/test (70/15/15)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

    # Save splits
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    print(f"Saved train set: {train_path} ({len(train_df)} samples)")
    print(f"Saved validation set: {val_path} ({len(val_df)} samples)")
    print(f"Saved test set: {test_path} ({len(test_df)} samples)")

    # Print sample data
    print("\nSample data:")
    print(f"Chunk {df.iloc[0]['chunk_no']}:")
    print(f"Transcript preview: {df.iloc[0]['transcript'][:200]}...")
    print(f"Summary preview: {df.iloc[0]['summary'][:200]}...")


def main():
    """Main function to create and split training data"""

    # Adjust this path to your data directory
    # Based on your screenshots, it looks like: mlops-demo-project/data
    data_dir = "./"  # Change this to your actual data directory path

    print("Creating training data from transcript and summary files...")
    training_data = create_training_data(data_dir)

    if training_data:
        print(f"\nCollected {len(training_data)} training samples")
        split_and_save_data(training_data)
        print("\nTraining data creation complete!")
    else:
        print("Failed to create training data. Please check your file paths.")


if __name__ == "__main__":
    main()
