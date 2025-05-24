#!/usr/bin/env python3
"""
Combine individual segment-level CSV transcripts into 20-minute chunks for D&D summarization training.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List


class SegmentCombiner:
    def __init__(self, transcript_dir: str = "./transcripts", files_per_chunk: int = 4):
        """Initialize the segment combiner.

        Args:
            transcript_dir: Directory containing segment CSV files
            files_per_chunk: Number of audio files to combine (4 = ~20 minutes)
        """
        self.transcript_dir = Path(transcript_dir)
        self.files_per_chunk = files_per_chunk
        # self.output_dir = Path("./combined_segments")
        # self.output_dir.mkdir(exist_ok=True)

    def get_segment_csv_files(self) -> List[Path]:
        """Get all segment CSV files, sorted by filename numerically."""
        csv_files = list(self.transcript_dir.glob("*_segments.csv"))

        # Sort numerically by extracting numbers from filename
        def extract_number(filename):
            import re

            # Look for patterns like "chunk_0", "segment_001", etc.
            match = re.search(r"chunk_(\d+)|segment_(\d+)", filename.stem)
            if match:
                return int(match.group(1) or match.group(2))
            return 0

        return sorted(csv_files, key=extract_number)

    def read_segments_csv(self, csv_file: Path) -> Dict:
        """Read segments from a CSV file.

        Args:
            csv_file: Path to the segments CSV file

        Returns:
            Dict with file info and segments
        """
        try:
            segments = []
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    segments.append(
                        {
                            "segment_id": int(row["segment_id"]),
                            "start_time": float(row["start_time"]),
                            "end_time": float(row["end_time"]),
                            "duration": float(row["duration"]),
                            "text": row["text"],
                        }
                    )

            # Calculate file duration from segments
            file_duration = max(seg["end_time"] for seg in segments) if segments else 0

            return {
                "filename": csv_file.stem.replace("_segments", ""),
                "segments": segments,
                "segment_count": len(segments),
                "file_duration": file_duration,
                "total_text": " ".join(seg["text"] for seg in segments),
            }

        except Exception as e:
            print(f"❌ Error reading {csv_file}: {e}")
            return None

    def combine_segment_files(self, csv_files: List[Path]) -> Dict:
        """Combine multiple segment CSV files into one chunk.

        Args:
            csv_files: List of segment CSV files to combine

        Returns:
            Combined segment data with global timing
        """
        combined_data = {
            "source_files": [],
            "file_info": [],
            "all_segments": [],
            "combined_text": "",
            "total_segments": 0,
            "total_duration": 0,
            "global_start_time": 0,
            "global_end_time": 0,
        }

        current_time_offset = 0.0

        for file_idx, csv_file in enumerate(csv_files):
            file_data = self.read_segments_csv(csv_file)
            if not file_data:
                continue

            # Store file info
            combined_data["source_files"].append(file_data["filename"])
            combined_data["file_info"].append(
                {
                    "file_number": file_idx + 1,
                    "filename": file_data["filename"],
                    "segment_count": file_data["segment_count"],
                    "file_duration": file_data["file_duration"],
                    "global_start_offset": current_time_offset,
                    "global_end_offset": current_time_offset + file_data["file_duration"],
                }
            )

            # Add file separator to combined text
            file_separator = f"\\n\\n=== File {file_idx + 1}: {file_data['filename']} (t={current_time_offset:.1f}s) ===\\n"
            combined_data["combined_text"] += file_separator

            # Process each segment with global timing
            for segment in file_data["segments"]:
                global_segment = {
                    "original_file": file_data["filename"],
                    "file_number": file_idx + 1,
                    "original_segment_id": segment["segment_id"],
                    "global_segment_id": combined_data["total_segments"] + 1,
                    # Original timing (within the file)
                    "local_start_time": segment["start_time"],
                    "local_end_time": segment["end_time"],
                    "local_duration": segment["duration"],
                    # Global timing (across all files)
                    "global_start_time": current_time_offset + segment["start_time"],
                    "global_end_time": current_time_offset + segment["end_time"],
                    "global_duration": segment["duration"],
                    "text": segment["text"],
                }

                combined_data["all_segments"].append(global_segment)
                combined_data["total_segments"] += 1

                # Add segment text with timing info
                segment_header = f"[{global_segment['global_start_time']:.1f}s - {global_segment['global_end_time']:.1f}s] "
                combined_data["combined_text"] += segment_header + segment["text"] + "\\n"

            # Update time offset for next file
            current_time_offset += file_data["file_duration"]

        # Set global timing info
        if combined_data["all_segments"]:
            combined_data["global_start_time"] = combined_data["all_segments"][0]["global_start_time"]
            combined_data["global_end_time"] = combined_data["all_segments"][-1]["global_end_time"]
            combined_data["total_duration"] = combined_data["global_end_time"] - combined_data["global_start_time"]

        return combined_data

    def save_combined_chunk(self, chunk_data: Dict, chunk_number: int):
        """Save a combined chunk to files.

        Args:
            chunk_data: Combined segment data
            chunk_number: Chunk number for filename
        """
        base_filename = f"chunk_{chunk_number:02d}"

        # Save comprehensive JSON for training
        json_file = self.output_dir / f"{base_filename}_full.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)

        # Save simplified JSON for easier processing
        simple_data = {
            "chunk_number": chunk_number,
            "source_files": chunk_data["source_files"],
            "total_duration_minutes": chunk_data["total_duration"] / 60,
            "global_start_time": chunk_data["global_start_time"],
            "global_end_time": chunk_data["global_end_time"],
            "segment_count": chunk_data["total_segments"],
            "combined_transcript": chunk_data["combined_text"],
        }

        simple_json_file = self.output_dir / f"{base_filename}_simple.json"
        with open(simple_json_file, "w", encoding="utf-8") as f:
            json.dump(simple_data, f, indent=2, ensure_ascii=False)

        # Save human-readable text
        txt_file = self.output_dir / f"{base_filename}.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(f"D&D Session Chunk {chunk_number}\\n")
            f.write(
                f"Duration: {chunk_data['total_duration']/60:.1f} minutes ({chunk_data['global_start_time']:.1f}s - {chunk_data['global_end_time']:.1f}s)\\n"
            )
            f.write(f"Files: {', '.join(chunk_data['source_files'])}\\n")
            f.write(f"Segments: {chunk_data['total_segments']}\\n")
            f.write("=" * 60 + "\\n\\n")
            f.write(chunk_data["combined_text"])

        # Save segments with global timing as CSV
        segments_csv = self.output_dir / f"{base_filename}_segments.csv"
        with open(segments_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "global_segment_id",
                "original_file",
                "file_number",
                "original_segment_id",
                "global_start_time",
                "global_end_time",
                "global_duration",
                "local_start_time",
                "local_end_time",
                "local_duration",
                "text",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(chunk_data["all_segments"])

        print(f"💾 Saved chunk {chunk_number}: {chunk_data['total_duration']/60:.1f} minutes, {chunk_data['total_segments']} segments")
        return simple_json_file

    def create_chunks_summary(self, chunk_files: List[Path]):
        """Create a summary of all combined chunks."""
        summary_file = self.output_dir / "chunks_summary.csv"

        with open(summary_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "chunk_number",
                "duration_minutes",
                "segment_count",
                "source_files",
                "json_file",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i, chunk_file in enumerate(chunk_files, 1):
                try:
                    with open(chunk_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    writer.writerow(
                        {
                            "chunk_number": i,
                            "duration_minutes": data["total_duration_minutes"],
                            "segment_count": data["segment_count"],
                            "source_files": "; ".join(data["source_files"]),
                            "json_file": chunk_file.name,
                        }
                    )
                except Exception as e:
                    print(f"❌ Error processing {chunk_file} for summary: {e}")

        print(f"📊 Chunks summary saved: {summary_file}")

    def run_combination(self):
        """Run the complete segment combination process."""
        print("🔄 Starting segment combination...")

        # Get all segment CSV files
        csv_files = self.get_segment_csv_files()
        if not csv_files:
            print(f"❌ No segment CSV files found in {self.transcript_dir}")
            return

        print(f"📁 Found {len(csv_files)} segment CSV files")

        # Group files into chunks
        chunk_files = []
        for i in range(0, len(csv_files), self.files_per_chunk):
            chunk = csv_files[i : i + self.files_per_chunk]

            chunk_number = (i // self.files_per_chunk) + 1
            print(f"\\n📝 Creating chunk {chunk_number} from {len(chunk)} files:")
            for file in chunk:
                print(f"   - {file.name}")

            # Combine segments
            combined_data = self.combine_segment_files(chunk)

            # Save combined chunk
            json_file = self.save_combined_chunk(combined_data, chunk_number)
            chunk_files.append(json_file)

        # Create summary
        self.create_chunks_summary(chunk_files)

        print("\\n🎉 Combination complete!")
        print(f"📊 Created {len(chunk_files)} combined chunks")
        print(f"📂 Chunks saved in: {self.output_dir}")


def main():
    """Main function to run segment combination."""
    import argparse

    parser = argparse.ArgumentParser(description="Combine segment CSV files into chunks with global timing")
    parser.add_argument(
        "--transcript-dir",
        default="./transcripts",
        help="Directory containing segment CSV files (default: ./transcripts)",
    )
    parser.add_argument(
        "--output-dir",
        default="./combined_segments",
        help="Output directory for combined chunks (default: ./combined_segments)",
    )
    parser.add_argument(
        "--files-per-chunk",
        type=int,
        default=4,
        help="Number of files per chunk (default: 4)",
    )

    args = parser.parse_args()

    # Create combiner and run
    combiner = SegmentCombiner(transcript_dir=args.transcript_dir, files_per_chunk=args.files_per_chunk)
    combiner.output_dir = Path(args.output_dir)
    combiner.output_dir.mkdir(exist_ok=True)

    combiner.run_combination()


if __name__ == "__main__":
    main()
