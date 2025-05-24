#!/usr/bin/env python3
"""
Batch transcription script for D&D audio segments.
Transcribes audio files and saves segment-level transcripts to CSV.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class BatchTranscriber:
    def __init__(
        self,
        api_url: str = "http://localhost:3569",
        audio_dir: str = "./audio_segments",
    ):
        """Initialize the batch transcriber.

        Args:
            api_url: URL of the Parakeet STT API
            audio_dir: Directory containing audio files
        """
        self.api_url = api_url
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path("./transcripts")
        self.output_dir.mkdir(exist_ok=True)

        # Setup requests session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=1,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def check_api_health(self) -> bool:
        """Check if the API is running and healthy."""
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=10)
            return response.status_code == 200 and response.json().get("status") == "healthy"
        except Exception as e:
            print(f"❌ API health check failed: {e}")
            return False

    def get_audio_files(self) -> List[Path]:
        """Get all audio files from the directory, sorted by name."""
        audio_extensions = {".wav", ".flac"}
        audio_files = []

        for ext in audio_extensions:
            audio_files.extend(self.audio_dir.glob(f"*{ext}"))

        # Sort by filename to maintain order
        return sorted(audio_files)

    def transcribe_file(self, audio_file: Path) -> Optional[Dict]:
        """Transcribe a single audio file.

        Args:
            audio_file: Path to the audio file

        Returns:
            Transcription result dict or None if failed
        """
        try:
            print(f"🎤 Transcribing: {audio_file.name}")

            with open(audio_file, "rb") as f:
                files = {"file": (audio_file.name, f, "audio/wav")}
                data = {"return_timestamps": True}

                response = self.session.post(
                    f"{self.api_url}/transcribe",
                    files=files,
                    data=data,
                    timeout=300,  # 5 minutes timeout
                )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success: {audio_file.name} ({result.get('processing_time_sec', 0):.2f}s)")
                return result
            else:
                print(f"❌ Failed: {audio_file.name} - {response.status_code}: {response.text}")
                return None

        except Exception as e:
            print(f"❌ Error transcribing {audio_file.name}: {e}")
            return None

    def save_segments_to_csv(self, audio_file: Path, transcript_data: Dict):
        """Save segment-level transcript data to CSV.

        Args:
            audio_file: Original audio file path
            transcript_data: Transcription result from API
        """
        csv_filename = self.output_dir / f"{audio_file.stem}_segments.csv"

        # Extract segment timestamps from the API response
        segments = []

        if transcript_data.get("text") and isinstance(transcript_data["text"], list) and transcript_data["text"][0].get("timestamp", {}).get("segment"):

            # Use the segment timestamps from the API
            segment_timestamps = transcript_data["text"][0]["timestamp"]["segment"]

            for i, segment in enumerate(segment_timestamps):
                segments.append(
                    {
                        "segment_id": i + 1,
                        "start_time": segment.get("start", 0),
                        "end_time": segment.get("end", 0),
                        "duration": segment.get("end", 0) - segment.get("start", 0),
                        "text": segment.get("segment", "").strip(),
                    }
                )

        else:
            # Fallback: create one segment with the full transcript
            full_text = ""
            if isinstance(transcript_data.get("text"), list) and transcript_data["text"]:
                full_text = transcript_data["text"][0].get("text", "")
            elif isinstance(transcript_data.get("text"), str):
                full_text = transcript_data["text"]

            segments.append(
                {
                    "segment_id": 1,
                    "start_time": 0,
                    "end_time": 300,  # Assume 5 minutes
                    "duration": 300,
                    "text": full_text.strip(),
                }
            )

        # Write segments to CSV
        with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["segment_id", "start_time", "end_time", "duration", "text"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(segments)

        print(f"💾 Saved {len(segments)} segments: {csv_filename}")
        return len(segments)

    def create_batch_summary(self, results: List[Dict]):
        """Create a summary CSV of all transcriptions."""
        summary_file = self.output_dir / "batch_summary.csv"

        with open(summary_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "filename",
                "status",
                "segment_count",
                "processing_time_sec",
                "first_segment_preview",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                writer.writerow(
                    {
                        "filename": result["filename"],
                        "status": "Success" if result["success"] else "Failed",
                        "segment_count": result.get("segment_count", 0),
                        "processing_time_sec": result.get("processing_time_sec", 0),
                        "first_segment_preview": result.get("first_segment_preview", "N/A"),
                    }
                )

        print(f"📊 Batch summary saved: {summary_file}")

    def run_batch_transcription(self):
        """Run the complete batch transcription process."""
        print("🚀 Starting batch transcription...")

        # Check API health
        if not self.check_api_health():
            print("❌ API is not healthy. Please start the backend service.")
            return

        # Get audio files
        audio_files = self.get_audio_files()
        if not audio_files:
            print(f"❌ No audio files found in {self.audio_dir}")
            return

        print(f"📁 Found {len(audio_files)} audio files")

        # Process each file
        results = []
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing {audio_file.name}")

            # Check if already processed
            csv_output = self.output_dir / f"{audio_file.stem}_segments.csv"
            if csv_output.exists():
                print(f"⏭️  Skipping {audio_file.name} (already transcribed)")
                # Read existing file to get segment count
                try:
                    with open(csv_output, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        segment_count = sum(1 for _ in reader)
                    results.append(
                        {
                            "filename": audio_file.name,
                            "success": True,
                            "segment_count": segment_count,
                            "processing_time_sec": 0,
                            "first_segment_preview": "Already processed",
                        }
                    )
                except Exception as e:
                    print(f"⚠️  Error reading existing file {csv_output}: {e}")
                    results.append(
                        {
                            "filename": audio_file.name,
                            "success": True,
                            "segment_count": 0,
                            "processing_time_sec": 0,
                            "first_segment_preview": "Error reading existing file",
                        }
                    )
                continue

            # Transcribe
            transcript_data = self.transcribe_file(audio_file)

            if transcript_data:
                segment_count = self.save_segments_to_csv(audio_file, transcript_data)

                # Get first segment preview for summary
                first_segment_preview = "No segments found"
                try:
                    csv_output = self.output_dir / f"{audio_file.stem}_segments.csv"
                    with open(csv_output, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        first_row = next(reader)
                        preview_text = first_row["text"][:100]
                        first_segment_preview = preview_text + "..." if len(preview_text) == 100 else preview_text
                except Exception as e:
                    print(f"⚠️  Error reading segments CSV for preview: {e}")
                    pass

                results.append(
                    {
                        "filename": audio_file.name,
                        "success": True,
                        "segment_count": segment_count,
                        "processing_time_sec": transcript_data.get("processing_time_sec", 0),
                        "first_segment_preview": first_segment_preview,
                    }
                )
            else:
                results.append(
                    {
                        "filename": audio_file.name,
                        "success": False,
                        "segment_count": 0,
                        "processing_time_sec": 0,
                        "first_segment_preview": "Transcription failed",
                    }
                )

            # # Small delay to be nice to the API
            # time.sleep(1)

        # Create summary
        self.create_batch_summary(results)

        # Final statistics
        successful = sum(1 for r in results if r["success"])
        total_segments = sum(r.get("segment_count", 0) for r in results if r["success"])
        total_time = sum(r.get("processing_time_sec", 0) for r in results if r["success"])

        print("\n🎉 Batch transcription complete!")
        print(f"✅ Successful: {successful}/{len(audio_files)}")
        print(f"📝 Total segments: {total_segments}")
        print(f"⏱️  Total processing time: {total_time:.2f} seconds")
        print(f"📂 Transcripts saved in: {self.output_dir}")


def main():
    """Main function to run batch transcription."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch transcribe audio files to segment-level CSV")
    parser.add_argument(
        "--audio-dir",
        default="./audio_segments",
        help="Directory containing audio files (default: ./audio_segments)",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:3569",
        help="Parakeet STT API URL (default: http://localhost:3569)",
    )
    parser.add_argument(
        "--output-dir",
        default="./transcripts",
        help="Output directory for transcripts (default: ./transcripts)",
    )

    args = parser.parse_args()

    # Create transcriber and run
    transcriber = BatchTranscriber(api_url=args.api_url, audio_dir=args.audio_dir)
    transcriber.output_dir = Path(args.output_dir)
    transcriber.output_dir.mkdir(exist_ok=True)

    transcriber.run_batch_transcription()


if __name__ == "__main__":
    main()
