#!/usr/bin/env python3
"""
Generate D&D session summaries from transcript chunks using OpenAI GPT.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SummaryGenerator:
    def __init__(
        self,
        chunks_dir: str = "./combined_segments",
        output_dir: str = "./summaries",
        model: str = "gpt-4o",
    ):
        """Initialize the summary generator.

        Args:
            chunks_dir: Directory containing combined transcript chunks
            output_dir: Directory to save generated summaries
            model: OpenAI model to use for generation
        """
        self.chunks_dir = Path(chunks_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.model = model

        # Initialize OpenAI client
        self.client = openai.OpenAI()

        # Verify API key is loaded
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables. Check your .env file.")

        print(f"✅ OpenAI client initialized with model: {model}")

    def get_chunk_files(self) -> List[Path]:
        """Get all chunk JSON files, sorted numerically."""
        chunk_files = list(self.chunks_dir.glob("chunk_*_simple.json"))

        # Sort numerically by chunk number
        def extract_chunk_number(filename):
            import re

            match = re.search(r"chunk_(\d+)", filename.stem)
            return int(match.group(1)) if match else 0

        return sorted(chunk_files, key=extract_chunk_number)

    def load_chunk(self, chunk_file: Path) -> Dict:
        """Load a chunk file and extract transcript data.

        Args:
            chunk_file: Path to the chunk JSON file

        Returns:
            Dict with chunk data
        """
        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunk_data = json.load(f)

            return {
                "chunk_number": chunk_data.get("chunk_number", 0),
                "source_files": chunk_data.get("source_files", []),
                "duration_minutes": chunk_data.get("total_duration_minutes", 0),
                "transcript": chunk_data.get("combined_transcript", ""),
                "segment_count": chunk_data.get("segment_count", 0),
            }
        except Exception as e:
            print(f"❌ Error loading {chunk_file}: {e}")
            return None

    def create_summary_prompt(self, chunk_data: Dict) -> str:
        """Create the GPT prompt for summarizing a chunk.

        Args:
            chunk_data: Chunk information and transcript

        Returns:
            Formatted prompt string
        """
        transcript = chunk_data["transcript"]
        duration = chunk_data["duration_minutes"]

        prompt = f"""You are summarizing a Critical Role D&D session transcript. Create an engaging summary (300-500 words) that captures:

                STORY: What major plot points advanced? What did the party discover or accomplish?
                CHARACTERS: What did each party member do? Any great roleplay or character development moments?
                GAMEPLAY: Combat encounters, critical rolls, creative problem solving, spells used
                SETUP: How does this set up future story threads or hooks?

                Write in an engaging style that would help someone catch up on what they missed. Focus on the most important and interesting moments.

                If you receive an empty segment or one with no meaningful content, return a short note like "No significant events occurred in this segment."

                Duration: {duration:.1f} minutes
                Transcript:
                {transcript}

                Summary:"""

        return prompt

    def generate_summary(self, chunk_data: Dict) -> Optional[Dict]:
        """Generate a summary for a single chunk.

        Args:
            chunk_data: Chunk information and transcript

        Returns:
            Dict with summary and metadata, or None if failed
        """
        try:
            prompt = self.create_summary_prompt(chunk_data)

            print(f"🤖 Generating summary for chunk {chunk_data['chunk_number']} ({chunk_data['duration_minutes']:.1f} min)")

            # Make API request
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=800,
            )

            generation_time = time.time() - start_time
            summary_text = response.choices[0].message.content

            # Calculate metrics
            word_count = len(summary_text.split())

            result = {
                "chunk_number": chunk_data["chunk_number"],
                "source_files": chunk_data["source_files"],
                "duration_minutes": chunk_data["duration_minutes"],
                "segment_count": chunk_data["segment_count"],
                "summary": summary_text,
                "word_count": word_count,
                "generation_time_sec": generation_time,
                "model_used": self.model,
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": (response.usage.completion_tokens if response.usage else 0),
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            print(f"✅ Generated summary: {word_count} words, {generation_time:.2f}s, {result['total_tokens']} tokens")
            return result

        except Exception as e:
            print(f"❌ Error generating summary for chunk {chunk_data['chunk_number']}: {e}")
            return None

    def save_summary(self, summary_data: Dict):
        """Save a summary to files.

        Args:
            summary_data: Summary data and metadata
        """
        chunk_num = summary_data["chunk_number"]

        # Save as JSON (for training data)
        json_file = self.output_dir / f"chunk_{chunk_num:02d}_summary.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        # Save as text (for easy reading)
        txt_file = self.output_dir / f"chunk_{chunk_num:02d}_summary.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(f"D&D Session Summary - Chunk {chunk_num}\n")
            f.write(f"Duration: {summary_data['duration_minutes']:.1f} minutes\n")
            f.write(f"Source Files: {', '.join(summary_data['source_files'])}\n")
            f.write(f"Word Count: {summary_data['word_count']}\n")
            f.write("=" * 60 + "\n\n")
            f.write(summary_data["summary"])

        print(f"💾 Saved summary: {json_file.name}")

    def create_batch_summary(self, results: List[Dict]):
        """Create a summary of the batch generation process.

        Args:
            results: List of summary results
        """
        summary_file = self.output_dir / "batch_summary.json"

        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        batch_data = {
            "total_chunks": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "total_generation_time": sum(r.get("generation_time_sec", 0) for r in successful),
            "total_tokens_used": sum(r.get("total_tokens", 0) for r in successful),
            "average_summary_length": (sum(r.get("word_count", 0) for r in successful) / len(successful) if successful else 0),
            "model_used": self.model,
            "failed_chunks": [r["chunk_number"] for r in failed] if failed else [],
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(batch_data, f, indent=2)

        # Also create CSV summary
        csv_file = self.output_dir / "summaries_overview.csv"
        import csv

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "chunk_number",
                "status",
                "duration_minutes",
                "word_count",
                "tokens_used",
                "generation_time_sec",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                writer.writerow(
                    {
                        "chunk_number": result["chunk_number"],
                        "status": "Success" if result["success"] else "Failed",
                        "duration_minutes": result.get("duration_minutes", 0),
                        "word_count": result.get("word_count", 0),
                        "tokens_used": result.get("total_tokens", 0),
                        "generation_time_sec": result.get("generation_time_sec", 0),
                    }
                )

        print(f"📊 Batch summary saved: {summary_file}")
        print(f"📈 Overview CSV saved: {csv_file}")

    def run_batch_generation(self, delay_between_requests: float = 1.0):
        """Run the complete batch summary generation process.

        Args:
            delay_between_requests: Seconds to wait between API calls
        """
        print("🚀 Starting batch summary generation...")

        # Get all chunk files
        chunk_files = self.get_chunk_files()
        if not chunk_files:
            print(f"❌ No chunk files found in {self.chunks_dir}")
            return

        print(f"📁 Found {len(chunk_files)} chunk files")

        # Process each chunk
        results = []
        for i, chunk_file in enumerate(chunk_files, 1):
            print(f"\n[{i}/{len(chunk_files)}] Processing {chunk_file.name}")

            # Check if already processed
            chunk_number = int(chunk_file.stem.split("_")[1])
            json_output = self.output_dir / f"chunk_{chunk_number:02d}_summary.json"

            if json_output.exists():
                print(f"⏭️  Skipping chunk {chunk_number} (already processed)")
                # Load existing data for batch summary
                try:
                    with open(json_output, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                    results.append(
                        {
                            "chunk_number": chunk_number,
                            "success": True,
                            "duration_minutes": existing_data.get("duration_minutes", 0),
                            "word_count": existing_data.get("word_count", 0),
                            "total_tokens": existing_data.get("total_tokens", 0),
                            "generation_time_sec": existing_data.get("generation_time_sec", 0),
                        }
                    )
                except Exception as e:
                    print(f"❌ Error loading existing summary for chunk {chunk_number}: {e}")
                    results.append({"chunk_number": chunk_number, "success": False})
                continue

            # Load chunk data
            chunk_data = self.load_chunk(chunk_file)
            if not chunk_data:
                results.append({"chunk_number": chunk_number, "success": False})
                continue

            # Generate summary
            summary_data = self.generate_summary(chunk_data)
            if summary_data:
                self.save_summary(summary_data)
                results.append(
                    {
                        "chunk_number": chunk_number,
                        "success": True,
                        "duration_minutes": summary_data["duration_minutes"],
                        "word_count": summary_data["word_count"],
                        "total_tokens": summary_data["total_tokens"],
                        "generation_time_sec": summary_data["generation_time_sec"],
                    }
                )
            else:
                results.append({"chunk_number": chunk_number, "success": False})

            # Rate limiting
            if i < len(chunk_files):  # Don't delay after last request
                time.sleep(delay_between_requests)

        # Create batch summary
        self.create_batch_summary(results)

        # Final statistics
        successful = sum(1 for r in results if r["success"])
        total_time = sum(r.get("generation_time_sec", 0) for r in results if r["success"])
        total_tokens = sum(r.get("total_tokens", 0) for r in results if r["success"])

        print("\n🎉 Batch generation complete!")
        print(f"✅ Successful: {successful}/{len(chunk_files)}")
        print(f"⏱️  Total generation time: {total_time:.2f} seconds")
        print(f"🪙 Total tokens used: {total_tokens:,}")
        print(f"📂 Summaries saved in: {self.output_dir}")


def main():
    """Main function to run batch summary generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate D&D summaries from transcript chunks using GPT")
    parser.add_argument(
        "--chunks-dir",
        default="./combined_segments",
        help="Directory containing chunk JSON files (default: ./combined_segments)",
    )
    parser.add_argument(
        "--output-dir",
        default="./summaries",
        help="Output directory for summaries (default: ./summaries)",
    )
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use (default: gpt-4o)")
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API requests in seconds (default: 1.0)",
    )

    args = parser.parse_args()

    # Create generator and run
    generator = SummaryGenerator(chunks_dir=args.chunks_dir, output_dir=args.output_dir, model=args.model)

    generator.run_batch_generation(delay_between_requests=args.delay)


if __name__ == "__main__":
    main()
