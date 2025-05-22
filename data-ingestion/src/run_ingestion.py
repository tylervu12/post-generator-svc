import os
import argparse
from .ingestion_service import IngestionService

def main():
    parser = argparse.ArgumentParser(description='Process CSV files for post ingestion')
    parser.add_argument('file_path', help='Path to the CSV file to process')
    args = parser.parse_args()

    # Validate file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File {args.file_path} does not exist")
        return

    # Initialize and run service
    service = IngestionService()
    try:
        stats = service.process_file(args.file_path)
        print("\nProcessing Summary:")
        print(f"Total rows: {stats['total_rows']}")
        print(f"Successfully processed: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main() 