#!/usr/bin/env python3
"""
Bulk Restaurant Processing CLI

Process restaurants in bulk with support for multiple input formats,
processing modes, and parallel execution.

Usage:
    # From CSV file
    python scripts/bulk_process.py --input restaurants.csv --mode refresh-stale
    
    # From JSON file
    python scripts/bulk_process.py --input restaurants.json --mode net-new
    
    # All restaurants in database
    python scripts/bulk_process.py --db-all --mode refresh-stale
    
    # Specific components only
    python scripts/bulk_process.py --db-all --mode refresh-stale --components description,embedding
    
    # Dry run (preview without changes)
    python scripts/bulk_process.py --db-all --mode force --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.database import SessionLocal
from app.services.bulk_service import (
    ProcessingMode,
    Component,
    RestaurantInput,
    BulkProcessingReport,
    ProcessingResult,
    parse_csv_input,
    parse_json_input,
    get_restaurants_from_db,
    run_bulk_processing,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Bulk process restaurants for AI analysis and caching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all restaurants that need updates (missing OR outdated data)
  python scripts/bulk_process.py --db-all
  
  # Import new restaurants from CSV (auto mode handles everything)
  python scripts/bulk_process.py --input restaurants.csv
  
  # Force regenerate descriptions only
  python scripts/bulk_process.py --db-all --force --components description
  
  # Run quality scoring only (people, lighting, blur detection)
  python scripts/bulk_process.py --db-all --components quality
  
  # Preview what would be processed
  python scripts/bulk_process.py --db-all --dry-run
  
  # Completely refresh a restaurant's images (delete + re-fetch)
  python scripts/bulk_process.py --place-ids ChIJxxx --refresh-images
  
  # High concurrency for faster processing
  python scripts/bulk_process.py --db-all --concurrency 10
        """
    )
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i",
        type=str,
        help="Path to CSV or JSON file with restaurant data"
    )
    input_group.add_argument(
        "--db-all",
        action="store_true",
        help="Process all restaurants in the database"
    )
    input_group.add_argument(
        "--place-ids",
        type=str,
        nargs="+",
        help="Specific place_ids to process"
    )
    
    # Processing mode
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["auto", "force", "net-new", "refresh-stale"],
        default="auto",
        help="Processing mode: auto (smart default - missing OR outdated), force (regenerate everything)"
    )
    
    # Shorthand for --mode force
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Shorthand for --mode force (regenerate everything regardless of state)"
    )
    
    # Components to process
    parser.add_argument(
        "--components", "-c",
        type=str,
        default="all",
        help="Comma-separated list of components: category,tags,description,embedding,quality,all (default: all)"
    )
    
    # Concurrency settings
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Overall concurrency limit (default: 5)"
    )
    parser.add_argument(
        "--openai-concurrency",
        type=int,
        default=3,
        help="OpenAI API concurrency limit (default: 3)"
    )
    parser.add_argument(
        "--places-concurrency",
        type=int,
        default=5,
        help="Google Places API concurrency limit (default: 5)"
    )
    
    # Output options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be processed without making changes"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file for results report"
    )
    parser.add_argument(
        "--output-failures",
        type=str,
        help="Output JSON file for failed items (for retry)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output (no progress bar)"
    )
    parser.add_argument(
        "--photo-provider",
        type=str,
        choices=["serpapi", "google"],
        default=None,
        help="Photo provider: serpapi (100+ photos) or google (10 photos). Default: serpapi"
    )
    parser.add_argument(
        "--refresh-images",
        action="store_true",
        help="Delete existing images and re-fetch from API (use with --place-ids or --db-all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with detailed progress"
    )
    
    return parser.parse_args()


def parse_components(components_str: str) -> list[Component]:
    """Parse comma-separated components string."""
    components = []
    for comp in components_str.split(","):
        comp = comp.strip().lower()
        if comp == "all":
            return [Component.ALL]
        elif comp == "category":
            components.append(Component.CATEGORY)
        elif comp == "tags":
            components.append(Component.TAGS)
        elif comp == "description":
            components.append(Component.DESCRIPTION)
        elif comp == "embedding":
            components.append(Component.EMBEDDING)
        elif comp == "quality":
            components.append(Component.QUALITY)
        else:
            raise ValueError(f"Unknown component: {comp}")
    return components if components else [Component.ALL]


def get_db_factory():
    """Return a factory function for creating database sessions."""
    def factory():
        return SessionLocal()
    return factory


def load_restaurants(args) -> list[RestaurantInput]:
    """Load restaurants from the specified input source."""
    if args.input:
        input_path = Path(args.input)
        if input_path.suffix.lower() == ".csv":
            return parse_csv_input(str(input_path))
        elif input_path.suffix.lower() == ".json":
            return parse_json_input(str(input_path))
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    elif args.db_all:
        db = SessionLocal()
        try:
            mode = ProcessingMode(args.mode)
            components = parse_components(args.components)
            return get_restaurants_from_db(db, mode, components)
        finally:
            db.close()
    
    elif args.place_ids:
        return [RestaurantInput(place_id=pid) for pid in args.place_ids]
    
    return []


def create_progress_callback(args, total: int):
    """Create a progress callback based on output settings."""
    if args.quiet:
        return None
    
    try:
        from tqdm import tqdm
        pbar = tqdm(total=total, desc="Processing", unit="restaurant")
        
        def callback(current: int, total: int, result: ProcessingResult):
            pbar.update(1)
            if args.verbose:
                status = "✓" if result.success else "✗"
                components = ", ".join(result.components_processed) if result.components_processed else "skipped"
                tqdm.write(f"  {status} {result.name[:30]}: {components}")
        
        return callback
    except ImportError:
        # Fallback without tqdm
        last_percent = [0]
        
        def callback(current: int, total: int, result: ProcessingResult):
            percent = int(current / total * 100)
            if percent >= last_percent[0] + 10:
                print(f"Progress: {percent}% ({current}/{total})")
                last_percent[0] = percent
            if args.verbose:
                status = "✓" if result.success else "✗"
                components = ", ".join(result.components_processed) if result.components_processed else "skipped"
                print(f"  {status} {result.name[:30]}: {components}")
        
        return callback


def print_summary(report: BulkProcessingReport, args):
    """Print a summary of the processing results."""
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    
    print(f"\nTotal restaurants: {report.total}")
    print(f"  Processed: {report.processed}")
    print(f"  Skipped:   {report.skipped}")
    print(f"  Failed:    {report.failed}")
    print(f"\nDuration: {report.duration_seconds:.1f} seconds")
    
    if report.processed > 0:
        avg_time = report.duration_seconds / report.processed
        print(f"Average time per restaurant: {avg_time:.2f}s")
    
    # Show cost breakdown
    if report.total_cost > 0:
        print(f"\n💰 Cost Summary:")
        print(f"  Total: ${report.total_cost:.4f}")
        for provider, cost in report.cost_by_provider.items():
            calls = report.api_calls_by_provider.get(provider, 0)
            print(f"  {provider}: ${cost:.4f} ({calls} API calls)")
    
    # Show failures
    failures = [r for r in report.results if not r.success]
    if failures:
        print(f"\nFailed restaurants ({len(failures)}):")
        for f in failures[:10]:  # Show first 10
            print(f"  - {f.name or f.place_id}: {f.error}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN - No changes were made")


def save_results(report: BulkProcessingReport, output_path: str):
    """Save results to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


def save_failures(report: BulkProcessingReport, output_path: str):
    """Save failed items to a JSON file for retry."""
    failures = [
        {
            "place_id": r.place_id,
            "name": r.name,
            "error": r.error
        }
        for r in report.results if not r.success
    ]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)
    print(f"Failures saved to: {output_path}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Parse mode and components
    # --force flag overrides --mode
    if args.force:
        mode = ProcessingMode.FORCE
    else:
        mode = ProcessingMode(args.mode)
    components = parse_components(args.components)
    
    # Load restaurants
    print(f"Loading restaurants...")
    try:
        restaurants = load_restaurants(args)
    except Exception as e:
        print(f"Error loading restaurants: {e}")
        sys.exit(1)
    
    if not restaurants:
        print("No restaurants to process.")
        sys.exit(0)
    
    # Determine photo provider
    photo_provider = args.photo_provider
    if not photo_provider:
        # Default to serpapi if SERPAPI_API_KEY is set, otherwise google
        import os
        if os.getenv("SERPAPI_API_KEY"):
            photo_provider = "serpapi"
        else:
            photo_provider = "google"
    
    print(f"Found {len(restaurants)} restaurants to process")
    print(f"Mode: {mode.value}")
    print(f"Components: {', '.join(c.value for c in components)}")
    print(f"Photo Provider: {photo_provider} {'(100+ photos)' if photo_provider == 'serpapi' else '(10 photos max)'}")
    print(f"Concurrency: {args.concurrency} (OpenAI: {args.openai_concurrency}, Places: {args.places_concurrency})")
    if args.refresh_images:
        print(f"⚠️  REFRESH IMAGES: Will delete existing images and re-fetch from API")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No changes will be made\n")
    
    # Create progress callback
    progress_callback = create_progress_callback(args, len(restaurants))
    
    # Run processing
    print("\nStarting processing...\n")
    
    try:
        report = run_bulk_processing(
            db_factory=get_db_factory(),
            restaurants=restaurants,
            mode=mode,
            components=components,
            concurrency=args.concurrency,
            openai_concurrency=args.openai_concurrency,
            places_concurrency=args.places_concurrency,
            dry_run=args.dry_run,
            progress_callback=progress_callback,
            photo_provider=photo_provider,
            refresh_images=args.refresh_images,
        )
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Close tqdm progress bar if used
    try:
        from tqdm import tqdm
        tqdm._instances.clear()
    except:
        pass
    
    # Print summary
    print_summary(report, args)
    
    # Save outputs
    if args.output:
        save_results(report, args.output)
    
    if args.output_failures and report.failed > 0:
        save_failures(report, args.output_failures)
    
    # Exit with appropriate code
    if report.failed > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

