#!/usr/bin/env python3
"""
Debug script for image quality scores.

Shows quality scores for images to help debug filtering issues.

Usage:
    # Show all images with scores for a restaurant
    python scripts/debug_quality_scores.py --place-id ChIJxxx
    
    # Show images that PASSED filtering but might be problematic
    python scripts/debug_quality_scores.py --place-id ChIJxxx --show-passed
    
    # Show images that FAILED filtering (to see if good ones are being rejected)
    python scripts/debug_quality_scores.py --place-id ChIJxxx --show-failed
    
    # Show score distribution across all restaurants
    python scripts/debug_quality_scores.py --distribution
    
    # Re-score specific images manually (for testing)
    python scripts/debug_quality_scores.py --rescore-image IMAGE_ID
    
    # Show images with people_score between 0.5 and 0.7 (edge cases)
    python scripts/debug_quality_scores.py --people-edge-cases
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import func
from app.database import SessionLocal
from app.models.database import Restaurant, RestaurantImage
from app.config.ai_versions import (
    QUALITY_PEOPLE_THRESHOLD,
    QUALITY_LIGHTING_THRESHOLD,
    QUALITY_BLUR_THRESHOLD,
)


def get_restaurant_images(db, place_id: str = None, with_scores: bool = True):
    """Get images with quality scores."""
    query = db.query(RestaurantImage).join(Restaurant)
    
    if place_id:
        query = query.filter(Restaurant.place_id == place_id)
    
    if with_scores:
        query = query.filter(RestaurantImage.quality_version != None)  # noqa
    
    return query.all()


def show_images_for_restaurant(db, place_id: str, filter_type: str = None):
    """Show images for a restaurant with their quality scores."""
    restaurant = db.query(Restaurant).filter(Restaurant.place_id == place_id).first()
    
    if not restaurant:
        print(f"❌ Restaurant with place_id '{place_id}' not found")
        return
    
    print(f"\n{'='*80}")
    print(f"Restaurant: {restaurant.name}")
    print(f"Place ID: {place_id}")
    print(f"{'='*80}")
    
    print(f"\nThresholds:")
    print(f"  People:   > {QUALITY_PEOPLE_THRESHOLD} (to pass)")
    print(f"  Lighting: > {QUALITY_LIGHTING_THRESHOLD} (to pass)")
    print(f"  Blur:     > {QUALITY_BLUR_THRESHOLD} (to pass)")
    
    images = db.query(RestaurantImage).filter(
        RestaurantImage.restaurant_id == restaurant.id
    ).order_by(RestaurantImage.id).all()
    
    if not images:
        print("\n⚠️  No images found for this restaurant")
        return
    
    # Count images by status
    scored = [img for img in images if img.quality_version]
    unscored = [img for img in images if not img.quality_version]
    displayed = [img for img in images if img.is_displayed]
    
    print(f"\nImage counts:")
    print(f"  Total:     {len(images)}")
    print(f"  Scored:    {len(scored)}")
    print(f"  Unscored:  {len(unscored)}")
    print(f"  Displayed: {len(displayed)}")
    
    # Categorize by pass/fail
    passed = []
    failed = []
    
    for img in scored:
        passes_people = img.people_confidence_score > QUALITY_PEOPLE_THRESHOLD
        passes_lighting = img.lighting_confidence_score > QUALITY_LIGHTING_THRESHOLD
        passes_blur = img.blur_confidence_score > QUALITY_BLUR_THRESHOLD
        
        if passes_people and passes_lighting and passes_blur:
            passed.append(img)
        else:
            failed.append(img)
    
    print(f"  Passed:    {len(passed)}")
    print(f"  Failed:    {len(failed)}")
    
    # Filter based on argument
    if filter_type == "passed":
        images_to_show = passed
        header = "PASSED IMAGES (might have issues)"
    elif filter_type == "failed":
        images_to_show = failed
        header = "FAILED IMAGES (might be incorrectly rejected)"
    else:
        images_to_show = scored
        header = "ALL SCORED IMAGES"
    
    print(f"\n{header}:")
    print("-" * 80)
    
    for img in images_to_show:
        passes_people = img.people_confidence_score > QUALITY_PEOPLE_THRESHOLD
        passes_lighting = img.lighting_confidence_score > QUALITY_LIGHTING_THRESHOLD
        passes_blur = img.blur_confidence_score > QUALITY_BLUR_THRESHOLD
        
        status = "✅ PASS" if (passes_people and passes_lighting and passes_blur) else "❌ FAIL"
        displayed_mark = "📺" if img.is_displayed else "  "
        
        print(f"\n{displayed_mark} Image ID: {img.id} | Category: {img.category or 'N/A'}")
        print(f"   {status}")
        print(f"   People:   {img.people_confidence_score:.2f} {'✅' if passes_people else '❌'} (need > {QUALITY_PEOPLE_THRESHOLD})")
        print(f"   Lighting: {img.lighting_confidence_score:.2f} {'✅' if passes_lighting else '❌'} (need > {QUALITY_LIGHTING_THRESHOLD})")
        print(f"   Blur:     {img.blur_confidence_score:.2f} {'✅' if passes_blur else '❌'} (need > {QUALITY_BLUR_THRESHOLD})")
        print(f"   URL: {img.gcs_url}")


def show_score_distribution(db):
    """Show score distribution across all images."""
    images = db.query(RestaurantImage).filter(
        RestaurantImage.quality_version != None  # noqa
    ).all()
    
    if not images:
        print("No scored images found")
        return
    
    print(f"\n{'='*80}")
    print(f"SCORE DISTRIBUTION ACROSS {len(images)} IMAGES")
    print(f"{'='*80}")
    
    # People score distribution
    print("\nPEOPLE SCORE (1.0 = no people, 0.0 = people are main subject):")
    print(f"  Threshold: > {QUALITY_PEOPLE_THRESHOLD}")
    
    people_ranges = {
        "0.0-0.3 (people dominant)": 0,
        "0.3-0.5 (people prominent)": 0,
        "0.5-0.7 (edge cases)": 0,
        "0.7-0.9 (minor people)": 0,
        "0.9-1.0 (no people)": 0,
    }
    
    for img in images:
        score = img.people_confidence_score
        if score <= 0.3:
            people_ranges["0.0-0.3 (people dominant)"] += 1
        elif score <= 0.5:
            people_ranges["0.3-0.5 (people prominent)"] += 1
        elif score <= 0.7:
            people_ranges["0.5-0.7 (edge cases)"] += 1
        elif score <= 0.9:
            people_ranges["0.7-0.9 (minor people)"] += 1
        else:
            people_ranges["0.9-1.0 (no people)"] += 1
    
    for range_name, count in people_ranges.items():
        bar = "█" * int(count / len(images) * 40)
        pct = count / len(images) * 100
        print(f"    {range_name}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Lighting score distribution
    print("\nLIGHTING SCORE (1.0 = well lit, 0.0 = too dark):")
    print(f"  Threshold: > {QUALITY_LIGHTING_THRESHOLD}")
    
    lighting_ranges = {
        "0.0-0.3 (very dark)": 0,
        "0.3-0.5 (dark)": 0,
        "0.5-0.7 (adequate)": 0,
        "0.7-0.9 (good)": 0,
        "0.9-1.0 (excellent)": 0,
    }
    
    for img in images:
        score = img.lighting_confidence_score
        if score <= 0.3:
            lighting_ranges["0.0-0.3 (very dark)"] += 1
        elif score <= 0.5:
            lighting_ranges["0.3-0.5 (dark)"] += 1
        elif score <= 0.7:
            lighting_ranges["0.5-0.7 (adequate)"] += 1
        elif score <= 0.9:
            lighting_ranges["0.7-0.9 (good)"] += 1
        else:
            lighting_ranges["0.9-1.0 (excellent)"] += 1
    
    for range_name, count in lighting_ranges.items():
        bar = "█" * int(count / len(images) * 40)
        pct = count / len(images) * 100
        print(f"    {range_name}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Pass/fail summary
    print("\n" + "="*80)
    print("PASS/FAIL SUMMARY")
    
    passed = failed_people = failed_lighting = failed_blur = 0
    
    for img in images:
        passes_people = img.people_confidence_score > QUALITY_PEOPLE_THRESHOLD
        passes_lighting = img.lighting_confidence_score > QUALITY_LIGHTING_THRESHOLD
        passes_blur = img.blur_confidence_score > QUALITY_BLUR_THRESHOLD
        
        if passes_people and passes_lighting and passes_blur:
            passed += 1
        else:
            if not passes_people:
                failed_people += 1
            if not passes_lighting:
                failed_lighting += 1
            if not passes_blur:
                failed_blur += 1
    
    print(f"\n  Passed all thresholds:  {passed:4d} ({passed/len(images)*100:.1f}%)")
    print(f"  Failed people filter:   {failed_people:4d} ({failed_people/len(images)*100:.1f}%)")
    print(f"  Failed lighting filter: {failed_lighting:4d} ({failed_lighting/len(images)*100:.1f}%)")
    print(f"  Failed blur filter:     {failed_blur:4d} ({failed_blur/len(images)*100:.1f}%)")


def show_people_edge_cases(db, limit: int = 20):
    """Show images with people_score in the edge case range (0.5-0.8)."""
    images = db.query(RestaurantImage).join(Restaurant).filter(
        RestaurantImage.quality_version != None,  # noqa
        RestaurantImage.people_confidence_score > 0.5,
        RestaurantImage.people_confidence_score <= 0.8,
    ).order_by(RestaurantImage.people_confidence_score).limit(limit).all()
    
    print(f"\n{'='*80}")
    print(f"PEOPLE EDGE CASES (scores 0.5-0.8, threshold is {QUALITY_PEOPLE_THRESHOLD})")
    print(f"{'='*80}")
    print("\nThese images passed the people filter but might still have prominent people.")
    print("Review these to check if the threshold is too low.\n")
    
    for img in images:
        restaurant = db.query(Restaurant).filter(Restaurant.id == img.restaurant_id).first()
        passes = img.people_confidence_score > QUALITY_PEOPLE_THRESHOLD
        status = "✅ PASSES" if passes else "❌ FAILS"
        
        print(f"Image ID: {img.id} | Score: {img.people_confidence_score:.2f} | {status}")
        print(f"  Restaurant: {restaurant.name if restaurant else 'N/A'}")
        print(f"  Category: {img.category or 'N/A'}")
        print(f"  URL: {img.gcs_url}")
        print()


def rescore_image(db, image_id: int):
    """Re-score a specific image and show the result."""
    from app.services.quality_service import score_image_quality_from_url, update_image_quality_scores
    
    image = db.query(RestaurantImage).filter(RestaurantImage.id == image_id).first()
    
    if not image:
        print(f"❌ Image with ID {image_id} not found")
        return
    
    print(f"\n{'='*80}")
    print(f"RE-SCORING IMAGE ID: {image_id}")
    print(f"{'='*80}")
    
    print(f"\nCurrent scores:")
    if image.quality_version:
        print(f"  People:   {image.people_confidence_score:.2f}")
        print(f"  Lighting: {image.lighting_confidence_score:.2f}")
        print(f"  Blur:     {image.blur_confidence_score:.2f}")
    else:
        print("  (not yet scored)")
    
    print(f"\nURL: {image.gcs_url}")
    print("\nRe-scoring with GPT-4 Vision...")
    
    try:
        scores = score_image_quality_from_url(image.gcs_url)
        
        print(f"\nNew scores:")
        print(f"  People:   {scores.people_score:.2f}")
        print(f"  Lighting: {scores.lighting_score:.2f}")
        print(f"  Blur:     {scores.blur_score:.2f}")
        
        passes = scores.passes_thresholds()
        print(f"\nWould pass: {'✅ YES' if passes else '❌ NO'}")
        
        print("\nSave these scores to database? [y/N] ", end="")
        response = input().strip().lower()
        
        if response == 'y':
            update_image_quality_scores(db, image_id, scores)
            print("✅ Scores saved")
        else:
            print("⏭️ Skipped")
            
    except Exception as e:
        print(f"❌ Error scoring image: {e}")


def show_displayed_images_with_low_scores(db, limit: int = 30):
    """Show displayed images that might have problematic scores."""
    images = db.query(RestaurantImage).join(Restaurant).filter(
        RestaurantImage.is_displayed == True,  # noqa
        RestaurantImage.quality_version != None,  # noqa
    ).order_by(RestaurantImage.people_confidence_score).limit(limit).all()
    
    print(f"\n{'='*80}")
    print(f"DISPLAYED IMAGES WITH LOWEST PEOPLE SCORES")
    print(f"(These are being shown to users - check if any have people as main subject)")
    print(f"{'='*80}")
    
    for img in images:
        restaurant = db.query(Restaurant).filter(Restaurant.id == img.restaurant_id).first()
        
        print(f"\nImage ID: {img.id}")
        print(f"  Restaurant: {restaurant.name if restaurant else 'N/A'}")
        print(f"  Category: {img.category or 'N/A'}")
        print(f"  People:   {img.people_confidence_score:.2f}")
        print(f"  Lighting: {img.lighting_confidence_score:.2f}")
        print(f"  Blur:     {img.blur_confidence_score:.2f}")
        print(f"  URL: {img.gcs_url}")


def main():
    parser = argparse.ArgumentParser(description="Debug image quality scores")
    
    parser.add_argument("--place-id", type=str, help="Show images for specific restaurant")
    parser.add_argument("--show-passed", action="store_true", help="Only show images that passed filtering")
    parser.add_argument("--show-failed", action="store_true", help="Only show images that failed filtering")
    parser.add_argument("--distribution", action="store_true", help="Show score distribution across all images")
    parser.add_argument("--people-edge-cases", action="store_true", help="Show images with borderline people scores")
    parser.add_argument("--rescore-image", type=int, help="Re-score a specific image")
    parser.add_argument("--displayed-low-scores", action="store_true", help="Show displayed images with lowest people scores")
    
    args = parser.parse_args()
    
    db = SessionLocal()
    
    try:
        if args.place_id:
            filter_type = None
            if args.show_passed:
                filter_type = "passed"
            elif args.show_failed:
                filter_type = "failed"
            show_images_for_restaurant(db, args.place_id, filter_type)
            
        elif args.distribution:
            show_score_distribution(db)
            
        elif args.people_edge_cases:
            show_people_edge_cases(db)
            
        elif args.rescore_image:
            rescore_image(db, args.rescore_image)
            
        elif args.displayed_low_scores:
            show_displayed_images_with_low_scores(db)
            
        else:
            # Default: show distribution
            show_score_distribution(db)
            print("\n\nTip: Use --place-id <place_id> to see images for a specific restaurant")
            print("     Use --displayed-low-scores to see displayed images that might be problematic")
            
    finally:
        db.close()


if __name__ == "__main__":
    main()

