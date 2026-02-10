#!/usr/bin/env python
"""Check Docker image availability for SWE-bench_Verified dataset."""

import argparse
import sys
from collections import defaultdict

import pandas as pd

try:
    from ..utils.docker_utils import build_image_name, check_docker_image_exists
except ImportError:
    from swe_agent.utils.docker_utils import build_image_name, check_docker_image_exists


def main():
    parser = argparse.ArgumentParser(
        description="Check Docker image availability for SWE-bench_Verified instances"
    )
    parser.add_argument(
        "parquet_path",
        help="Path to SWE-bench_Verified parquet file",
    )
    parser.add_argument(
        "--arch",
        default="x86_64",
        help="Docker architecture (default: x86_64)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-instance results",
    )
    args = parser.parse_args()

    print(f"Loading data from: {args.parquet_path}")
    df = pd.read_parquet(args.parquet_path)
    total_instances = len(df)

    print(f"Total instances: {total_instances}")
    print("Checking Docker images...\n")

    available = []
    missing = []
    repo_stats = defaultdict(lambda: {"total": 0, "available": 0})

    for idx, row in df.iterrows():
        instance_id = row["instance_id"]
        repo = row.get("repo", "unknown")
        image_name = build_image_name(instance_id, arch=args.arch)

        repo_stats[repo]["total"] += 1

        if check_docker_image_exists(image_name):
            available.append((instance_id, image_name))
            repo_stats[repo]["available"] += 1
            if args.verbose:
                print(f"✓ {instance_id}: {image_name}")
        else:
            missing.append((instance_id, image_name))
            if args.verbose:
                print(f"✗ {instance_id}: {image_name} (MISSING)")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total instances: {total_instances}")
    print(f"Available images: {len(available)} ({len(available)/total_instances*100:.1f}%)")
    print(f"Missing images: {len(missing)} ({len(missing)/total_instances*100:.1f}%)")

    print("\n" + "="*70)
    print("PER-REPO BREAKDOWN")
    print("="*70)
    for repo in sorted(repo_stats.keys()):
        stats = repo_stats[repo]
        coverage = stats["available"] / stats["total"] * 100
        print(f"{repo:40s} {stats['available']:3d}/{stats['total']:3d} ({coverage:5.1f}%)")

    # Print missing instances if not verbose
    if not args.verbose and missing:
        print("\n" + "="*70)
        print(f"MISSING INSTANCES (showing first 10 of {len(missing)})")
        print("="*70)
        for instance_id, image_name in missing[:10]:
            print(f"  {instance_id}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    if len(missing) > 0:
        print("• Use --skip_missing_images flag in prepare_data.py")
        print("• Or build missing images using SWE-bench docker_build.py")
    else:
        print("• All images available! Ready for training.")

    # Exit code
    sys.exit(0 if len(available) > 0 else 1)


if __name__ == "__main__":
    main()
