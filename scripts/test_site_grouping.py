#!/usr/bin/env python3
"""
Test script to verify site grouping logic.
"""

def get_base_site(site: str) -> str:
    """
    Get the base site name, removing suffixes like '-2'.
    
    Examples:
        'A4-2' -> 'A4'
        'A4' -> 'A4'
        'H9-2' -> 'H9'
    """
    if '-' in site and site.split('-')[-1].isdigit():
        return site.rsplit('-', 1)[0]
    return site


# Test cases
test_sites = [
    'A4',
    'A4-2',
    'H9',
    'H9-2',
    'G2-2',
    'Site-Name',  # Should not be split
    'Site-Name-2',  # Should be -> 'Site-Name'
    'ABC',
]

print("Site grouping test:")
print("-" * 50)
for site in test_sites:
    base = get_base_site(site)
    print(f"{site:20} -> {base}")

# Show grouping
from collections import defaultdict
groups = defaultdict(list)
for site in test_sites:
    base = get_base_site(site)
    groups[base].append(site)

print("\n" + "=" * 50)
print("Site groups:")
print("=" * 50)
for base, variants in sorted(groups.items()):
    print(f"{base}: {variants}")
