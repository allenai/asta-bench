#!/usr/bin/env python3
"""
Script to migrate annotation files from old list format to new dict format.

Old format: row_annotations[row_idx] = [None, ClaimAnnotation, None, ...]
New format: row_annotations[row_idx] = {claim_idx: ClaimAnnotation, ...}
"""

import json
from pathlib import Path
import shutil
from datetime import datetime


def migrate_annotation_file(file_path: Path) -> bool:
    """
    Migrate a single annotation file from old to new format.
    Returns True if migration was performed, False if already in new format.
    """
    print(f"Processing {file_path.name}...")
    
    # Create backup
    backup_path = file_path.with_suffix('.json.bak')
    shutil.copy2(file_path, backup_path)
    print(f"  Created backup: {backup_path.name}")
    
    # Load the annotation data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Check if migration is needed
    needs_migration = False
    if 'row_annotations' in data:
        for row_idx, annotations in data['row_annotations'].items():
            if isinstance(annotations, list):
                needs_migration = True
                break
    
    if not needs_migration:
        print(f"  Already in new format, skipping migration")
        backup_path.unlink()  # Remove unnecessary backup
        return False
    
    # Perform migration
    new_row_annotations = {}
    for row_idx, annotations in data['row_annotations'].items():
        if isinstance(annotations, list):
            # Convert list to dict, skipping None values
            new_annotations = {}
            for claim_idx, claim_annotation in enumerate(annotations):
                if claim_annotation is not None:
                    new_annotations[str(claim_idx)] = claim_annotation
            new_row_annotations[row_idx] = new_annotations
        else:
            # Already a dict, keep as is
            new_row_annotations[row_idx] = annotations
    
    data['row_annotations'] = new_row_annotations
    
    # Save the migrated data
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  Migration completed successfully")
    return True


def main():
    """Migrate all annotation files in the annotations directory."""
    annotations_dir = Path(__file__).parent / "citation_eval_ui" / "annotations"
    
    if not annotations_dir.exists():
        print(f"Annotations directory not found: {annotations_dir}")
        return
    
    # Find all JSON files
    json_files = list(annotations_dir.glob("*.json"))
    
    if not json_files:
        print("No annotation files found")
        return
    
    print(f"Found {len(json_files)} annotation files")
    print("-" * 50)
    
    migrated_count = 0
    for json_file in json_files:
        if json_file.suffix == '.json' and not json_file.name.endswith('.bak'):
            if migrate_annotation_file(json_file):
                migrated_count += 1
            print()
    
    print("-" * 50)
    print(f"Migration complete: {migrated_count}/{len(json_files)} files migrated")
    print("Backup files created with .bak extension")


if __name__ == "__main__":
    main()