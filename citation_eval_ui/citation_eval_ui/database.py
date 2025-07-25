"""SQLite database module for storing annotations with proper concurrency control."""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from contextlib import contextmanager
import threading
import time

from .models import FileAnnotation, ClaimAnnotation, CitationAnnotation


class AnnotationDatabase:
    """Thread-safe SQLite database for storing annotations."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,  # 30 second timeout for locks
                isolation_level='IMMEDIATE'  # Use immediate locking for writes
            )
            # Enable WAL mode for better concurrency
            self._local.conn.execute('PRAGMA journal_mode=WAL')
            self._local.conn.execute('PRAGMA synchronous=NORMAL')
            # Enable foreign keys
            self._local.conn.execute('PRAGMA foreign_keys=ON')
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    def _init_db(self):
        """Initialize database schema."""
        with self.transaction() as conn:
            # File annotations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_annotations (
                    filename TEXT PRIMARY KEY,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Claim annotations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS claim_annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    row_index INTEGER NOT NULL,
                    claim_index INTEGER NOT NULL,
                    claim_text TEXT NOT NULL,
                    missing_citations TEXT,
                    notes TEXT,
                    is_fully_supported_annotation TEXT,
                    initial_supporting TEXT,
                    initial_non_supporting TEXT,
                    initial_is_fully_supported BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(filename, row_index, claim_index),
                    FOREIGN KEY (filename) REFERENCES file_annotations(filename)
                )
            """)
            
            # Citation annotations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS citation_annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    claim_annotation_id INTEGER NOT NULL,
                    citation_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    FOREIGN KEY (claim_annotation_id) REFERENCES claim_annotations(id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for better performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_claim_annotations_lookup 
                ON claim_annotations(filename, row_index, claim_index)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_citation_annotations_claim 
                ON citation_annotations(claim_annotation_id)
            """)
    
    def save_annotation(self, filename: str, row_index: int, claim_index: int, 
                       claim_annotation: ClaimAnnotation) -> None:
        """Save or update a claim annotation atomically."""
        with self.transaction() as conn:
            # Ensure file annotation exists
            conn.execute("""
                INSERT OR REPLACE INTO file_annotations (filename, last_updated) 
                VALUES (?, CURRENT_TIMESTAMP)
            """, (filename,))
            
            # Check if claim annotation already exists
            existing = conn.execute("""
                SELECT id FROM claim_annotations 
                WHERE filename = ? AND row_index = ? AND claim_index = ?
            """, (filename, row_index, claim_index)).fetchone()
            
            # Prepare JSON arrays for initial values
            initial_supporting = json.dumps(claim_annotation.initial_supporting)
            initial_non_supporting = json.dumps(claim_annotation.initial_non_supporting)
            
            if existing:
                # Update existing annotation
                claim_id = existing['id']
                conn.execute("""
                    UPDATE claim_annotations 
                    SET claim_text = ?, missing_citations = ?, notes = ?, 
                        is_fully_supported_annotation = ?, initial_supporting = ?,
                        initial_non_supporting = ?, initial_is_fully_supported = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    claim_annotation.claim_text,
                    claim_annotation.missing_citations,
                    claim_annotation.notes,
                    claim_annotation.is_fully_supported_annotation,
                    initial_supporting,
                    initial_non_supporting,
                    claim_annotation.initial_is_fully_supported,
                    claim_id
                ))
                
                # Delete existing citation annotations
                conn.execute("DELETE FROM citation_annotations WHERE claim_annotation_id = ?", (claim_id,))
            else:
                # Insert new annotation
                cursor = conn.execute("""
                    INSERT INTO claim_annotations (
                        filename, row_index, claim_index, claim_text, missing_citations,
                        notes, is_fully_supported_annotation, initial_supporting,
                        initial_non_supporting, initial_is_fully_supported
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    filename, row_index, claim_index,
                    claim_annotation.claim_text,
                    claim_annotation.missing_citations,
                    claim_annotation.notes,
                    claim_annotation.is_fully_supported_annotation,
                    initial_supporting,
                    initial_non_supporting,
                    claim_annotation.initial_is_fully_supported
                ))
                claim_id = cursor.lastrowid
            
            # Insert citation annotations
            for citation_ann in claim_annotation.citation_annotations:
                conn.execute("""
                    INSERT INTO citation_annotations (claim_annotation_id, citation_id, label)
                    VALUES (?, ?, ?)
                """, (claim_id, citation_ann.citation_id, citation_ann.label))
    
    def load_annotations(self, filename: str) -> FileAnnotation:
        """Load all annotations for a file."""
        conn = self._get_connection()
        
        # Get file annotation
        file_row = conn.execute("""
            SELECT last_updated FROM file_annotations WHERE filename = ?
        """, (filename,)).fetchone()
        
        if not file_row:
            # No annotations exist yet
            return FileAnnotation(
                filename=filename,
                timestamp=datetime.now().isoformat(),
                row_annotations={}
            )
        
        # Get all claim annotations for this file
        claim_rows = conn.execute("""
            SELECT * FROM claim_annotations 
            WHERE filename = ? 
            ORDER BY row_index, claim_index
        """, (filename,)).fetchall()
        
        row_annotations: Dict[int, Dict[int, ClaimAnnotation]] = {}
        
        for claim_row in claim_rows:
            # Get citation annotations for this claim
            citation_rows = conn.execute("""
                SELECT citation_id, label FROM citation_annotations
                WHERE claim_annotation_id = ?
            """, (claim_row['id'],)).fetchall()
            
            citation_annotations = [
                CitationAnnotation(citation_id=row['citation_id'], label=row['label'])
                for row in citation_rows
            ]
            
            # Parse JSON arrays
            initial_supporting = json.loads(claim_row['initial_supporting'] or '[]')
            initial_non_supporting = json.loads(claim_row['initial_non_supporting'] or '[]')
            
            claim_annotation = ClaimAnnotation(
                claim_index=claim_row['claim_index'],
                claim_text=claim_row['claim_text'],
                citation_annotations=citation_annotations,
                missing_citations=claim_row['missing_citations'],
                notes=claim_row['notes'],
                is_fully_supported_annotation=claim_row['is_fully_supported_annotation'],
                initial_supporting=initial_supporting,
                initial_non_supporting=initial_non_supporting,
                initial_is_fully_supported=bool(claim_row['initial_is_fully_supported'])
            )
            
            row_idx = claim_row['row_index']
            if row_idx not in row_annotations:
                row_annotations[row_idx] = {}
            
            row_annotations[row_idx][claim_row['claim_index']] = claim_annotation
        
        return FileAnnotation(
            filename=filename,
            timestamp=file_row['last_updated'],
            row_annotations=row_annotations
        )
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None