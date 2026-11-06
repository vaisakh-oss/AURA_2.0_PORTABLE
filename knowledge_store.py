import datetime
import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_FILE = "aura.db"
KNOWLEDGE_DIR = Path("knowledge_data").resolve()
DOCUMENTS_DIR = (KNOWLEDGE_DIR / "documents").resolve()


def get_db_connection(db_path: Optional[str] = None, allow_admin_ddl: bool = False) -> sqlite3.Connection:
    """Connect to SQLite database with timeout, dict row factory, and least-privilege security authorizer."""
    try:
        import storage
        return storage.get_db_connection(db_path=db_path, allow_admin_ddl=allow_admin_ddl)
    except ImportError:
        target_db = db_path if db_path is not None else (os.environ.get("AURA_DB_PATH") or DB_FILE)
        conn = sqlite3.connect(target_db, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn


def get_knowledge_dir() -> Path:
    """Ensure knowledge data directories exist and return base path."""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    return KNOWLEDGE_DIR


def initialize_knowledge_tables(db_path: Optional[str] = None) -> None:
    """
    Initialize SQLite knowledge workspace tables: documents & document_chunks.
    Safe migration if tables do not exist.
    """
    get_knowledge_dir()
    target_db = db_path if db_path is not None else (os.environ.get("AURA_DB_PATH") or DB_FILE)
    with get_db_connection(target_db, allow_admin_ddl=True) as conn:
        cursor = conn.cursor()

        # Documents metadata table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            content_hash TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        );
        """)

        # Chunks and embeddings table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL,
            page_number INTEGER,
            metadata_json TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        );
        """)

        conn.commit()


def get_document_by_hash(content_hash: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve document metadata by content SHA-256 hash for deduplication."""
    initialize_knowledge_tables(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, filename, file_type, file_size, content_hash, created_at FROM documents WHERE content_hash = ?", (content_hash,))
        row = cursor.fetchone()
        return dict(row) if row else None


def add_document(filename: str, file_type: str, file_size: int, content_hash: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Register a new document entry in SQLite."""
    initialize_knowledge_tables(db_path)
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO documents (id, filename, file_type, file_size, content_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (doc_id, filename, file_type, file_size, content_hash, now_iso)
        )
        conn.commit()

    return {
        "id": doc_id,
        "filename": filename,
        "file_type": file_type,
        "file_size": file_size,
        "content_hash": content_hash,
        "created_at": now_iso
    }


def add_chunks(document_id: str, chunks: List[Dict[str, Any]], db_path: Optional[str] = None) -> int:
    """Store document chunks and serialized vector embeddings into SQLite."""
    initialize_knowledge_tables(db_path)
    if not chunks:
        return 0

    inserted_count = 0
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        for idx, chunk in enumerate(chunks):
            embedding_data = chunk.get("embedding", [])
            embedding_json = json.dumps(embedding_data) if isinstance(embedding_data, list) else str(embedding_data)
            meta_json = json.dumps(chunk.get("metadata", {}))

            cursor.execute(
                """
                INSERT INTO document_chunks (document_id, chunk_index, text, embedding, page_number, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    chunk.get("chunk_index", idx),
                    chunk.get("text", ""),
                    embedding_json,
                    chunk.get("page_number"),
                    meta_json
                )
            )
            inserted_count += 1
        conn.commit()

    return inserted_count


def list_documents(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all knowledge documents with chunk counts and formatted metadata."""
    initialize_knowledge_tables(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT d.id, d.filename, d.file_type, d.file_size, d.content_hash, d.created_at,
                   COUNT(c.id) as chunk_count
            FROM documents d
            LEFT JOIN document_chunks c ON d.id = c.document_id
            GROUP BY d.id
            ORDER BY d.created_at DESC
        """)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_document(document_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve document metadata by ID."""
    initialize_knowledge_tables(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, filename, file_type, file_size, content_hash, created_at FROM documents WHERE id = ?", (document_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def delete_document(document_id: str, db_path: Optional[str] = None) -> bool:
    """Delete a document and all associated chunks from SQLite and storage directory."""
    initialize_knowledge_tables(db_path)
    doc = get_document(document_id, db_path=db_path)
    if not doc:
        return False

    # Safe deletion of local file if saved
    safe_filename = Path(doc["filename"]).name
    file_path = DOCUMENTS_DIR / f"{document_id}_{safe_filename}"
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception:
            pass

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
        cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        conn.commit()
        return cursor.rowcount > 0


def clear_knowledge(db_path: Optional[str] = None) -> int:
    """Clear all documents and chunks from knowledge base."""
    initialize_knowledge_tables(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM document_chunks")
        cursor.execute("DELETE FROM documents")
        count = cursor.rowcount
        conn.commit()

    if DOCUMENTS_DIR.exists():
        for item in DOCUMENTS_DIR.glob("*"):
            if item.is_file():
                try:
                    item.unlink()
                except Exception:
                    pass

    return count


def get_all_chunks(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all chunks with document filename & page metadata for similarity search."""
    initialize_knowledge_tables(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.id, c.document_id, c.chunk_index, c.text, c.embedding, c.page_number, c.metadata_json,
                   d.filename, d.file_type
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
        """)
        rows = cursor.fetchall()
        chunks = []
        for row in rows:
            d = dict(row)
            try:
                d["embedding"] = json.loads(d["embedding"])
            except Exception:
                d["embedding"] = []
            try:
                d["metadata"] = json.loads(d["metadata_json"]) if d["metadata_json"] else {}
            except Exception:
                d["metadata"] = {}
            chunks.append(d)
        return chunks
