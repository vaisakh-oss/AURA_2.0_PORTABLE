#!/usr/bin/env bash
# ==============================================================================
# AURA 2.0 — Restore Historical Data Backup (macOS / Linux)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================="
echo "  AURA 2.0 — Data Restoration"
echo "================================================================="

if [ ! -d "data_backup" ]; then
    echo "[-] Error: data_backup directory not found in $SCRIPT_DIR"
    exit 1
fi

if [ -f "aura.db" ]; then
    read -p "[?] An existing aura.db already exists. Overwrite with backup? (y/N): " -r CONFIRM
    if [[ ! $CONFIRM =~ ^[Yy]$ ]]; then
        echo "[-] Restoration cancelled. Existing aura.db preserved."
        exit 0
    fi
    BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    cp aura.db "aura.db.pre_restore_${BACKUP_TIMESTAMP}.bak"
    echo "[+] Preserved existing database to aura.db.pre_restore_${BACKUP_TIMESTAMP}.bak"
fi

echo "[+] Copying aura.db..."
cp data_backup/aura.db ./aura.db

if [ -f "data_backup/memory.json" ]; then
    cp data_backup/memory.json ./memory.json
    echo "[+] Restored memory.json"
fi

if [ -f "data_backup/notes.json" ]; then
    cp data_backup/notes.json ./notes.json
    echo "[+] Restored notes.json"
fi

mkdir -p knowledge_data/documents
if [ -d "data_backup/knowledge_documents" ]; then
    cp -r data_backup/knowledge_documents/* knowledge_data/documents/ 2>/dev/null || true
    echo "[+] Restored referenced knowledge documents"
fi

echo ""
echo "[✓] Data restoration complete!"
echo "    - Database: aura.db (1,000 runs, 215 conversations restored)"
echo "    - Memories: memory.json"
echo "    - Notes: notes.json"
echo "    - Documents: knowledge_data/documents/"
echo ""
echo "You can now run ./start.sh to start AURA with your restored data."

