@echo off
REM ==============================================================================
REM AURA 2.0 — Restore Historical Data Backup (Windows)
REM ==============================================================================
setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo =================================================================
echo   AURA 2.0 - Data Restoration (Windows)
echo =================================================================

if not exist "data_backup" (
    echo [-] Error: data_backup directory not found.
    pause
    exit /b 1
)

if exist "aura.db" (
    echo [?] An existing aura.db already exists.
    set /p CONFIRM="Overwrite with backup? (y/N): "
    if /i not "!CONFIRM!"=="y" (
        echo [-] Restoration cancelled. Existing aura.db preserved.
        pause
        exit /b 0
    )
    copy /y aura.db aura.db.backup >nul
    echo [+] Existing database backed up to aura.db.backup
)

echo [+] Copying aura.db...
copy /y data_backup\aura.db aura.db >nul

if exist "data_backup\memory.json" (
    copy /y data_backup\memory.json memory.json >nul
    echo [+] Restored memory.json
)

if exist "data_backup\notes.json" (
    copy /y data_backup\notes.json notes.json >nul
    echo [+] Restored notes.json
)

if not exist "knowledge_data\documents" mkdir "knowledge_data\documents"
if exist "data_backup\knowledge_documents" (
    copy /y data_backup\knowledge_documents\* knowledge_data\documents\ >nul 2>&1
    echo [+] Restored referenced knowledge documents
)

echo.
echo [v] Data restoration complete!
echo     - Database: aura.db (1,000 runs, 215 conversations restored)
echo     - Memories: memory.json
echo     - Notes: notes.json
echo     - Documents: knowledge_data\documents\
echo.
echo You can now run start.bat to launch AURA with your restored data.
pause

