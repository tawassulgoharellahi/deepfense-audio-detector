import sqlite3
import datetime
import os

class QuotaManager:
    def __init__(self, db_path=None):
        if db_path is None:
            # Resolve db path relative to this file's directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(base_dir, "quota.db")
        else:
            self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        # Ensure parent directories exist
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scan_quota (
                    identifier TEXT,
                    scan_date TEXT,
                    scan_count INTEGER DEFAULT 0,
                    PRIMARY KEY (identifier, scan_date)
                )
            """)
            conn.commit()
            
    def consume_quota(self, identifier: str, limit: int = 10) -> tuple[bool, int, int]:
        """
        Atomically checks and increments user quota for the current date.
        Returns: (is_allowed, current_count, remaining_scans)
        """
        scan_date = datetime.date.today().isoformat()  # YYYY-MM-DD
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Select current count
            cursor.execute(
                "SELECT scan_count FROM scan_quota WHERE identifier = ? AND scan_date = ?",
                (identifier, scan_date)
            )
            row = cursor.fetchone()
            
            if row:
                current_count = row["scan_count"]
            else:
                current_count = 0
                
            if current_count >= limit:
                return False, current_count, 0
                
            # Increment count
            new_count = current_count + 1
            cursor.execute(
                """
                INSERT INTO scan_quota (identifier, scan_date, scan_count)
                VALUES (?, ?, ?)
                ON CONFLICT(identifier, scan_date) DO UPDATE SET scan_count = ?
                """,
                (identifier, scan_date, new_count, new_count)
            )
            conn.commit()
            
            remaining = limit - new_count
            return True, new_count, remaining

    def get_quota_status(self, identifier: str, limit: int = 10) -> tuple[int, int]:
        """
        Returns (current_count, remaining) without incrementing.
        """
        scan_date = datetime.date.today().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT scan_count FROM scan_quota WHERE identifier = ? AND scan_date = ?",
                (identifier, scan_date)
            )
            row = cursor.fetchone()
            current_count = row[0] if row else 0
            remaining = max(0, limit - current_count)
            return current_count, remaining
