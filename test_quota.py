import os
import unittest
from quota_manager import QuotaManager

class TestQuotaManager(unittest.TestCase):
    def setUp(self):
        self.db_path = "/Users/tge/Documents/ai_audio_detector/test_quota.db"
        # Clean up any leftover database
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.quota_manager = QuotaManager(db_path=self.db_path)

    def tearDown(self):
        # Clean up database file after test
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_daily_limit_enforcement(self):
        user_email = "test_user@example.com"
        limit = 10
        
        # 1. Run 10 sequential successful queries
        for i in range(1, limit + 1):
            allowed, count, remaining = self.quota_manager.consume_quota(user_email, limit=limit)
            self.assertTrue(allowed, f"Query {i} should be allowed.")
            self.assertEqual(count, i, f"Count should be {i} after query {i}.")
            self.assertEqual(remaining, limit - i, f"Remaining should be {limit - i} after query {i}.")
            
        # 2. Run the 11th query, which should be rejected
        allowed, count, remaining = self.quota_manager.consume_quota(user_email, limit=limit)
        self.assertFalse(allowed, "Query 11 should be rejected.")
        self.assertEqual(count, 10, "Count should remain capped at 10.")
        self.assertEqual(remaining, 0, "Remaining should be 0.")

        # 3. Check status reporting without decrementing
        current_count, current_remaining = self.quota_manager.get_quota_status(user_email, limit=limit)
        self.assertEqual(current_count, 10, "Status count should be 10.")
        self.assertEqual(current_remaining, 0, "Status remaining should be 0.")

if __name__ == "__main__":
    unittest.main()
