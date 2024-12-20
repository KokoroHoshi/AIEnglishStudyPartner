import sqlite3
from typing import Tuple, Any

class RelationalDB:
    def __init__(self, db_name: str):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()

    def _get_primary_key_name(self, table_name: str) -> str:
        """Get the primary key column name for the given table."""
        self.cursor.execute(f"PRAGMA table_info({table_name});")
        columns = self.cursor.fetchall()
        for col in columns:
            if col[5] > 0:  # col[5] is the pk column indicator
                return col[1]  # col[1] is the column name
        raise ValueError(f"No primary key found for table {table_name}")

    def create_table(self, table_name: str, columns: str):
        """Create a table with the given name and columns."""
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
        self.cursor.execute(create_table_query)
        self.connection.commit()

    def insert_data(self, table_name: str, columns: str, values: Tuple):
        """Insert data into the given table."""
        placeholders = ", ".join(["?" for _ in values])
        insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        self.cursor.execute(insert_query, values)
        self.connection.commit()

    def update_data(self, table_name: str, set_clause: str, condition: str, values: Tuple):
        """Update data in the given table based on a condition."""
        update_query = f"UPDATE {table_name} SET {set_clause} WHERE {condition}"
        self.cursor.execute(update_query, values)
        self.connection.commit()

    def delete_data(self, table_name: str, condition: str, values: Tuple):
        """Delete data from the given table based on a condition."""
        delete_query = f"DELETE FROM {table_name} WHERE {condition}"
        self.cursor.execute(delete_query, values)
        self.connection.commit()

    def upsert_data(self, table_name: str, columns: str, values: Tuple):
        """Insert or update data in the given table."""
        placeholders = ", ".join(["?" for _ in values])
        insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders}) " \
                       f"ON CONFLICT(id) DO UPDATE SET {', '.join([f'{col}=excluded.{col}' for col in columns.split(', ') if col != 'id'])}"
        self.cursor.execute(insert_query, values)
        self.connection.commit()

    def exists(self, table_name: str, key_value: Any) -> bool:
        """Check if a primary key exists in the given table."""
        primary_key_name = self._get_primary_key_name(table_name)
        query = f"SELECT 1 FROM {table_name} WHERE {primary_key_name} = ? LIMIT 1"
        self.cursor.execute(query, (key_value,))
        return self.cursor.fetchone() is not None

    def get_data_by_primary_key(self, table_name: str, key_value: Any, columns: str) -> Tuple:
        """Get data from the given table based on the primary key."""
        primary_key_name = self._get_primary_key_name(table_name)
        query = f"SELECT {columns} FROM {table_name} WHERE {primary_key_name} = ?"
        self.cursor.execute(query, (key_value,))
        return self.cursor.fetchone()
    
    def update_column_by_primary_key(self, table_name: str, key_value: Any, column_name: str, new_value: Any):
        """Update a specific column for based on the primary key."""
        primary_key_name = self._get_primary_key_name(table_name)
        update_query = f"UPDATE {table_name} SET {column_name} = ? WHERE {primary_key_name} = ?"
        self.cursor.execute(update_query, (new_value, key_value))
        self.connection.commit()

    def print_table(self, table_name: str):
        """Print all data from the given table."""
        select_query = f"SELECT * FROM {table_name}"
        self.cursor.execute(select_query)
        rows = self.cursor.fetchall()
        for row in rows:
            print(row)

    def close(self):
        """Close the database connection."""
        self.connection.close()

if __name__ == "__main__":
    db = RelationalDB('test.db')

    # Creating tables based on ERD
    db.create_table('user', 'user_id TEXT PRIMARY KEY, user_name TEXT, profile_photo TEXT')
    db.create_table('user_setting', 'user_id TEXT PRIMARY KEY, current_mode TEXT, english_level TEXT, '
                                    'FOREIGN KEY(user_id) REFERENCES user(user_id)')
    db.create_table('user_notification', 'user_id TEXT PRIMARY KEY, notification_enabled INTEGER, notification_day TEXT, notification_time TEXT, '
                                         'FOREIGN KEY(user_id) REFERENCES user(user_id)')
    db.create_table('conversation_history', 'conversation_id TEXT PRIMARY KEY, speaker TEXT, conversation_data TEXT, timestamp TEXT')
    db.create_table('user_conversation_relation', 'user_id TEXT, conversation_id TEXT, '
                                                  'FOREIGN KEY(user_id) REFERENCES user(user_id), '
                                                  'FOREIGN KEY(conversation_id) REFERENCES conversation_history(conversation_id)')
    db.create_table('user_cache', 'cache_id TEXT PRIMARY KEY, cache_type TEXT, cache_data TEXT, timestamp TEXT')
    db.create_table('user_cache_relation', 'user_id TEXT, cache_id TEXT, '
                                           'FOREIGN KEY(user_id) REFERENCES user(user_id), '
                                           'FOREIGN KEY(cache_id) REFERENCES user_cache(cache_id)')

    # Example of inserting data
    db.insert_data('user', 'user_id, user_name, profile_photo', ('u1', 'Alice', 'photo1.png'))
    db.insert_data('user_setting', 'user_id, current_mode, english_level',
                   ('u1', 'Chat', 'Intermediate'))
    db.insert_data('user_notification', 'user_id, notification_enabled, notification_day, notification_time',
                    ('u1', 1, 'Monday', '08:00'))
    db.insert_data('conversation_history', 'conversation_id, speaker, conversation_data, timestamp', ('c1', 'Alice', 'Hello, how are you?', '2024-12-20 10:05:00'))
    db.insert_data('user_conversation_relation', 'user_id, conversation_id', ('u1', 'c1'))
    db.insert_data('user_cache', 'cache_id, cache_type, cache_data, timestamp', ('cache1', 'TypeA', 'Cache data here', '2024-12-20 10:07:00'))
    db.insert_data('user_cache_relation', 'user_id, cache_id', ('u1', 'cache1'))

    # Closing the database
    db.close()