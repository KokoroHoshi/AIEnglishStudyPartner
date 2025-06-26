import sqlite3
from pathlib import Path
from typing import Tuple, Any

class RelationalDB:
    def __init__(self, db_name: str):
        self.connection = sqlite3.connect(Path(db_name))
        self.cursor = self.connection.cursor()

    def _get_primary_key_name(self, table_name: str) -> str:
        """Get the primary key column name for the given table."""
        self.cursor.execute(f"PRAGMA table_info({table_name});")
        columns = self.cursor.fetchall()
        for col in columns:
            if col[5] > 0:  # col[5] is the pk column indicator
                return col[1]  # col[1] is the column name
        raise ValueError(f"No primary key found for table {table_name}")

    def _get_table_columns(self, table_name: str):
        """Get the list of column names for a given table."""
        query = f"PRAGMA table_info({table_name})"
        self.cursor.execute(query)
        columns = self.cursor.fetchall()
        return [column[1] for column in columns]  # column[1] is the column name in SQLite
    
    def _validate_column(self, table_name: str, column: str) -> bool:
        """Validate that the column exists in the table to prevent SQL injection."""
        allowed_columns = self._get_table_columns(table_name)
        return column in allowed_columns

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
        """
        Update data in the specified table based on a given condition. The method ensures that column names are validated 
        and SQL injection is prevented by using parameterized queries.

        Arguments:
        table_name (str): The name of the table in which the data will be updated.
        set_clause (str): A comma-separated list of column-value pairs to update. For example: "level = ?, notification_enabled = ?".
        condition (str): A condition to determine which rows to update. This could be something like "user_id = ?" or "email = ?".
        values (Tuple): A tuple containing the values to be used for the placeholders in the query. The order of values should 
                        correspond to the placeholders in `set_clause` and `condition`.

        Usage:
        This method uses parameterized queries (with `?` placeholders) to protect against SQL injection. 
        The column names are validated against a list of allowed columns for the given table, and the condition is 
        assumed to be a valid SQL expression. The values are passed separately to ensure safe execution.

        Example:
        db.update_data(
            table_name="user_notification",
            set_clause="notification_enabled = ?, notifacation_day = ?",
            condition="user_id = ?",
            values=(True, "0111110", "12345")
        )
        This would produce the query: 
        "UPDATE user_notification SET notification_enabled = ?, notifacation_day = ? WHERE user_id = ?"

        Safe Usage Considerations:
        1. **Column Validation**: This method validates the column names in the `set_clause` to ensure they are allowed columns 
        for the given table. This helps to prevent SQL injection via unauthorized column names.
        2. **Parameterization**: Values are safely passed through placeholders (`?`), preventing any malicious input from 
        modifying the structure of the SQL query.
        3. **Condition Handling**: The `condition` argument should be a valid SQL expression. It is assumed that the caller 
        has ensured its correctness. 
        4. **No Direct String Concatenation for Values**: Values are passed in the `values` tuple and not directly concatenated 
        into the SQL query, thus protecting against SQL injection.

        Notes:
        - If you need additional validation (e.g., check if `condition` contains only basic SQL operators), it should be done 
        outside this method, based on your application logic.
        - This method is designed for updates that involve a condition (e.g., updating based on a `user_id` or `email`).
        """

        # Step 1: Split the set_clause into individual column-value pairs (for example, "level = ?")
        set_columns = set_clause.split(",")  # Split by comma to separate multiple columns
        valid_columns = []

        # Step 2: Validate each column name in the set_clause
        for column_value in set_columns:
            column_name, _ = column_value.split("=")  # Extract the column name from "column_name = ?"
            column_name = column_name.strip()  # Clean any extra spaces around the column name
            
            # Validate the column name against allowed columns for the specific table
            if not self.validate_column(table_name, column_name):
                raise ValueError(f"Invalid column name: {column_name}")  # Raise an error if invalid column
            
            valid_columns.append(column_name)  # Collect valid column names

        # Step 3: Construct the final SET clause for the query
        set_clause_final = ", ".join([f"{col} = ?" for col in valid_columns])  # Prepare placeholders for values

        # Step 4: Safely execute the query using parameterized values
        update_query = f"UPDATE {table_name} SET {set_clause_final} WHERE {condition}"

        # Using parameterized queries to avoid SQL injection
        self.cursor.execute(update_query, values)  # `values` are safely passed as parameters
        self.connection.commit()  # Commit the changes to the database

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
        """Update a specific column based on the primary key."""
        try:
            # Validate column name to avoid SQL injection
            if not self._validate_column(table_name, column_name):
                raise ValueError(f"Invalid column name: {column_name}")

            primary_key_name = self._get_primary_key_name(table_name)

            # Prepare and execute the update query
            update_query = f"UPDATE {table_name} SET {column_name} = ? WHERE {primary_key_name} = ?"
            self.cursor.execute(update_query, (new_value, key_value))
            self.connection.commit()
        
        except ValueError as e:
            # If ValueError is raised (invalid column name), log the error or re-raise it
            raise ValueError(f"Column validation failed: {str(e)}")
        except Exception as e:
            # Catch any unexpected exceptions and re-raise them
            raise Exception(f"An unexpected error occurred: {str(e)}")

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

    def get_all_users(self) -> list:
        """
        Get a list of all user_id and user_name from the 'user' table.
        """
        query = """
            SELECT user_id, user_name
            FROM user;
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()
    
    def get_users_for_notification(self, current_weekday: int) -> list:
        """
        Retrieve user_ids from the 'user_notification' table where notifications are enabled
        for the given weekday.

        :param current_weekday: Current day of the week (0 = Sunday, 6 = Saturday)
        :return: List of user_ids that should be notified
        """
        # Convert current_weekday to index for the day string, add 1 because SQL uses 1-based indexing
        day_index = current_weekday + 1
        
        query = """
            SELECT user_id
            FROM user_notification
            WHERE notification_enabled = 'true'
              AND SUBSTR(notification_days, ?, 1) = '1';
        """
        self.cursor.execute(query, (str(day_index),))
        return [row[0] for row in self.cursor.fetchall()]

    def get_users_for_notification(self, current_weekday: int, current_hour: int, current_minute: int) -> list:
        """
        Retrieve user_ids from the 'user_notification' table where notifications are enabled
        for the given weekday, hour, and minute.

        :param current_weekday: Current day of the week (0 = Sunday, 6 = Saturday)
        :param current_hour: Current hour in 24-hour format
        :param current_minute: Current minute
        :return: List of user_ids that should be notified
        """
        # Convert current_weekday to index for the day string
        day_index = current_weekday + 1  # Add 1 because SQL uses 1-based indexing
        # Format time string for matching in the database
        time_str = f"{current_hour:02d}:{current_minute:02d}"

        query = """
            SELECT user_id
            FROM user_notification
            WHERE notification_enabled = 'True'
              AND SUBSTR(notification_days, ?, 1) = '1'
              AND notification_time = ?;
        """
        self.cursor.execute(query, (str(day_index), time_str))
        return [row[0] for row in self.cursor.fetchall()]

    def get_conversations_by_user(self, user_id: str, max_history_length: int) -> list:
            """
            Get the latest 'max_history_length' conversations related to a specific user, ordered from old to new.
            Return (speaker, conversation_data)
            """
            query = """
                SELECT speaker, conversation_data
                FROM (
                    SELECT c.speaker, c.conversation_data, c.timestamp
                    FROM conversation_history c
                    JOIN user_conversation_relation ucr ON c.conversation_id = ucr.conversation_id
                    WHERE ucr.user_id = ?
                    ORDER BY c.timestamp DESC
                    LIMIT ?
                ) AS subquery
                ORDER BY subquery.timestamp ASC;
            """
            self.cursor.execute(query, (user_id, max_history_length))
            return self.cursor.fetchall()


if __name__ == "__main__":
    db = RelationalDB('test.db')

    # Creating tables based on ERD
    db.create_table('user', 'user_id TEXT PRIMARY KEY, user_name TEXT, profile_photo TEXT')
    db.create_table('user_settings', 'user_id TEXT PRIMARY KEY, current_mode TEXT, english_level TEXT, '
                                    'FOREIGN KEY(user_id) REFERENCES user(user_id)')
    db.create_table('user_notification', 'user_id TEXT PRIMARY KEY, notification_enabled TEXT, notification_days TEXT, notification_time TEXT, '
                                         'FOREIGN KEY(user_id) REFERENCES user(user_id)')
    db.create_table('conversation_history', 'conversation_id TEXT PRIMARY KEY, speaker TEXT, conversation_data TEXT, timestamp TEXT')
    db.create_table('user_conversation_relation', 'user_id TEXT, conversation_id TEXT, '
                                                  'FOREIGN KEY(user_id) REFERENCES user(user_id), '
                                                  'FOREIGN KEY(conversation_id) REFERENCES conversation_history(conversation_id)')
    db.create_table('user_cache', 'cache_id TEXT PRIMARY KEY, cache_type TEXT, cache_data BLOB, timestamp TEXT')
    db.create_table('user_cache_relation', 'user_id TEXT, cache_id TEXT, '
                                           'FOREIGN KEY(user_id) REFERENCES user(user_id), '
                                           'FOREIGN KEY(cache_id) REFERENCES user_cache(cache_id)')

    # Example of inserting data
    db.insert_data('user', 'user_id, user_name, profile_photo', ('u1', 'Alice', 'photo1.png'))
    db.insert_data('user', 'user_id, user_name, profile_photo', ('u2', 'Bob', 'photo2.png'))
    db.insert_data('user_settings', 'user_id, current_mode, english_level',
                   ('u1', 'Chat', 'Intermediate'))
    db.insert_data('user_settings', 'user_id, current_mode, english_level',
                   ('u2', 'Chat', 'Intermediate'))
    db.insert_data('user_notification', 'user_id, notification_enabled, notification_days, notification_time',
                    ('u1', 'True', 'Monday', '08:00'))
    db.insert_data('user_notification', 'user_id, notification_enabled, notification_days, notification_time',
                    ('u2', 'False', 'Monday', '08:00'))
    db.insert_data('conversation_history', 'conversation_id, speaker, conversation_data, timestamp', ('c1', 'Alice', 'Hello, how are you?', '2024-12-20 06:06:00'))
    db.insert_data('user_conversation_relation', 'user_id, conversation_id', ('u1', 'c1'))
    db.insert_data('conversation_history', 'conversation_id, speaker, conversation_data, timestamp', ('c2', 'AI', 'Hi, I\'m good. How about you?', '2024-12-20 06:07:00'))
    db.insert_data('user_conversation_relation', 'user_id, conversation_id', ('u1', 'c2'))
    db.insert_data('conversation_history', 'conversation_id, speaker, conversation_data, timestamp', ('c3', 'Bob', 'Hi?', '2024-12-20 06:10:00'))
    db.insert_data('user_conversation_relation', 'user_id, conversation_id', ('u2', 'c3'))
    db.insert_data('conversation_history', 'conversation_id, speaker, conversation_data, timestamp', ('c4', 'Alice', 'I\'m happy because we have a conversation now.', '2024-12-20 06:16:00'))
    db.insert_data('user_conversation_relation', 'user_id, conversation_id', ('u1', 'c4'))
    # change cache_data TEXT to BLOB so cannot use this example
    # db.insert_data('user_cache', 'cache_id, cache_type, cache_data, timestamp', ('cache1', 'TypeA', 'Cache data here', '2024-12-20 10:07:00'))
    db.insert_data('user_cache_relation', 'user_id, cache_id', ('u1', 'cache1'))

    # Select test
    print(db.get_conversations_by_user('u1'))

    # Closing the database
    db.close()