import sqlite3

# Create a new SQLite database (or connect if it exists)
conn = sqlite3.connect(r"c:\Users\DELL\Desktop\aftar-daftar\sample_data.db")
cursor = conn.cursor()

# Create table
cursor.execute('''
CREATE TABLE IF NOT EXISTS sample_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    value INTEGER NOT NULL
)
''')

# Insert sample data if table is empty
cursor.execute('SELECT COUNT(*) FROM sample_data')
if cursor.fetchone()[0] == 0:
    for i in range(1, 21):
        cursor.execute("INSERT INTO sample_data (name, value) VALUES (?, ?)", (f'Name{i}', i*10))
    conn.commit()

cursor.execute('SELECT * FROM sample_data ORDER BY id LIMIT 10')
top_10_rows = cursor.fetchall()

# Print results
print("id\tname\tvalue")
for row in top_10_rows:
    print(f"{row[0]}\t{row[1]}\t{row[2]}")

conn.close()