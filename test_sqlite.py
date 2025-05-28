import psycopg2
try:
    conn = psycopg2.connect(
        dbname="drpsrao3_1",
        user="drpsrao3_1_user",
        password="UpKaDI52KyFuzbQGxB8CocKOHo3c3kXh",
        host="dpg-d0o8a1gdl3ps73abg0i0-a.oregon-postgres.render.com",
        port="5432"
    )
    print("Connected to PostgreSQL")
    conn.close()
except Exception as e:
    print(f"Error: {str(e)}")