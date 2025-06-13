import psycopg2
try:
    conn = psycopg2.connect(
        dbname="drpsrao3_1",
        user="drpsrao3_1_user",
        password="9494",
        host="dpg-d0o8a1gdl3ps73abg0i0-a.oregon-postgres.render.com",
        port="5432",
        sslmode="require"
    )
    print("Connected to PostgreSQL")
    conn.close()
except Exception as e:
    print(f"Error: {str(e)}")