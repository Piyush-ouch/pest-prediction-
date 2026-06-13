"""
Debug script to see exactly what the last 47 entries look like
"""
from firebase_admin import credentials, db
import firebase_admin

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://esp32---demo-ac37f-default-rtdb.europe-west1.firebasedatabase.app"
})

user_id = "vQ92EkhiW4dueXGwvEqkz08uBf43"
field_id = "field_A"

# Fetch exactly like the app does
logs_ref = db.reference(f"users/{user_id}/historical_logs/{field_id}")
logs = logs_ref.order_by_key().limit_to_last(47).get()

print(f"Fetched {len(logs)} entries")
print("\nChecking each entry:")

valid_count = 0
for i, (key, entry) in enumerate(logs.items()):
    temp = None
    hum = None
    
    # Try new format
    if isinstance(entry, dict) and "env" in entry:
        env = entry["env"]
        temp = env.get("temp")
        hum = env.get("hum")
        if temp and hum:
            valid_count += 1
            print(f"  Entry {i+1}: NEW format - temp={temp}, hum={hum}")
    
    # Try old format
    elif isinstance(entry, dict) and "environment" in entry:
        environment = entry["environment"]
        temp = environment.get("temperature")
        hum = environment.get("humidity")
        if temp and hum:
            valid_count += 1
            print(f"  Entry {i+1}: OLD format - temp={temp}, hum={hum}")
    else:
        print(f"  Entry {i+1}: INVALID - {entry}")

print(f"\nValid entries: {valid_count}/47")
print(f"With current live data: {valid_count + 1}/48")
