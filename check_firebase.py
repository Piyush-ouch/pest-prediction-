"""
Quick script to check Firebase historical_logs date-wise structure and count
"""
from firebase_admin import credentials, db
import firebase_admin

# Initialize
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://esp32---demo-ac37f-default-rtdb.europe-west1.firebasedatabase.app"
})

user_id = "vQ92EkhiW4dueXGwvEqkz08uBf43"
field_id = "field_A"

# Check historical_logs - date-wise structure
dates_ref = db.reference(f"users/{user_id}/historical_logs/{field_id}")
dates = dates_ref.get(shallow=True)

if dates:
    sorted_dates = sorted(dates.keys(), reverse=True)
    print(f"Available dates: {sorted_dates}")
    print(f"Latest date: {sorted_dates[0]}")
    
    # Fetch entries from the latest date
    latest_date = sorted_dates[0]
    logs_ref = db.reference(f"users/{user_id}/historical_logs/{field_id}/{latest_date}")
    logs = logs_ref.get()
    
    if logs:
        print(f"\nTotal entries on {latest_date}: {len(logs)}")
        print(f"\nFirst 3 entries:")
        for i, (key, value) in enumerate(list(logs.items())[:3]):
            print(f"\n  Entry {key}:")
            # Show env data
            if isinstance(value, dict) and "env" in value:
                env = value["env"]
                print(f"    env/temp: {env.get('temp')}")
                print(f"    env/hum: {env.get('hum')}")
            # Show probe data
            if isinstance(value, dict) and "probes" in value:
                probes = value["probes"]
                for probe_id, probe_data in probes.items():
                    print(f"    probes/{probe_id}: {probe_data}")
        
        # Count valid entries
        valid_count = 0
        for entry in logs.values():
            if isinstance(entry, dict) and "env" in entry:
                env = entry["env"]
                if "temp" in env and "hum" in env:
                    valid_count += 1
        
        print(f"\nValid entries with env.temp and env.hum: {valid_count}")
    else:
        print(f"No entries found for {latest_date}")
else:
    print("No historical_logs dates found!")
