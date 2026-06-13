"""
Script to populate Firebase historical_logs with mock data to reach 48 entries.
This allows the ML prediction model to start working immediately.
"""

import firebase_admin
from firebase_admin import credentials, db
import time
import random

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://esp32---demo-ac37f-default-rtdb.europe-west1.firebasedatabase.app"
})

def generate_realistic_data():
    """Generate realistic temperature and humidity values."""
    # Realistic ranges for agricultural fields
    temp = round(random.uniform(20, 35), 1)  # 20-35°C
    hum = round(random.uniform(40, 85), 1)   # 40-85% humidity
    return temp, hum

def add_mock_historical_data(user_id, field_id, num_entries=48):
    """
    Add mock historical data to Firebase.
    
    Args:
        user_id: Firebase user ID
        field_id: Field identifier
        num_entries: Number of entries to add (default 48)
    """
    print(f"Adding {num_entries} mock entries to historical_logs...")
    
    # Reference to historical_logs
    logs_ref = db.reference(f"users/{user_id}/historical_logs/{field_id}")
    
    # Check existing entries
    existing = logs_ref.get()
    existing_count = len(existing) if existing else 0
    print(f"Found {existing_count} existing entries")
    
    # Calculate how many we need to add
    needed = num_entries - existing_count
    if needed <= 0:
        print(f"✅ Already have {existing_count} entries. No mock data needed!")
        return
    
    print(f"Need to add {needed} more entries...")
    
    # Current timestamp in milliseconds
    current_time = int(time.time() * 1000)
    
    # Add entries going backwards in time (1 hour intervals)
    hour_in_ms = 3600 * 1000
    
    for i in range(needed):
        # Timestamp going backwards from current time
        timestamp = current_time - (hour_in_ms * (needed - i))
        
        # Generate realistic data
        temp, hum = generate_realistic_data()
        
        # Create entry
        entry = {
            "env": {
                "temp": temp,
                "hum": hum
            },
            "timestamp": timestamp
        }
        
        # Write to Firebase
        logs_ref.child(str(timestamp)).set(entry)
        print(f"  Added entry {i+1}/{needed}: temp={temp}°C, hum={hum}%")
    
    print(f"\n✅ Successfully added {needed} mock entries!")
    print(f"Total entries now: {existing_count + needed}")
    print("\nYour ML predictions should work now! 🎉")

if __name__ == "__main__":
    # Configuration
    USER_ID = "vQ92EkhiW4dueXGwvEqkz08uBf43"
    FIELD_ID = "field_A"
    
    print("=" * 60)
    print("Mock Data Generator for Pest Prediction System")
    print("=" * 60)
    print(f"\nUser ID: {USER_ID}")
    print(f"Field ID: {FIELD_ID}")
    print()
    
    add_mock_historical_data(USER_ID, FIELD_ID, num_entries=48)
    
    print("\n" + "=" * 60)
    print("Done! You can now run your prediction backend.")
    print("=" * 60)
