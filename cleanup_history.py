
import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://esp32---demo-ac37f-default-rtdb.europe-west1.firebasedatabase.app"
})

def clean_history():
    print("Cleaning up live_status history...")
    users = db.reference("users").get(shallow=True) or {}
    for user_id in users.keys():
        ls_ref = db.reference(f"users/{user_id}/live_status")
        ls_data = ls_ref.get()
        if not ls_data: continue
        
        for field_id in ls_data.keys():
            # Check if history exists in ROOT history
            hist_ref = db.reference(f"users/{user_id}/history/{field_id}")
            if hist_ref.get(shallow=True):
                print(f"Deleting history from users/{user_id}/history/{field_id}...")
                hist_ref.delete()
                print("Deleted.")

if __name__ == "__main__":
    clean_history()
