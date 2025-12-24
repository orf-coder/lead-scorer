import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
import os
import json

# Lazy-import Google libraries so module can be imported even when they are
# not available (e.g., on Streamlit Community Cloud when not needed).
GOOGLE_LIBS_AVAILABLE = True
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
except Exception:
    GOOGLE_LIBS_AVAILABLE = False

# Scopes for Google Sheets API
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# Path to credentials file
SERVICE_ACCOUNT_FILE = os.path.join(os.path.dirname(__file__), 'credentials.json')

# Initialize defaults
SERVICE_ACCOUNT_EMAIL = None
PROJECT_ID = None
service = None

# Attempt to load credentials and build service only if google libs are present
if GOOGLE_LIBS_AVAILABLE:
    try:
        if os.path.exists(SERVICE_ACCOUNT_FILE):
            with open(SERVICE_ACCOUNT_FILE, 'r') as f:
                creds_info = json.load(f)
            SERVICE_ACCOUNT_EMAIL = creds_info.get('client_email')
            PROJECT_ID = creds_info.get('project_id')
            try:
                creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
                service = build('sheets', 'v4', credentials=creds)
                print(f"Google Sheets service initialized for {SERVICE_ACCOUNT_EMAIL}")
            except Exception as e:
                print(f"Failed to initialize Google Sheets service: {e}")
                service = None
        else:
            print(f"Google credentials file not found at {SERVICE_ACCOUNT_FILE}; Google Sheets disabled")
    except Exception as e:
        print(f"Error reading Google credentials: {e}")
        service = None
else:
    print("Google API libraries not available; Google Sheets integration disabled")

def append_entry_to_sheet(name, company, job_title, source, message, label):
    Appends a new entry (name, company, job_title, source, message, label) to the Google Sheet.
    This function can be called whenever a new entry is saved to the training data.
    """
    if service is None:
        print("Google Sheets service not initialized")
        return None

    try:
        # Check if the sheet has headers
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range='Sheet1!A1:F1'
        ).execute()
        values = result.get('values', [])
        if not values or not values[0]:
            # Sheet is empty, add headers first
            header_body = {
                'values': [['Name', 'Company', 'Job Title', 'Source', 'Message', 'Label']]
            }
            service.spreadsheets().values().update(
                spreadsheetId=SPREADSHEET_ID,
                range='Sheet1!A1:F1',
                valueInputOption='RAW',
                body=header_body
            ).execute()
            print("Added headers to Google Sheet")
    except Exception as e:
        print(f"Error checking/adding headers to Google Sheet: {e}")
        print(f"Make sure the sheet '{SPREADSHEET_ID}' exists and is shared with {SERVICE_ACCOUNT_EMAIL}")
        return None

    # Prepare the values to append
    values = [[name, company, job_title, source, message, label]]

    # Body for the append request
    body = {
        'values': values
    }

    # Append to the sheet (columns A to F)
    range_name = 'Sheet1!A:F'

    try:
        result = service.spreadsheets().values().append(
            spreadsheetId=SPREADSHEET_ID,
            range=range_name,
            valueInputOption='RAW',
            body=body
        ).execute()
        print(f"Successfully appended entry: {name} - {message[:50]}... with label {label}")
        return result
    except Exception as e:
        print(f"Error appending to Google Sheet: {e}")
        print(f"Check that the sheet is shared with {SERVICE_ACCOUNT_EMAIL} and Google Sheets API is enabled")
        return None

# Example usage (uncomment to test)
append_entry_to_sheet("Test Name", "Test Company", "Test Job", "email", "Test message", "Hot")