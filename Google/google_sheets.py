import os
import json

# Lazy-import Google libraries so module can be imported even when they are
# not available (e.g., on Streamlit Community Cloud when Google Sheets is not used).
GOOGLE_LIBS_AVAILABLE = True
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
except Exception:
    GOOGLE_LIBS_AVAILABLE = False

# Scopes for Google Sheets API
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# Defaults and configuration
# Spreadsheet ID can be set via env var `GOOGLE_SPREADSHEET_ID` or keep the
# existing default below (change if needed).
SPREADSHEET_ID = os.environ.get('GOOGLE_SPREADSHEET_ID', '1IMDjdviTB43-sJrIM4x4TpSyaAR6KnIOCi9MVlgaf4U')

# Path to credentials file (for local dev). In deployed apps prefer secrets/env.
SERVICE_ACCOUNT_FILE = os.path.join(os.path.dirname(__file__), 'credentials.json')

SERVICE_ACCOUNT_EMAIL = None
PROJECT_ID = None
service = None


def _init_google_service():
    """Initialize and return Google Sheets service or None.

    Order of attempts:
      1. Read JSON from env var `GOOGLE_SERVICE_ACCOUNT_JSON` (full JSON string).
      2. Read from Streamlit secrets: `st.secrets.get('google_service_account')` if available.
      3. Read local `Google/credentials.json` file (development).
    """
    global SERVICE_ACCOUNT_EMAIL, PROJECT_ID, service

    if not GOOGLE_LIBS_AVAILABLE:
        print("Google API libraries not installed; Google Sheets disabled")
        return None

    creds_info = None

    # 1) Environment variable (preferred for deployments)
    env_json = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON')
    if env_json:
        try:
            creds_info = json.loads(env_json)
        except Exception as e:
            print(f"Failed to parse GOOGLE_SERVICE_ACCOUNT_JSON: {e}")

    # 2) Streamlit secrets (if running inside Streamlit and configured)
    if creds_info is None:
        try:
            import streamlit as _st
            s = _st.secrets
            # Common patterns: st.secrets['google_service_account'] or st.secrets['google']['service_account']
            if 'google_service_account' in s:
                creds_info = s['google_service_account']
            elif 'google' in s and 'service_account' in s['google']:
                creds_info = s['google']['service_account']
        except Exception:
            pass

    # 3) Local file (development)
    if creds_info is None and os.path.exists(SERVICE_ACCOUNT_FILE):
        try:
            with open(SERVICE_ACCOUNT_FILE, 'r') as f:
                creds_info = json.load(f)
        except Exception as e:
            print(f"Failed to read {SERVICE_ACCOUNT_FILE}: {e}")

    if creds_info is None:
        print("No Google service account credentials found; Google Sheets disabled")
        return None

    SERVICE_ACCOUNT_EMAIL = creds_info.get('client_email')
    PROJECT_ID = creds_info.get('project_id')

    try:
        # from_service_account_info allows using an in-memory dict instead of a file
        creds = service_account.Credentials.from_service_account_info(creds_info, scopes=SCOPES)
        service = build('sheets', 'v4', credentials=creds)
        print(f"Google Sheets service initialized for {SERVICE_ACCOUNT_EMAIL}")
        return service
    except Exception as e:
        print(f"Failed to initialize Google Sheets service: {e}")
        service = None
        return None


def append_entry_to_sheet(name, company, job_title, source, message, label):
    """Append a row to the configured Google Sheet (columns A-F).

    This function is safe to call even when Google integration is not configured;
    it will return None and log a message instead of raising.
    """
    global service

    if not GOOGLE_LIBS_AVAILABLE:
        print("Google libs not available; skipping append to Google Sheet")
        return None

    if service is None:
        _init_google_service()

    if service is None:
        print("Google Sheets service not initialized; skipping append")
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
    