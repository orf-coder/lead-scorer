import os
import sys
sys.path.append(os.path.dirname(__file__))
import base64
import re
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from HubSpot.hubspot_integration import score_contact
import hubspot
from hubspot.crm.contacts import SimplePublicObjectInput, PublicObjectSearchRequest, Filter, FilterGroup
from datetime import datetime
import time
from email_automation import send_lead_email
from dotenv import load_dotenv
load_dotenv()

# Gmail API setup
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
CREDENTIALS_FILE = 'Google/google_credentials.json'
TOKEN_FILE = 'Google/token.json'

# HubSpot setup
HUBSPOT_ACCESS_TOKEN = os.getenv('HUBSPOT_ACCESS_TOKEN')
if not HUBSPOT_ACCESS_TOKEN:
    raise ValueError("HUBSPOT_ACCESS_TOKEN not found in environment variables")
client = hubspot.Client(access_token=HUBSPOT_ACCESS_TOKEN)

def get_gmail_service():
    """
    The function `get_gmail_service` retrieves Gmail service credentials and returns a Gmail API service
    object.
    :return: The function `get_gmail_service()` returns a Gmail service object that is authenticated
    with the provided credentials.
    """
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            from google_auth_oauthlib.flow import InstalledAppFlow
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def find_contact_by_email(email):
    """
    The function `find_contact_by_email` searches for an existing contact by email using a provided
    email address.
    
    :param email: The `find_contact_by_email` function is designed to search for an existing contact by
    email. When you call this function and pass an email address as the `email` parameter, it will use
    this email address to search for a contact in the CRM system
    :return: The function `find_contact_by_email` is returning either the first contact found with the
    specified email address or `None` if no contact is found or an error occurs during the search.
    """
    """Search for existing contact by email."""
    filter_obj = Filter(property_name="email", operator="EQ", value=email)
    filter_group = FilterGroup(filters=[filter_obj])
    search_request = PublicObjectSearchRequest(filter_groups=[filter_group])
    try:
        response = client.crm.contacts.search_api.do_search(search_request)
        if response.results:
            return response.results[0]
        return None
    except Exception as e:
        print(f"Error searching for contact by email {email}: {e}")
        return None

# Removed fixed format parsing - now uses entire email content

def process_email(service, msg_id):
    """
    The `process_email` function processes an email message, extracts relevant information, scores the
    contact, updates or creates a contact in a CRM system, and sends an email for leads with specific
    labels.
    
    :param service: The `service` parameter in the `process_email` function is typically an instance of
    a Google API service, specifically the Gmail API service in this case. It is used to interact with
    the Gmail API to retrieve and process email messages
    :param msg_id: The `msg_id` parameter in the `process_email` function is the unique identifier of
    the email message that you want to process. This identifier is used to retrieve the specific email
    message from the email service using the Gmail API
    :return: The `process_email` function processes an email message retrieved from a Gmail service
    using the provided `msg_id`. It extracts information such as the subject, sender's email address,
    sender's name, and email body from the message. It then scores the contact based on the extracted
    information, maps the score to a label, and checks if the contact already exists in a CRM system.
    """
    try:
        message = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
        payload = message['payload']
        headers = payload['headers']

        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
        from_header = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')

        # Parse email and name from From header
        email_match = re.search(r'<([^>]+)>', from_header)
        if email_match:
            from_email = email_match.group(1)
            sender_name = from_header.replace(f'<{from_email}>', '').strip()
        else:
            from_email = from_header
            sender_name = from_header.split('@')[0].replace('.', ' ').title()

        # Skip no-reply emails and self-emails
        if 'noreply' in from_email.lower() or 'no-reply' in from_email.lower():
            print(f"Skipping no-reply email from {from_email}")
            return
        sender_email = os.getenv('SENDER_EMAIL', '').lower()
        if from_email.lower() == sender_email:  # Skip self-emails
            print(f"Skipping self-email from {from_email}")
            return

        # Get email body
        if 'parts' in payload:
            parts = payload['parts']
            data = parts[0]['body']['data']
        else:
            data = payload['body']['data']

        email_body = base64.urlsafe_b64decode(data).decode('utf-8')

        # Use entire email content for scoring
        full_message = f"{subject}\n\n{email_body}".strip()

        # Create contact dict for scoring
        contact = {
            'properties': {
                'firstname': sender_name.split()[0] if sender_name else '',
                'lastname': ' '.join(sender_name.split()[1:]) if len(sender_name.split()) > 1 else '',
                'company': '',  # Will be empty since no fixed format
                'jobtitle': '',  # Will be empty since no fixed format
                'notes': '',  # No notes
                'message': full_message
            }
        }

        # Score the lead
        score, label, reason = score_contact(contact)

        # Map label
        label_mapping = {
            'hot': 'hot_lead',
            'Hot': 'hot_lead',
            'warm': 'warm_lead',
            'Warm': 'warm_lead',
            'cold': 'cold_lead',
            'Cold': 'cold_lead'
        }
        mapped_label = label_mapping.get(label, label)

        # Check if contact exists, update or create
        existing_contact = find_contact_by_email(from_email)
        properties = {
            'firstname': contact['properties']['firstname'],
            'lastname': contact['properties']['lastname'],
            'company': '',
            'jobtitle': '',
            'message': full_message,
            'lead_score': score,
            'lead_label': mapped_label,
            'email': from_email
        }

        simple_public_object_input = SimplePublicObjectInput(properties=properties)

        if existing_contact:
            # Update existing contact
            contact_id = existing_contact.id
            try:
                client.crm.contacts.basic_api.update(contact_id, simple_public_object_input)
                print(f"Updated existing contact {contact_id}: {sender_name}, Score: {score}, Label: {mapped_label}")
            except Exception as e:
                print(f"Error updating contact {contact_id}: {e}")
        else:
            # Create new contact
            try:
                response = client.crm.contacts.basic_api.create(simple_public_object_input)
                print(f"Created new contact: {sender_name}, Score: {score}, Label: {mapped_label}")
            except Exception as e:
                print(f"Error creating contact: {e}")

        # Send email for all leads (Hot, Warm, Cold), but skip HubSpot accounts
        if label in ['Hot', 'Warm', 'Cold'] and not from_email.lower().endswith('@hubspot.com'):
            send_lead_email(from_email, label, sender_name, company='')

    except Exception as e:
        print(f"Error processing email {msg_id}: {e}")

def main():
    """
    The main function retrieves recent Gmail messages, processes new emails, and saves the processed
    email IDs for future reference in a loop with error handling.
    """
    service = get_gmail_service()

    # Track processed emails
    processed_file = 'processed_emails.txt'
    if os.path.exists(processed_file):
        with open(processed_file, 'r') as f:
            processed = set(f.read().splitlines())
    else:
        processed = set()

    while True:
        try:
            # Get recent messages (last 1 hour)
            query = f"newer_than:1h"
            results = service.users().messages().list(userId='me', q=query).execute()
            messages = results.get('messages', [])

            for msg in messages:
                msg_id = msg['id']
                if msg_id not in processed:
                    process_email(service, msg_id)
                    processed.add(msg_id)

            # Save processed emails
            with open(processed_file, 'w') as f:
                f.write('\n'.join(processed))

            print(f"Checked for new emails at {datetime.now()}. Next check in 1 minute...")
            time.sleep(60)  # Check every minute

        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(60)

if __name__ == '__main__':
    main()