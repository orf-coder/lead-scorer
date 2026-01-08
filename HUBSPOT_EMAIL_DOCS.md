# HubSpot Integration & Email Automation Documentation

This document provides detailed documentation for the HubSpot integration and email automation components.

## Files Overview

### `.env` - Environment Configuration File

**Purpose**: Stores sensitive configuration and environment-specific settings for the application.

**Location**: Root directory (not committed to version control)

**Structure**:
```bash
# HubSpot Configuration
HUBSPOT_ACCESS_TOKEN=your_hubspot_token_here

# Email Configuration
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Scoring Thresholds
HOT_CONFIDENCE_THRESHOLD=0.75
COLD_CONFIDENCE_THRESHOLD=0.5

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=lead_scorer.log
```

**Security Notes**:
- Add `.env` to `.gitignore` to prevent committing sensitive data
- Use Gmail App Passwords (not regular passwords) for email authentication
- Rotate HubSpot access tokens regularly
- Never share or expose this file

### `hubspot_sync.py` - HubSpot Synchronization Script

**Purpose**: Command-line entry point for running the HubSpot lead synchronization process.

**Location**: Root directory

**Key Features**:
- Imports the main sync functionality from `HubSpot/hubspot_integration.py`
- Sets up logging configuration
- Provides error handling for the sync process
- Simple command-line interface

**Usage**:
```bash
python hubspot_sync.py
```

**Expected Output**:
```
INFO - Starting HubSpot lead sync process
INFO - Successfully fetched X contacts
INFO - Contact XXX scored: rule_score=X, final_label=X, final_conf=X.XX
INFO - Sync complete: X/X contacts processed successfully
HubSpot sync complete.
```

**Error Handling**:
- Catches and logs any exceptions during sync
- Provides user-friendly error messages
- Ensures clean exit on failures

### `HubSpot/hubspot_integration.py` - Core HubSpot Integration Module

**Purpose**: Handles all HubSpot API interactions, lead scoring, and automated email processing.

**Location**: `HubSpot/` directory

**Key Functions**:

#### `fetch_contacts(limit=100)`
```python
def fetch_contacts(limit=100):
    """Fetch recent contacts from HubSpot with error handling and logging."""
```
- **Purpose**: Retrieves contact data from HubSpot CRM
- **Parameters**: `limit` (int) - Maximum contacts to fetch (default: 100, but overridden to 10000)
- **Returns**: List of contact objects
- **Features**:
  - Automatic retry logic (3 attempts) for API failures
  - Comprehensive error logging
  - Handles different API response formats

#### `validate_contact_data(properties)`
```python
def validate_contact_data(properties):
    """Validate and sanitize contact data."""
```
- **Purpose**: Cleans and validates contact data from HubSpot API
- **Parameters**: `properties` (dict) - Raw contact properties
- **Returns**: Sanitized properties dictionary
- **Features**:
  - Converts `None` values to empty strings
  - Ensures string type safety
  - Length limiting for large text fields

#### `score_contact(contact)`
```python
def score_contact(contact):
    """Score a single contact using lead scorer logic."""
```
- **Purpose**: Applies hybrid rule-based + ML scoring to individual contacts
- **Parameters**: `contact` - HubSpot contact object
- **Returns**: Tuple of (score, label, reason)
- **Features**:
  - Data validation and sanitization
  - Rule-based keyword scoring
  - ML model predictions with caching
  - Confidence aggregation and threshold-based labeling

#### `update_contact_score(contact_id, score, label)`
```python
def update_contact_score(contact_id, score, label):
    """Update the contact with score and label in HubSpot."""
```
- **Purpose**: Updates contact records in HubSpot with scoring results
- **Parameters**:
  - `contact_id` (str) - HubSpot contact identifier
  - `score` (int/float) - Calculated lead score
  - `label` (str) - Lead classification (Hot/Warm/Cold)
- **Features**:
  - Label mapping to HubSpot dropdown options
  - Retry logic for API reliability
  - Comprehensive error handling

#### `record_email_sent(contact_id, subject, body)`
```python
def record_email_sent(contact_id, subject, body):
    """Record the sent email details in HubSpot contact's notes."""
```
- **Purpose**: Logs email sending activity in HubSpot contact notes
- **Parameters**:
  - `contact_id` (str) - HubSpot contact identifier
  - `subject` (str) - Email subject line
  - `body` (str) - Email body content
- **Features**:
  - Appends to existing notes (preserves history)
  - Truncates body for HubSpot field limits
  - API retry logic

#### `generate_email_content(label, contact_name, company)`
```python
def generate_email_content(label, contact_name, company):
    """Generate email subject and body based on lead label."""
```
- **Purpose**: Creates personalized email content based on lead classification
- **Parameters**:
  - `label` (str) - Lead type (Hot/Warm/Cold)
  - `contact_name` (str) - Contact's full name
  - `company` (str) - Company name (optional)
- **Returns**: Tuple of (subject, body)
- **Templates**:
  - **Hot**: Focuses on scheduling calls and immediate engagement
  - **Warm**: Follows up on recent interest
  - **Cold**: Provides value through resources and nurturing

#### `process_contact(contact)`
```python
def process_contact(contact):
    """Process a single contact: score, update, and send email."""
```
- **Purpose**: Orchestrates complete contact processing pipeline
- **Parameters**: `contact` - HubSpot contact object
- **Returns**: Boolean success indicator
- **Workflow**:
  1. Score the contact
  2. Update HubSpot with score/label
  3. Send personalized email (Hot/Warm/Cold)
  4. Log email in HubSpot notes

#### `sync_leads()`
```python
def sync_leads():
    """Main function to fetch, score, and update leads."""
```
- **Purpose**: Main orchestration function for bulk lead processing
- **Features**:
  - Fetches all available contacts
  - Processes each contact individually
  - Provides processing statistics
  - Comprehensive error handling and logging

**Configuration**:
- Uses environment variables for all sensitive data
- Configurable confidence thresholds via `.env`
- Flexible logging levels and output destinations

**Dependencies**:
- `hubspot-api-client` - HubSpot CRM API
- `python-dotenv` - Environment variable management
- `lead_scorer.py` - Scoring algorithms
- `email_automation.py` - Email sending functionality

### `email_automation.py` - Email Automation Module

**Purpose**: Handles automated email sending via Gmail SMTP with comprehensive error handling.

**Location**: Root directory

**Key Functions**:

#### `validate_email_params(recipient_email, lead_label, contact_name)`
```python
def validate_email_params(recipient_email, lead_label, contact_name):
    """Validate email sending parameters."""
```
- **Purpose**: Validates email parameters before sending
- **Parameters**:
  - `recipient_email` (str) - Recipient email address
  - `lead_label` (str) - Lead classification
  - `contact_name` (str) - Contact name
- **Raises**: `ValueError` for invalid parameters
- **Validation**:
  - Email format checking
  - Required field presence
  - Basic input sanitization

#### `send_lead_email(recipient_email, lead_label, contact_name, company='', custom_message='')`
```python
def send_lead_email(recipient_email, lead_label, contact_name, company='', custom_message=''):
    """Send automated email based on lead label using Gmail SMTP."""
```
- **Purpose**: Sends personalized emails based on lead classification
- **Parameters**:
  - `recipient_email` (str) - Recipient email address
  - `lead_label` (str) - Lead type (Hot/Warm/Cold)
  - `contact_name` (str) - Contact's name
  - `company` (str) - Company name (optional)
  - `custom_message` (str) - Additional message content (optional)
- **Returns**: Boolean success indicator
- **Features**:
  - SMTP connection with TLS encryption
  - Retry logic for delivery failures
  - Specific error handling for authentication and recipient issues
  - Comprehensive logging

**Email Templates**:
- **Hot Leads**: "Exciting Opportunity, {contact_name}"
- **Warm Leads**: "Follow Up, {contact_name}"
- **Cold Leads**: "Exclusive Resources to Accelerate Your Business Growth, {contact_name}"

**Security Features**:
- Gmail App Password authentication
- TLS-encrypted SMTP connections
- Input validation and sanitization

**Configuration**:
- SMTP server settings via environment variables
- Sender credentials from `.env`
- Timeout and retry configurations

### `gmail_lead_scorer.py` - Legacy Gmail Integration

**Purpose**: Original Gmail API integration for processing leads from email sources.

**Location**: Root directory

**Status**: Legacy code - appears unused in current HubSpot-focused workflow.

**Note**: This file contains older Gmail API integration code and may not be actively maintained or used in the current system architecture.

## System Architecture

```
HubSpot CRM API
       ↓
hubspot_sync.py
       ↓
HubSpot/hubspot_integration.py
    ├── fetch_contacts()
    ├── score_contact()
    ├── update_contact_score()
    ├── generate_email_content()
    └── record_email_sent()
       ↓
email_automation.py
    └── send_lead_email()
```

## Data Flow

1. **Initialization**: Load environment configuration and setup logging
2. **Contact Retrieval**: Fetch contacts from HubSpot with properties (name, email, company, etc.)
3. **Data Processing**: Validate and sanitize contact data
4. **Lead Scoring**: Apply hybrid rule-based + ML scoring algorithm
5. **CRM Updates**: Update HubSpot contact records with scores and labels
6. **Email Generation**: Create personalized email content based on lead classification
7. **Email Delivery**: Send emails via Gmail SMTP with error handling
8. **Audit Logging**: Record email details in HubSpot contact notes

## Configuration Management

All sensitive configuration managed through environment variables:

- **HUBSPOT_ACCESS_TOKEN**: HubSpot API authentication token
- **SENDER_EMAIL/SENDER_PASSWORD**: Gmail SMTP credentials
- **SMTP_SERVER/SMTP_PORT**: Email server configuration
- **HOT_CONFIDENCE_THRESHOLD/COLD_CONFIDENCE_THRESHOLD**: Scoring decision boundaries
- **LOG_LEVEL/LOG_FILE**: Logging configuration

## Error Handling & Resilience

- **API Retries**: Automatic retry with configurable attempts for HubSpot API calls
- **SMTP Recovery**: Retry logic for email delivery failures with specific error types
- **Data Sanitization**: Handles null/missing values from external APIs
- **Logging**: Structured logging with different levels (DEBUG, INFO, ERROR, WARNING)
- **Graceful Degradation**: Continues processing other contacts when individual operations fail

## Security Considerations

- **Credential Management**: Environment variables prevent hardcoded secrets
- **API Authentication**: OAuth tokens with regular rotation requirements
- **Email Security**: App passwords and TLS encryption for SMTP
- **Data Validation**: Input sanitization prevents injection attacks
- **Audit Trail**: Email logging in CRM for compliance and tracking

## Usage Examples

### Basic HubSpot Sync
```bash
python hubspot_sync.py
```

### Environment Setup
```bash
# Create .env file
cp .env.example .env

# Edit with your credentials
# HUBSPOT_ACCESS_TOKEN=your_token
# SENDER_EMAIL=your_email@gmail.com
# SENDER_PASSWORD=your_app_password
```

### Expected Log Output
```
2026-01-06 11:30:00,000 - HubSpot.hubspot_integration - INFO - Starting HubSpot lead sync process
2026-01-06 11:30:01,000 - HubSpot.hubspot_integration - INFO - Successfully fetched 150 contacts
2026-01-06 11:30:02,000 - HubSpot.hubspot_integration - INFO - Contact 12345 scored: rule_score=45, final_label=Hot, final_conf=0.82
2026-01-06 11:30:03,000 - HubSpot.hubspot_integration - INFO - Sync complete: 148/150 contacts processed successfully
```

## Troubleshooting

### Common Issues

1. **HubSpot API Authentication Errors**
   - Verify `HUBSPOT_ACCESS_TOKEN` is valid and has proper permissions
   - Check token expiration and refresh if needed

2. **Email Delivery Failures**
   - Confirm Gmail App Password is correct (not regular password)
   - Verify SMTP settings in `.env`
   - Check Gmail account security settings

3. **Scoring Errors**
   - Ensure ML model files exist in project root
   - Check contact data format from HubSpot
   - Verify scoring thresholds in configuration

4. **Null Value Errors**
   - Ensure `validate_contact_data()` is properly sanitizing inputs
   - Check HubSpot contact data quality

### Debug Mode
Set `LOG_LEVEL=DEBUG` in `.env` for detailed operation logs including:
- API request/response details
- Scoring calculation breakdowns
- Email content generation
- Step-by-step processing flow

## Performance Considerations

- **ML Model Caching**: `@lru_cache` prevents repeated model loading
- **Batch Processing**: Processes contacts individually to manage memory
- **API Rate Limiting**: Built-in delays and retry logic
- **Logging Overhead**: Configurable log levels for production optimization

## Future Enhancements

- **Batch Email Processing**: Queue-based system for high-volume sending
- **Advanced Templates**: Dynamic content based on lead profiles and history
- **CRM Integration**: Bidirectional sync with additional HubSpot properties
- **Analytics Dashboard**: Lead scoring performance and conversion metrics
- **Multi-Channel Support**: Additional email providers beyond Gmail
- **Webhook Integration**: Real-time processing triggers from HubSpot

## Dependencies

**Core Requirements**:
- `hubspot-api-client` - HubSpot CRM API client
- `python-dotenv` - Environment variable management
- `scikit-learn` - ML model loading (inherited from lead_scorer.py)

**Standard Library**:
- `os`, `sys`, `logging` - System and logging utilities
- `functools` - Caching decorators
- `smtplib`, `email` - Email functionality

## File Structure

```
project_root/
├── .env                           # Environment configuration
├── hubspot_sync.py               # Sync orchestration script
├── email_automation.py           # Email sending module
├── gmail_lead_scorer.py          # Legacy Gmail integration
├── HubSpot/
│   └── hubspot_integration.py    # Core HubSpot integration
└── lead_classifier_*.pkl         # ML model files
```

---

*Document Version: 1.0*
*Last Updated: January 6, 2026*
*Applies to: HubSpot Integration & Email Automation System*