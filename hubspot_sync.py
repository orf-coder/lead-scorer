import logging
from HubSpot.hubspot_integration import sync_leads

# Setup logging (will inherit from hubspot_integration.py setup)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # Run the HubSpot sync process
    try:
        sync_leads()
        print("HubSpot sync complete.")
    except Exception as e:
        logging.error(f"Sync failed: {e}")
        print(f"HubSpot sync failed: {e}")