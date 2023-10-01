import anvil.server
import anvil.secrets
import pandas as pd
from google.cloud import bigquery
from google.auth.transport.requests import Request
import requests
import json
from google.oauth2 import service_account

@anvil.server.callable
def df_as_markdown():
    # Load the service account credentials from Anvil App Secrets
    service_account_info = json.loads(anvil.secrets.get_secret('argolis-test-service-account-project-owner'))
    
    # Authenticate using the provided method
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    credentials.refresh(Request())
    token = credentials.token
    
    # Set up BigQuery client with the authenticated credentials
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    
    # Define your BigQuery SQL query here
    query = "SELECT * FROM `showvik-argolis-test-project.test_datasets.denmark_test` LIMIT 30"  # Modify this query as per your needs
    
    # Execute the query and get the result as a DataFrame
    df = client.query(query).to_dataframe()
    
    # Convert the DataFrame to markdown and return
    return df.to_markdown()
