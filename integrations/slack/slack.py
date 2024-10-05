import os
import json
from slack import WebClient

# Load the configuration from the config.json file
with open('config.json') as f:
    config = json.load(f)

# Set the Slack API token and channel ID
slack_token = config['slack_token']
slack_channel = config['slack_channel']

# Create a Slack client
slack_client = WebClient(token=slack_token)

def send_message(message):
    # Send a message to the Slack channel
    slack_client.chat_postMessage(channel=slack_channel, text=message)

def receive_message():
    # Receive a message from the Slack channel
    response = slack_client.chat_getPermalink(channel=slack_channel)
    return response['message']['text']
