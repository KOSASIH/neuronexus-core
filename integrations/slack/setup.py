import os

# Set up the Slack integration
def setup():
    # Create the config.json file if it doesn't exist
    if not os.path.exists('config.json'):
        with open('config.json', 'w') as f:
            json.dump({'slack_token': '', 'slack_channel': ''}, f)

    # Install the required libraries
    os.system('pip install -r requirements.txt')

if __name__ == '__main__':
    setup()
