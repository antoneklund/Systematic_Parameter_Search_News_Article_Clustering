import os

PATH = os.path.dirname(os.path.realpath(__file__))
SETTINGS_FOLDER_NAME = os.path.basename(PATH)
PROJECT_PATH = os.path.dirname(os.path.realpath(SETTINGS_FOLDER_NAME))
LOGGING_CONF = os.path.join(PROJECT_PATH, SETTINGS_FOLDER_NAME, 'logging.conf')
