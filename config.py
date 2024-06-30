from dotenv import load_dotenv
import os

load_dotenv(".env")
DATA_DIR = os.getenv("DATA_DIR")
