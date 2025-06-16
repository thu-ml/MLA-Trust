import os

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.getenv("DOTENV_PATH", ".env"))
