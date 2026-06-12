from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == "__main__":
    fre = os.getenv("BATCH_SIZE")
    print(fre)
    print(type(fre))