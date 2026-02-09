import os
from dotenv import load_dotenv

def to_excel(df, filename):
    load_dotenv(override=True)
    results_path = os.getenv("RESULTS_PATH")
    output_path = os.path.join(results_path, filename)
    df.to_excel(output_path, index=False)
    return output_path
