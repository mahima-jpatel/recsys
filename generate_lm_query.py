import pandas as pd
import openai
import os
import time
from tqdm import tqdm

# Step 1: Load your data
print("Loading CSV file...")
df = pd.read_csv("collected_vs_retrieved_soft_0320_0900_updated.csv")  # Replace with your actual file path
print(f"Loaded {len(df)} rows.")

# Step 2: Sort by session and timestamp
print("Sorting data by session and timestamp...")
df = df.sort_values(by=["session_id", "timestamp"])
print("Sorting complete.")

# Step 3: Define prompt generation function
def generate_query_from_history(history_titles):
    joined_titles = "; ".join(history_titles)
    prompt = f"""
A user has browsed the following sequence of pages:

{joined_titles}

Based on this browsing history, generate a concise and meaningful search query that captures the user's likely next interest or intent. Focus on summarizing what the user might be looking for next. Avoid copying titles verbatim — infer the intent.
"""
    return prompt.strip()

# Step 4: Setup OpenAI API client
print("Reading API key from environment variable 'RTGIS'...")
api_key = os.getenv("RTGIS")
if not api_key:
    raise ValueError("Environment variable 'RTGIS' not found. Please set it using: export RTGIS='your-key'")

client = openai.OpenAI(api_key=api_key)
print("OpenAI client initialized.")

# Step 5: Initialize query list
queries = []

# Step 6: Get unique sessions and iterate
all_sessions = list(df["session_id"].unique())
print(f"Found {len(all_sessions)} unique sessions. Starting query generation...")

for session_id in tqdm(all_sessions, desc="Sessions"):
    group = df[df["session_id"] == session_id]
    titles = group["retrieved_page_title"].fillna("").tolist()
    urls = group["retrieved_url"].fillna("").tolist()
    for i in range(1, len(titles)):
        history = titles[:i]
        target = titles[i]
        prompt = generate_query_from_history(history)
        target_url = urls[i]

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that writes search queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            query = response.choices[0].message.content.strip()
            print(f"Generated query for session {session_id}, index {i}")
        except Exception as e:
            print(f"Error in session {session_id}, index {i}: {e}")
            query = ""

        queries.append({
            "session_id": session_id,
            "history": history,
            "target_title": target,
            "target_url": target_url,
            "query": query
        })

        time.sleep(1)  # Delay to avoid rate limits

# Step 7: Save results to CSV
print("Saving generated queries to 'generated_queries.csv'...")
query_df = pd.DataFrame(queries)
query_df.to_csv("generated_queries_turbo.csv", index=False)
print("File saved. Script completed successfully.")
