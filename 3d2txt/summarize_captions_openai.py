import csv
import openai
import argparse
import os
from dotenv import load_dotenv
import time

load_dotenv(dotenv_path='D:/MyProjects/.env', override=True)
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print("API key loaded from .env")
else:
    print("API key NOT loaded from .env")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default='D:/MyProjects/partFeatureAnalysisVLM/3d2txt/example_material/Cap3D_captions/Cap3d_captions_final.csv')
    parser.add_argument('--output_csv', type=str, default='D:/MyProjects/partFeatureAnalysisVLM/3d2txt/example_material/Cap3D_captions/summarize.csv')
    parser.add_argument('--openai_api_key', type=str, default=api_key)
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--max_retries', type=int, default=3)
    return parser.parse_args()

args = parse_args()
client = openai.OpenAI(api_key=api_key)

# Helper: summarize caption using OpenAI
def summarize_captions_openai(captions, model='gpt-4o', max_retries=3):
    prompt = (
        "Given a set of descriptions about the same 3D object, distill these descriptions into one concise caption. "
        "The descriptions are as follows: '" + ' | '.join(captions) + "'. "
        "Avoid describing background, surface, and posture. The caption should be:"
    )
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": prompt}],
                max_tokens=100,
                temperature=0.2,
            )
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                return f"[ERROR: {str(e)}]"
            time.sleep(2)
    return "[ERROR: No response]"


# Read input CSV, summarize, and write output CSV
with open(args.input_csv, 'r') as infile, \
     open(args.output_csv, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    header = next(reader)
    writer.writerow(['i', 'obj_id', 'summary'])

    for row in reader:
        if len(row) < 3:
            continue
        i, obj_id, obj_final_caption = row[0], row[1], row[2]
        #print (i)
        #print (obj_id)
        #print(obj_final_caption)
        summary = summarize_captions_openai(obj_final_caption, args.model, args.max_retries)
        print(f"{i}, {obj_id}, {summary}")
        writer.writerow([i, obj_id, summary])
        outfile.flush()
        os.fsync(outfile.fileno())
print(f"Summarization complete. Output written to {args.output_csv}") 