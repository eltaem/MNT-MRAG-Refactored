import json
import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

def main():
    # Load JSON
    json_qa_path = Path(os.getenv("JSON_QA_PATH"))
    csv_qa_path = Path(os.getenv("CSV_QA_PATH"))

    if csv_qa_path.exists():
        print("File JSON yang ke pase ke CSV sudah ada di", csv_qa_path.relative_to(Path.cwd()))
        return

    with open(json_qa_path, "r", encoding="utf-8") as f:
        data:list[dict[str, str]] = json.load(f)

    rows = []

    for item in data:
        anthropometric = item.get("anthropometric_assessment", "").strip()
        anthropometric = " ".join(anthropometric.split())
        
        diagnosis = item.get("medical_diagnosis", "").strip()
        diagnosis = " ".join(diagnosis.split())
        
        original_question = item.get("question", "").strip()
        original_question = " ".join(original_question.split())
        
        question = f"{anthropometric} | {diagnosis} | {original_question}"
        
        answer_data = item.get("answer", {})
        answer_text = answer_data.get("text", [])

        if isinstance(answer_text, list):
            answer = " ".join([a.strip() for a in answer_text])
        else:
            answer = str(answer_text).strip()

        answer = " ".join(answer.split())

        rows.append({
            "question": question,
            "answer": answer
        })

    df = pd.DataFrame(rows)
    csv_qa_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_qa_path, index=False, encoding="utf-8-sig")
    print(df.head())
    print(f"\nTotal QA: {len(df)}")
    print("JSON parsed ke csv di folder:", csv_qa_path.relative_to(Path.cwd()))

if __name__ == "__main__":
    main()