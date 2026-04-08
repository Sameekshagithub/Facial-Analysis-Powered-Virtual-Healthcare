from docx import Document
import pandas as pd

def extract_qa_from_docx(docx_path):
    doc = Document(docx_path)
    data = []
    current_emotion = None
    question = None

    for para in doc.paragraphs:
        text = para.text.strip()

        if not text:
            continue  # skip empty lines

        # Check if it's an emotion header
        if text.upper() in ['HAPPY', 'SAD', 'FEAR', 'ANGRY', 'SURPRISE', 'DISGUST', 'CONTEMPT', 'LOVE', 'GUILTY', 'DEPRESSION']:
            current_emotion = text.capitalize()
            continue

        # Check if it's a question (heuristically starts with WH words or numbered)
        if text.endswith("?"):
            question = text
            continue

        # If it's an answer and there's a current question and emotion
        if question and current_emotion:
            data.append({
                'emotion': current_emotion,
                'question': question,
                'answer': text
            })
            question = None  # Reset for next pair

    df = pd.DataFrame(data)
    return df

# Convert and save to CSV
df_qa = extract_qa_from_docx("Chatbot Q&A-2.docx")
df_qa.to_csv("chatbot_qa.csv", index=False)
print("✅ Q&A extracted and saved to chatbot_qa.csv")
