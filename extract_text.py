from pdfminer.high_level import extract_text
import os

input_dir = 'data/pdfs'
output_dir = 'data/texts'
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith('.pdf'):
        text = extract_text(os.path.join(input_dir, file))
        with open(os.path.join(output_dir, file.replace('.pdf', '.txt')), 'w', encoding='utf-8') as f:
            f.write(text)
