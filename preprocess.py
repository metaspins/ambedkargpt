import os
from datasets import Dataset

text_dir = 'data/texts'
all_texts = []

for file in os.listdir(text_dir):
    with open(os.path.join(text_dir, file), 'r', encoding='utf-8') as f:
        content = f.read()
        chunks = content.split('\n\n')
        for chunk in chunks:
            cleaned = chunk.strip().replace('\n', ' ')
            if len(cleaned) > 100:
                all_texts.append({'text': cleaned})

dataset = Dataset.from_list(all_texts)
dataset.save_to_disk('data/ambedkar_dataset')
