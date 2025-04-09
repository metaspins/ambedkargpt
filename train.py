from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments, DataCollatorForLanguageModeling)

model_name = "gpt2"  # Or "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

dataset = load_from_disk('data/ambedkar_dataset')
tokenized = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=512), batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="models/ambedkar-gpt",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    evaluation_strategy="no",
    no_cuda=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
