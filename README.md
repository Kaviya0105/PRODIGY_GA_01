# PRODIGY_GA_01
!pip install transformers datasets torch
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
from google.colab import files
uploaded = files.upload()
def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )
    return dataset

train_dataset = load_dataset("custom_dataset.txt", tokenizer)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)
trainer.train()
trainer.save_model("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")
prompt = "What are the key components of a healthy diet?"
input_ids = fine_tuned_tokenizer.encode(prompt, return_tensors='pt')
attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)

output = fine_tuned_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=100,
    num_return_sequences=1,
    pad_token_id=fine_tuned_tokenizer.eos_token_id
)

print(fine_tuned_tokenizer.decode(output[0], skip_special_tokens=True)) 
