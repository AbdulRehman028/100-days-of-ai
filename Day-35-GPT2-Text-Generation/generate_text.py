from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_path = "./gpt2-finetuned"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

prompt = "Artificial intelligence will reshape the future because"

inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
    **inputs,
    max_length=120,
    temperature=0.8,
    top_p=0.9,
    num_return_sequences=1,
    do_sample=True,
)

print("\nGenerated Text:\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))