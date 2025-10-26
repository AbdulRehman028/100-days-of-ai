### 🧠 **story_generator.py**
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_story(prompt, max_length=200, temperature=0.8, top_k=50, top_p=0.95):
    # Load pre-trained GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate story
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and return generated text
    story = tokenizer.decode(output[0], skip_special_tokens=True)
    return story

if __name__ == "__main__":
    print("🧠 Welcome to AI Story Generator!")
    prompt = input("Enter a story prompt: ")
    print("\n✨ Generating story...\n")

    story = generate_story(prompt)
    print(story)