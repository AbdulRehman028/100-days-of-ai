### 🧠 **story_generator.py**
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time

def load_model_with_retry(model_name="gpt2", max_retries=3):
    """Load model with retry logic for unstable connections."""
    for attempt in range(max_retries):
        try:
            print(f"📥 Loading model (attempt {attempt + 1}/{max_retries})...")
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
            print("✅ Model loaded successfully!")
            return tokenizer, model
        except Exception as e:
            print(f"⚠️ Download failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise Exception("❌ Failed to download model after multiple attempts. Check your internet connection.")

def generate_story(prompt, max_length=200, temperature=0.8, top_k=50, top_p=0.95):
    # Load pre-trained GPT-2 model and tokenizer with retry logic
    tokenizer, model = load_model_with_retry()

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