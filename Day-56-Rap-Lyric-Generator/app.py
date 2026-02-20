import os
import re
import json
import time
import random
import torch
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ============================================
# CONFIG
# ============================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "lyrics_data"
MODEL_DIR = BASE_DIR / "fine_tuned_model"
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))

# ============================================
# RAP LYRICS DATASET BUILDER
# ============================================
class LyricsDataManager:
    """Manages the lyrics dataset for fine-tuning"""

    def __init__(self):
        self.data_file = DATA_DIR / "rap_lyrics.txt"
        self.meta_file = DATA_DIR / "metadata.json"
        self.metadata = {"total_lines": 0, "themes": [], "fine_tuned": False}
        self.ensure_dataset()

    def ensure_dataset(self):
        """Make sure a lyrics dataset exists"""
        if not self.data_file.exists():
            print("📝 Creating sample rap lyrics dataset...")
            self.create_sample_dataset()
        else:
            print(f"✅ Lyrics dataset found: {self.data_file}")
        self._load_metadata()

    def _load_metadata(self):
        try:
            with open(self.meta_file, 'r') as f:
                self.metadata = json.load(f)
        except Exception:
            self.metadata = {"total_lines": 0, "themes": [], "fine_tuned": False}

    def _save_metadata(self, **kwargs):
        self.metadata.update(kwargs)
        with open(self.meta_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def create_sample_dataset(self):
        """Create a sample rap lyrics training corpus. 
        These are ORIGINAL lyrics, not copied from any artist."""
        lyrics = """[Theme: Hustle]
I wake up every morning with a mission on my mind
Grinding through the struggle leaving yesterday behind
Stack the paper up, never let them see me fall
Built this empire brick by brick, standing ten feet tall
Every setback is a setup for a comeback twice as strong
Writing history in the making, this is where I belong

[Theme: Hustle]
Started from the bottom now I'm climbing every day
They doubted every step but I kept walking anyway
The city lights are calling and the money never sleeps
I plant the seeds of greatness and I harvest what I reap
No shortcuts on this highway, I do it the hard way
Every hour of the grind is building up my payday

[Theme: Street Life]
Corner store philosophy from scholars of the block
Midnight conversations while we're watching for the cops
Streetlights flicker, another story unfolds
The truth is always bitter but the lies feel like gold
Concrete jungle where the strong survive the night
Every shadow has a story, every corner has a fight

[Theme: Street Life]
Sidewalk sermons from the preacher on the stoop
Kids are playing hopscotch while the sirens loop
Grandma's cooking dinner, smell it down the hall
Murals painted beautiful on every broken wall
This neighborhood made me, every crack and every scar
Even broken pavement leads you to a star

[Theme: Dreams]
I've been dreaming of the day when the world knows my name
When the verses that I write turn into eternal flame
Chasing clouds of purple haze through silver-lined skies
The vision in my head is clearer than these tired eyes
They told me I should quit and get a regular career
But the music in my soul is the only voice I hear

[Theme: Dreams]
Close my eyes and picture everything I want to be
Penthouse view, the ocean, total serenity
But the grind won't let me rest, the hunger keeps me up
Pouring rhymes into the mic like coffee in a cup
Manifestation station, I believe it so it's true
Every single dream I had is slowly coming through

[Theme: Love]
She walked in like a melody I'd never heard before
Every word she spoke was like a key to every door
Heart was beating drum lines in a rhythm built for two
In a world of black and white, she painted everything blue
Late night conversations, stars replacing ceiling lights
She's the chorus to my verses, she makes everything right

[Theme: Love]
We were strangers in the rain under neon-colored lights
She smiled and the city noise went quiet for the night
Her laughter is a sample that I loop inside my head
Every moment with her feels like every word unsaid
Two hearts beating bass lines in a symphony of trust
In a world that's filled with rust she's the gold beneath the dust

[Theme: Flex]
Pull up in the chariot, the engine start to roar
Diamond-studded confidence, they never seen before
Ice around my wrist like winter came in June
Every beat I step on turns into a platinum tune
They talk but never walk it, I'm the proof inside the booth
Every bar I spit is nothing but the truth

[Theme: Flex]
Chain heavy, pockets loaded, yeah the lifestyle's elite
Every verse I drop is fire, feel the heat in the street
Designer on the outside, warrior within
Champagne celebrations, let the winning begin
They wanna know the secret but the recipe is sealed
Hard work mixed with talent on a never-ending reel

[Theme: Struggle]
Mama worked two jobs just to keep the lights on
Dinner was a prayer answered when the check was gone
Hand-me-down ambition, secondhand respect
But I took the broken pieces and I built an architect
Struggle is the teacher and the lesson is the pain
Every single tear I shed became a drop of rain

[Theme: Struggle]
Empty refrigerator, full imagination
Living in survival mode across the nation
Bills piled up like skyscrapers in the sky
But I learned to build a ladder every time they asked me why
The weight upon my shoulders made my backbone strong
This struggle is the intro to my victory song

[Theme: Party]
DJ spin that record let the bass line drop
Whole club jumping, yeah we never gonna stop
Hands up in the air like we just don't care
Confetti in the spotlight, magic everywhere
Friday night is calling and the speakers start to bump
Every single soul in here is feeling the jump

[Theme: Party]
Bottle service, VIP, the night is young
Turn the music up until we feel it in our lungs
Dancing like tomorrow doesn't even exist
Neon lights and laser beams cutting through the mist
The rhythm takes control and everybody moves
Life is just a party when you're locked into the groove

[Theme: Legacy]
When I'm gone I want the words to echo through the years
A library of verses built from joy and fears
Every page a chapter of the story that I've lived
The gift of honest music is the greatest I can give
Legacy is measured not by wealth but by the hearts
You touch with every lyric, every verse, every part

[Theme: Legacy]
Carve my name in marble made of beats and rhymes
A monument to every word I spit between the lines
When the final curtain falls and the stage goes dark
Remember that the music was a flame, not a spark
I came, I saw, I conquered every single beat
Left my footprints tattooed on every single street

[Theme: Hustle]
Alarm clock at five, no time for hesitation
Building up my kingdom with relentless dedication
Coffee black as midnight, focus sharp as steel
Every single obstacle is just another wheel
Turning me towards destiny, I see it crystal clear
The top is getting closer with every passing year

[Theme: Confidence]
I walk in any room and own the atmosphere
Every word I speak is engineered to persevere
They tried to write me off but I'm the author of the book
One look is all it takes, go ahead and take a look
Self-made, self-paid, the blueprint is my DNA
I'm the definition of a legendary day

[Theme: Confidence]
Mirror mirror on the wall, I already know the truth
Every scar upon my skin is living breathing proof
That I survived the fires, walked through every storm
Came out the other side in legendary form
Talk is cheap but action speaks in volumes that are loud
I'm the lightning and the thunder breaking through the cloud

[Theme: Night Life]
City never sleeps and neither do the dreamers here
Midnight is the canvas, neon paint is crystal clear
Every rooftop has a view of possibilities
The skyline is a playlist full of melodies
Bass bumping from the underground to penthouses above
The night belongs to those who do the things they love

[Theme: Night Life]
Three AM the city got a different kind of pulse
Running through the avenues on pure adrenaline results
Taxicabs and limousines all racing through the dark
Every intersection is an after-midnight spark
The moon becomes the spotlight for the ones who come alive
When the sun goes down, that's when we truly thrive

[Theme: Motivation]
Get up, get out, and get something going
The seeds you plant today are the gardens you'll be growing
No excuses, no delays, just put the work in
Every champion was once a beginner who kept searching
The mountain looks impossible until you start the climb
One step at a time, one bar at a time, one rhyme at a time

[Theme: Motivation]
Stop waiting for permission, give it to yourself
Your potential's gathering dust up on the shelf
Take it down and use it, let the fire burn
Every lesson that you learn is another page to turn
The world will try to stop you, let it fuel the flame
You didn't come this far to only come this far — reclaim your name

[Theme: Reality]
Not everything that glitters has a golden core
Behind the fame and fortune there's a revolving door
The pressure of perfection weighs a thousand tons
Smiling for the cameras while inside you want to run
Reality is brutal but it keeps you grounded, friend
The real ones stay beside you from beginning to the end

[Theme: Reality]
Social media painted pictures that are never real
Filters on the trauma, nobody shows how they feel
Behind the highlight reel there's a blooper tape of pain
Sunshine after midnight only comes after the rain
Keep it one hundred, that's the code I live by
Real recognize real, no need to justify
"""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            f.write(lyrics)

        themes = ["Hustle", "Street Life", "Dreams", "Love", "Flex",
                   "Struggle", "Party", "Legacy", "Confidence",
                   "Night Life", "Motivation", "Reality"]

        line_count = len([l for l in lyrics.strip().split('\n') if l.strip() and not l.startswith('[')])
        self._save_metadata(total_lines=line_count, themes=themes, fine_tuned=False)
        print(f"✅ Created dataset: {line_count} lines across {len(themes)} themes")

    def get_stats(self):
        """Get dataset statistics"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                text = f.read()
            lines = [l for l in text.strip().split('\n') if l.strip() and not l.startswith('[')]
            themes = list(set(re.findall(r'\[Theme: (.+?)\]', text)))
            return {
                "total_lines": len(lines),
                "total_words": sum(len(l.split()) for l in lines),
                "themes": themes,
                "fine_tuned": self.metadata.get("fine_tuned", False),
                "file_size": f"{os.path.getsize(self.data_file) / 1024:.1f} KB"
            }
        except Exception:
            return {"total_lines": 0, "total_words": 0, "themes": [], "fine_tuned": False}

    def add_lyrics(self, theme: str, lyrics_text: str):
        """Add custom lyrics to the dataset"""
        with open(self.data_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\n[Theme: {theme}]\n{lyrics_text}\n")
        self._load_metadata()
        stats = self.get_stats()
        self._save_metadata(total_lines=stats["total_lines"], themes=stats["themes"])
        return stats


# ============================================
# FINE-TUNING ENGINE
# ============================================
class FineTuner:
    """Fine-tune GPT-2 on rap lyrics"""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.is_fine_tuned = False
        self.training_log = []
        self._load_model()

    def _load_model(self):
        """Load base or fine-tuned GPT-2"""
        try:
            if (MODEL_DIR / "config.json").exists():
                print("📦 Loading fine-tuned rap model...")
                self.tokenizer = GPT2Tokenizer.from_pretrained(str(MODEL_DIR))
                self.model = GPT2LMHeadModel.from_pretrained(str(MODEL_DIR))
                self.is_fine_tuned = True
                print("✅ Fine-tuned model loaded!")
            else:
                print("📦 Loading base GPT-2 model...")
                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
                    self.model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
                except Exception:
                    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                    self.model = GPT2LMHeadModel.from_pretrained('gpt2')
                print("✅ Base GPT-2 loaded (not fine-tuned yet)")

            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def fine_tune(self, data_file: str, epochs: int = 3, batch_size: int = 2, learning_rate: float = 5e-5):
        """Fine-tune GPT-2 on the lyrics dataset"""
        print(f"\n🎤 Starting fine-tuning...")
        print(f"   Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}")
        start_time = time.time()

        try:
            # Prepare dataset
            dataset = TextDataset(
                tokenizer=self.tokenizer,
                file_path=data_file,
                block_size=128
            )

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )

            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(MODEL_DIR / "checkpoints"),
                overwrite_output_dir=True,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_steps=50,
                weight_decay=0.01,
                logging_dir=str(MODEL_DIR / "logs"),
                logging_steps=10,
                save_steps=500,
                save_total_limit=2,
                prediction_loss_only=True,
                fp16=torch.cuda.is_available(),
                report_to="none",
            )

            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=dataset,
            )

            # Train
            train_result = trainer.train()

            # Save the fine-tuned model
            self.model.save_pretrained(str(MODEL_DIR))
            self.tokenizer.save_pretrained(str(MODEL_DIR))
            self.is_fine_tuned = True

            elapsed = round(time.time() - start_time, 2)
            loss = train_result.training_loss

            log_entry = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epochs": epochs,
                "loss": round(loss, 4),
                "elapsed_seconds": elapsed,
                "status": "success"
            }
            self.training_log.append(log_entry)

            print(f"✅ Fine-tuning complete!")
            print(f"   Loss: {loss:.4f}")
            print(f"   Time: {elapsed}s")
            print(f"   Model saved to: {MODEL_DIR}")

            return {
                "success": True,
                "loss": round(loss, 4),
                "elapsed": elapsed,
                "epochs": epochs,
                "message": "Fine-tuning complete! Model saved."
            }

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            print(f"❌ Fine-tuning error: {e}")
            return {
                "success": False,
                "error": str(e),
                "elapsed": elapsed
            }

    def get_status(self):
        """Get model status"""
        return {
            "is_fine_tuned": self.is_fine_tuned,
            "model_name": "GPT-2 (Fine-tuned on Rap Lyrics)" if self.is_fine_tuned else "GPT-2 (Base)",
            "model_size": f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M params",
            "training_log": self.training_log[-5:],
            "device": "CUDA" if torch.cuda.is_available() else "CPU"
        }


# ============================================
# LYRICS GENERATOR
# ============================================
class RapGenerator:
    """Generate rap lyrics from themes/prompts using few-shot in-context learning"""

    # Each theme has a seed line PLUS few-shot examples for in-context learning
    THEME_DATA = {
        "hustle": {
            "seed": "I wake up every morning with a mission to grind",
            "examples": [
                "Stack the paper up, never let them see me fall",
                "Built this empire brick by brick, standing ten feet tall",
                "Every setback is a setup for a comeback twice as strong",
                "Grinding through the struggle leaving yesterday behind",
                "No shortcuts on this highway, I do it the hard way",
                "Every hour of the grind is building up my payday",
                "Writing history in the making, this is where I belong",
                "Started from the bottom now I'm climbing every day",
                "They doubted every step but I kept walking anyway",
                "The city lights are calling and the money never sleeps",
                "I plant the seeds of greatness and I harvest what I reap",
                "Alarm clock at five, no time for hesitation",
                "Building up my kingdom with relentless dedication",
                "Coffee black as midnight, focus sharp as steel",
                "Every single obstacle is just another wheel",
                "The top is getting closer with every passing year",
            ]
        },
        "street life": {
            "seed": "Walking through the blocks where the stories never end",
            "examples": [
                "Streetlights flicker, another story unfolds",
                "The truth is always bitter but the lies feel like gold",
                "Concrete jungle where the strong survive the night",
                "Every shadow has a story, every corner has a fight",
                "Corner store philosophy from scholars of the block",
                "Midnight conversations while we're watching for the cops",
                "Sidewalk sermons from the preacher on the stoop",
                "Kids are playing hopscotch while the sirens loop",
                "Grandma's cooking dinner, smell it down the hall",
                "Murals painted beautiful on every broken wall",
                "This neighborhood made me, every crack and every scar",
                "Even broken pavement leads you to a star",
            ]
        },
        "dreams": {
            "seed": "I close my eyes and see the future that I'm building",
            "examples": [
                "Visions of the top, manifesting every night",
                "They said I'd never make it, watch me prove them wrong",
                "Dreaming with my eyes open, chasing what is real",
                "The stars are just the ceiling for the dreams I'm chasing",
                "Sleepless nights and early mornings, that's the sacrifice",
                "One day they'll tell my story and it's going to be right",
                "Chasing clouds of purple haze through silver-lined skies",
                "The vision in my head is clearer than these tired eyes",
                "The music in my soul is the only voice I hear",
                "Penthouse view, the ocean, total serenity",
                "Pouring rhymes into the mic like coffee in a cup",
                "Every single dream I had is slowly coming through",
            ]
        },
        "love": {
            "seed": "She walked into my life and changed the game forever",
            "examples": [
                "Heart on my sleeve but I wear it like armor",
                "Every beat reminds me of the way you used to smile",
                "Love is like a melody that never stops playing",
                "Through the ups and downs you stayed, never walked away",
                "Two hearts beating like a drum in perfect rhythm",
                "You're the verse I never wrote but always wanted to",
                "She walked in like a melody I'd never heard before",
                "Every word she spoke was like a key to every door",
                "Late night conversations, stars replacing ceiling lights",
                "She's the chorus to my verses, she makes everything right",
                "Her laughter is a sample that I loop inside my head",
                "Two hearts beating bass lines in a symphony of trust",
                "In a world that's filled with rust she's the gold beneath the dust",
                "Every moment with her feels like every word unsaid",
            ]
        },
        "flex": {
            "seed": "Pull up with the confidence that money cannot buy",
            "examples": [
                "Dripping head to toe, you already know the vibe",
                "Ice around my neck, shine bright when I walk in",
                "They can't match the energy, I'm on another level",
                "Earned everything I got, never had it handed to me",
                "Luxury is just the standard where I'm coming from",
                "Watch me do it bigger every single time I'm up",
                "Chain heavy, pockets loaded, yeah the lifestyle's elite",
                "Every verse I drop is fire, feel the heat in the street",
                "Diamond-studded confidence, they never seen before",
                "Every beat I step on turns into a platinum tune",
                "Hard work mixed with talent on a never-ending reel",
                "Champagne celebrations, let the winning begin",
            ]
        },
        "struggle": {
            "seed": "Through the pain and the hardship I keep pushing forward",
            "examples": [
                "Scars on my hands from the battles that I've fought",
                "They counted me out before I even had a chance",
                "Every wound is just a lesson that I carry with me",
                "Broken but not finished, I refuse to stay down",
                "The struggle made me stronger than they'll ever understand",
                "Rock bottom was the foundation where I built my throne",
                "Mama worked two jobs just to keep the lights on",
                "Hand-me-down ambition, secondhand respect",
                "I took the broken pieces and I built an architect",
                "Struggle is the teacher and the lesson is the pain",
                "Every single tear I shed became a drop of rain",
                "The weight upon my shoulders made my backbone strong",
                "This struggle is the intro to my victory song",
            ]
        },
        "party": {
            "seed": "Turn the music up tonight we celebrate the win",
            "examples": [
                "Hands up in the air, we don't care about tomorrow",
                "Bass is shaking walls, feel the rhythm in your soul",
                "Bottles popping, vibes are right, living in the moment",
                "Dance floor packed, everybody moving to the beat",
                "Tonight we own the city, let them hear us from the top",
                "No regrets, no worries, just the music and the night",
                "DJ spin that record let the bass line drop",
                "Whole club jumping, yeah we never gonna stop",
                "Confetti in the spotlight, magic everywhere",
                "Dancing like tomorrow doesn't even exist",
                "Neon lights and laser beams cutting through the mist",
                "Life is just a party when you're locked into the groove",
            ]
        },
        "legacy": {
            "seed": "When I'm gone remember me through every verse I wrote",
            "examples": [
                "Building something bigger than myself, that's the mission",
                "My words will echo long after I'm gone",
                "Plant the seeds today so the next generation eats",
                "History remembers those who dared to be different",
                "Leave a mark so deep they can't erase my name",
                "The legacy is more than money, it's the impact that I made",
                "A library of verses built from joy and fears",
                "The gift of honest music is the greatest I can give",
                "Carve my name in marble made of beats and rhymes",
                "When the final curtain falls and the stage goes dark",
                "I came, I saw, I conquered every single beat",
                "Left my footprints tattooed on every single street",
            ]
        },
        "confidence": {
            "seed": "I walk in any room and own the atmosphere",
            "examples": [
                "Self-made, self-paid, I don't need validation",
                "Crown on my head but I built it with my hands",
                "Speak with conviction, every word is like a weapon",
                "They tried to dim my light but I just shine brighter",
                "Unstoppable force, I don't know how to quit",
                "Born to lead, I set the pace and they follow",
                "Every word I speak is engineered to persevere",
                "Mirror mirror on the wall, I already know the truth",
                "I survived the fires, walked through every storm",
                "Came out the other side in legendary form",
                "Talk is cheap but action speaks in volumes that are loud",
                "I'm the lightning and the thunder breaking through the cloud",
            ]
        },
        "night life": {
            "seed": "The city comes alive when the sun goes down",
            "examples": [
                "Neon lights reflecting off the midnight rain",
                "Shadows dancing underneath the city glow",
                "Late night drive with the windows down and bass up",
                "Every corner has a story after dark",
                "The moon is watching while we run the town tonight",
                "Night owls plotting while the rest of them are sleeping",
                "Midnight is the canvas, neon paint is crystal clear",
                "The skyline is a playlist full of melodies",
                "Three AM the city got a different kind of pulse",
                "Running through the avenues on pure adrenaline",
                "Every intersection is an after-midnight spark",
                "When the sun goes down, that's when we truly thrive",
            ]
        },
        "motivation": {
            "seed": "Get up and chase the dream, no more excuses left",
            "examples": [
                "Rise above the noise, stay focused on the goal",
                "Discipline is freedom, that's the truth they never teach",
                "Every champion was once a fighter nobody believed in",
                "The grind doesn't stop just because you're tired",
                "Keep moving forward, backward isn't an option",
                "Light the fire inside and never let it die",
                "The seeds you plant today are the gardens you'll be growing",
                "The mountain looks impossible until you start the climb",
                "One step at a time, one bar at a time, one rhyme at a time",
                "Stop waiting for permission, give it to yourself",
                "Your potential's gathering dust up on the shelf",
                "You didn't come this far to only come this far",
            ]
        },
        "reality": {
            "seed": "Behind the fame and glory there's a human being",
            "examples": [
                "Real life hits different when the cameras stop rolling",
                "Not everything that shines is meant to be gold",
                "The truth will set you free but first it's going to hurt",
                "Smile on the outside but the weight is getting heavy",
                "Life is not a highlight reel, remember that",
                "Reality check, not everything goes according to plan",
                "Filters on the trauma, nobody shows how they feel",
                "Sunshine after midnight only comes after the rain",
                "Keep it one hundred, that's the code I live by",
                "Real recognize real, no need to justify",
                "Not everything that glitters has a golden core",
                "The pressure of perfection weighs a thousand tons",
            ]
        },
        "freestyle": {
            "seed": "Let me spit it real quick, no filter and no script",
            "examples": [
                "Off the top of my dome, every word is raw",
                "No rehearsal, no practice, just the mic and me",
                "Words are flowing like a river that can't be stopped",
                "Freestyle king, I don't need a written verse",
                "Watch me flip the syllables and bend the flow",
                "This is how it sounds when talent meets the moment",
                "I ride the beat like waves crashing on the shore",
                "Every punchline hits different when it's off the dome",
                "Lyrically untouchable, rhythmically insane",
                "The mic is my canvas and the words are my art",
                "Improvising greatness, that's a skill you can't fake",
                "When the beat drops I go in, there's no going back",
            ]
        },
    }

    # Keep backward compat
    THEME_PROMPTS = {k: v["seed"] for k, v in THEME_DATA.items()}

    BEAT_STYLES = {
        "boom_bap": {"temp": 0.85, "top_k": 50, "rep_penalty": 1.3, "desc": "Classic 90s hip-hop feel"},
        "trap": {"temp": 0.95, "top_k": 60, "rep_penalty": 1.2, "desc": "Modern trap with heavy bass"},
        "lo_fi": {"temp": 0.80, "top_k": 40, "rep_penalty": 1.4, "desc": "Chill, mellow vibes"},
        "drill": {"temp": 1.0, "top_k": 70, "rep_penalty": 1.1, "desc": "Dark, aggressive energy"},
        "old_school": {"temp": 0.75, "top_k": 45, "rep_penalty": 1.5, "desc": "Golden era storytelling"},
        "melodic": {"temp": 0.9, "top_k": 55, "rep_penalty": 1.25, "desc": "Smooth, melodic flow"},
    }

    def __init__(self, fine_tuner: FineTuner, data_manager: 'LyricsDataManager'):
        self.fine_tuner = fine_tuner
        self.data_manager = data_manager

    def _load_corpus_lines(self, theme: str) -> list:
        """Load matching lines from the lyrics corpus file"""
        try:
            with open(self.data_manager.data_file, 'r', encoding='utf-8') as f:
                text = f.read()
            # Parse corpus into theme-grouped lines
            blocks = re.split(r'\[Theme:\s*(.*?)\]', text)
            theme_lines = []
            all_lines = []
            for i in range(1, len(blocks), 2):
                block_theme = blocks[i].strip().lower()
                lines = [l.strip() for l in blocks[i+1].strip().split('\n') if l.strip() and len(l.strip()) > 10]
                all_lines.extend(lines)
                if block_theme == theme.lower():
                    theme_lines.extend(lines)
            return theme_lines, all_lines
        except Exception:
            return [], []

    def generate(self, theme: str = "freestyle", beat_style: str = "boom_bap",
                 num_lines: int = 16, custom_prompt: str = "", temperature: float = None):
        """Generate rap lyrics using hybrid approach: GPT-2 + corpus"""
        start_time = time.time()

        model = self.fine_tuner.model
        tokenizer = self.fine_tuner.tokenizer

        # Get beat style params
        beat = self.BEAT_STYLES.get(beat_style, self.BEAT_STYLES["boom_bap"])
        temp = temperature if temperature else beat["temp"]
        top_k = beat["top_k"]
        rep_penalty = beat["rep_penalty"]

        theme_lower = theme.lower()
        theme_info = self.THEME_DATA.get(theme_lower, self.THEME_DATA["freestyle"])

        if self.fine_tuner.is_fine_tuned:
            # ===== FINE-TUNED MODEL: use standard generation =====
            if custom_prompt.strip():
                prompt_text = f"[Theme: {theme.title()}]\n{custom_prompt.strip()}"
            else:
                prompt_text = f"[Theme: {theme.title()}]\n{theme_info['seed']}"

            input_ids = tokenizer.encode(prompt_text, return_tensors='pt')
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max(num_lines * 25, 250),
                    temperature=temp,
                    top_k=top_k,
                    top_p=0.92,
                    repetition_penalty=rep_penalty,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = tokenizer.decode(output[0], skip_special_tokens=True)
            lyrics = self._format_lyrics(generated, prompt_text, num_lines, theme_info)
        else:
            # ===== BASE MODEL: Hybrid approach =====
            # Use corpus lines + GPT-2 to create a blended verse
            lyrics = self._hybrid_generate(
                theme=theme, theme_info=theme_info,
                custom_prompt=custom_prompt,
                num_lines=num_lines,
                model=model, tokenizer=tokenizer,
                temp=temp, top_k=top_k, rep_penalty=rep_penalty
            )

        elapsed = round(time.time() - start_time, 2)
        analysis = self._analyze_lyrics(lyrics)

        return {
            "success": True,
            "lyrics": lyrics,
            "theme": theme.title(),
            "beat_style": beat_style.replace("_", " ").title(),
            "beat_desc": beat["desc"],
            "num_lines": len(lyrics.strip().split('\n')),
            "word_count": len(lyrics.split()),
            "generation_time": elapsed,
            "temperature": temp,
            "analysis": analysis,
            "is_fine_tuned": self.fine_tuner.is_fine_tuned
        }

    def _hybrid_generate(self, theme, theme_info, custom_prompt, num_lines,
                          model, tokenizer, temp, top_k, rep_penalty):
        """Hybrid generation: feed GPT-2 real rap lines from corpus as prompt,
        then generate continuations line-by-line to maintain rap format."""
        theme_lines, all_lines = self._load_corpus_lines(theme)
        seed_lines = theme_lines if theme_lines else all_lines
        random.shuffle(seed_lines)

        # Collect generated bars
        generated_bars = []

        # Start with custom prompt or seed
        if custom_prompt.strip():
            generated_bars.append(custom_prompt.strip())
        else:
            generated_bars.append(theme_info["seed"])

        # Use corpus examples as context, generate new lines one at a time
        context_lines = seed_lines[:6]  # Use 6 corpus lines as context

        for i in range(num_lines - 1):
            # Build a prompt: several real rap lines + the ones we've generated so far
            # This teaches GPT-2 the line-by-line format in-context
            prompt_lines = context_lines + generated_bars[-4:]  # last 4 generated
            prompt_text = '\n'.join(prompt_lines) + '\n'

            input_ids = tokenizer.encode(prompt_text, return_tensors='pt')
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

            try:
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=40,  # Just enough for ~1 line
                        temperature=temp,
                        top_k=top_k,
                        top_p=0.92,
                        repetition_penalty=rep_penalty,
                        do_sample=True,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                new_text = tokenizer.decode(output[0], skip_special_tokens=True)
                # Extract only the NEW text after the prompt
                if new_text.startswith(prompt_text):
                    new_text = new_text[len(prompt_text):]

                # Get just the first line from generated text
                new_line = self._extract_first_bar(new_text)
                if new_line and new_line not in generated_bars and new_line not in context_lines:
                    generated_bars.append(new_line)
                else:
                    # If GPT-2 produced junk, use a corpus line we haven't used
                    unused = [l for l in seed_lines if l not in generated_bars]
                    if unused:
                        generated_bars.append(unused[0])
                        seed_lines.remove(unused[0])
                    else:
                        # Use theme examples
                        ex = theme_info["examples"]
                        unused_ex = [e for e in ex if e not in generated_bars]
                        if unused_ex:
                            generated_bars.append(random.choice(unused_ex))
            except Exception:
                # Fallback to corpus
                unused = [l for l in seed_lines if l not in generated_bars]
                if unused:
                    generated_bars.append(unused[0])

            if len(generated_bars) >= num_lines:
                break

        # Pad if needed
        while len(generated_bars) < num_lines:
            unused = [l for l in seed_lines + theme_info["examples"] if l not in generated_bars]
            if unused:
                generated_bars.append(random.choice(unused))
            else:
                break

        return '\n'.join(generated_bars[:num_lines])

    def _extract_first_bar(self, text: str) -> str:
        """Extract the first valid rap bar from generated text"""
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Clean up
            line = re.sub(r'\[.*?\]', '', line).strip()
            line = re.sub(r'"[^"]*?"\s*[-–—]\s*[A-Z].*', '', line).strip()
            line = re.sub(r'\(.*?\)', '', line).strip()
            line = re.sub(r'https?://\S+', '', line).strip()
            line = re.sub(r'\.{2,}', '', line).strip()

            # Must be reasonable length (5-14 words)
            word_count = len(line.split())
            if word_count < 5 or word_count > 14:
                continue

            lower = line.lower()

            # REJECT: contains quotation marks
            if '"' in line or '\u201c' in line or '\u201d' in line:
                continue

            # REJECT: contains these junk markers
            junk_patterns = [
                r'http|www\.|\.com|\.org',
                r'\b(anime|manga|novel|chapter|episode|movie|film|book)\b',
                r':\)|;\)|:D|lol|omg',
                r'".*"',
                r' - ',  # prose dashes
                r'\b(born|died|married)\b.*\d{4}',
                r'\b(scientist|professor|university|research|study)\b',
                r'\.{3}|…',  # trailing ellipsis = prose
            ]
            if any(re.search(p, lower) for p in junk_patterns):
                continue

            # REJECT: ends with period, question mark, or exclamation (rap bars don't)
            if line.endswith(('.', '?', '!')):
                continue

            # POSITIVE CHECK: must contain at least 2 "rap signal" words/patterns
            # This separates genuine rap bars from generic prose
            rap_signals = [
                r'\b(grind|hustle|stack|bars|flow|spit|verse|rhyme|beat)\b',
                r'\b(money|cash|paper|bread|chains|ice|drip|fire)\b',
                r'\b(dream|chase|rise|climb|fight|battle|struggle|pain)\b',
                r'\b(king|queen|crown|throne|empire|legacy)\b',
                r'\b(street|block|hood|city|corner|midnight|neon)\b',
                r'\b(heart|soul|mind|eyes|voice|hands|scars)\b',
                r'\b(never|every|always|forever)\b',
                r'\b(real|truth|fake|prove|believe)\b',
                r'\b(mic|booth|record|lyric)\b',
                r'\b(shine|glow|flame|lightning|thunder)\b',
                r'\b(vibe|energy|power|strength|grind)\b',
            ]
            signal_count = sum(1 for p in rap_signals if re.search(p, lower))
            if signal_count < 2:
                continue

            # Capitalize and clean
            line = line[0].upper() + line[1:]
            return line

        return ""

    def _format_lyrics(self, raw_text: str, prompt_text: str, target_lines: int, theme_info: dict = None) -> str:
        """Clean and format generated lyrics into proper rap bars"""
        # Remove the prompt prefix
        text = raw_text
        if text.startswith(prompt_text):
            text = text[len(prompt_text):]

        # Remove any theme markers, quotes, attributions, URLs
        text = re.sub(r'\[Theme:.*?\]', '', text)
        text = re.sub(r'\[.*?\]', '', text)  # Remove any brackets
        text = re.sub(r'https?://\S+', '', text)  # URLs
        text = re.sub(r'[-–—]{2,}', '', text)  # Dashes
        text = re.sub(r'"[^"]*?" ?[-–—] ?[A-Z][^\n]*', '', text)  # Quoted attributions like "text" - Author
        text = re.sub(r'\([^)]*\)', '', text)  # Parenthetical

        # First: try splitting by newlines (works for fine-tuned model)
        lines = text.strip().split('\n')
        cleaned = []
        for line in lines:
            line = self._clean_line(line)
            if line:
                cleaned.append(line)
            if len(cleaned) >= target_lines:
                break

        # If base model produced prose (few/no newlines), split smartly
        if len(cleaned) < target_lines // 2:
            full_text = ' '.join(cleaned) if cleaned else text.strip()
            cleaned = self._split_prose_to_bars(full_text, target_lines)

        # If still not enough lines, pad with themed examples
        if len(cleaned) < target_lines and theme_info:
            examples = theme_info.get("examples", [])
            random.shuffle(examples)
            for ex in examples:
                if ex not in cleaned:
                    cleaned.append(ex)
                if len(cleaned) >= target_lines:
                    break

        # If we still have nothing, use the seed
        if not cleaned:
            cleaned = [theme_info["seed"]] if theme_info else ["No lyrics generated"]

        return '\n'.join(cleaned[:target_lines])

    def _clean_line(self, line: str) -> str:
        """Clean a single line of lyrics"""
        line = line.strip()
        if not line:
            return ""
        # Skip lines that look like prose/article content
        skip_patterns = [
            r'^(http|www\.)',  # URLs
            r'^(source|credit|image|photo|via|copyright)',  # Attributions
            r'^\d+\.',  # Numbered lists  
            r'^(he |she |they |it |the |a |an |this |that |these |those ).*\.$',  # Prose sentences ending in period
        ]
        for pat in skip_patterns:
            if re.match(pat, line, re.IGNORECASE):
                return ""
        # Must have at least 4 words to be a bar
        if len(line.split()) < 4:
            return ""
        # Capitalize first letter
        line = line[0].upper() + line[1:]
        # Remove trailing period (rap bars don't end with periods usually)
        line = line.rstrip('.')
        return line

    def _split_prose_to_bars(self, text: str, target_lines: int) -> list:
        """Split prose text into rap-bar-length lines"""
        # First try splitting by sentence boundaries
        fragments = re.split(r'(?<=[.!?,;:…])\s+', text)
        cleaned = []
        for frag in fragments:
            frag = frag.strip()
            if not frag or len(frag.split()) < 3:
                continue
            words = frag.split()
            # If fragment is long, split into ~8-word chunks
            if len(words) > 12:
                for i in range(0, len(words), 8):
                    chunk = ' '.join(words[i:i+8]).strip()
                    if chunk and len(chunk.split()) >= 3:
                        chunk = chunk[0].upper() + chunk[1:]
                        chunk = chunk.rstrip('.')
                        cleaned.append(chunk)
                    if len(cleaned) >= target_lines:
                        return cleaned
            else:
                frag = frag[0].upper() + frag[1:]
                frag = frag.rstrip('.')
                cleaned.append(frag)
            if len(cleaned) >= target_lines:
                return cleaned

        # Last resort: split every ~8 words
        if len(cleaned) < 2:
            words = text.split()
            cleaned = []
            for i in range(0, len(words), 8):
                chunk = ' '.join(words[i:i+8]).strip()
                if chunk and len(chunk.split()) >= 3:
                    chunk = chunk[0].upper() + chunk[1:]
                    chunk = chunk.rstrip('.')
                    cleaned.append(chunk)
                if len(cleaned) >= target_lines:
                    break

        return cleaned

    def _score_lyrics(self, lyrics: str) -> float:
        """Score lyrics quality: prefer more lines, more unique words, no weird tokens"""
        lines = [l for l in lyrics.split('\n') if l.strip()]
        words = lyrics.split()
        if not words:
            return 0
        unique = len(set(w.lower() for w in words))
        line_score = len(lines) * 2
        vocab_score = unique / max(len(words), 1)
        # Penalize gibberish (non-ascii, too many special chars)
        gibberish_penalty = len(re.findall(r'[^\w\s,\'!?.\-]', lyrics)) * 0.5
        return line_score + vocab_score * 10 - gibberish_penalty

    def _analyze_lyrics(self, lyrics: str) -> dict:
        """Analyze the generated lyrics"""
        lines = lyrics.strip().split('\n')
        words = lyrics.split()

        # Find potential rhymes (last words of consecutive lines)
        last_words = [line.strip().split()[-1].lower().rstrip('.,!?') if line.strip().split() else '' for line in lines]
        rhyme_pairs = 0
        for i in range(0, len(last_words) - 1, 2):
            if last_words[i] and last_words[i + 1]:
                # Simple rhyme check: same ending
                if last_words[i][-2:] == last_words[i + 1][-2:] and last_words[i] != last_words[i + 1]:
                    rhyme_pairs += 1

        # Vocabulary richness
        unique_words = len(set(w.lower().rstrip('.,!?') for w in words))
        vocab_richness = round(unique_words / max(len(words), 1) * 100, 1)

        # Average syllables per line (rough estimate)
        def estimate_syllables(word):
            word = word.lower().rstrip('.,!?')
            count = len(re.findall(r'[aeiouy]+', word))
            return max(count, 1)

        avg_syllables = round(np.mean([
            sum(estimate_syllables(w) for w in line.split())
            for line in lines if line.strip()
        ]), 1) if lines else 0

        return {
            "line_count": len(lines),
            "word_count": len(words),
            "rhyme_pairs": rhyme_pairs,
            "vocab_richness": vocab_richness,
            "avg_syllables_per_line": avg_syllables,
            "unique_words": unique_words
        }

    def get_themes(self):
        return list(self.THEME_PROMPTS.keys())

    def get_beat_styles(self):
        return {k: v["desc"] for k, v in self.BEAT_STYLES.items()}


# ============================================
# INITIALIZE
# ============================================
print("=" * 50)
print("🎤 Rap Lyric Generator — Fine-Tuned LLM")
print("=" * 50)

data_manager = LyricsDataManager()
fine_tuner = FineTuner()
rap_generator = RapGenerator(fine_tuner, data_manager)

print(f"\n🌐 Starting server at http://localhost:5000")
print("=" * 50)


# ============================================
# ROUTES
# ============================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    """Generate rap lyrics"""
    data = request.json
    theme = data.get('theme', 'freestyle')
    beat_style = data.get('beat_style', 'boom_bap')
    num_lines = min(int(data.get('num_lines', 16)), 32)
    custom_prompt = data.get('custom_prompt', '')
    temperature = data.get('temperature', None)
    if temperature:
        temperature = float(temperature)

    result = rap_generator.generate(
        theme=theme,
        beat_style=beat_style,
        num_lines=num_lines,
        custom_prompt=custom_prompt,
        temperature=temperature
    )
    return jsonify(result)


@app.route('/fine-tune', methods=['POST'])
def fine_tune():
    """Fine-tune the model on lyrics dataset"""
    data = request.json
    epochs = min(int(data.get('epochs', 3)), 10)
    batch_size = int(data.get('batch_size', 2))
    learning_rate = float(data.get('learning_rate', 5e-5))

    result = fine_tuner.fine_tune(
        data_file=str(data_manager.data_file),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    if result["success"]:
        data_manager._save_metadata(fine_tuned=True)

    return jsonify(result)


@app.route('/status')
def status():
    """Get system status"""
    model_status = fine_tuner.get_status()
    dataset_stats = data_manager.get_stats()
    return jsonify({
        "model": model_status,
        "dataset": dataset_stats,
        "themes": rap_generator.get_themes(),
        "beat_styles": rap_generator.get_beat_styles()
    })


@app.route('/add-lyrics', methods=['POST'])
def add_lyrics():
    """Add custom lyrics to the dataset"""
    data = request.json
    theme = data.get('theme', 'Custom')
    lyrics_text = data.get('lyrics', '').strip()

    if not lyrics_text:
        return jsonify({"error": "Please provide lyrics"}), 400

    stats = data_manager.add_lyrics(theme, lyrics_text)
    return jsonify({"success": True, "stats": stats, "message": "Lyrics added to dataset!"})


@app.route('/themes')
def get_themes():
    return jsonify({
        "themes": rap_generator.get_themes(),
        "beat_styles": rap_generator.get_beat_styles()
    })


# ============================================
# RUN
# ============================================
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
