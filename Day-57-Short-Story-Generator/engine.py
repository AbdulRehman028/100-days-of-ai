"""
StoryEngine — LangChain LCEL story generation engine.
Handles model loading, chain construction, generation pipeline,
and all post-processing (cleaning, trimming, analysis).
"""

import re
import json
import time
import random
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import (
    PRIMARY_MODEL, FALLBACK_MODEL, LAST_RESORT_MODEL,
    GENRES, TONES, LENGTHS, GENRE_SETTINGS, GENRE_PROTAGONISTS,
    STORIES_DIR,
)
from seeds import build_narrative_seed


class StoryEngine:
    """
    Multi-chain story generator using LangChain LCEL + TinyLlama-1.1B-Chat.

    With an instruction-following model we can use direct prompts
    instead of the narrative-seed hack GPT-2 required.

    Pipeline:
      1. Plan   (deterministic)  — build outline, pick setting/character
      2. Write  (Story Chain)    — LLM generates complete story
      3. Name   (Title Chain)    — LLM generates a title
      4. Polish (deterministic)  — clean, format, analyse
    """

    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.pipe = None
        self.model_name = None
        self.model_type = None  # "causal" or "seq2seq"
        self.chains = {}
        self._load_model()
        self._build_chains()

    # ─────────────────────────────
    # MODEL LOADING
    # ─────────────────────────────
    def _load_model(self):
        """Try TinyLlama → GPT-2 Medium → Flan-T5-Base."""

        # --- Attempt 1: TinyLlama-1.1B-Chat (instruction-following) ---
        try:
            print(f"Trying {PRIMARY_MODEL}...")
            tok = AutoTokenizer.from_pretrained(PRIMARY_MODEL, local_files_only=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                PRIMARY_MODEL, local_files_only=True, torch_dtype=torch.float32)
            self.tokenizer = tok
            self.model_name = PRIMARY_MODEL
            self.model_type = "causal"

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.pipe = pipeline(
                "text-generation",
                model=mdl,
                tokenizer=self.tokenizer,
                max_new_tokens=500,
                temperature=0.8, top_k=50, top_p=0.92,
                repetition_penalty=1.15, no_repeat_ngram_size=3,
                do_sample=True, return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            param_count = sum(p.numel() for p in mdl.parameters()) / 1e6
            print(f"   Loaded {PRIMARY_MODEL} ({param_count:.0f}M params)")
            self.llm = HuggingFacePipeline(pipeline=self.pipe)
            print(f"Model ready: {self.model_name} [{self.model_type}]")
            return
        except Exception as e:
            print(f"   TinyLlama not ready: {e}")

        # --- Attempt 2: GPT-2 Medium (creative text, 355M) ---
        try:
            print(f"Trying {FALLBACK_MODEL}...")
            tok = AutoTokenizer.from_pretrained(FALLBACK_MODEL, local_files_only=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                FALLBACK_MODEL, local_files_only=True, torch_dtype=torch.float32)
            self.tokenizer = tok
            self.model_name = FALLBACK_MODEL
            self.model_type = "causal"

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.pipe = pipeline(
                "text-generation",
                model=mdl,
                tokenizer=self.tokenizer,
                max_new_tokens=500,
                min_new_tokens=100,
                temperature=0.65, top_k=30, top_p=0.85,
                repetition_penalty=1.4, no_repeat_ngram_size=4,
                do_sample=True, return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            param_count = sum(p.numel() for p in mdl.parameters()) / 1e6
            print(f"   Loaded {FALLBACK_MODEL} ({param_count:.0f}M params)")
            self.llm = HuggingFacePipeline(pipeline=self.pipe)
            print(f"Model ready: {self.model_name} [{self.model_type}]")
            return
        except Exception as e:
            print(f"   GPT-2 Medium not available: {e}")

        # --- Attempt 3: Flan-T5-Base (instruction-following, 250M) ---
        print(f"Trying {LAST_RESORT_MODEL}...")
        try:
            tok = AutoTokenizer.from_pretrained(LAST_RESORT_MODEL, local_files_only=True)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(
                LAST_RESORT_MODEL, local_files_only=True, torch_dtype=torch.float32)
        except Exception:
            print(f"   Downloading {LAST_RESORT_MODEL}...")
            tok = AutoTokenizer.from_pretrained(LAST_RESORT_MODEL)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(
                LAST_RESORT_MODEL, torch_dtype=torch.float32)

        self.tokenizer = tok
        self.model_name = LAST_RESORT_MODEL
        self.model_type = "seq2seq"

        self.pipe = pipeline(
            "text2text-generation",
            model=mdl,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            min_new_tokens=150,
            temperature=0.9, top_k=60, top_p=0.95,
            repetition_penalty=1.2, no_repeat_ngram_size=3,
            do_sample=True,
        )
        param_count = sum(p.numel() for p in mdl.parameters()) / 1e6
        print(f"   Loaded {LAST_RESORT_MODEL} ({param_count:.0f}M params)")
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        print(f"Model ready: {self.model_name} [{self.model_type}]")

    # ─────────────────────────────
    # LCEL CHAIN CONSTRUCTION
    # ─────────────────────────────
    def _build_chains(self):
        """Build LCEL chains — prompt format depends on model type."""
        parser = StrOutputParser()

        if self.model_name == PRIMARY_MODEL:
            # ── TinyLlama: single chain with instruction prompt ──
            story_tpl = (
                "<|system|>\n"
                "You are a talented fiction author who writes vivid, immersive "
                "short stories. {style_hint} "
                "Write in third person with rich sensory details and natural "
                "dialogue. Every story must have a compelling beginning, "
                "rising tension, and a satisfying ending.</s>\n"
                "<|user|>\n"
                "Write a complete {genre} short story of approximately "
                "{word_target} words. Tone: {tone}.\n\n"
                "Setting: {setting}\n"
                "Main character: {protagonist}\n"
                "Premise: {plot}\n\n"
                "Begin the story immediately with an engaging opening scene. "
                "Do not include a title, headers, or any commentary. "
                "Write only the story.</s>\n"
                "<|assistant|>\n"
            )
            self.chains["story"] = (
                PromptTemplate(
                    input_variables=["genre", "tone", "word_target",
                                     "plot", "style_hint", "setting", "protagonist"],
                    template=story_tpl,
                ) | self.llm | parser
            )
            title_tpl = (
                "<|system|>\n"
                "You create short, evocative titles for fiction. "
                "Respond with only the title, nothing else.</s>\n"
                "<|user|>\n"
                "Create a creative title (2-5 words) for this {genre} story:\n\n"
                "\"{excerpt}\"\n\n"
                "Title:</s>\n"
                "<|assistant|>\n"
            )
            self.chains["title"] = (
                PromptTemplate(input_variables=["excerpt", "genre"], template=title_tpl)
                | self.llm | parser
            )

        elif self.model_name == FALLBACK_MODEL:
            # ── GPT-2 Medium: single chain with long immersive seed ──
            self.chains["story"] = (
                PromptTemplate(
                    input_variables=["narrative_seed"],
                    template="{narrative_seed}",
                ) | self.llm | parser
            )
            self.chains["title"] = (
                PromptTemplate(
                    input_variables=["excerpt", "genre"],
                    template="A {genre} story titled \"",
                ) | self.llm | parser
            )

        else:
            # ── Flan-T5: instruction format ──
            story_tpl = (
                "Write a vivid {genre} short story of about {word_target} words. "
                "{style_hint}\n\n"
                "Tone: {tone}\nSetting: {setting}\n"
                "Main character: {protagonist}\nPremise: {plot}\n\n"
                "Write the complete story with rich details and dialogue:"
            )
            self.chains["story"] = (
                PromptTemplate(
                    input_variables=["genre", "tone", "word_target",
                                     "plot", "style_hint", "setting", "protagonist"],
                    template=story_tpl,
                ) | self.llm | parser
            )
            self.chains["title"] = (
                PromptTemplate(
                    input_variables=["excerpt", "genre"],
                    template="Create a creative title (2-5 words) for this {genre} story:\n\n\"{excerpt}\"\n\nTitle:",
                ) | self.llm | parser
            )

        chain_names = list(self.chains.keys())
        print(f"LangChain LCEL chains built [{self.model_name}]: {' -> '.join(chain_names)}")

    # ─────────────────────────────
    # OUTPUT CLEANING
    # ─────────────────────────────
    @staticmethod
    def _clean_output(text: str) -> str:
        """Clean LLM output: strip chat tokens, web artifacts, meta-text."""
        # Strip leftover chat tokens
        for tok in ['</s>', '<|assistant|>', '<|user|>', '<|system|>', '<s>',
                     '<|end|>', '<|endoftext|>']:
            text = text.replace(tok, '')

        # Strip metadata primer markers [Genre: ...] [Plot: ...]
        text = re.sub(r'\[Genre:[^\]]*\]', '', text)
        text = re.sub(r'\[Plot:[^\]]*\]', '', text)
        text = re.sub(r'^A \w+ story about [^\n]+\n+', '', text, flags=re.IGNORECASE)

        # Remove non-printable / garbage Unicode (keep basic ASCII + common punct)
        text = re.sub(r'[^\x20-\x7E\n\r\t\u2013\u2014\u2018\u2019\u201C\u201D\u2026]', '', text)

        # Strip web / metadata artifacts
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'Copyright.*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bAdvertisement\b.*', '', text, flags=re.IGNORECASE)

        # Remove meta-commentary / instructions that leaked
        lines = text.split('\n')
        story_lines = []
        for ln in lines:
            s = ln.strip()
            if not s:
                story_lines.append('')
                continue
            if re.match(
                r'^(Genre|Tone|Style|Setting|Plot|Outline|Chapter \d|Title|Note|Write a|Author|Source|Tags|Published|END|---)',
                s, re.IGNORECASE,
            ):
                continue
            if s.startswith('#') or s.startswith('http'):
                continue
            story_lines.append(s)

        text = '\n'.join(story_lines).strip()

        # Fix excessive punctuation
        text = re.sub(r'\.{4,}', '...', text)
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)

        return text.strip()

    @staticmethod
    def _quality_trim(text: str) -> str:
        """Detect quality cliffs (franchise contamination, meta-text, news) and trim."""
        # Hard-stop patterns: cut everything from here onward
        hard_stop = [
            r'\b(Skyrim|Elder Scrolls|Dovahkiin|Dragonborn|Tamriel)\b',
            r'\b(RWBY|Rias|Kurono|Gremory|Akeno)\b',
            r'\b(Frozen|Elsa|Anna|Arendelle|Olaf)\b',
            r'\b(Harry Potter|Hogwarts|Voldemort|Dumbledore|Hermione)\b',
            r'\b(Star Wars|Jedi|Sith|Darth|Skywalker)\b',
            r'\b(Pokemon|Pikachu|Naruto|Sasuke|Goku|Dragon Ball)\b',
            r'\b(Marvel|Avengers|Thanos|Tony Stark|Iron Man)\b',
            r'\b(Game of Thrones|Westeros|Lannister|Winterfell)\b',
            r'\b(Lord of the Rings|Gandalf|Frodo|Mordor|Sauron)\b',
            r'\b(Zelda|Hyrule|Link|Minecraft|Fortnite|Overwatch)\b',
            r'\b(PlayStation|Xbox|Nintendo|Steam|DLC|mod|quest log)\b',
        ]
        # Soft-skip patterns: skip this sentence only
        soft_skip = [
            r'\bChapter \d+\b',
            r'\bPart \d+\b',
            r'\bEpisode \d+\b',
            r'\b(Prologue|Epilogue):\b',
            r'\b(Reddit|Twitter|Facebook|YouTube|Wikipedia|Instagram|TikTok)\b',
            r'\b(click here|subscribe|read more|next page|sign up)\b',
            r'\b(www\.|\.com|\.org|\.net|\.io)\b',
            r'\b(according to|the article|the report|researchers say)\b',
            r'\b(percent|million|billion) (increase|decrease|growth|decline)\b',
            r'\bThe (game|movie|film|series|show|anime|manga|novel) (is|was|has)\b',
            r'\b(In this (essay|article|review|post|chapter|guide|tutorial))\b',
            r'\b(Loading|Buffering|Error \d|Page \d|404|login)\b',
            r'\b(Copyright|All rights reserved|Terms of|Privacy Policy)\b',
            r'\bTable of Contents\b',
            r'\b(Fig\.|Figure \d|Table \d|Appendix)\b',
            r'\b(Credits|Cast|Directed by|Produced by|Written by)\b',
            r'^\s*\d+[\.\)]\s',
            r'\b(FAQ|Q&A|Pro tip|Disclaimer)\b',
            r'\bFrom "',
            r'\bNote:',
            r'\bSee also',
            r'\b(by [A-Z][a-z]+ [A-Z][a-z]+)\)',
        ]
        hard_re = '|'.join(f'({p})' for p in hard_stop)
        soft_re = '|'.join(f'({p})' for p in soft_skip)

        sentences = re.split(r'(?<=[.!?])\s+', text)
        clean = []
        consecutive_skips = 0

        for s in sentences:
            # Hard stop: franchise contamination → cut everything
            if re.search(hard_re, s, re.IGNORECASE):
                break
            # Soft skip: meta-text → skip this sentence
            if re.search(soft_re, s, re.IGNORECASE):
                consecutive_skips += 1
                if consecutive_skips >= 2:
                    break  # Two bad sentences in a row → stop
                continue
            consecutive_skips = 0
            clean.append(s)

        if len(clean) >= 2:
            return ' '.join(clean)
        return text  # If too short after trim, keep original

    @staticmethod
    def _trim_to_complete(text: str) -> str:
        """Trim to the last complete sentence."""
        if not text:
            return text
        # Find last sentence-ending punctuation (. ! ?)
        # Also accept closing quote immediately after sentence-ender: ." or !'
        last = -1
        for i in range(len(text) - 1, -1, -1):
            ch = text[i]
            if ch in '.!?':
                # Check if followed by closing quote
                if i + 1 < len(text) and text[i + 1] in '"\'\u201D\u2019':
                    last = i + 1
                else:
                    last = i
                break
        if last > len(text) // 4:
            text = text[:last + 1]
        return text.strip()

    @staticmethod
    def _deduplicate(text: str) -> str:
        """Remove duplicate sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        seen = set()
        unique = []
        for s in sentences:
            key = re.sub(r'\s+', ' ', s.strip().lower())
            if len(key) < 8:
                unique.append(s)
                continue
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return ' '.join(unique)

    # ─────────────────────────────
    # OUTLINE (deterministic)
    # ─────────────────────────────
    def _build_outline(self, plot, genre, setting, protagonist, tone):
        return (
            f"Genre: {genre}\n"
            f"Setting: {setting}\n"
            f"Protagonist: {protagonist}\n"
            f"Tone: {tone}\n"
            f"Premise: {plot}\n"
            f"Arc: Introduction → Conflict → Resolution"
        )

    # ─────────────────────────────
    # MAIN GENERATION PIPELINE
    # ─────────────────────────────
    def generate_story(self, plot: str, genre: str = "fantasy",
                       tone: str = "light", length: str = "short",
                       creativity: float = 0.82) -> dict:
        """
        Full pipeline:
          1. Plan   (deterministic)
          2. Write  (Story LCEL Chain(s) — multi-chunk for GPT-2)
          3. Name   (Title LCEL Chain)
          4. Polish (deterministic — clean, format, analyse)
        """
        start_time = time.time()
        genre_info = GENRES.get(genre, GENRES["fantasy"])
        genre_label = genre_info["label"]
        tone_label = TONES.get(tone, "Light and uplifting")
        length_info = LENGTHS.get(length, LENGTHS["short"])

        # Adjust pipeline params
        temp = max(0.5, min(creativity, 1.0))
        if self.model_name == FALLBACK_MODEL:
            # GPT-2 needs lower temperature for coherence
            temp = max(0.5, min(creativity * 0.75, 0.75))
        self.pipe._forward_params["temperature"] = temp

        chain_results = {}

        try:
            # ═══════════════════════════════
            # STEP 1  —  PLAN (deterministic)
            # ═══════════════════════════════
            setting = random.choice(GENRE_SETTINGS.get(genre_label, GENRE_SETTINGS["Fantasy"]))
            protagonist = random.choice(GENRE_PROTAGONISTS.get(genre_label, GENRE_PROTAGONISTS["Fantasy"]))
            outline = self._build_outline(plot, genre_label, setting, protagonist, tone_label)
            chain_results["outline"] = outline
            print(f"\n--- Plan: {genre_label} / {tone_label} / {length_info['label']} ---")
            print(f"    Setting: {setting}")
            print(f"    Protagonist: {protagonist}")

            # ═══════════════════════════════
            # STEP 2  —  WRITE (Story Chain(s))
            # ═══════════════════════════════
            if self.model_name == FALLBACK_MODEL:
                # GPT-2: single long-seed generation
                # Cap at 350 tokens — GPT-2 degrades rapidly beyond that
                max_tok = min(length_info["tokens"], 350)
                self.pipe._forward_params["max_new_tokens"] = max_tok
                self.pipe._forward_params["min_new_tokens"] = max(60, max_tok // 4)

                narrative_seed = build_narrative_seed(
                    genre_label, setting, protagonist, plot, tone_label
                )
                print(f"   Narrative seed: {len(narrative_seed.split())} words")
                print("   Chain: Generating story from seed...")

                raw_story = self.chains["story"].invoke({
                    "narrative_seed": narrative_seed,
                })
                # Prepend the seed prose (skip the metadata primer)
                seed_lines = narrative_seed.strip().split('\n')
                prose_lines = [l for l in seed_lines if not l.strip().startswith('[')]
                prose_part = '\n'.join(prose_lines).strip()
                full = prose_part + "\n\n" + raw_story.strip()
                story = self._clean_output(full)
                story = self._quality_trim(story)
                story = self._trim_to_complete(story)
                story = self._deduplicate(story)
            else:
                # Single-chain generation (TinyLlama / Flan-T5)
                max_tok = length_info["tokens"]
                if self.model_type == "seq2seq":
                    max_tok = min(max_tok, 512)
                self.pipe._forward_params["max_new_tokens"] = max_tok

                print("Chain 1/2: Generating story...")
                raw_story = self.chains["story"].invoke({
                    "genre": genre_label,
                    "tone": tone_label,
                    "word_target": str(length_info["words"]),
                    "plot": plot,
                    "style_hint": genre_info["style"],
                    "setting": setting,
                    "protagonist": protagonist,
                })
                story = self._clean_output(raw_story)
                story = self._quality_trim(story)
                story = self._trim_to_complete(story)
                story = self._deduplicate(story)

            chain_results["story"] = story
            wc = len(story.split())
            print(f"   Story: {wc} words")

            # ═══════════════════════════════
            # STEP 3  —  NAME (Title Chain)
            # ═══════════════════════════════
            print("Title chain: Generating title...")
            self.pipe._forward_params["max_new_tokens"] = 20
            if "min_new_tokens" in self.pipe._forward_params:
                self.pipe._forward_params["min_new_tokens"] = 3

            excerpt = story[:300].replace('"', "'")
            raw_title = self.chains["title"].invoke({
                "excerpt": excerpt,
                "genre": genre_label,
            })
            title = self._extract_title(raw_title, genre_label)
            chain_results["title"] = title
            print(f"   Title: '{title}'")

            # ═══════════════════════════════
            # STEP 4  —  POLISH (deterministic)
            # ═══════════════════════════════
            formatted = self._format_paragraphs(story)
            word_count = len(formatted.split())
            elapsed = round(time.time() - start_time, 2)
            analysis = self._analyze(formatted)

            result = {
                "success": True,
                "title": title,
                "story": formatted,
                "outline": outline,
                "genre": genre_label,
                "genre_icon": genre_info["icon"],
                "tone": tone_label,
                "length_label": length_info["label"],
                "word_count": word_count,
                "generation_time": elapsed,
                "creativity": creativity,
                "chain_steps": [
                    {"name": "Plan",   "status": "success", "output_len": len(outline)},
                    {"name": "Story",  "status": "success", "output_len": len(formatted)},
                    {"name": "Title",  "status": "success", "output_len": len(title)},
                    {"name": "Polish", "status": "success", "output_len": word_count},
                ],
                "analysis": analysis,
            }
            self._save_story(result)
            print(f"Done: {word_count} words in {elapsed}s")
            return result

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "generation_time": elapsed,
                "chain_results": chain_results,
            }

    # ─────────────────────────────
    # TITLE EXTRACTION
    # ─────────────────────────────
    def _extract_title(self, raw: str, genre: str) -> str:
        """Extract a clean title from the LLM output."""
        text = raw.strip()
        # Strip chat tokens that may remain
        for tok in ['</s>', '<|assistant|>', '<|user|>']:
            text = text.replace(tok, '')

        # Take first line, strip quotes / markers
        text = text.split('\n')[0].strip()
        text = text.split('"')[0].strip()
        # Remove "Title:" prefix if echoed
        text = re.sub(r'^Title:\s*', '', text, flags=re.IGNORECASE)
        text = text.strip('"\'.,!:-').strip()
        text = re.sub(r'[^a-zA-Z0-9\s\'\-:,&]', '', text).strip()

        # Limit length and avoid trailing stop words
        words = text.split()
        if len(words) > 6:
            stop = {'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'is',
                     'was', 'no', 'and', 'or', 'but', 'when', 'where', 'who'}
            best = min(6, len(words))
            for i in range(min(6, len(words)), 2, -1):
                if words[i - 1].lower() not in stop:
                    best = i
                    break
            text = ' '.join(words[:best])

        # Fallback title bank
        if len(text.split()) < 2 or len(text) < 4:
            bank = {
                "Fantasy":   ["The Last Enchantment",   "Beyond the Silver Gate"],
                "Sci-Fi":    ["Signal from the Void",   "The Last Transmission"],
                "Mystery":   ["The Unsigned Letter",    "Shadows at Midnight"],
                "Romance":   ["Between Two Hearts",     "The Letter Never Sent"],
                "Horror":    ["What Waits Below",       "The Silent Room"],
                "Adventure": ["The Uncharted Path",     "Beyond the Horizon"],
                "Drama":     ["The Weight of Silence",  "Unspoken Truths"],
                "Comedy":    ["The Accidental Hero",    "Murphy's Best Day"],
            }
            text = random.choice(bank.get(genre, ["The Untold Story"]))

        return text.title()

    # ─────────────────────────────
    # FORMATTING / ANALYSIS
    # ─────────────────────────────
    @staticmethod
    def _format_paragraphs(story: str) -> str:
        paragraphs = [p.strip() for p in story.split('\n') if p.strip()]
        formatted = []
        for p in paragraphs:
            if p and p[0].islower():
                p = p[0].upper() + p[1:]
            if p and p[-1] not in '.!?"\'':
                p += '.'
            formatted.append(p)
        return '\n\n'.join(formatted)

    @staticmethod
    def _analyze(story: str) -> dict:
        words = story.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', story) if len(s.strip()) > 3]
        paragraphs = [p.strip() for p in story.split('\n\n') if p.strip()]
        unique_words = len(set(w.lower().strip('.,!?"\'()-') for w in words if len(w) > 1))
        vocab = round(unique_words / max(len(words), 1) * 100, 1)
        avg_sent = round(len(words) / max(len(sentences), 1), 1)
        dialogue = len(re.findall(r'"[^"]{3,}"', story))
        desc_re = r'\b(ancient|dark|bright|beautiful|mysterious|silent|golden|silver|vast|tiny|enormous|cold|warm|strange|old|young|tall|deep|loud|quiet|fierce|gentle|dim|faint|heavy|soft|sharp|bitter|hollow|pale|crimson|iron|thick|thin|bare|rough|smooth|wet|dry|empty|broken|rusted|worn|dusty|frozen|burning|trembling)\b'
        desc = len(re.findall(desc_re, story.lower()))
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "vocab_richness": vocab,
            "unique_words": unique_words,
            "avg_sentence_length": avg_sent,
            "dialogue_lines": dialogue,
            "descriptive_words": desc,
        }

    # ─────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────
    def _save_story(self, result: dict):
        try:
            fp = STORIES_DIR / f"story_{int(time.time())}.json"
            data = {k: result[k] for k in
                    ["title", "story", "genre", "tone", "word_count", "generation_time"]}
            with open(fp, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"   Warning: save failed: {e}")

    def get_status(self) -> dict:
        saved = list(STORIES_DIR.glob("*.json"))
        return {
            "model": f"{self.model_name} [{self.model_type}]",
            "framework": "LangChain + HuggingFace",
            "chains": ["Story Chain (LCEL)", "Title Chain (LCEL)"],
            "device": "CUDA" if torch.cuda.is_available() else "CPU",
            "stories_generated": len(saved),
        }

    def get_saved_stories(self) -> list:
        stories = []
        for f in sorted(STORIES_DIR.glob("*.json"), reverse=True)[:20]:
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                    data["filename"] = f.name
                    stories.append(data)
            except Exception:
                pass
        return stories
