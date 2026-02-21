"""
Configuration constants for the Short Story Generator.
Genres, tones, lengths, settings, protagonists, and model names.
"""

from pathlib import Path

# ============================================
# PATHS
# ============================================
BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
STORIES_DIR = BASE_DIR / "generated_stories"

TEMPLATE_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
STORIES_DIR.mkdir(exist_ok=True)

# ============================================
# MODEL NAMES
# ============================================
PRIMARY_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FALLBACK_MODEL = "gpt2-medium"
LAST_RESORT_MODEL = "google/flan-t5-base"

# ============================================
# GENRES
# ============================================
GENRES = {
    "fantasy":   {"label": "Fantasy",   "icon": "🧙", "desc": "Magic, dragons, and epic quests",
                  "style": "Use vivid magical imagery, mythical creatures, and enchanted settings."},
    "scifi":     {"label": "Sci-Fi",    "icon": "🚀", "desc": "Space, technology, and the future",
                  "style": "Use futuristic technology, space exploration, and scientific wonder."},
    "mystery":   {"label": "Mystery",   "icon": "🔍", "desc": "Clues, suspense, and whodunits",
                  "style": "Build suspense, plant clues, and create a sense of intrigue."},
    "romance":   {"label": "Romance",   "icon": "💕", "desc": "Love, relationships, and emotions",
                  "style": "Focus on emotions, chemistry between characters, and heartfelt moments."},
    "horror":    {"label": "Horror",    "icon": "👻", "desc": "Fear, darkness, and the unknown",
                  "style": "Create an atmosphere of dread, suspense, and disturbing imagery."},
    "adventure": {"label": "Adventure", "icon": "⚔️", "desc": "Action, exploration, and daring feats",
                  "style": "High energy action, exploration of unknown lands, and heroic moments."},
    "drama":     {"label": "Drama",     "icon": "🎭", "desc": "Human conflict and emotional depth",
                  "style": "Focus on character development, moral dilemmas, and emotional conflict."},
    "comedy":    {"label": "Comedy",    "icon": "😂", "desc": "Humor, wit, and funny situations",
                  "style": "Use wit, irony, funny situations, and comedic timing."},
}

# ============================================
# TONES
# ============================================
TONES = {
    "dark":        "Dark and gritty",
    "light":       "Light and uplifting",
    "suspenseful": "Tense and suspenseful",
    "whimsical":   "Playful and whimsical",
    "emotional":   "Deep and emotional",
    "action":      "Fast-paced and action-packed",
}

# ============================================
# LENGTHS
# ============================================
LENGTHS = {
    "flash":  {"label": "Flash Fiction",  "tokens": 300,  "words": 200,  "desc": "~150-250 words"},
    "short":  {"label": "Short Story",    "tokens": 500,  "words": 350,  "desc": "~250-450 words"},
    "medium": {"label": "Medium Story",   "tokens": 750,  "words": 550,  "desc": "~450-700 words"},
}

# ============================================
# GENRE SETTINGS & PROTAGONISTS
# ============================================
GENRE_SETTINGS = {
    "Fantasy":   ["an enchanted forest", "a crumbling tower", "the Crystal Caverns", "a floating citadel", "the Library of Lost Tongues", "Dragon's Peak"],
    "Sci-Fi":    ["space station Artemis-7", "a colony ship drifting through a nebula", "a research outpost on Europa", "a terraforming platform on Mars", "orbital habitat Elysium", "a derelict alien vessel"],
    "Mystery":   ["an abandoned manor house", "a rain-soaked harbor district", "a locked study", "a dusty city archive", "a warehouse at the docks", "a quiet town with a dark secret"],
    "Romance":   ["a corner bookshop", "a rain-lit cafe", "a Tuscan vineyard", "a quiet coastal town", "a moonlit botanical garden", "a university campus"],
    "Horror":    ["an old house at the end of a dead-end road", "an underground bunker", "an abandoned hospital", "a fog-covered lake house", "a Victorian estate", "a cellar beneath an old church"],
    "Adventure": ["a hidden island", "a mountain pass", "ancient temple ruins", "a dense jungle", "a volcanic archipelago", "a desert canyon"],
    "Drama":     ["a family home", "a hospital waiting room", "an empty apartment", "courthouse steps", "a bus station at midnight", "a childhood kitchen"],
    "Comedy":    ["an office break room", "the worst restaurant in town", "a neighbourhood karaoke bar", "an airport security line", "an escape room", "a disastrous cooking class"],
}

GENRE_PROTAGONISTS = {
    "Fantasy":   ["a young sorcerer", "a wandering knight", "an exiled queen", "a half-elf scholar", "a reluctant chosen one", "a blind oracle"],
    "Sci-Fi":    ["an astro-engineer", "a xenobiologist", "a ship's navigator", "an AI researcher", "a colony archivist", "a deep-space pilot"],
    "Mystery":   ["a detective", "an amateur sleuth", "a retired journalist", "a forensic archivist", "a private investigator", "a librarian"],
    "Romance":   ["a bookshop owner", "a travelling musician", "a pastry chef", "a marine biologist", "an architecture student", "a florist"],
    "Horror":    ["a night-shift nurse", "a new homeowner", "a paranormal researcher", "a lone hiker", "a school teacher", "an antique collector"],
    "Adventure": ["a cartographer", "a treasure hunter", "a shipwrecked sailor", "a mountain guide", "a bush pilot", "an archaeologist"],
    "Drama":     ["an estranged father", "an aspiring actress", "a retired teacher", "a war veteran", "an immigrant mother", "a widowed musician"],
    "Comedy":    ["an over-caffeinated intern", "a hapless wedding planner", "a retired spy", "a confused tourist", "an unlucky inventor", "a substitute teacher"],
}

# ============================================
# SAMPLE PROMPTS
# ============================================
SAMPLE_PROMPTS = [
    {"genre": "fantasy",   "plot": "A young apprentice discovers they can hear the thoughts of dragons"},
    {"genre": "scifi",     "plot": "The last human on a generation ship wakes up to find the AI has changed course"},
    {"genre": "mystery",   "plot": "A detective receives a letter from a victim written before their disappearance"},
    {"genre": "romance",   "plot": "Two rival bookshop owners discover they've been anonymous pen pals"},
    {"genre": "horror",    "plot": "A family moves into a house where every mirror shows a different version of them"},
    {"genre": "adventure", "plot": "A mapmaker finds a blank island that shouldn't exist on any chart"},
    {"genre": "drama",     "plot": "A retired musician hears their stolen melody played on the radio thirty years later"},
    {"genre": "comedy",    "plot": "A time traveler keeps accidentally preventing their own first date"},
    {"genre": "fantasy",   "plot": "The last librarian guards books that rewrite themselves every midnight"},
    {"genre": "scifi",     "plot": "A message from the future arrives but it's addressed to someone who doesn't exist yet"},
    {"genre": "mystery",   "plot": "Every resident of a small town wakes up with a tattoo they don't remember getting"},
    {"genre": "romance",   "plot": "A musician and a deaf artist discover a language only they understand"},
]
