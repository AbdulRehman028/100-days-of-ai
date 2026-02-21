"""
Genre-specific narrative seed openings for GPT-2 fallback.
Each genre has 3 atmospheric prose templates that lock GPT-2 into
literary fiction register, preventing drift into web-text.
"""

import re
import random


# ============================================
# GENRE OPENINGS (3 per genre)
# ============================================
GENRE_OPENINGS = {
    "Fantasy": [
        (
            "Deep within {setting}, an ancient power hummed — one that {protagonist} could feel "
            "in the marrow of every bone. Runes carved into stone flickered to life as "
            "the last light of dusk bled across the horizon. {protagonist_cap} tightened "
            "the strap of the worn leather satchel and stepped forward. The air shimmered, "
            "and a voice older than the mountains whispered a name that had not been spoken "
            "in a thousand years."
        ),
        (
            "Few mortals had ever stood within {setting}, yet {protagonist} stood there "
            "now, heart pounding, staring at the seal of silver and shadow. The legends "
            "had warned of this place — not as a destination but as a threshold. A cold "
            "wind swept the dust from the stone floor, and somewhere in the distance, "
            "a bell tolled once."
        ),
        (
            "Night gathered like a living thing around {setting}. {protagonist_cap} knelt beside "
            "the stream, watching starlight ripple across the dark water. Something stirred "
            "beneath the surface — something that should not have been awake. The quest "
            "was about to begin, and there would be no turning back."
        ),
    ],
    "Sci-Fi": [
        (
            "The viewport of {setting} framed a silence that stretched for light-years. "
            "{protagonist} ran a gloved hand across the console, watching readouts flicker "
            "in amber and blue. The oxygen counter ticked downward. A soft alarm chimed — "
            "the kind that meant something fundamental had changed, something the "
            "ship's protocols had no plan for."
        ),
        (
            "Inside {setting}, the hum of the life-support system was the only sound. "
            "{protagonist} pulled up the navigation hologram: the trajectory had shifted "
            "again, a deviation that could not be explained by drift alone. Somewhere in "
            "the forward compartment, a door opened that should have stayed sealed."
        ),
        (
            "Gravity pressed against the hull of {setting} as {protagonist} stared at the "
            "transmission log. The message was seven minutes old and already impossible. "
            "The AI's status light blinked green, but its voice, when it finally "
            "spoke, carried something disturbingly close to hesitation."
        ),
    ],
    "Mystery": [
        (
            "Rain hammered the windows of {setting}. {protagonist} set down the photograph "
            "and exhaled slowly. The victim had been dead for three weeks, but the letter "
            "on the desk was dated yesterday. Every clue pointed in a different "
            "direction, and each direction led to another question."
        ),
        (
            "The floor of {setting} creaked under each careful step. {protagonist} crouched "
            "beside the overturned chair and studied the scuff marks. Someone had been "
            "dragged. A single thread caught the lamplight — crimson silk, expensive, "
            "and completely out of place."
        ),
        (
            "At half past midnight, {setting} felt like the loneliest place on earth. "
            "{protagonist} spread the documents across the table and traced the timeline. "
            "There was a gap — forty-three minutes unaccounted for. That gap was "
            "where the truth was hiding."
        ),
    ],
    "Romance": [
        (
            "The afternoon light in {setting} fell in warm amber stripes across the floor. "
            "{protagonist} looked up from the counter and their breath caught. The stranger "
            "standing in the doorway smiled — not politely, but as if recognizing someone "
            "from a dream. Neither of them moved for a long, electric moment."
        ),
        (
            "It had rained all morning, but inside {setting} the air was warm with the "
            "scent of coffee and old pages. {protagonist} kept stealing glances toward the "
            "window table, where the same person had sat every Tuesday for three months. "
            "Today, for the first time, they looked back."
        ),
        (
            "The evening crowd at {setting} had thinned to a quiet murmur. {protagonist} "
            "was about to close up for the night when the bell above the door rang one "
            "last time. Something about that voice — honest, a little nervous — "
            "made them set the keys back down."
        ),
    ],
    "Horror": [
        (
            "The silence in {setting} was not the peaceful kind. {protagonist} stood in "
            "the doorway, flashlight carving a thin beam through the dark. The air "
            "tasted of damp stone and something sweetly rotten. The door behind "
            "them swung shut with a click that was far too deliberate."
        ),
        (
            "At first, {protagonist} mistook the sound for wind scraping against {setting}. "
            "But wind did not breathe. Wind did not pause and listen. The "
            "temperature dropped three degrees in under a second, and the flashlight "
            "flickered once, twice, then held — barely."
        ),
        (
            "Nobody had lived in {setting} for twelve years, which made the fresh "
            "footprints on the cellar stairs impossible. {protagonist} photographed them "
            "with shaking hands. Upstairs, a child's music box began to play, "
            "the melody thin and wrong, as though the notes had been rearranged."
        ),
    ],
    "Adventure": [
        (
            "The map ended where {setting} began. {protagonist} folded the cracked paper "
            "into a pocket and climbed the last ridge. The valley below was green and "
            "vast and unmarked on any chart. A warm wind carried the scent of "
            "rain and possibility."
        ),
        (
            "The rope groaned as {protagonist} swung across the ravine toward {setting}. "
            "Below, the river roared white against black rock. One hand caught the ledge; "
            "stone crumbled, but held. Somewhere ahead, the path split into three, "
            "and only one led to the surface."
        ),
        (
            "Dawn broke crimson over {setting}. {protagonist} shouldered the pack and "
            "checked the compass — the needle wobbled, then pointed east, away from "
            "civilization. There was no trail, but the carved marker "
            "half-buried in moss was proof: someone had been here before."
        ),
    ],
    "Drama": [
        (
            "The kitchen in {setting} smelled of cold coffee and unfinished conversations. "
            "{protagonist} stared at the empty chair across the table. The letter sat "
            "between them — opened, read, and reread until the creases were soft. "
            "The clock on the wall ticked, each second heavier than the last."
        ),
        (
            "Silence filled {setting} like water filling a glass. {protagonist} sat on the "
            "edge of the bed and pressed both palms against closed eyes. The phone call "
            "had lasted eleven seconds — enough to dismantle twenty years. "
            "Somewhere outside, a car started, pulled away, and was gone."
        ),
        (
            "The waiting room of {setting} had pale walls and fluorescent light that made "
            "everything look uncertain. {protagonist} held a paper cup of water without "
            "drinking it. A nurse appeared at the end of the corridor, "
            "and the world narrowed to the expression on her face."
        ),
    ],
    "Comedy": [
        (
            "It was only Tuesday, and {protagonist} had already caused two small fires "
            "and one diplomatic incident at {setting}. The fire extinguisher sat empty "
            "in the corner — a monument to Monday. Things were about to get "
            "significantly worse, which, all things considered, was impressive."
        ),
        (
            "The look on the manager's face told {protagonist} everything. The thing at "
            "{setting} had not gone according to plan. None of the things at {setting} "
            "ever went according to plan. Somewhere in the back, someone was laughing "
            "so hard they had started to hiccup."
        ),
        (
            "{protagonist} stared at the wreckage of {setting} with the calm of someone "
            "who had already accepted their fate. The sprinklers were still going. The "
            "cake — what was left of it — clung to the ceiling. \"This,\" "
            "they whispered to no one, \"is exactly how I imagined it would go.\""
        ),
    ],
}


# ============================================
# SEED BUILDER
# ============================================
def build_narrative_seed(genre: str, setting: str, protagonist: str,
                         plot: str, tone: str) -> str:
    """
    Build a long, immersive narrative seed (~80-120 words) for GPT-2.

    Structure:
      Line 1 — context primer (tells GPT-2 what the story is about)
      Line 2 — blank
      Line 3+ — pure genre fiction prose (this is what GPT-2 continues)

    The primer gives thematic direction; the prose locks the model into
    literary fiction register.  We strip the primer from the final output.
    """
    # Pick a random seed for the genre
    seeds = GENRE_OPENINGS.get(genre, GENRE_OPENINGS["Fantasy"])
    template = random.choice(seeds)

    # Capitalize protagonist for sentence starts
    protagonist_cap = protagonist[0].upper() + protagonist[1:] if protagonist else protagonist

    # Format seed with setting + protagonist (no plot in prose)
    prose = template.format(
        setting=setting,
        protagonist=protagonist,
        protagonist_cap=protagonist_cap,
    )

    # Auto-capitalize protagonist after sentence boundaries (. ! ?)
    prose = re.sub(
        r'([.!?])\s+' + re.escape(protagonist),
        lambda m: m.group(1) + ' ' + protagonist_cap,
        prose,
    )

    # Context primer: give GPT-2 thematic direction via metadata markers.
    seed = f"[Genre: {genre}] [Plot: {plot}]\n\n{prose}\n\n"

    return seed
