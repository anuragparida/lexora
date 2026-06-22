"""
Anki deck builder using genanki.

Creates beautiful, styled Anki decks from the vocabulary database.
Each word becomes 1 note with 2 cards (German->English and English->German).
Anki handles the card generation natively - this is the proper way to do bidirectional cards.
"""

import os
import random
from datetime import datetime
from typing import List, Optional

import genanki
from sqlalchemy.orm import Session

from app import models

# Create decks directory. Path is env-overridable so tests (and any
# future host-mount layout) can redirect to a writable location. The
# container default stays /app/generated_decks because that's where
# docker-compose.yml mounts the volume.
DECKS_DIR = os.getenv("LEXORA_DECKS_DIR", "/app/generated_decks")
os.makedirs(DECKS_DIR, exist_ok=True)


# Anki model CSS
CARD_CSS = """
.card {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    font-size: 20px;
    text-align: center;
    color: #e2e8f0;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    padding: 20px;
    min-height: 100vh;
    line-height: 1.6;
}

.word {
    font-size: 36px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 12px;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

.word-type {
    display: inline-block;
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 600;
    margin: 4px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.level-badge {
    display: inline-block;
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 600;
    margin: 4px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.frequency-badge {
    display: inline-block;
    background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 600;
    margin: 4px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.translation {
    font-size: 28px;
    color: #fbbf24;
    font-weight: 600;
    margin: 16px 0;
    padding: 12px;
    background: rgba(251, 191, 36, 0.1);
    border-radius: 12px;
    border: 2px solid rgba(251, 191, 36, 0.3);
}

.divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #475569, transparent);
    margin: 20px 0;
}

.section-title {
    font-size: 14px;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 20px 0 12px 0;
}

.examples-container {
    background: rgba(30, 41, 59, 0.8);
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
    border: 1px solid rgba(71, 85, 105, 0.5);
}

.example-german {
    font-size: 18px;
    color: #f1f5f9;
    font-weight: 600;
    margin-bottom: 4px;
}

.example-english {
    font-size: 16px;
    color: #94a3b8;
    font-style: italic;
}

.conjugation-box {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.15) 100%);
    border: 2px solid rgba(245, 158, 11, 0.4);
    border-radius: 12px;
    padding: 16px;
    margin: 16px 0;
}

.conjugation-title {
    font-size: 14px;
    font-weight: 700;
    color: #fbbf24;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 12px;
}

.conjugation-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}

.conjugation-item {
    background: rgba(0, 0, 0, 0.2);
    padding: 8px;
    border-radius: 8px;
}

.conjugation-label {
    font-size: 11px;
    color: #fbbf24;
    margin-bottom: 4px;
    font-weight: 600;
}

.conjugation-value {
    font-size: 16px;
    color: #f1f5f9;
    font-weight: 600;
}

.additional-info {
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 8px;
    padding: 10px;
    margin-top: 12px;
    font-size: 15px;
    color: #93c5fd;
}

.hint {
    font-size: 14px;
    color: #64748b;
    margin-top: 20px;
    font-style: italic;
}

/* Night mode compatibility */
.nightMode .card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
}
"""

# Card 1: German -> English
# Front shows German word, back shows English + examples + conjugation
CARD_1_FRONT = """
<div class="word">{{GermanWord}}</div>
<div>
    {{#WordType}}<span class="word-type">{{WordType}}</span>{{/WordType}}
    {{#Level}}<span class="level-badge">{{Level}}</span>{{/Level}}
    {{#Frequency}}<span class="frequency-badge">Freq: {{Frequency}}</span>{{/Frequency}}
</div>
<div class="hint">Tap to reveal answer</div>
"""

CARD_1_BACK = """
{{FrontSide}}

<div class="divider"></div>

<div class="translation">{{EnglishTranslation}}</div>

{{#AdditionalInfo}}
<div class="additional-info">{{AdditionalInfo}}</div>
{{/AdditionalInfo}}

{{#HasExamples}}
<div class="section-title">Examples</div>
<div class="examples-container">
    <div class="example-german">{{ExampleGerman}}</div>
    <div class="example-english">{{ExampleEnglish}}</div>
</div>
{{/HasExamples}}

{{#HasConjugation}}
<div class="conjugation-box">
    <div class="conjugation-title">Conjugation</div>
    <div class="conjugation-grid">
        {{#Present3rd}}
        <div class="conjugation-item">
            <div class="conjugation-label">3rd Person Present</div>
            <div class="conjugation-value">{{Present3rd}}</div>
        </div>
        {{/Present3rd}}
        {{#SimplePast}}
        <div class="conjugation-item">
            <div class="conjugation-label">Simple Past</div>
            <div class="conjugation-value">{{SimplePast}}</div>
        </div>
        {{/SimplePast}}
        {{#Participle}}
        <div class="conjugation-item">
            <div class="conjugation-label">Participle</div>
            <div class="conjugation-value">{{Participle}}</div>
        </div>
        {{/Participle}}
    </div>
</div>
{{/HasConjugation}}
"""

# Card 2: English -> German
# Front shows English word, back shows German + examples + conjugation
CARD_2_FRONT = """
<div class="word">{{EnglishTranslation}}</div>
<div>
    {{#WordType}}<span class="word-type">{{WordType}}</span>{{/WordType}}
    {{#Level}}<span class="level-badge">{{Level}}</span>{{/Level}}
    {{#Frequency}}<span class="frequency-badge">Freq: {{Frequency}}</span>{{/Frequency}}
</div>
<div class="hint">Tap to reveal answer</div>
"""

CARD_2_BACK = """
{{FrontSide}}

<div class="divider"></div>

<div class="translation">{{GermanWord}}</div>

{{#AdditionalInfo}}
<div class="additional-info">{{AdditionalInfo}}</div>
{{/AdditionalInfo}}

{{#HasExamples}}
<div class="section-title">Examples</div>
<div class="examples-container">
    <div class="example-german">{{ExampleGerman}}</div>
    <div class="example-english">{{ExampleEnglish}}</div>
</div>
{{/HasExamples}}

{{#HasConjugation}}
<div class="conjugation-box">
    <div class="conjugation-title">Conjugation</div>
    <div class="conjugation-grid">
        {{#Present3rd}}
        <div class="conjugation-item">
            <div class="conjugation-label">3rd Person Present</div>
            <div class="conjugation-value">{{Present3rd}}</div>
        </div>
        {{/Present3rd}}
        {{#SimplePast}}
        <div class="conjugation-item">
            <div class="conjugation-label">Simple Past</div>
            <div class="conjugation-value">{{SimplePast}}</div>
        </div>
        {{/SimplePast}}
        {{#Participle}}
        <div class="conjugation-item">
            <div class="conjugation-label">Participle</div>
            <div class="conjugation-value">{{Participle}}</div>
        </div>
        {{/Participle}}
    </div>
</div>
{{/HasConjugation}}
"""

# Create the model with 2 card templates (Anki's native bidirectional support)
GERMAN_VOCAB_MODEL = genanki.Model(
    1607392319,
    "German Vocabulary",
    css=CARD_CSS,
    fields=[
        {"name": "GermanWord"},
        {"name": "EnglishTranslation"},
        {"name": "WordType"},
        {"name": "Level"},
        {"name": "Frequency"},
        {"name": "HasExamples"},
        {"name": "ExampleGerman"},
        {"name": "ExampleEnglish"},
        {"name": "HasConjugation"},
        {"name": "Present3rd"},
        {"name": "SimplePast"},
        {"name": "Participle"},
        {"name": "AdditionalInfo"},
    ],
    templates=[
        {
            "name": "German -> English",
            "qfmt": CARD_1_FRONT,
            "afmt": CARD_1_BACK,
        },
        {
            "name": "English -> German",
            "qfmt": CARD_2_FRONT,
            "afmt": CARD_2_BACK,
        },
    ],
)


def get_filtered_words(
    db: Session,
    word_types: Optional[List[str]] = None,
    frequencies: Optional[List[str]] = None,
) -> List[models.Word]:
    """Get all words matching the filter criteria."""
    query = db.query(models.Word)

    if word_types:
        query = query.filter(models.Word.word_type.in_(word_types))
    if frequencies:
        query = query.filter(models.Word.frequency.in_(frequencies))

    return query.all()


def create_anki_deck(
    db: Session,
    word_types: Optional[List[str]] = None,
    frequencies: Optional[List[str]] = None,
    deck_name: Optional[str] = None,
    direction: str = "both",
) -> str:
    """
    Create an Anki deck from filtered words.
    Each word becomes 1 note with 2 cards (Anki handles card generation natively).

    Args:
        direction: 'both', 'de-en', or 'en-de'

    Returns the path to the generated .apkg file.
    """
    # Get all filtered words
    words = get_filtered_words(db, word_types, frequencies)

    if not words:
        raise ValueError("No words found matching the criteria")

    # Group words by concept (German word + English translation pair)
    word_pairs = {}
    for word in words:
        key = tuple(sorted([word.word, word.translations or ""]))
        if key not in word_pairs:
            word_pairs[key] = []
        word_pairs[key].append(word)

    # Get unique concepts (pairs that have both directions)
    concepts = []
    for pair in word_pairs.values():
        if len(pair) == 2:
            # Find German->English and English->German cards
            german_chars = set("äöüßÄÖÜ")
            de_en = None
            en_de = None
            for w in pair:
                if any(c in w.word for c in german_chars):
                    de_en = w
                else:
                    en_de = w
            if de_en and en_de:
                concepts.append(
                    {
                        "german": de_en.word,
                        "english": de_en.translations or "",
                        "word_type": de_en.word_type or "",
                        "level": de_en.level or "",
                        "frequency": de_en.frequency or "",
                        "examples": de_en.examples,
                        "conjugation": de_en.verb_conjugation,
                        "additional_info": de_en.additional_info or "",
                    }
                )

    if not concepts:
        raise ValueError("No complete word pairs found")

    # Generate deck name if not provided
    if not deck_name:
        parts = []
        if word_types:
            parts.append("".join(word_types))
        else:
            parts.append("alltypes")

        if frequencies:
            parts.append("".join([f"freq{f}" for f in frequencies]))
        else:
            parts.append("allfreqs")

        dir_map = {"both": "bothdir", "de-en": "detoen", "en-de": "entode"}
        parts.append(dir_map.get(direction, direction))

        parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
        deck_name = "_".join(parts)

    # Create deck
    deck_id = random.randrange(1 << 30, 1 << 31)
    deck = genanki.Deck(deck_id, f"German Vocabulary::{deck_name}")

    # Add notes - 1 note per concept with 2 cards
    for concept in concepts:
        # Build examples
        has_examples = "1" if concept["examples"] else ""
        example_german = ""
        example_english = ""
        if concept["examples"]:
            example_german = concept["examples"][0].german
            example_english = concept["examples"][0].english

        # Build conjugation
        has_conjugation = ""
        present_3rd = ""
        simple_past = ""
        participle = ""
        if concept["conjugation"]:
            has_conjugation = "1"
            present_3rd = concept["conjugation"].present_3rd_person or ""
            simple_past = concept["conjugation"].simple_past or ""
            participle = concept["conjugation"].participle or ""

        note = genanki.Note(
            model=GERMAN_VOCAB_MODEL,
            fields=[
                concept["german"],
                concept["english"],
                concept["word_type"],
                concept["level"],
                concept["frequency"],
                has_examples,
                example_german,
                example_english,
                has_conjugation,
                present_3rd,
                simple_past,
                participle,
                concept["additional_info"],
            ],
        )
        deck.add_note(note)

    # Save deck
    filename = f"{deck_name}.apkg"
    filepath = os.path.join(DECKS_DIR, filename)
    genanki.Package(deck).write_to_file(filepath)

    return filepath
