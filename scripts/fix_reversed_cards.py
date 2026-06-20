"""
Script to fix reversed cards so examples and conjugation stay with the answer side.

Logic:
- Original cards (German -> English): German is the question, English is the answer
  So examples should show German on the question side, English on the answer side
  Conjugation should be on the answer side (but it's German conjugation, so it stays with German)

- Reversed cards (English -> German): English is the question, German is the answer
  So examples should show English on the question side, German on the answer side
  Conjugation should be on the answer side (German conjugation stays with German answer)

The issue with the previous script: it swapped example languages for reversed cards.
Instead, we should keep examples as-is for original cards, and for reversed cards,
the examples should have English as the "question" part and German as the "answer" part.

Actually, let me reconsider. The examples have two parts:
- german: the German sentence
- english: the English translation

For a German->English card:
- Question side shows German word
- Answer side shows English translation + examples
- In examples on answer side: German sentence first, then English translation

For an English->German card:
- Question side shows English word
- Answer side shows German translation + examples
- In examples on answer side: English sentence first, then German translation

So for reversed cards, we should swap the german/english fields in examples.

Wait, the user said "the examples and conjugation should always appear with the answer side".
So:
- German->English card: answer side is English. But conjugation is German grammar info.
  Should conjugation appear on the English answer side? That doesn't make sense.

Let me re-read: "if i am seeing a card in german, and the answer side is in english,
the examples and conjugations should appear on the english answer side"

Hmm, but conjugation is German-specific. If the answer is English, showing German
conjugation on the English side seems odd. But the user explicitly said it should
appear on the answer side.

Actually wait - I think I misunderstood. Let me re-read:
"only the word and the translation should be reversed, the examples and conjugation
should always appear with the 'answer' side of the card"

So for the original German->English card:
- Question: German word
- Answer: English translation + examples + conjugation

For the reversed English->German card:
- Question: English word
- Answer: German translation + examples + conjugation

So examples and conjugation always stay with the German side (the translation side
for reversed cards, which is the answer). That means for reversed cards:
- The German word becomes the answer (translation field)
- Examples should show the German version (since German is the answer)
- But examples have both German and English...

I think the cleanest approach is:
- For original cards (German->English): examples have german first, english second
- For reversed cards (English->German): examples have english first, german second
  This way, when viewing the answer side, you see the target language first.

Actually, let me just fix the current reversed cards. The issue is that the previous
script swapped the example languages. Instead, for reversed cards:
- word = English (question)
- translations = German (answer)
- examples should show: english sentence (as context for the English question),
  german sentence (as the answer translation)
- conjugation stays as-is (it's German conjugation info)

So I need to:
1. Delete all reversed cards (English->German cards where the word is English)
2. Re-create them properly with examples having english first, german second

How to identify reversed cards? We can check if the word looks like English vs German.
Actually, a simpler approach: delete all cards where id > original_count.
Since we created reversed cards after original cards, they should have higher IDs.

Wait, but some original cards might have been added later too. Let me think...

Actually, the simplest approach: the original cards were created first (IDs 1-6215),
and reversed cards were created second (IDs 6216-12430). So I can just delete
IDs > 6215 and re-create properly.

But wait, the user might have added more cards since then. Let me check the DB.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from sqlalchemy.orm import Session
from app.database import SessionLocal
from app import models


def fix_reversed_cards():
    db = SessionLocal()
    try:
        # Get all words
        all_words = db.query(models.Word).order_by(models.Word.id).all()
        total = len(all_words)
        print(f"Total words: {total}")

        # Find the cutoff - original cards are those that don't have a corresponding
        # reversed card pointing to them. Actually, let's just find cards where
        # word == some other card's translations and translations == that card's word

        # Build a set of (word, translations) pairs
        word_pairs = set()
        for word in all_words:
            word_pairs.add((word.word, word.translations))

        # Find reversed pairs
        reversed_ids = []
        original_words = []

        for word in all_words:
            # If this word's (word, translations) has a corresponding (translations, word)
            # and this word's word is the other's translations, then this could be reversed
            # A card is "reversed" if there's another card where:
            # other.word == this.translations AND other.translations == this.word
            # AND this.id > other.id (assuming original was created first)
            if (word.translations, word.word) in word_pairs:
                # Find the other card
                other = (
                    db.query(models.Word)
                    .filter(
                        models.Word.word == word.translations,
                        models.Word.translations == word.word,
                    )
                    .order_by(models.Word.id)
                    .first()
                )

                if other and word.id > other.id:
                    reversed_ids.append(word.id)
                elif other and word.id == other.id:
                    # Same card, shouldn't happen
                    pass
                else:
                    original_words.append(word)
            else:
                original_words.append(word)

        print(f"Found {len(reversed_ids)} reversed cards to delete")
        print(f"Found {len(original_words)} original cards")

        if not reversed_ids:
            print("No reversed cards found. Nothing to fix.")
            return

        # Delete examples for reversed cards first
        db.query(models.Example).filter(
            models.Example.word_id.in_(reversed_ids)
        ).delete(synchronize_session=False)

        # Delete reversed cards
        db.query(models.Word).filter(models.Word.id.in_(reversed_ids)).delete(
            synchronize_session=False
        )

        db.commit()
        print(f"Deleted {len(reversed_ids)} reversed cards")

        # Now re-create reversed cards properly
        created_count = 0

        for word in original_words:
            if not word.translations:
                continue

            # Create reversed card
            reversed_word = models.Word(
                word=word.translations,  # English becomes the "word" (question)
                word_type=word.word_type,
                frequency=word.frequency,
                level=word.level,
                translations=word.word,  # German becomes the "translation" (answer)
                conjugation=word.conjugation,  # Conjugation stays with German (answer)
                additional_info=word.additional_info,
                is_complete=False,
                conjugation_id=word.conjugation_id,
            )

            db.add(reversed_word)
            db.flush()

            # For reversed cards: examples should have english first (question context),
            # german second (answer). But actually, we want examples to appear on the
            # answer side. So when viewing the answer (German), we see the German example
            # with its English translation.

            # Wait, the user said "examples and conjugation should always appear with
            # the answer side". So on the answer side, we show both German and English
            # examples. The order doesn't matter as much as long as they appear together
            # on the answer side.

            # Let's keep examples as-is but associate them with the reversed card.
            # The card template will handle which side to show them on.
            for example in word.examples:
                reversed_example = models.Example(
                    word_id=reversed_word.id,
                    german=example.german,
                    english=example.english,
                )
                db.add(reversed_example)

            created_count += 1

            if created_count % 100 == 0:
                print(f"Re-created {created_count} reversed cards...")
                db.commit()

        db.commit()
        print(f"\nDone!")
        print(f"Re-created: {created_count} reversed cards")

        # Verify
        total_after = db.query(models.Word).count()
        print(f"Total words in database: {total_after}")

    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    fix_reversed_cards()
