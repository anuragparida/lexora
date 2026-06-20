"""
Script to reverse all vocabulary cards and add them as duplicates.

For each existing card (German -> English):
- Creates a new card (English -> German)
- Swaps word and translations fields
- Keeps examples and conjugation with the "answer" side (German side)
- Preserves all other metadata (word_type, frequency, level, etc.)
"""

import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from sqlalchemy.orm import Session
from app.database import SessionLocal, engine
from app import models


def reverse_cards():
    db = SessionLocal()
    try:
        # Get all existing words
        words = db.query(models.Word).all()
        print(f"Found {len(words)} existing words")

        created_count = 0
        skipped_count = 0

        for word in words:
            # Skip if translations is empty
            if not word.translations:
                skipped_count += 1
                continue

            # Check if reversed card already exists (to avoid duplicates on re-runs)
            existing = (
                db.query(models.Word)
                .filter(
                    models.Word.word == word.translations,
                    models.Word.translations == word.word,
                )
                .first()
            )

            if existing:
                skipped_count += 1
                continue

            # Create reversed card
            reversed_word = models.Word(
                word=word.translations,  # English becomes the "word"
                word_type=word.word_type,
                frequency=word.frequency,
                level=word.level,
                translations=word.word,  # German becomes the "translation"
                conjugation=word.conjugation,
                additional_info=word.additional_info,
                is_complete=False,
                conjugation_id=word.conjugation_id,
            )

            db.add(reversed_word)
            db.flush()  # Get the ID without committing

            # Copy examples - they stay with the German side (answer side)
            # For the reversed card, we swap german/english in the examples
            for example in word.examples:
                reversed_example = models.Example(
                    word_id=reversed_word.id,
                    german=example.english,  # English example becomes the "question"
                    english=example.german,  # German example becomes the "answer"
                )
                db.add(reversed_example)

            created_count += 1

            if created_count % 100 == 0:
                print(f"Created {created_count} reversed cards...")
                db.commit()

        db.commit()
        print(f"\nDone!")
        print(f"Created: {created_count} reversed cards")
        print(f"Skipped: {skipped_count} (empty translations or already exists)")

        # Verify
        total = db.query(models.Word).count()
        print(f"Total words in database: {total}")

    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    reverse_cards()
