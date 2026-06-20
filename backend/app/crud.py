from sqlalchemy.orm import Session
from app import models


def get_words(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    word_types: list = None,
    frequencies: list = None,
):
    query = db.query(models.Word)

    if word_types:
        query = query.filter(models.Word.word_type.in_(word_types))
    if frequencies:
        query = query.filter(models.Word.frequency.in_(frequencies))

    total = query.count()
    words = query.offset(skip).limit(limit).all()
    return {
        "items": words,
        "total": total,
        "page": skip // limit + 1 if limit > 0 else 1,
        "page_size": limit,
    }


def get_word(db: Session, word_id: int):
    return db.query(models.Word).filter(models.Word.id == word_id).first()


def search_words(
    db: Session,
    query: str,
    skip: int = 0,
    limit: int = 100,
    word_types: list = None,
    frequencies: list = None,
):
    db_query = db.query(models.Word).filter(models.Word.word.contains(query))

    if word_types:
        db_query = db_query.filter(models.Word.word_type.in_(word_types))
    if frequencies:
        db_query = db_query.filter(models.Word.frequency.in_(frequencies))

    total = db_query.count()
    words = db_query.offset(skip).limit(limit).all()
    return {
        "items": words,
        "total": total,
        "page": skip // limit + 1 if limit > 0 else 1,
        "page_size": limit,
    }


def get_word_types(db: Session):
    types = db.query(models.Word.word_type).distinct().all()
    return [t[0] for t in types if t[0]]


def get_frequencies(db: Session):
    freqs = db.query(models.Word.frequency).distinct().all()
    return sorted(
        [f[0] for f in freqs if f[0]], key=lambda x: int(x) if x.isdigit() else x
    )
