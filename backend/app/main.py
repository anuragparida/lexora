import os
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
from app import crud, models, schemas, anki_builder
from app.database import get_db, engine, run_migrations

models.Base.metadata.create_all(bind=engine)
run_migrations()

app = FastAPI(title="German Vocabulary API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "German Vocabulary API", "version": "0.1.0"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/words", response_model=schemas.WordListResponse)
def read_words(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    word_types: List[str] = Query(None),
    frequencies: List[str] = Query(None),
    db: Session = Depends(get_db),
):
    return crud.get_words(
        db,
        skip=skip,
        limit=limit,
        word_types=word_types,
        frequencies=frequencies,
    )


@app.get("/words/search", response_model=schemas.WordListResponse)
def search_words(
    q: str = Query(..., min_length=1),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    word_types: List[str] = Query(None),
    frequencies: List[str] = Query(None),
    db: Session = Depends(get_db),
):
    return crud.search_words(
        db,
        query=q,
        skip=skip,
        limit=limit,
        word_types=word_types,
        frequencies=frequencies,
    )


@app.get("/words/filters/options")
def get_filter_options(db: Session = Depends(get_db)):
    return {
        "word_types": crud.get_word_types(db),
        "frequencies": crud.get_frequencies(db),
    }


@app.get("/words/{word_id}", response_model=schemas.WordResponse)
def read_word(word_id: int, db: Session = Depends(get_db)):
    word = crud.get_word(db, word_id=word_id)
    if word is None:
        raise HTTPException(status_code=404, detail="Word not found")
    return word


@app.post("/decks/generate")
def generate_deck(
    word_types: Optional[List[str]] = Query(None),
    frequencies: Optional[List[str]] = Query(None),
    direction: str = Query("both"),
    db: Session = Depends(get_db),
):
    """Generate an Anki deck from filtered words."""
    try:
        filepath = anki_builder.create_anki_deck(
            db,
            word_types=word_types,
            frequencies=frequencies,
            direction=direction,
        )
        filename = filepath.split("/")[-1]
        return {
            "message": "Deck generated successfully",
            "filename": filename,
            "filepath": filepath,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/decks/list")
def list_decks():
    """List all generated decks."""
    decks = []
    if os.path.exists(anki_builder.DECKS_DIR):
        for filename in os.listdir(anki_builder.DECKS_DIR):
            if filename.endswith(".apkg"):
                filepath = os.path.join(anki_builder.DECKS_DIR, filename)
                decks.append(
                    {
                        "filename": filename,
                        "created": os.path.getctime(filepath),
                        "size": os.path.getsize(filepath),
                    }
                )
    return {"decks": sorted(decks, key=lambda x: x["created"], reverse=True)}
