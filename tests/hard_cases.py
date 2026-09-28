"""Hard test cases for the IMDB sentiment models.
Run from nlp-project/:  uv run python tests/hard_cases.py
"""
import sys
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from utils.text_preprocessing import preprocess

CASES = [
    # (category, review, expected)
    ("Negation", "This movie was not good at all.", "negative"),
    ("Negation", "I didn't like it, and I didn't hate it either, but mostly I didn't care.", "negative"),
    ("Negation", "Not a single moment of this film was boring.", "positive"),
    ("Double negation", "It's not that I didn't enjoy it, I really did.", "positive"),
    ("Not bad", "Honestly not bad. Not bad at all.", "positive"),
    ("Negated praise", "Never have I been so disappointed by a cast this talented.", "negative"),
    ("Sarcasm", "Oh great, another two hours of my life I will never get back. Brilliant.", "negative"),
    ("Sarcasm", "Wow, what a masterpiece. I only fell asleep three times.", "negative"),
    ("Sarcasm", "If you love clichés, wooden acting and a plot full of holes, this is the perfect movie for you.", "negative"),
    ("Contrast (but)", "The visuals were stunning and the music was beautiful, but the story was a complete mess and I hated the ending.", "negative"),
    ("Contrast (but)", "The first half was slow and boring, but the last hour was absolutely incredible.", "positive"),
    ("Contrast (although)", "Although the acting was terrible, I still loved every minute of it.", "positive"),
    ("Expectation", "I expected it to be terrible. It wasn't. It was wonderful.", "positive"),
    ("Expectation", "Everyone said it was a masterpiece, so I was expecting a great film. What a letdown.", "negative"),
    ("Comparison", "The original was a masterpiece. This remake is nothing like it.", "negative"),
    ("Comparison", "Better than I expected, and far better than the terrible sequel.", "positive"),
    ("Rating only", "3/10", "negative"),
    ("Rating only", "10/10 would watch again", "positive"),
    ("Rating in text", "I give it 2 out of 10. The only good part was the popcorn.", "negative"),
    ("Recommendation", "Save your money and wait for it to come out on TV.", "negative"),
    ("Recommendation", "Do yourself a favour and go see this in a cinema.", "positive"),
    ("Negative words, positive meaning", "A dark, brutal, disturbing and terrifying horror film that kept me awake all night. Loved it.", "positive"),
    ("Positive words, negative meaning", "The best thing about this movie was the end credits.", "negative"),
    ("Positive words, negative meaning", "Great actors, great director, great budget. How did they make something this bad?", "negative"),
    ("Plot summary (little opinion)", "A young man returns to his hometown after his father dies and discovers a family secret that changes everything.", "positive"),
    ("Very short", "Meh.", "negative"),
    ("Very short", "Wow.", "positive"),
    ("Slang", "This film slaps. Absolute banger, no cap.", "positive"),
    ("Slang", "Mid. Totally mid.", "negative"),
    ("Typos", "ths movi was sooo borring and the acter was terible", "negative"),
    ("Emoji", "😍😍😍 🔥🔥", "positive"),
    ("Emoji", "👎👎 🤮", "negative"),
    ("Question", "Why would anyone pay to watch this?", "negative"),
    ("Mixed / neutral", "It was okay. Some parts were good, some were bad. Average.", "negative"),
    ("Sequel", "The sequel is somehow worse than the first one, and the first one was awful.", "negative"),
    ("Hindsight", "I used to think this was the worst film ever made, but after a rewatch I now think it is a hidden gem.", "positive"),
]

MODELS = {
    "CV+LR": ("count_vectorizer", "cv_logistic_regression"),
    "TFIDF+LR": ("tfidf_vectorizer", "tfidf_logistic_regression"),
    "CV+NB": ("count_vectorizer", "cv_multinomial_nb"),
    "TFIDF+NB": ("tfidf_vectorizer", "tfidf_multinomial_nb"),
}

def main():
    loaded = {name: (joblib.load(BASE_DIR / "Model/vectorizer" / f"{v}.joblib"),
                     joblib.load(BASE_DIR / "Model/models" / f"{m}.joblib"))
              for name, (v, m) in MODELS.items()}
    rows = []
    for cat, text, expected in CASES:
        processed = preprocess(text)
        row = {"Category": cat, "Review": text, "Expected": expected,
               "Processed": processed or "(empty)"}
        for name, (vec, model) in loaded.items():
            proba = model.predict_proba(vec.transform([processed]))[0]
            pos = proba[list(model.classes_).index("positive")]
            label = "positive" if pos >= 0.5 else "negative"
            row[name] = f"{'OK ' if label == expected else 'XX '}{label[:3]} {pos:.0%}"
            row[name + "_ok"] = label == expected
        rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print(df[["Category", "Review", "Expected"] + list(MODELS)].to_string(index=False))
    print("\nCorrect out of", len(df))
    for name in MODELS:
        print(f"  {name:9s} {df[name + '_ok'].sum()}/{len(df)}")
    return df

if __name__ == "__main__":
    main()
