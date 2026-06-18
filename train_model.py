from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


FAKE_PATH = Path("Fake.csv")
TRUE_PATH = Path("True.csv")
MODEL_PATH = Path("model/fake_news_model.joblib")


def clean_text(text):
    text = str(text)

    # Remove Reuters-style tags so the model does not depend only on source words
    text = re.sub(r"\(Reuters\)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"^[A-Z\s]+-\s+", " ", text)

    # Remove extra symbols and spaces
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def load_dataset():
    if not FAKE_PATH.exists() or not TRUE_PATH.exists():
        print("ERROR: Fake.csv and True.csv must be in the same folder.")
        exit()

    fake_df = pd.read_csv(FAKE_PATH)
    true_df = pd.read_csv(TRUE_PATH)

    fake_df["label"] = "FAKE"
    true_df["label"] = "REAL"

    df = pd.concat([fake_df, true_df], ignore_index=True)

    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["content"] = df["title"] + " " + df["text"]

    df["content"] = df["content"].apply(clean_text)

    df = df[df["content"].str.len() > 30]

    return df[["content", "label"]]


def add_extra_training_examples(df):
    extra_data = [
        # Extra REAL examples
        {
            "content": "government announces a new scholarship scheme for engineering students to support higher education",
            "label": "REAL"
        },
        {
            "content": "the university has released the semester examination timetable on its official website",
            "label": "REAL"
        },
        {
            "content": "nasa published satellite data showing changes in global climate patterns",
            "label": "REAL"
        },
        {
            "content": "the health department advised citizens to drink clean water and maintain hygiene during the rainy season",
            "label": "REAL"
        },
        {
            "content": "the reserve bank released its quarterly report on inflation and economic growth",
            "label": "REAL"
        },
        {
            "content": "the education ministry announced new guidelines for digital learning in schools and colleges",
            "label": "REAL"
        },
        {
            "content": "the railway department announced additional special trains during the festival season",
            "label": "REAL"
        },
        {
            "content": "scientists published a peer reviewed study on renewable energy storage systems",
            "label": "REAL"
        },
        {
            "content": "the municipal corporation started a new waste management awareness program in the city",
            "label": "REAL"
        },
        {
            "content": "the government launched an online portal for students to apply for academic scholarships",
            "label": "REAL"
        },

        # Extra FAKE examples
        {
            "content": "one magic tablet can cure all diseases overnight without any medical treatment",
            "label": "FAKE"
        },
        {
            "content": "scientists discovered that drinking hot water once can make humans live for 200 years",
            "label": "FAKE"
        },
        {
            "content": "breaking news a secret alien city has been found under the ocean with unlimited gold",
            "label": "FAKE"
        },
        {
            "content": "a celebrity doctor claims that one spoon of powder can replace all medicines permanently",
            "label": "FAKE"
        },
        {
            "content": "students can get free marks by clicking an unknown link shared on social media",
            "label": "FAKE"
        },
        {
            "content": "government gives free laptops to everyone who shares this message immediately",
            "label": "FAKE"
        },
        {
            "content": "a miracle drink can remove every disease from the body in five minutes",
            "label": "FAKE"
        },
        {
            "content": "aliens secretly helped humans build all modern technology according to viral message",
            "label": "FAKE"
        },
        {
            "content": "click this link to receive free money from a hidden government lottery",
            "label": "FAKE"
        },
        {
            "content": "doctors confirm that sleeping for one hour a day is enough for perfect health",
            "label": "FAKE"
        }
    ]

    extra_df = pd.DataFrame(extra_data)

    # Repeat extra examples to help model learn simple student-style examples
    extra_df = pd.concat([extra_df] * 20, ignore_index=True)

    final_df = pd.concat([df, extra_df], ignore_index=True)

    return final_df


def train_model():
    df = load_dataset()
    df = add_extra_training_examples(df)

    X = df["content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                lowercase=True,
                ngram_range=(1, 2),
                max_df=0.85,
                min_df=1
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ])

    print("Training started...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nTraining completed.")
    print("Accuracy:", accuracy_score(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved successfully.")
    print("Saved as:", MODEL_PATH)


if __name__ == "__main__":
    train_model()
