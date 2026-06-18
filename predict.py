import joblib
from pathlib import Path

model_path = Path("model/fake_news_model.joblib")

if not model_path.exists():
    print("Model not found. First run train_model.py")
    exit()

model = joblib.load(model_path)

print("Fake News Detector")
print("------------------")

text = input("Enter news headline or article: ")

if len(text.strip()) < 10:
    print("Please enter more news text.")
else:
    prediction = model.predict([text])[0]
    confidence = model.predict_proba([text])[0].max() * 100

    print("Prediction:", prediction)
    print("Confidence:", round(confidence, 2), "%")

    if prediction == "FAKE":
        print("Warning: This news may be fake. Verify before sharing.")
    else:
        print("This news appears real based on the trained model.")
