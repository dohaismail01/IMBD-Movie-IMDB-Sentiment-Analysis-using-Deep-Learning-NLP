# To run the app: streamlit run app.py  # quick run instruction
#  # spacer
import json  # standard library: load config JSON
from pathlib import Path  # standard library: filesystem paths
from typing import Dict, Tuple  # typing helpers
#  # spacer
import numpy as np  # numerical arrays
import streamlit as st  # Streamlit UI framework
#  # spacer
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"  # folder that contains saved models/vectorizers
#  # spacer
TEST_CASES = [  # simple manual test prompts you can paste
    {"expected": "positive", "text": "Absolutely loved this movie — great acting, tight script, and an ending that really worked."},  # 1
    {"expected": "negative", "text": "This was painfully boring. The plot went nowhere and the performances were wooden."},  # 2
    {"expected": "positive", "text": "Funny, smart, and surprisingly emotional. I’d happily watch it again."},  # 3
    {"expected": "negative", "text": "A complete mess: terrible pacing, confusing story, and the dialogue was cringe."},  # 4
    {"expected": "positive", "text": "Beautiful cinematography and a powerful soundtrack. The whole experience felt inspiring."},  # 5
    {"expected": "negative", "text": "I regret wasting my time. Predictable twists, bad jokes, and zero chemistry."},  # 6
    {"expected": "positive (mixed)", "text": "It starts slow, but once it gets going it’s engaging and ends on a strong note."},  # 7
    {"expected": "negative (mixed)", "text": "There are a couple of good scenes, but overall it’s dull and way too long."},  # 8
    {"expected": "negative (sarcasm)", "text": "What a masterpiece… if the goal was to make the worst movie of the year."},  # 9
    {"expected": "ambiguous", "text": "Not sure how I feel. Some parts were brilliant, others were awful — it’s complicated."},  # 10
]  # end test cases
#  # spacer

@st.cache_resource  # cache across reruns for speed
def load_config() -> Dict:  # load config.json (max_len, vocab_size, label map)
    config_path = ARTIFACTS_DIR / "config.json"  # config file path
    with config_path.open("r", encoding="utf-8") as f:  # open config for reading
        return json.load(f)  # parse JSON into dict
#  # spacer


@st.cache_resource  # cache baseline artifacts across reruns
def load_tfidf_baseline():  # load TF-IDF vectorizer and logistic regression classifier
    import joblib  # local import so app can still start even if TF stack fails elsewhere
#  # spacer
    vectorizer = joblib.load(ARTIFACTS_DIR / "tfidf_vectorizer.joblib")  # load saved TF-IDF vectorizer
    clf = joblib.load(ARTIFACTS_DIR / "tfidf_logreg.joblib")  # load saved logistic regression model
    return vectorizer, clf  # return both objects
#  # spacer


@st.cache_resource  # cache tokenizer across reruns
def load_tokenizer():  # load Keras tokenizer saved via tokenizer.to_json()
    from tensorflow.keras.preprocessing.text import tokenizer_from_json  # convert JSON string back to tokenizer
#  # spacer
    tok_path = ARTIFACTS_DIR / "tokenizer.json"  # tokenizer file path
    with tok_path.open("r", encoding="utf-8") as f:  # open tokenizer json
        tok_json = f.read()  # read full JSON string
#  # spacer
    return tokenizer_from_json(tok_json)  # rebuild tokenizer object
#  # spacer


@st.cache_resource  # cache Keras model across reruns
def load_keras_model(model_name: str):  # load a saved Keras model from disk
    from tensorflow.keras.models import load_model  # Keras model loader
#  # spacer
    if model_name == "scratch":  # scratch CNN-BiLSTM
        path = ARTIFACTS_DIR / "scratch_model.keras"  # scratch model path
    elif model_name == "glove_finetuned":  # GloVe fine-tuned model
        path = ARTIFACTS_DIR / "glove_finetuned_model.keras"  # glove model path
    else:  # unknown option
        raise ValueError(f"Unknown model_name: {model_name}")  # fail fast for invalid names
#  # spacer
    # For inference we do NOT need to compile the model.  # avoids optimizer deserialization issues
    # This also avoids optimizer deserialization issues across Keras/TensorFlow versions.  # extra context
    return load_model(path, compile=False)  # load without compile for inference-only
#  # spacer


def ensure_artifacts_present() -> Tuple[bool, str]:  # verify all required files exist
    required = [  # list of required artifact paths
        ARTIFACTS_DIR / "config.json",  # config
        ARTIFACTS_DIR / "tokenizer.json",  # tokenizer
        ARTIFACTS_DIR / "tfidf_vectorizer.joblib",  # TF-IDF vectorizer
        ARTIFACTS_DIR / "tfidf_logreg.joblib",  # baseline classifier
        ARTIFACTS_DIR / "scratch_model.keras",  # scratch model
        ARTIFACTS_DIR / "glove_finetuned_model.keras",  # glove model
    ]  # end required list
#  # spacer
    missing = [p.name for p in required if not p.exists()]  # compute missing files
    if missing:  # if anything is missing
        return False, "Missing artifacts in ./artifacts/: " + ", ".join(missing)  # return error message
    return True, ""  # otherwise ok
#  # spacer


def predict_with_tfidf(text: str) -> Tuple[str, float, np.ndarray]:  # baseline prediction function
    vectorizer, clf = load_tfidf_baseline()  # load TF-IDF + classifier
#  # spacer
    X = vectorizer.transform([text])  # vectorize input text
    proba = clf.predict_proba(X)[0]  # predict probabilities: [neg, pos]
    pos_prob = float(proba[1])  # extract positive probability
    label = "positive" if pos_prob >= 0.5 else "negative"  # threshold at 0.5
#  # spacer
    return label, pos_prob, proba  # return label, P(pos), and full proba vector
#  # spacer


def predict_with_keras(text: str, model_name: str) -> Tuple[str, float, np.ndarray]:  # Keras model prediction
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # pad sequences to fixed length
#  # spacer
    config = load_config()  # load config dict
    max_len = int(config["max_len"])  # get max_len used in training
#  # spacer
    tokenizer = load_tokenizer()  # load saved tokenizer
    seq = tokenizer.texts_to_sequences([text])  # convert text to token ids
    X = pad_sequences(seq, maxlen=max_len)  # pad/truncate to max_len
#  # spacer
    model = load_keras_model(model_name)  # load selected Keras model
    proba = model.predict(X, verbose=0)[0]  # run inference and take first example
#  # spacer
    # Notebook uses Dense(2, softmax) => [neg, pos]  # expected output shape
    proba = np.asarray(proba).reshape(-1)  # flatten to 1D
    if proba.size == 1:  # handle sigmoid-style binary output (just in case)
        pos_prob = float(proba[0])  # interpret single value as P(pos)
        label = "positive" if pos_prob >= 0.5 else "negative"  # threshold at 0.5
        return label, pos_prob, np.array([1 - pos_prob, pos_prob], dtype=float)  # synthesize [neg,pos]
#  # spacer
    pos_prob = float(proba[1])  # extract positive prob
    label = "positive" if int(np.argmax(proba)) == 1 else "negative"  # choose argmax class
    return label, pos_prob, proba  # return label, P(pos), vector
#  # spacer


def main() -> None:  # Streamlit entry
    st.set_page_config(page_title="IMDB Sentiment Classifier", page_icon="🎬", layout="centered")  # page metadata
#  # spacer
    st.title("IMDB Sentiment Classifier")  # header
    st.caption("Enter a movie review and get a predicted sentiment.")  # short description
#  # spacer
    ok, err = ensure_artifacts_present()  # validate artifacts
    if not ok:  # if missing artifacts
        st.error(err)  # show error
        st.stop()  # stop execution
#  # spacer
    model_choice = st.radio(  # model selector
        "Choose model",  # label
        options=[  # choices
            "TF-IDF + Logistic Regression (baseline)",  # baseline
            "Keras: Scratch CNN-BiLSTM",  # scratch
            "Keras: GloVe fine-tuned",  # glove
        ],  # end options
        index=2,  # default to glove
    )  # end radio
#  # spacer
    with st.expander("Show 10 test cases", expanded=False):  # optional test cases
        for i, tc in enumerate(TEST_CASES, start=1):  # loop tests
            st.write(f"{i}. Expected: **{tc['expected']}**")  # show expected
            st.code(tc["text"])  # show text
#  # spacer
    if "review_text" not in st.session_state:  # initialize session state for text area
        st.session_state["review_text"] = ""  # default empty
#  # spacer
    example = "I absolutely loved this movie. The acting was great and the story kept me engaged."  # placeholder text
    text = st.text_area("Movie review", key="review_text", height=200, placeholder=example)  # input box
#  # spacer
    col1, col2 = st.columns([1, 1])  # layout columns
    with col1:  # left column
        predict_clicked = st.button("Predict", type="primary")  # predict button
    with col2:  # right column
        st.button("Clear", on_click=lambda: st.session_state.update({"review_text": ""}))  # clear input
#  # spacer
    if predict_clicked:  # if user clicked predict
        cleaned = (text or "").strip()  # strip whitespace
        if not cleaned:  # empty input
            st.warning("Please paste/type a review first.")  # prompt user
            st.stop()  # stop
#  # spacer
        with st.spinner("Running inference..."):  # spinner while predicting
            if model_choice.startswith("TF-IDF"):  # baseline path
                label, pos_prob, proba = predict_with_tfidf(cleaned)  # TF-IDF prediction
            elif "Scratch" in model_choice:  # scratch model
                label, pos_prob, proba = predict_with_keras(cleaned, model_name="scratch")  # Keras prediction
            else:  # glove model
                label, pos_prob, proba = predict_with_keras(cleaned, model_name="glove_finetuned")  # Keras prediction
#  # spacer
        st.subheader("Result")  # result section
        st.write(f"**Predicted sentiment:** {label}")  # show predicted label
        st.write(f"**P(positive):** {pos_prob:.3f}")  # show positive probability
#  # spacer
        st.subheader("Probabilities")  # probabilities section
        st.write({"negative": float(proba[0]), "positive": float(proba[1])})  # show both probs
#  # spacer
        st.divider()  # horizontal divider
        st.caption("Artifacts loaded from ./artifacts")  # footer note
#  # spacer


if __name__ == "__main__":  # run main only when executed directly
    main()  # start Streamlit app logic
