# ============================================================
#   HuggingFace Transformers - All Pipeline Tasks
#   Author  : HEMANTHSELVA A K
#   Task    : Sourcesys Technologies - Data Science Internship
#   Topics  : QA, Token Classification, Text Classification,
#             Zero-Shot, Summarization, Text Generation,
#             Sentence Similarity, Feature Extraction
# ============================================================

# Install (run once in terminal / Colab):
# pip install transformers torch sentence-transformers

from transformers import pipeline
import torch

DIVIDER = "\n" + "=" * 60 + "\n"


# ─────────────────────────────────────────────────────────────
# 1. QUESTION ANSWERING
# ─────────────────────────────────────────────────────────────
def task_question_answering():
    print(DIVIDER + "TASK 1: QUESTION ANSWERING" + DIVIDER)

    qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    context = """
    Artificial Intelligence (AI) is the simulation of human intelligence processes by machines,
    especially computer systems. These processes include learning, reasoning, and self-correction.
    Machine Learning is a subset of AI that gives computers the ability to learn without being
    explicitly programmed. Deep Learning is a subset of Machine Learning that uses neural networks
    with many layers to analyze data.
    """

    questions = [
        "What is Artificial Intelligence?",
        "What is Machine Learning a subset of?",
        "What does Deep Learning use to analyze data?"
    ]

    for q in questions:
        result = qa(question=q, context=context)
        print(f"Q: {q}")
        print(f"A: {result['answer']}  (Score: {result['score']:.4f})\n")


# ─────────────────────────────────────────────────────────────
# 2. TOKEN CLASSIFICATION (Named Entity Recognition)
# ─────────────────────────────────────────────────────────────
def task_token_classification():
    print(DIVIDER + "TASK 2: TOKEN CLASSIFICATION (NER)" + DIVIDER)

    ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english",
                   aggregation_strategy="simple")

    sentences = [
        "Elon Musk founded SpaceX and Tesla in California.",
        "Barack Obama was the 44th President of the United States.",
        "Google was founded in Menlo Park by Larry Page and Sergey Brin."
    ]

    for sentence in sentences:
        print(f"Sentence: {sentence}")
        entities = ner(sentence)
        for entity in entities:
            print(f"  Entity: {entity['word']:20s} | Type: {entity['entity_group']:5s} | Score: {entity['score']:.4f}")
        print()


# ─────────────────────────────────────────────────────────────
# 3. TEXT CLASSIFICATION (Sentiment Analysis)
# ─────────────────────────────────────────────────────────────
def task_text_classification():
    print(DIVIDER + "TASK 3: TEXT CLASSIFICATION (Sentiment Analysis)" + DIVIDER)

    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

    texts = [
        "I absolutely love this product! It works perfectly.",
        "This is the worst experience I've ever had.",
        "The movie was okay, nothing special.",
        "Fantastic service and amazing quality!",
        "I am very disappointed with this purchase."
    ]

    for text in texts:
        result = classifier(text)[0]
        print(f"Text   : {text}")
        print(f"Label  : {result['label']}  |  Score: {result['score']:.4f}\n")


# ─────────────────────────────────────────────────────────────
# 4. ZERO-SHOT CLASSIFICATION
# ─────────────────────────────────────────────────────────────
def task_zero_shot_classification():
    print(DIVIDER + "TASK 4: ZERO-SHOT CLASSIFICATION" + DIVIDER)

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    examples = [
        {
            "text": "The new iPhone 15 comes with a powerful A17 chip and improved camera system.",
            "labels": ["technology", "sports", "politics", "food", "entertainment"]
        },
        {
            "text": "The team scored three goals in the final match to win the championship.",
            "labels": ["technology", "sports", "politics", "food", "health"]
        },
        {
            "text": "The government announced new tax reforms to boost the economy.",
            "labels": ["technology", "sports", "politics", "food", "science"]
        }
    ]

    for ex in examples:
        result = classifier(ex["text"], candidate_labels=ex["labels"])
        print(f"Text   : {ex['text']}")
        print(f"Labels : {ex['labels']}")
        print(f"Top    : {result['labels'][0]}  (Score: {result['scores'][0]:.4f})")
        print("All Scores:")
        for label, score in zip(result['labels'], result['scores']):
            print(f"  {label:15s}: {score:.4f}")
        print()


# ─────────────────────────────────────────────────────────────
# 5. SUMMARIZATION
# ─────────────────────────────────────────────────────────────
def task_summarization():
    print(DIVIDER + "TASK 5: SUMMARIZATION" + DIVIDER)

    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    article = """
    Climate change refers to long-term shifts in global temperatures and weather patterns.
    While some climate change is natural, since the 1800s, human activities have been the
    main driver of climate change, primarily due to the burning of fossil fuels like coal,
    oil and gas. Burning fossil fuels generates greenhouse gas emissions that act like a
    blanket wrapped around the Earth, trapping the sun's heat and raising temperatures.
    The main greenhouse gases that are causing climate change include carbon dioxide and
    methane. These come from using gasoline for driving a car or coal for heating a building.
    Clearing land and forests can also release carbon dioxide. Landfills for garbage are a
    major source of methane emissions. Energy, industry, transport, buildings, agriculture
    and land use are among the main sectors causing greenhouse gases. Climate change effects
    include intense droughts, water scarcity, severe fires, rising sea levels, flooding,
    melting polar ice, catastrophic storms and declining biodiversity.
    """

    summary = summarizer(article, max_length=80, min_length=30, do_sample=False)
    print("Original Article:")
    print(article.strip())
    print(f"\nSummary:\n{summary[0]['summary_text']}\n")


# ─────────────────────────────────────────────────────────────
# 6. TEXT GENERATION
# ─────────────────────────────────────────────────────────────
def task_text_generation():
    print(DIVIDER + "TASK 6: TEXT GENERATION" + DIVIDER)

    generator = pipeline("text-generation", model="gpt2")

    prompts = [
        "Artificial Intelligence will change the future by",
        "Once upon a time in a world full of data,",
        "The best way to learn machine learning is"
    ]

    for prompt in prompts:
        result = generator(
            prompt,
            max_new_tokens=60,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            truncation=True
        )
        print(f"Prompt    : {prompt}")
        print(f"Generated : {result[0]['generated_text']}\n")


# ─────────────────────────────────────────────────────────────
# 7. SENTENCE SIMILARITY
# ─────────────────────────────────────────────────────────────
def task_sentence_similarity():
    print(DIVIDER + "TASK 7: SENTENCE SIMILARITY" + DIVIDER)

    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentence_pairs = [
        ("I love playing football.", "Soccer is my favorite sport."),
        ("The cat sat on the mat.", "Dogs are loyal companions."),
        ("Machine learning is a branch of AI.", "AI includes deep learning and ML."),
        ("It is raining today.", "The weather is sunny and bright.")
    ]

    for s1, s2 in sentence_pairs:
        emb1 = model.encode(s1, convert_to_tensor=True)
        emb2 = model.encode(s2, convert_to_tensor=True)
        score = util.cos_sim(emb1, emb2).item()
        print(f"Sentence 1 : {s1}")
        print(f"Sentence 2 : {s2}")
        print(f"Similarity : {score:.4f}  {'(High)' if score > 0.6 else '(Low)'}\n")


# ─────────────────────────────────────────────────────────────
# 8. FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────
def task_feature_extraction():
    print(DIVIDER + "TASK 8: FEATURE EXTRACTION" + DIVIDER)

    extractor = pipeline("feature-extraction", model="distilbert-base-uncased")

    texts = [
        "Natural Language Processing is a field of AI.",
        "Python is the most popular programming language for data science.",
        "Neural networks are inspired by the human brain."
    ]

    for text in texts:
        features = extractor(text)
        import torch as _torch
        tensor = _torch.tensor(features[0])        # shape: [seq_len, hidden_size]
        sentence_embedding = tensor.mean(dim=0)    # mean pooling -> [hidden_size]
        print(f"Text        : {text}")
        print(f"Shape       : {tensor.shape}  (tokens x hidden_dim)")
        print(f"Sentence Vec: first 8 dims -> {sentence_embedding[:8].tolist()}\n")


# ─────────────────────────────────────────────────────────────
# MAIN - Run All Tasks
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   HuggingFace Transformers - All Pipeline Tasks Demo")
    print("   Author: HEMANTHSELVA A K | Sourcesys Technologies")
    print("=" * 60)

    task_question_answering()
    task_token_classification()
    task_text_classification()
    task_zero_shot_classification()
    task_summarization()
    task_text_generation()
    task_sentence_similarity()
    task_feature_extraction()

    print(DIVIDER + "All 8 Tasks Completed Successfully!" + DIVIDER)
