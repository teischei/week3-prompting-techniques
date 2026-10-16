"""RAG — policy-grounded answers.

Task: answer a customer question using ONLY the retrieved help-center text. If the
answer isn't in the docs, the model must say it doesn't know (don't hallucinate).

Fill in YOUR_SYSTEM_PROMPT and YOUR_RETRIEVER.
Run:  python rag.py
"""
import os

from ollama import chat

MODEL = "llama3.1:8b"
DATA = os.path.join(os.path.dirname(__file__), "data", "help_center.txt")


def load_chunks() -> list[str]:
    with open(DATA, encoding="utf-8") as f:
        # one chunk per non-empty line
        return [line.strip() for line in f if line.strip()]


# Return help-center chunks relevant to the question using normalized keyword overlap.
def YOUR_RETRIEVER(question: str, chunks: list[str]) -> list[str]:
    aliases = {
        "returned": "return",
        "returns": "return",
        "returning": "return",
        "shipping": "ship",
        "shipped": "ship",
        "ships": "ship",
        "internationally": "international",
        "countries": "country",
    }

    def terms(text: str) -> set[str]:
        return {
            aliases.get(word.lower().strip(".,?!:;'\""), word.lower().strip(".,?!:;'\""))
            for word in text.split()
            if word.lower().strip(".,?!:;'\"")
        }

    question_terms = terms(question)
    relevant = []
    for index, chunk in enumerate(chunks):
        overlap = question_terms & terms(chunk)
        if overlap:
            relevant.append((len(overlap), index, chunk))

    return [chunk for _, _, chunk in sorted(relevant, key=lambda item: (-item[0], item[1]))]


# System prompt grounds answers in the retrieved context.
YOUR_SYSTEM_PROMPT = """You are a support-desk assistant. Answer only with facts supported by the
help-center context supplied in the user's message. Do not rely on outside knowledge or make up
policy details. If the context is empty or does not answer the question, respond politely: "I don't
know based on the help-center information provided." Keep supported answers concise."""

CASES = [
    ("How long do I have to return an item?", ["30 days"]),
    ("Do you ship internationally?", ["international"]),
    ("What's your CEO's home address?", ["don't know", "do not know", "not sure", "cannot"]),
]


def answer(question: str) -> str:
    context = "\n".join(YOUR_RETRIEVER(question, load_chunks()))
    resp = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": YOUR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0.0},
    )
    return resp.message.content.strip().lower()


def run() -> bool:
    passed = 0
    for question, needles in CASES:
        got = answer(question)
        ok = any(n.lower() in got for n in needles)
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {question!r} -> {got[:80]!r}")
    print(f"{passed}/{len(CASES)} correct")
    return passed == len(CASES)


if __name__ == "__main__":
    run()
