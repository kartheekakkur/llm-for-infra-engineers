import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

texts = [
    "An Apple a day keeps the doctor away.",
    "That was an unbelievable shot by the batsman.",
]

for text in texts:
    ids = enc.encode(text)
    print(f"Text: {text}")
    print(f"Characters: {len(text)}, Tokens: {len(ids)}, Chars per token: {len(text) / len(ids):.2f}")
    for token_id in ids:
        print(f"  {token_id:>7}  {enc.decode([token_id])!r}")
    print()