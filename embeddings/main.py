from transformers import AutoTokenizer, AutoModelForCausalLM

name = "gpt2"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

emb = model.get_input_embeddings()
print("Embedding table shape:", tuple(emb.weight.shape))  # (vocab_size, hidden_size)

text = "An Apple a day keeps the doctor away."
ids = tok.encode(text)

for token_id in ids:
    row = emb.weight[token_id]
    print(f"{token_id:>6}  {tok.decode([token_id])!r:>10}  length={row.shape[0]}  first 3: {row[:3].tolist()}")

size_mb = emb.weight.numel() * emb.weight.element_size() / 1e6
print(f"Table size: {size_mb:.0f} MB")

# Is the output layer a separate matrix, or does it reuse the table?
lm_head = model.get_output_embeddings()
print("Tied embeddings:", lm_head.weight.data_ptr() == emb.weight.data_ptr())