"""Prepara wikitext-2 test como .txt (para llama-perplexity)"""
import os
from datasets import load_dataset

OUT = "/mnt/Data_3TB/project Marla/quant-paper/results/wikitext2_test.txt"
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n".join(ds["text"])
with open(OUT, "w") as f:
    f.write(text)
print(f"wikitext-2 test -> {OUT} ({len(text):,} chars)")
