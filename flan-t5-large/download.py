from transformers import T5Tokenizer, T5ForConditionalGeneration

T5Tokenizer.from_pretrained("google/flan-t5-large")
T5ForConditionalGeneration.from_pretrained("google/flan-t5-large",device_map="auto", offload_folder="offload")

print("Download complete")