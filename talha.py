from transformers import T5Tokenizer, T5ForConditionalGeneration
model_name = 't5-small'
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Dummy text for testing
dummy_text = "My name is talha and i'm a good boy as well as i'm a student of the university"
print("Input Text:", dummy_text)

# Preprocess and summarize the input text
preprocess_text = dummy_text.strip().replace("\n", "")
t5_input_text = f"summarize: {preprocess_text}"
tokenized_text = tokenizer.encode(t5_input_text, return_tensors="pt")
summary_ids = model.generate(
    tokenized_text, 
    num_beams=4, 
    no_repeat_ngram_size=2, 
    min_length=10, 
    max_length=50, 
    early_stopping=True
)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Print the summary in the terminal
print("Summarized Output::::::", summary)