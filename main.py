from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fastapi.templating import Jinja2Templates


app = FastAPI()

# ------------------- Mount the static directory for serving HTML ------------------- #
# app.mount("/static", StaticFiles(directory="static"), name="static")

templates=Jinja2Templates(directory="templates")

# ------------------- Load T5 model & tokenizer ------------------- #
model_name = 't5-small'
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)


# ------------------- Root endpoint: redirect to the HTML page ------------------- #
@app.get("/")
def root(request:Request):
    print("-------Hello world------")
    return templates.TemplateResponse("index.html", {"request": request})


# ------------------- Summarization endpoint ------------------- #
@app.post("/summarize")
async def summarize_text(request: Request):
    # Retrieve JSON data from request
    data = await request.json()
    user_text = data.get("text", "")

    # Preprocess user input
    preprocess_text = user_text.strip().replace("\n", "")
    t5_input_text = f"summarize: {preprocess_text}"

    # Tokenize and generate summary
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

    return {"summary": summary}
