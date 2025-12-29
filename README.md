# NEXT-WORD-PREDICTION-

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "gpt2-large"

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


def generate_next_three_words(prompt, temperature, top_k):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    output = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + 3,
        temperature=temperature,
        top_k=top_k,
        num_beams=5,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )


    generated_tokens = output[0][-3:].tolist()
    next_words = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return next_words

