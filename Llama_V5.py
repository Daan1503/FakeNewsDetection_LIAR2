import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from datetime import datetime

# Load Hugging Face token from environment
my_secret_key = os.environ.get('HF_TOKEN')

# Load the dataset and check columns
df_train = pd.read_csv("train_dataV3.csv")
print("Column names:", df_train.columns)  # Print columns to verify the column name

# Filter out rows with NaN values in the "statement" column
df_train = df_train.dropna(subset=["statement"])

# Define the model configuration
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=my_secret_key)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", use_auth_token=my_secret_key)

# Set up the text generation pipeline
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
)

def get_response(statement):
    prompt = f"""Expand on the following statement: "{statement}". Keep the original statement and add more infornation that has the same sentiment as the statement.
Please make a text with consecutive sentences, no bullet points or paragraphs. The text must end with a period.
"""

    # Calculate token count for the statement
    statement_tokens = len(tokenizer.encode(statement, add_special_tokens=False))

    # Calculate 30% and 50% more than the statement length
    min_expansion_tokens = max(int(statement_tokens * 0.3), 25)
    max_expansion_tokens = int(statement_tokens * 1.5)

    # Set max_new_tokens as statement length + min_expansion_tokens, capped at max_expansion_tokens
    max_new_tokens = statement_tokens + min_expansion_tokens
    max_new_tokens = min(max_new_tokens, max_expansion_tokens) + 20  # Increase the buffer for completion

    # Generate response with the calculated max_new_tokens
    response = text_generator(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"]

    # Remove the prompt from the generated text
    response = response[len(prompt):]

    # Post-process to ensure the response ends with a complete sentence and period
    last_period_index = response.rfind('.')
    if last_period_index != -1:
        response = response[:last_period_index + 1]
    else:
        # If there's no period, try adding more tokens and regenerate
        max_new_tokens += 15  # Allow a bit more room for a complete sentence
        response = text_generator(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"][len(prompt):]
        
        # Check again for sentence completion
        last_period_index = response.rfind('.')
        if last_period_index != -1:
            response = response[:last_period_index + 1]
        else:
            response = response.rstrip() + '.'

    return response

# Iterate over each statement and generate expanded responses
expanded_texts = []
for statement in df_train["statement"]:
    expanded_text = get_response(statement)
    expanded_texts.append(expanded_text)

# Add the expanded responses to the DataFrame
df_train["expanded_statement"] = expanded_texts

# Save the updated DataFrame with expanded statements
df_train.to_csv("expanded_train_df.csv", index=False, encoding='utf-8')