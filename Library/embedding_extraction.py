import os
import pickle
import time

import torch
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

model_name = "sentence-transformers/all-mpnet-base-v2"

# Load model and tokenizer from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name,
    output_hidden_states=True,
    output_attentions=True
).to(DEVICE)


def mean_pooling(last_hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(dim=-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return torch.nn.functional.normalize(sum_embeddings / sum_mask, p=2, dim=1)


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Function to get embeddings for a batch of texts
def get_OpenAI_embeddings(texts, model="text-embedding-3-small"):
    texts = [text.replace("\n", " ") for text in texts]  # Clean up text
    response = client.embeddings.create(input=texts, model=model)
    return [res_data.embedding for res_data in response.data]


def extract_embeddings(text, foldername, model_name="mpnet"):
    filename = "embed_data"

    if model_name != "gpt4":
        tokenized_input = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=384,  # Natural input length for all-mpnet-base-v2
            return_tensors='pt'
        )
    else:
        filename += "-gpt4"
        tokenized_input = None

    try:
        with open(f"{foldername}/{filename}.pickle", 'rb') as handle:
            embeddings = pickle.load(handle)
            print(f"File '{foldername}/{filename}.pickle' loaded successfully.")
    except FileNotFoundError:
        print(
            f"Could not find file '{foldername}/{filename}.pickle'."
            "Regenerating the embeddings."
        )

        if model_name == "gpt4":
            # Function to process texts in batches
            def batched_embeddings(texts, batch_size=16, model="text-embedding-3-small"):
                embeddings = []
                for i in tqdm(range(0, len(texts), batch_size)):
                    batch_texts = texts[i:i + batch_size]
                    batch_embeddings = get_OpenAI_embeddings(batch_texts, model=model)
                    embeddings.extend(batch_embeddings)
                    time.sleep(0.75)  # Prevent making too many requests too fast
                return torch.tensor(embeddings)

            embeddings = batched_embeddings(text, batch_size=16)
        else:
            batch_size = 64
            embeddings = []
            for idx in tqdm(range(0, len(text), batch_size)):
                batch_input_ids = tokenized_input["input_ids"][idx:idx + batch_size]
                batch_attn_masks = tokenized_input["attention_mask"][idx:idx + batch_size]

                # Compute token embeddings
                with torch.no_grad():
                    model_output = model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attn_masks
                    )

                # Perform pooling of the model's output as the embeddings
                embeddings.append(
                    mean_pooling(model_output.last_hidden_state, batch_attn_masks)
                )

                # Clear to save ram
                del model_output

            # Save the embeddings as a pickle file for later use
            embeddings = torch.cat(embeddings)

        with open(f"{foldername}/{filename}.pickle", 'wb') as handle:
            pickle.dump(
                obj=embeddings,
                file=handle,
                protocol=pickle.HIGHEST_PROTOCOL
            )

    return embeddings, tokenized_input
