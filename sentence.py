import hfendpoint
import torch
from sentence_transformers import SentenceTransformer

model_name = 'all-MiniLM-L6-v2'
device = torch.device("mps") if torch.backends.mps.is_available() and torch.backends.mps.is_built() else torch.device("cpu")
model = SentenceTransformer(model_name, device=device)
tokenizer = model.tokenizer

print(f"Model {model_name} loaded on {device}")

def embeddings_handler(request_data, send_chunk):
    try:
        texts = request_data["input"]
        tokens = sum(len(tokenizer.encode(t, add_special_tokens=True)) for t in texts)
        embeddings = model.encode(texts)
        send_chunk({
            'embeddings': [emb.tolist() for emb in embeddings],
            'usage': {
                'prompt_tokens': tokens,
                'total_tokens': tokens
            },
            'model': model_name
        })
    except Exception as e:
        print(f"Error in handler: {e}")

if __name__ == "__main__":
    hfendpoint.run({ "embeddings": embeddings_handler })
