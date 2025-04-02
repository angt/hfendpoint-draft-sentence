import hfendpoint
import torch
from sentence_transformers import SentenceTransformer

model_name = 'all-MiniLM-L6-v2'
device = torch.device("mps") if torch.backends.mps.is_available() and torch.backends.mps.is_built() else torch.device("cpu")
model = SentenceTransformer(model_name, device=device)
print(f"Model {model_name} loaded on {device}")

def embeddings_handler(request_data, send_chunk):
    try:
        input_data = request_data.get('input', '')
        texts = [input_data] if isinstance(input_data, str) else [str(item) for item in input_data]
        embeddings = [emb.tolist() for emb in model.encode(texts)] if texts else []
        send_chunk({
            'embeddings': embeddings,
            'usage': {'prompt_tokens': 0, 'total_tokens': 0}, # Fake
            'model': model_name
        })
    except Exception as e:
        print(f"Error in handler: {e}")

if __name__ == "__main__":
    hfendpoint.run({ "embeddings": embeddings_handler })
