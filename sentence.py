import hfendpoint
import torch
from sentence_transformers import SentenceTransformer
import os

model_name = os.getenv('MODEL_ID')
device = torch.device("mps") if torch.backends.mps.is_available() and torch.backends.mps.is_built() else torch.device("cpu")
model = SentenceTransformer(model_name, device=device)
tokenizer = model.tokenizer

encode_batch_size = int(os.getenv('HFENDPOINT_BATCH_SIZE', '32'))
n_threads = torch.get_num_threads()

print(f"Model {model_name} loaded on {device}, {n_threads} thread(s)")

def embeddings_handler(request_data, send_chunk):
    try:
        texts = request_data["input"]
        tokens = sum(len(tokenizer.encode(t, add_special_tokens=True)) for t in texts)
        embeddings = model.encode(
            texts,
            batch_size=encode_batch_size,
            device=device.type
        )
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
