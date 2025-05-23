import hfendpoint
import torch
from sentence_transformers import SentenceTransformer
import os

model_name = os.getenv('MODEL_ID')

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = SentenceTransformer(model_name, device=device.type)
model.eval()

tokenizer = model.tokenizer
max_length = model.get_max_seq_length()

if max_length is None:
    max_length = tokenizer.model_max_length

encode_batch_size = int(os.getenv('HFENDPOINT_BATCH_SIZE', '32'))
n_threads = torch.get_num_threads()

print(f"Model {model_name} loaded on {device}, {n_threads} thread(s)")

def embeddings(request, send_chunk):
    try:
        features = tokenizer(
            request["input"],
            add_special_tokens=True,
            padding='longest',
            truncation=True,
            max_length=max_length,
            return_attention_mask=True, # for total_tokens
            return_tensors='pt'
        )
        total_tokens = features['attention_mask'].sum().item()

        features_on_device = {}
        for k, v in features.items():
            if torch.is_tensor(v):
                features_on_device[k] = v.to(device)
            else:
                features_on_device[k] = v

        with torch.no_grad():
            if device.type == 'cpu':
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
                    output_features = model(features_on_device)
            else:
                output_features = model(features_on_device)

            embeddings = output_features['sentence_embedding']
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        send_chunk({
            'embeddings': embeddings.cpu().tolist(),
            'usage': {
                'prompt_tokens': total_tokens,
                'total_tokens': total_tokens
            },
            'model': model_name
        })
    except Exception as e:
        print(f"Error in embeddings: {e}")

if __name__ == "__main__":
    hfendpoint.run({
        "embeddings": embeddings,
    })
