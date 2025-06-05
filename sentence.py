import hfendpoint
import torch
from sentence_transformers import SentenceTransformer
import os

try:
    # ugly hack to hide stupid warnings
    stderr = os.dup(2)
    os.dup2(os.open(os.devnull, os.O_RDWR), 2)
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False
finally:
    if stderr != -1:
        os.dup2(stderr, 2)
        os.close(stderr)

model_name = os.getenv('MODEL_ID')

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = SentenceTransformer(model_name, device=device.type)
model.eval()

if IPEX_AVAILABLE and device.type == 'cpu':
    try:
        model = ipex.optimize(model, dtype=torch.bfloat16)
        print("IPEX optimization enabled")
    except Exception as e:
        print(f"IPEX optimization failed: {e}")
        IPEX_AVAILABLE = False

tokenizer = model.tokenizer
max_length = model.get_max_seq_length()

if max_length is None:
    max_length = tokenizer.model_max_length

encode_batch_size = int(os.getenv('HFENDPOINT_BATCH_SIZE', '32'))
n_threads = torch.get_num_threads()

print(f"Model {model_name} loaded on {device}, {n_threads} thread(s)")

def tokenize(request, send_chunk):
    try:
        features = tokenizer(
            request["input"],
            add_special_tokens=True,
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        send_chunk({
            "tokens": features['input_ids']
        })
    except Exception as e:
        print(f"Error in tokenize: {e}")

def embeddings(request, send_chunk):
    try:
        features = tokenizer.pad(
            {'input_ids': request["input"]},
            padding='longest',
            return_tensors='pt',
            return_attention_mask=True
        )
        features_on_device = {}

        for k, v in features.items():
            if torch.is_tensor(v):
                features_on_device[k] = v.to(device)
            else:
                features_on_device[k] = v

        with torch.no_grad():
            if device.type == 'cpu' and not IPEX_AVAILABLE:
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
                    output_features = model(features_on_device)
            else:
                output_features = model(features_on_device)

            embeddings = output_features['sentence_embedding']
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        send_chunk({
            'embeddings': embeddings.cpu().tolist()
        })
    except Exception as e:
        print(f"Error in embeddings: {e}")

if __name__ == "__main__":
    hfendpoint.run({
        "tokenize": tokenize,
        "embeddings": embeddings,
    })
