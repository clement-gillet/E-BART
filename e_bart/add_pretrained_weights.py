import torch
from transformers import AutoConfig

from model.modeling_e_bart import BartForConditionalGeneration

existing_weights = torch.load('/Users/clementgillet/Desktop/model.bin')

config = AutoConfig.from_pretrained("model/config.json")
model = BartForConditionalGeneration(config=config)

missing_keys, unexpected_keys = model.load_state_dict(existing_weights, strict=False)

print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)

layer_mapping = {}
for missing_key in missing_keys:
    source_key = missing_key.replace("g_", "x_")
    layer_mapping[missing_key] = source_key


for missing_key, source_key in layer_mapping.items():
    if source_key in existing_weights:
        existing_weights[missing_key] = existing_weights[source_key].clone()
    else:
        print(f"Source layer {source_key} not found in existing weights.")

model.load_state_dict(existing_weights, strict=False)

torch.save(model.state_dict(), "g_cross_attention_trained_model.bin")

print("Model updated and saved as : g_cross_attention_trained_model.bin")