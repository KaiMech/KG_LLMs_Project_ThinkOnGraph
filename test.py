from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype = "auto",
    device_map = "auto"
)

messages = [
    {"role": "system", "content": "You are a helpful assistant who keeps their answers clear and concise."},
    {"role": "user", "content": "write me a poem on knowledge graphs"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True
)

inputs = tokenizer(text, return_tensors = "pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature = 0
    )
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
