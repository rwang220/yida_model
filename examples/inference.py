"""Load Yida-Model-14B from Hugging Face and run a short chat generation."""

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "rwang220/Yida-Model-14B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [{"role": "user", "content": "请用通俗语言解释高血压的常见注意事项。"}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
inputs = tokenizer([text], return_tensors="pt").to(model.device)
output = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
)
print(tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True))
