from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_huggingface_gpu(model_name="gpt2"):
    if not torch.cuda.is_available():
        print("❌ GPU not available for Hugging Face")
        return

    print(f"📦 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()  # gpt2 uses fp32 by default

    input_text = "The capital of France is"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10)

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    print("🧠 Model output:", result)

# Run with a fast model for test purposes
test_huggingface_gpu("gpt2")
