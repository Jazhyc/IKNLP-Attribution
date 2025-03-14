from transformers import AutoModelForCausalLM, AutoTokenizer

def load_pretrained(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation='sdpa',
        torch_dtype='auto',
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer