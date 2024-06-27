from unsloth import FastLlamaModel
max_seq_length = 2048 
dtype = None 
load_in_4bit = False

model, tokenizer = FastLlamaModel.from_pretrained(
    model_name = "./kdd_cup/models/meta-llama/Meta-Llama-3-8B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")