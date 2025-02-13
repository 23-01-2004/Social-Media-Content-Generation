from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


base_model_path = 'src/base_models/falcon1b/model'
tokenizer = 'src/base_models/falcon1b/tokenizer'
lora_path = "results/llm_results/fine_tuning_results_v1/checkpoint-115"
output_model_path = "test_result/falcon"

# Load base model
model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="cpu")

# Load and merge LoRA
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()  # Merge LoRA with base model

# Save merged model
model.save_pretrained(output_model_path)
tokenizer = AutoTokenizer.from_pretrained('src/base_models/falcon1b/tokenizer')
tokenizer.save_pretrained(output_model_path)