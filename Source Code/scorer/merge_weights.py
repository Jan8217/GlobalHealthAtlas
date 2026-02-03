import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge_weights():
    base_model_path = "/path/to/Qwen/Qwen3-8B"
    fine_tuned_path = "aerovane0/GlobalHealthAtlas_Public_Evaluator"
    output_path = "/path/to/output/Qwen3-8B-Merged"

    print(f"Loading base model from: {base_model_path}")
    print(f"Loading fine-tuned weights from: {fine_tuned_path}")
    print(f"Output path: {output_path}")

    print("\nLoading base model and tokenizer...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    print("Loading fine-tuned weights...")
    fine_tuned_model = AutoModelForCausalLM.from_pretrained(
        fine_tuned_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

    print("Merging weights...")
    base_model.load_state_dict(fine_tuned_model.state_dict(), strict=False)

    print(f"Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    base_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"\nMerge completed successfully!")
    print(f"Merged model saved to: {output_path}")

if __name__ == "__main__":
    merge_weights()
