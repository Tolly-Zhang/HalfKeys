from datasets import load_dataset

dataset = load_dataset("openwebtext", cache_dir="./build-1/datasets/openwebtext/data", trust_remote_code=True)

print(dataset)  # Check dataset structure
print(dataset['train'][0])  # Inspect the first data sample
