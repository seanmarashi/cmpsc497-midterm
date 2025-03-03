from datasets import load_dataset

# Load the CoNLL-2003 
dataset = load_dataset("conll2003", trust_remote_code=True)

print(dataset)
