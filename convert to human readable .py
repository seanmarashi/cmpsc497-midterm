
from datasets import load_dataset

dataset = load_dataset("conll2003", trust_remote_code=True)

# Label mapping for NER tags
label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

#convert numerical labels to human-readable text
def decode_labels(labels):
    return [label_names[label] for label in labels]

#  Print the first sentence with its decoded NER labels
print(list(zip(dataset["train"][0]["tokens"], decode_labels(dataset["train"][0]["ner_tags"]))))
