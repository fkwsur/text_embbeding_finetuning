import csv
import random
from sentence_transformers.readers import InputExample

def load_train_kor_sts(filename):
    samples = []
    with open(filename, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row["score"]) / 5.0
            samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=score))
    return samples


def load_train_kor_nli(filename):
    data = {}
    
    def add_sampling(samplingA, samplingB, label):
        if  samplingA not in data:
            data[samplingA] = {"contradiction": set(), "entailment": set(), "neutral": set()}
        data[samplingA][label].add(samplingB)
        
    with open(filename, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                samplingA = row["sentence1"].strip()
                samplingB = row["sentence2"].strip()
                add_sampling(samplingA, samplingB, row["gold_label"])
                add_sampling(samplingB, samplingA, row["gold_label"])
        
    samples = []
        
    for sampling, etc in data.items():
            if len(etc["entailment"]) > 0 and len(etc["contradiction"]) > 0:
                samples.append(
                    InputExample(
                        texts=[
                            sampling,
                            random.choice(list(etc["entailment"])),
                            random.choice(list(etc["contradiction"])),
                        ]
                    )
                )
                samples.append(
                    InputExample(
                        texts=[
                            random.choice(list(etc["entailment"])),
                            sampling,
                            random.choice(list(etc["contradiction"])),
                        ]
                    )
                )
    return samples                