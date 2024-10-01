import glob
import logging
import math
import os
import random
from datetime import datetime
import numpy as np
import torch
from sentence_transformers import LoggingHandler, SentenceTransformer, datasets, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from data_pipe import load_train_kor_nli, load_train_kor_sts

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

# Modify as needed
model_path = "Alibaba-NLP/gte-multilingual-base"
max_seq_length = 8192
nli_batch_size = 64
sts_batch_size = 8
num_epochs = 10
eval_steps = 1000
learning_rate = 2e-5
seed = 500

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

if __name__ == "__main__":
    set_seeds(seed)
    
    model_save_path = os.path.join("output/gte-kor-turbo" + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # Use trust_remote_code=True when loading the SentenceTransformer model
    model = SentenceTransformer(model_path, trust_remote_code=True)
    
    # Use the model to set up the rest of the training pipeline
    base_model = model._first_module()
    pooling_model = models.Pooling(base_model.get_word_embedding_dimension(), pooling_mode="mean")
    model = SentenceTransformer(modules=[base_model, pooling_model])
    
    # load data
    nli_load_path = "dataset/KorNLI"
    sts_load_path = "dataset/KorSTS"
    
    logging.info("Read KorNLI, STS dataset")
    
    nli_train_datasets = glob.glob(os.path.join(nli_load_path, "*train.ko.tsv"))
    dev_sts_path = os.path.join(sts_load_path, "sts-dev.tsv")
    
    nli_train_samples = []
    for nli_train_data in nli_train_datasets:
        nli_train_samples += load_train_kor_nli(nli_train_data)

    nli_train_dataloader = datasets.NoDuplicatesDataLoader(nli_train_samples, batch_size=nli_batch_size)
    nli_train_loss = losses.MultipleNegativesRankingLoss(model)

    sts_dataset_path = "dataset/KorSTS"
    sts_train_file = os.path.join(sts_dataset_path, "sts-train.tsv")

    sts_train_samples = load_train_kor_sts(sts_train_file)
    sts_train_dataloader = DataLoader(sts_train_samples, shuffle=True, batch_size=sts_batch_size)
    sts_train_loss = losses.CosineSimilarityLoss(model=model)

    dev_samples = load_train_kor_sts(dev_sts_path)
    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=sts_batch_size, name="sts-dev")

    print("Length of NLI data loader:", len(nli_train_dataloader))
    print("Length of STS data loader:", len(sts_train_dataloader))
    steps_per_epoch = min(len(nli_train_dataloader), len(sts_train_dataloader))

    epoch_steps = math.ceil(steps_per_epoch * num_epochs * 0.1)  # 10% of train data for warm-up
    logging.info("epoch-steps: {}".format(epoch_steps))

    train_objectives = [(nli_train_dataloader, nli_train_loss), (sts_train_dataloader, sts_train_loss)]
    model.fit(
        train_objectives=train_objectives,
        evaluator=dev_evaluator,
        epochs=num_epochs,
        optimizer_params={"lr": learning_rate},
        evaluation_steps=eval_steps,
        warmup_steps=epoch_steps,
        output_path=model_save_path,
    )

    # Load the trained model with trust_remote_code=True
    model = SentenceTransformer(model_save_path, trust_remote_code=True)
    logging.info("Start benchmark test dataset")

    test_file = os.path.join(sts_dataset_path, "sts-test.tsv")
    test_samples = load_train_kor_sts(test_file)

    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name="sts-test")
    test_evaluator(model, output_path=model_save_path)