# get WebQSP data, we only save the testset here as we do not plan to finetune/ train anything but try zero shot
from datasets import load_dataset
import pandas as pd

import os
print(os.path.exists("./data"))


# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("stanfordnlp/web_questions")

df = ds["test"].to_pandas()
# print(df.head())
df.to_csv("./data/WebQSP_Test.csv")