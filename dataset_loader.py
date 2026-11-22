from datasets import load_dataset


def load_mbpp_dataset():
    return load_dataset("google-research-datasets/mbpp", "full")


def load_magicoder_python_dataset():
    dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    return dataset.filter(lambda lang: lang == "python", input_columns=["lang"])
