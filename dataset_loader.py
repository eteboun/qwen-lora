from datasets import load_dataset


def load_mbpp_dataset(*, revision: str | None = None):
    return load_dataset("google-research-datasets/mbpp", "full", revision=revision)


def load_magicoder_python_dataset(*, revision: str | None = None):
    dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", revision=revision)
    return dataset.filter(lambda lang: lang == "python", input_columns=["lang"])
