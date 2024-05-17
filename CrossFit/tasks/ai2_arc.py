import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset


class AI2_ARC(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "ai2_arc"
        self.task_type = "text to text"
        self.license = "unknown"

    def get_choices_and_answer_string(self, datapoint):
        answer_index = datapoint["answerKey"]
        choices_string = ""
        for i in range(len(datapoint["choices"]["label"])):
            if datapoint["choices"]["label"][i] == answer_index:
                answer_string = datapoint["choices"]["text"][i]
            choices_string += (
                " - "
                # " ("
                # + datapoint["choices"]["label"][i]
                # + ") "
                + datapoint["choices"]["text"][i]
                + " "
            )
        print(choices_string)
        return choices_string, answer_string

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            choices_string, answer_string = self.get_choices_and_answer_string(
                datapoint
            )
            lines.append((datapoint["question"] + choices_string, answer_string))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("ai2_arc", "ARC-Challenge")


def main():
    print("HERE")
    dataset = AI2_ARC()
    for seed in [100, 13, 21]:
        train, dev, test = dataset.generate_k_shot_data(
            k=32, seed=seed, path="../data/"
        )


def main_more_shots():
    dataset = AI2_ARC()

    for shots in [1, 2, 3, 4, 5]:
        for seed in [100, 13, 21, 2, 8, 33, 26, 19, 49, 2016]:
            train, dev, test = dataset.generate_k_shot_data(
                k=shots, seed=seed, path="../data_more_shots/{}_shot".format(str(shots))
            )


if __name__ == "__main__":
    # main()
    main_more_shots()
