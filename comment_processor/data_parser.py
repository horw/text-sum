import pandas


def read_file():
    df = pandas.read_json('/home/horw/playground/text-sum/dataset.jsonl',lines=True)
    return df

