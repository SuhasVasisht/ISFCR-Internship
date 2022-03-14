import pandas as pd
import argparse
from tqdm import tqdm
from client import Client


def load_inputs(data_path):
    df = pd.read_csv(data_path)
    df = df[['v2']]
    return df


def gen_output(df, ip, port, path_to_synth):
    client = Client(ip, port)
    inputs = df['v2'].values.tolist()

    synth_data = []

    print("Generating Synthetic outputs for ", len(inputs), " samples")

    for i in tqdm(range(len(inputs)-1000, len(inputs))):
        out = client(inputs[i])['result']
        synth_data.append(out)

    synth_df = pd.DataFrame(
        {'Result': synth_data,
         'Input': inputs[-1000:]
         })

    synth_df.to_csv(path_to_synth, index=False)

    print("Finished generating a synthetic dataset with ", len(inputs), " samples")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""python gen_synth [--path | -p] (str)""")
    parser.add_argument('--path', '-p', type=str, required=True,
                        help='The path to save the generated synthetic dataset.')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='The path to the input dataset')

    args = parser.parse_args()
    synth_data_path = args.path
    input_data_path = args.input

    df = load_inputs(input_data_path)
    gen_output(df, '127.0.0.1', 5000, synth_data_path)
