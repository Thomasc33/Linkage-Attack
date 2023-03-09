import argparse
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Random Forest Attack Model for Skeleton Anonymization')
# Data should be a dictionary where the key is the name of the actor and the value is a list of skeletons
parser.add_argument('--data', type=str, default='X.pkl', help='dataset name')

parser.add_argument('--model', type=str,
                    default='clf 30.15.pkl', help='model name')

parser.add_argument('--labels', type=str, default='Genders.csv',
                    help='Actor,Label CSV file')

parser.add_argument('--verbose', type=int, default=0, help='SKLearn Verbosity')

parser.add_argument('--source', type=str,
                    default='motion_privacy', help='skele_anon,motion_privacy')

args = parser.parse_args()


def load_model(model = args.model):
    print(f"Loading model from {model}")
    with open(model, 'rb') as f:
        m = pickle.load(f)
    print("Model loaded")
    return m


def load_data():
    print(f"Loading data from {args.data}")
    with open(args.data, 'rb') as f:
        Actors = pickle.load(f)
    print("Data loaded")
    return Actors


def anonymizer_to_rf(t):
    # (75)
    # [x:0,y:1,z:2][300][joints][actors]
    xyz, frames, joints, actors = t.shape
    # transpose to make loop simple
    # [frame][actors][joints][xyz]
    t = t.transpose(1, 3, 2, 0)
    # make empty array
    frames = []

    # crazy loop
    for frame in t:
        f = []
        for actor in frame:
            for joint in actor:
                for xyz in joint:
                    f.append(xyz)
      
        # Cut to 75
        if len(f) != 75:
            f = f[:75]

        frames.append(f)

    return frames


def gen_labels(Actors):
    # Load the Genders
    Genders = pd.read_csv(args.labels)

    # Convert M to 1 and F to 0
    Genders = Genders.replace('M', 1).replace('F', 0)

    # Convert dataframe to oject where P is the key, and Gender is the value
    Genders = Genders.set_index('P').T.to_dict('list')

    # Convert to X and Y
    print("Generating labels")
    X = []
    Y = []
    for actor in tqdm(Actors, desc="Actor"):
        # Account for skeleton anonymization source
        if args.source == 'skele_anon':
            for skeleton in Actors[actor]:
                s = anonymizer_to_rf(skeleton)
                if s is not None:
                    X.extend(s)
                    Y.extend([Genders[actor]]*len(s))
        else:
            X.extend(Actors[actor])
            Y.extend([Genders[actor]]*len(Actors[actor]))

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int8)
    print("Labels generated")
    return X, Y


if __name__ == '__main__':
    model = load_model()
    model.verbose = args.verbose
    Actors = load_data()
    X, Y = gen_labels(Actors)
    print(X.shape, Y.shape)
    try:
        assert len(X) == len(Y)
    except AssertionError:
        print("X and Y are not the same length")
    print("Evaluating model")
    print(f"Accuracy: {model.score(X, Y)}")


    # Use this to test multiple models
    # model2 = load_model('clf 10.5.pkl')
    # model3 = load_model('clf 20.10.pkl')

    # print(f"Accuracy 10.5: {model2.score(X, Y)}")
    # print(f"Accuracy 20.10: {model3.score(X, Y)}")

