from DQNSolver import DQNSolver
from util import csv2dict, tsv2dict, topk_accuarcy, imbalance_ratio
from math import ceil
from keras.layers import Dense
import numpy as np
from datasets import DATASET
from classified_env_dev import Env


def features_and_labels(samples):
    """ Returns features and labels for the given list of samples
    
    Arguments:
        samples {list} -- samples from features.csv
    """
    features = np.zeros((len(samples), 6))
    labels = np.zeros((len(samples), 1))

    for i, sample in enumerate(samples):
        features[i][0] = float(sample["rVSM_similarity"])
        features[i][1] = float(sample["collab_filter"])
        features[i][2] = float(sample["classname_similarity"])
        features[i][3] = float(sample["bug_recency"])
        features[i][4] = float(sample["bug_frequency"])
        features[i][5] = float(sample["dnn_relevancy_score"])
        labels[i] = float(sample["match"])
    return features, labels


def kfold_split_indexes(k, len_samples):
    """ Returns list of tuples for split start(inclusive) and 
        finish(exclusive) indexes.
    
    Arguments:
        k {integer} -- the number of folds
        len_samples {interger} -- the length of the sample list
    """
    step = ceil(len_samples / k)
    ret_list = [(start, start + step, step) for start in range(0, len_samples, step)]

    return ret_list


def kfold_split(bug_reports, i, k):
    """ Returns train samples and bug reports for test
    
    Arguments:
        bug_reports {list of dictionaries} -- list of all bug reports
        samples {list} -- samples from features.csv
        start {integer} -- start index for test fold
        finish {integer} -- start index for test fold
    """

    total_list = list(range(0, i)) + list(range(i + 1, k))
    test_path = str(DATASET.features) + '/' + str(DATASET.name) + 'test' + str(i) + '.csv'
    test_samples = csv2dict(test_path)
    total_1 = 0
    total_0 = 0
    train_data_1 = []
    train_data_0 = []
    fold_train = 9
    index_fold = 0

    for j, train_id in enumerate(total_list):
        if index_fold == fold_train:
            break
        else:
            train_path = str(DATASET.features) + '/' + str(DATASET.name) + 'train' + str(train_id) + '.csv'
            train_sample = csv2dict(train_path)
            for sample in train_sample:
                if int(sample["match"]) == 1:
                    total_1 += 1
                    train_data_1.append(sample)
                else:
                    total_0 += 1
                    train_data_0.append(sample)
            index_fold = index_fold + 1
    # print(len(train_data))
    N = total_1
    Sn = total_0 // N
    batch = Sn + 1
    train_data = []
    for i, value in enumerate(train_data_1):
        train_data.append(value)
        for j in range(i * Sn, i * Sn + Sn):
            train_data.append(train_data_0[j])

    test_br_ids = set([s["report_id"] for s in test_samples])
    test_bug_reports = [br for br in bug_reports if br["id"] in test_br_ids]

    return train_data, test_bug_reports, batch


# def calculate_Q_star(i, k, bug_reports):
#     GAMMA = 0.95
#     Q_star = [0] * training_samples
#
#     train_samples, test_bug_reports, batch = kfold_split(bug_reports, i, k)
#     X_train, y_train = features_and_labels(train_samples)
#     imb = imbalance_ratio(y_train)
#     print("imb =", imb)
#     env = Env(imb, X_train, y_train)
#
#     for i in range(training_samples):
#         if i == 0:
#             Q = env.calculate_Q_value(action_space, y_train[training_samples - i - 1])
#             max_Q = np.amax(Q)
#         else:
#             Q = env.calculate_Q_value(action_space, y_train[training_samples - i - 1])
#             for index in range(len(Q)):
#                 Q[index] = Q[index] + (GAMMA * Q_star[training_samples - i])
#             max_Q = np.amax(Q)
#
#         Q_star[training_samples - i - 1] = max_Q
#     return Q_star


def train_dnn(i, k, bug_reports):

    print("Training fold " + str(i + 1) + "...")
    train_samples, test_bug_reports, batch = kfold_split(bug_reports, i, k)
    X_train, y_train = features_and_labels(train_samples)
    imb = imbalance_ratio(y_train)
    print("Imb = " + str(imb))
    env = Env(imb, X_train, y_train)
    observation_space = 6
    sample = 0
    # training_samples = len(y_train) - 1
    training_samples = 100
    action_space = 2
    dqn_solver = DQNSolver(observation_space, action_space)
    # dqn_solver.model.add(Dense())

    # test = calculate_Q_star(i, k, bug_reports)

    observation = env.start()
    observation = np.reshape(observation, [1, observation_space])
    while True:
        if sample == training_samples:
            break

        # Using our observation, choose an action and take it in the environment
        action = dqn_solver.act(observation)
        next_observation, reward, done = env.step(action)

        # Add to memory
        next_observation = np.reshape(next_observation, [1, observation_space])
        dqn_solver.remember(observation, action, reward, done)
        # print("Done:", done)
        observation = next_observation

        sample += 1
        # print("Sample: " + str(sample))

    print("Load memory, fit model...")
    dqn_solver.experience_replay(batch)

    print("Calculate accuracy...")
    total_list = topk_accuarcy(test_bug_reports, i, dqn_solver.model)
    # total_list = topk_accuarcy(test_bug_reports,i, X_train, y_train)
    print("Fold ", i + 1)
    print("==========")
    print(total_list[0])
    print(total_list[1])
    print(total_list[2])

    return total_list, imb


def dnn_model_kfold(k):
    """ Run kfold cross validation in parallel

    Keyword Arguments:
        k {integer} -- the number of folds (default: {10})
    """
    # samples = csv2dict(DATASET.features)

    k_ids = list(range(0, k))
    # These collections are speed up the process while calculating top-k accuracy
    # sample_dict, bug_reports, br2files_dict = helper_collections(samples)

    # np.random.shuffle(samples)
    # total_samples = len(samples)
    # K-fold Cross Validation in parallel
    bug_reports = tsv2dict(DATASET.bug_repo)
    # total_lists = Parallel(n_jobs=-2)(  # Uses all cores but one
    #     delayed(train_dnn)(
    #         i, k, bug_reports
    #     )
    #     for i in k_ids
    # )
    total_lists = []
    imbs = []
    for i in k_ids:
        total_list, imb = train_dnn(i, k, bug_reports)
        total_lists.append(total_list)
        imbs.append(imb)
    print("==============")
    print(total_lists)
    print(imbs)
    # Calculating the average accuracy from all folds
    mrrs = []
    mean_avgps = []
    acc_dicts = []
    for i in range(0, len(total_lists)):
        mrrs.append(total_lists[i][0])
        mean_avgps.append(total_lists[i][1])
        acc_dicts.append(total_lists[i][2])
    avg_mrr = np.mean(mrrs)
    avg_mean_avgps = np.mean(mean_avgps)
    avg_acc_dict = {}
    for key in acc_dicts[0].keys():
        avg_acc_dict[key] = round(sum([d[key] for d in acc_dicts]) / len(acc_dicts), 3)
    print("====================")
    print('Top K accuracy total: ', avg_acc_dict)
    print('MRR_total: ', avg_mrr)
    print('MAP_total: ', avg_mean_avgps)
    return avg_acc_dict, avg_mrr, avg_mean_avgps
