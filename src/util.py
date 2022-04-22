import random
import timeit
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from preprocessing import *
from classified_env_dev import Env


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def imbalance_ratio(y: np.ndarray):
    min_classes = 0
    maj_classes = 0
    for label in y:
        if label == 1:
            min_classes += 1
        else:
            maj_classes += 1
    return min_classes / maj_classes


def git_clone(repo_url, clone_folder):
    """ Clones the git repo from 'repo_ur' into 'clone_folder'

    Arguments:
        repo_url {string} -- Url of git repository
        clone_folder {string} -- path of a local folder to clone the repository
    """
    repo_name = repo_url[repo_url.rfind("/") + 1: -4]
    if os.path.isdir(clone_folder + repo_name):
        print("Already cloned")
        return
    cwd = os.getcwd()
    if not os.path.isdir(clone_folder):
        os.mkdir(clone_folder)
    os.chdir(clone_folder)
    os.system("git clone {}".format(repo_url))
    os.chdir(cwd)


def tsv2dict2(tsv_path):
    reader = csv.DictReader(open(tsv_path, "r"), delimiter="\t")
    dict_list = []
    for line in reader:
        temp = line["file"].strip().split(".java ")
        length = len(temp)
        x = []
        for i, f in enumerate(temp):
            if i == (length - 1):
                x.append(os.path.normpath(f))
                continue
            x.append(os.path.normpath(f + ".java"))
        # print(x)
        line["file"] = x
        line["report_time"] = datetime.strptime(
            line["report_time"], "%Y-%m-%d %H:%M:%S"
        )

        dict_list.append(line)
    return dict_list


def tsv2dict(tsv_path):
    """ Converts a tab separated values (tsv) file into a list of dictionaries

    Arguments:
        tsv_path {string} -- path of the tsv file
    """
    reader = csv.DictReader(open(tsv_path, "r"), delimiter="\t")
    dict_list = []
    for line in reader:
        temp = line["files"].strip().split(".java ")
        length = len(temp)
        x = []
        for i, f in enumerate(temp):
            if i == (length - 1):
                x.append(os.path.normpath(f))
                continue
            x.append(os.path.normpath(f + ".java"))
        # print(x)
        line["files"] = x
        line["raw_text"] = line["summary"] + ' ' + line["description"]
        # line["summary"] = clean_and_split(line["summary"][11:])
        # line["description"] = clean_and_split(line["description"])
        line["report_time"] = datetime.strptime(
            line["report_time"], "%Y-%m-%d %H:%M:%S"
        )
        dict_list.append(line)
    return dict_list


def csv2dict(csv_path):
    """ Converts a comma separated values (csv) file into a dictionary

    Arguments:
        csv_path {string} -- path to csv file
    """
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, delimiter=",")
        csv_dict = list()
        for line in reader:
            csv_dict.append(line)

    return csv_dict


def extract_all_files(br, bug_reports, java_files, similarity_bug, semantic_bug):
    right_files = br["files"]
    br_raw_text = process_text(br["raw_text"])
    br_date = br["report_time"]
    br_summary = br["summary"]
    # Randomly sample 2*k files
    remain_sampled = set(java_files.keys()) - set(right_files)

    all_files = []
    for filename in remain_sampled:
        try:
            src = java_files[filename]
            keys = list(java_files.keys())
            index = keys.index(filename)
            rvsm = similarity_bug[index]
            prev_reports = previous_reports(filename, br_date, bug_reports)

            # Collaborative Filter Score
            cfs = collaborative_filtering_score(br_raw_text, prev_reports)

            # Class Name Similarity
            src_name = os.path.basename(filename)
            src_name = src_name.replace(".java", "")
            cns = class_name_similarity(br_summary, src_name)

            # Bug Fixing Recency
            bfr = bug_fixing_recency(br, prev_reports)

            # Bug Fixing Frequency
            bff = len(prev_reports)
            sss = semantic_bug[index]
            all_files.append((filename, rvsm, cfs, cns, bfr, bff, sss))
        except:
            pass

    return all_files


def extract_all_files1(br, bug_reports, java_files, similarity_bug, semantic_bug):
    right_files = br["files"]
    br_raw_text = process_text(br["raw_text"])
    br_date = br["report_time"]
    br_summary = br["summary"]

    # Randomly sample files
    """
        tomcat: 700
        aspect: 800
        swt: 1000
        eclipse: 1300
    """
    total_source_files = 700
    remain_sampled = random.sample(set(java_files.keys()) - set(right_files), total_source_files)

    all_files = []
    for filename in remain_sampled:
        try:
            src = java_files[filename]
            # new_src = process_text(src)
            keys = list(java_files.keys())
            index = keys.index(filename)
            rvsm = similarity_bug[index]
            # rvsm = cosine_sim(br_raw_text, new_src)
            # Previous Reports
            prev_reports = previous_reports(filename, br_date, bug_reports)
            # Collaborative Filter Score
            cfs = collaborative_filtering_score(br_raw_text, prev_reports)
            # Class Name Similarity
            src_name = os.path.basename(filename)
            src_name = src_name.replace(".java", "")
            cns = class_name_similarity(br_summary, src_name)
            # Bug Fixing Recency
            bfr = bug_fixing_recency(br, prev_reports)
            # Bug Fixing Frequency
            bff = len(prev_reports)
            sss = semantic_bug[index]
            all_files.append((filename, rvsm, cfs, cns, bfr, bff, sss))
        except:
            pass

    return all_files


def top_k_wrong_files(br, bug_reports, java_files, similarity_bug, k=300):
    """ Randomly samples 2*k from all wrong files and returns metrics
        for top k files according to rvsm similarity.

    Arguments:
        right_files {list} -- list of right files
        br_raw_text {string} -- raw text of the bug report
        java_files {dictionary} -- dictionary of source code files

    Keyword Arguments:
        k {integer} -- the number of files to return metrics (default: {50})
    """
    right_files = br["files"]
    br_raw_text = process_text(br["raw_text"])
    br_date = br["report_time"]

    # Randomly sample 2*k files
    randomly_sampled = random.sample(set(java_files.keys()) - set(right_files), 2 * k)

    all_files = []
    for filename in randomly_sampled:
        try:
            src = java_files[filename]
            # new_src = process_text(src)
            keys = list(java_files.keys())
            index = keys.index(filename)
            rvsm = similarity_bug[index]
            # rvsm = cosine_sim(br_raw_text, new_src)
            # Previous Reports
            prev_reports = previous_reports(filename, br_date, bug_reports)
            # Collaborative Filter Score
            cfs = collaborative_filtering_score(br_raw_text, prev_reports)
            # Class Name Similarity
            cns = class_name_similarity(br_raw_text, src)
            # Bug Fixing Recency
            bfr = bug_fixing_recency(br, prev_reports)
            # Bug Fixing Frequency
            bff = len(prev_reports)
            all_files.append((filename, rvsm, cfs, cns, bfr, bff))
        except:
            pass

    top_k_files = sorted(all_files, key=lambda x: x[1], reverse=True)[:k]

    return top_k_files


def top_k_wrong_files1(br, bug_reports, java_files, similarity_bug, semantic_bug, k=300):
    """ Randomly samples 2*k from all wrong files and returns metrics
        for top k files according to rvsm similarity.

    Arguments:
        right_files {list} -- list of right files
        br_raw_text {string} -- raw text of the bug report
        java_files {dictionary} -- dictionary of source code files

    Keyword Arguments:
        k {integer} -- the number of files to return metrics (default: {50})
    """
    right_files = br["files"]
    br_raw_text = process_text(br["raw_text"])
    br_date = br["report_time"]
    # Randomly sample 2*k files
    randomly_sampled = random.sample(set(java_files.keys()) - set(right_files), 2 * k)

    all_files = []
    for filename in randomly_sampled:
        try:
            src = java_files[filename]
            # new_src = process_text(src)
            keys = list(java_files.keys())
            index = keys.index(filename)
            rvsm = similarity_bug[index]
            # rvsm = cosine_sim(br_raw_text, new_src)
            # Previous Reports
            prev_reports = previous_reports(filename, br_date, bug_reports)
            # Collaborative Filter Score
            cfs = collaborative_filtering_score(br_raw_text, prev_reports)
            # Class Name Similarity
            cns = class_name_similarity(br_raw_text, src)
            # Bug Fixing Recency
            bfr = bug_fixing_recency(br, prev_reports)
            # Bug Fixing Frequency
            bff = len(prev_reports)
            dnn_rele = semantic_bug[index]
            all_files.append((filename, rvsm, cfs, cns, bfr, bff, dnn_rele))
        except:
            pass

    top_k_files = sorted(all_files, key=lambda x: x[1], reverse=True)[:k]

    return top_k_files


def cosine_sim(text1, text2):
    """ Cosine similarity with tfidf

    Arguments:
        text1 {string} -- first text
        text2 {string} -- second text
    """
    vectorizer = TfidfVectorizer(min_df=1)
    tfidf = vectorizer.fit_transform([text1, text2])
    sim = (tfidf * tfidf.T).A[0, 1]
    return sim


def get_all_source_code(start_dir):
    """ Creates corpus starting from 'start_dir'

    Arguments:
        start_dir {string} -- directory path to start
    """
    files = OrderedDict()
    start_dir = os.path.normpath(start_dir)
    for dir_, dir_names, file_names in os.walk(start_dir):
        # print(file_names)
        for filename in [f for f in file_names if f.endswith(".java")]:
            src_name = os.path.join(dir_, filename)
            # print(src_name)
            try:
                with open(src_name, "r") as src_file:
                    # print(src_name)
                    src = src_file.read()

                file_key = src_name.split(start_dir)[1]
                file_key = file_key[len(os.sep):]
                files[file_key] = src
            except:
                pass

    return files


def get_months_between(d1, d2):
    """ Calculates the number of months between two date strings

    Arguments:
        d1 {datetime} -- date 1
        d2 {datetime} -- date 2
    """

    diff_in_months = abs((d1.year - d2.year) * 12 + d1.month - d2.month)

    return diff_in_months


def most_recent_report(reports):
    """ Returns the most recently submitted previous report that shares a filename with the given bug report

    Arguments:
        filename {string} -- the name of the shared Java file
        current_date {datetime} -- until date
        bug_reports {list of dictionaries} -- list of all bug reports
    """

    if len(reports) > 0:
        return max(reports, key=lambda x: x.get("report_time"))

    return None


def previous_reports(filename, until, bug_reports):
    """ Returns a list of previously filed bug reports that share a file with the current bug report

    Arguments:
        filename {string} -- the name of the shared Java file
        until {datetime} -- until date
        bug_reports {list of dictionaries} -- list of all bug reports
    """
    return [
        br
        for br in bug_reports
        if (filename in br["files"] and br["report_time"] < until)
    ]


def bug_fixing_recency(br, prev_reports):
    """ Calculates the Bug Fixing Recency as defined by Lam et al.

    Arguments:
        report1 {dictionary} -- current bug report
        report2 {dictionary} -- most recent bug report
    """
    most_rr = most_recent_report(prev_reports)

    if br and most_rr:
        return 1 / float(
            get_months_between(br.get("report_time"), most_rr.get("report_time")) + 1
        )

    return 0


def collaborative_filtering_score(raw_text, prev_reports):
    """[summary]

    Arguments:
        raw_text {string} -- raw text of the bug report
        prev_reports {list} -- list of previous reports
    """

    prev_reports_merged_raw_text = ""
    for report in prev_reports:
        new_report = process_text(report["summary"])
        # prev_reports_merged_raw_text += report["raw_text"]
        prev_reports_merged_raw_text += new_report
    cfs = cosine_sim(raw_text, prev_reports_merged_raw_text)

    return cfs


def class_name_similarity(summary_text, src_name):
    # classes = source_code.split(" class ")[1:]
    # class_names = [c[: c.find(" ")] for c in classes]
    # class_names_text = " ".join(class_names)
    # new_class_names_text = process_text(class_names_text)
    # class_name_sim = cosine_sim(summary_text, new_class_names_text)
    if src_name in summary_text:
        class_name_sim = len(src_name)
    else:
        class_name_sim = 0
    return class_name_sim


def helper_collections(samples, only_rvsm=False):
    """ Generates helper function for calculations

    Arguments:
        samples {list} -- samples from features.csv

    Keyword Arguments:
        only_rvsm {bool} -- If True only 'rvsm' features are added to 'sample_dict'. (default: {False})
    """
    sample_dict = {}
    for s in samples:
        sample_dict[s["report_id"]] = []

    for s in samples:
        temp_dict = {}

        values = [float(s["rVSM_similarity"])]
        # values = [float(s["dnn_relevancy_score"])]
        if not only_rvsm:
            values += [
                float(s["collab_filter"]),
                float(s["classname_similarity"]),
                float(s["bug_recency"]),
                float(s["bug_frequency"]),
                float(s["dnn_relevancy_score"]),
            ]
        temp_dict[os.path.normpath(s["file"])] = values

        sample_dict[s["report_id"]].append(temp_dict)

    bug_reports = tsv2dict(DATASET.bug_repo)
    # bug_reports = tsv2dict("../data/AspectJ.txt")
    br2files_dict = {}

    for bug_report in bug_reports:
        br2files_dict[bug_report["id"]] = bug_report["files"]

    return sample_dict, bug_reports, br2files_dict


def topk_accuarcy(test_bug_reports, i, model):
    test_path = str(DATASET.features) + '/' + str(DATASET.name) + 'test' + str(i) + '.csv'
    test_sample = csv2dict(test_path)
    if model is None:
        sample_dict, bug_reports, br2files_dict = helper_collections(test_sample, True)
    else:
        sample_dict, bug_reports, br2files_dict = helper_collections(test_sample)
    topk_counters = [0] * 20
    negative_total = 0
    mrr = []
    mean_avgp = []
    for bug_report in test_bug_reports:
        dnn_input = []
        corresponding_files = []
        bug_id = bug_report["id"]
        # print('Bug data id:', bug_id)

        try:
            for temp_dict in sample_dict[bug_id]:
                java_file = list(temp_dict.keys())[0]
                features_for_java_file = list(temp_dict.values())[0]
                dnn_input.append(features_for_java_file)
                corresponding_files.append(java_file)
        except:
            negative_total += 1
            continue

        # Calculate relevancy for all files related to the bug report in features.csv
        # Remember that, in features.csv, there are 50 wrong(randomly chosen) files for each right(buggy)
        relevancy_list = []
        if model:  # dnn classifier
            for input in dnn_input:
                input = np.reshape(input, [1, 6])
                input_predict = model.predict(input)[0][1]
                relevancy_list.append(input_predict)
            # for y in relevancy_list:
            #     relevancy_list_new.append(y[0])
        else:  # rvsm
            relevancy_list = np.array(dnn_input).ravel()
            relevancy_list_new = relevancy_list
        x = list(np.argsort(relevancy_list, axis=0))
        x.reverse()
        # print(x)

        temp = []
        # print(br2files_dict[bug_id])
        for y in x:
            # t = y[0]
            temp.append(corresponding_files[y])
        # print(temp)
        # getting the ranks of reported fixed files
        relevant_ranks = sorted(temp.index(fixed) + 1
                                for fixed in br2files_dict[bug_id] if fixed in temp)
        if len(relevant_ranks) == 0:
            continue
        # MRR
        min_rank = relevant_ranks[0]
        mrr.append(1 / min_rank)

        # MAP
        mean_avgp.append(np.mean([len(relevant_ranks[:j + 1]) / rank
                                  for j, rank in enumerate(relevant_ranks)]))

        # Top-1, top-2 ... top-20 accuracy
        for i in range(1, 21):
            max_indices = np.argpartition(relevancy_list, -i)[-i:]
            # print(max_indices)
            for corresponding_file in np.array(corresponding_files)[max_indices]:
                if str(corresponding_file) in br2files_dict[bug_id]:
                    topk_counters[i - 1] += 1
                    break
    acc_dict = {}
    print('negative_total: ', negative_total)
    mrr1 = np.mean(mrr)
    mean_avgp1 = np.mean(mean_avgp)
    # print("MRR", np.mean(mrr))
    # print("MAP", np.mean(mean_avgp))
    # print("MRR ",mrr1)
    # print("MAP ",mean_avgp1)
    for i, counter in enumerate(topk_counters):
        acc = counter / (len(test_bug_reports) - negative_total)
        acc_dict[i + 1] = round(acc, 3)
    total_list = []
    total_list.append(mrr1)
    total_list.append(mean_avgp1)
    total_list.append(acc_dict)
    return total_list


def process_text(text):
    # tokenize raw_text
    tokens_text = nltk.wordpunct_tokenize(text)
    # split_camel
    returning_tokens = tokens_text[:]
    for token in tokens_text:
        split_tokens = re.split(fr'[{string.punctuation}]+', token)
        # if token is split into some other tokens
        if len(split_tokens) > 1:
            returning_tokens.remove(token)
            # camel case detection for new tokens
            for st in split_tokens:
                camel_split = inflection.underscore(st).split('_')
                if len(camel_split) > 1:
                    returning_tokens.append(st)
                    returning_tokens = returning_tokens + camel_split
                else:
                    returning_tokens.append(st)
        else:
            camel_split = inflection.underscore(token).split('_')
            if len(camel_split) > 1:
                # returning_tokens.remove(token)
                returning_tokens = returning_tokens + camel_split
    # normalize
    # build a translate table for punctuation and number removal
    punctual_table = str.maketrans({c: None for c in string.punctuation + string.digits})
    text_punctual = [token.translate(punctual_table) for token in returning_tokens]
    text_lower = [token.lower() for token in text_punctual if token]
    # remove stop_word
    text_rm_stopwords = [token for token in text_lower if token not in stop_words]
    # remove_java_keyword
    text_rm_javakeyword = [token for token in text_rm_stopwords if token not in java_keywords]
    # stem
    stemmer = PorterStemmer()
    processed_text = [stemmer.stem(token) for token in text_rm_javakeyword]
    listToStr = ' '.join([str(elem) for elem in processed_text])
    return listToStr


def process_text_not_stem(text):
    # tokenize raw_text
    tokens_text = nltk.wordpunct_tokenize(text)
    # split_camel
    returning_tokens = tokens_text[:]
    for token in tokens_text:
        split_tokens = re.split(fr'[{string.punctuation}]+', token)
        # if token is split into some other tokens
        if len(split_tokens) > 1:
            returning_tokens.remove(token)
            # camel case detection for new tokens
            for st in split_tokens:
                camel_split = inflection.underscore(st).split('_')
                if len(camel_split) > 1:
                    returning_tokens.append(st)
                    returning_tokens = returning_tokens + camel_split
                else:
                    returning_tokens.append(st)
        else:
            camel_split = inflection.underscore(token).split('_')
            if len(camel_split) > 1:
                # returning_tokens.remove(token)
                returning_tokens = returning_tokens + camel_split
    # normalize
    # build a translate table for punctuation and number removal
    punctual_table = str.maketrans({c: None for c in string.punctuation + string.digits})
    text_punctual = [token.translate(punctual_table) for token in returning_tokens]
    text_lower = [token.lower() for token in text_punctual if token]
    # remove stop_word
    text_rm_stopwords = [token for token in text_lower if token not in stop_words]
    # remove_java_keyword
    text_rm_javakeyword = [token for token in text_rm_stopwords if token not in java_keywords]

    listToStr = ' '.join([str(elem) for elem in text_rm_javakeyword])
    return listToStr


def process_tex_not_stem_not_tokenize(tokens_text):
    # split_camel
    returning_tokens = tokens_text[:]
    for token in tokens_text:
        split_tokens = re.split(fr'[{string.punctuation}]+', token)
        # if token is split into some other tokens
        if len(split_tokens) > 1:
            returning_tokens.remove(token)
            # camel case detection for new tokens
            for st in split_tokens:
                camel_split = inflection.underscore(st).split('_')
                if len(camel_split) > 1:
                    returning_tokens.append(st)
                    returning_tokens = returning_tokens + camel_split
                else:
                    returning_tokens.append(st)
        else:
            camel_split = inflection.underscore(token).split('_')
            if len(camel_split) > 1:
                # returning_tokens.remove(token)
                returning_tokens = returning_tokens + camel_split
    # normalize
    # build a translate table for punctuation and number removal
    punctual_table = str.maketrans({c: None for c in string.punctuation + string.digits})
    text_punctual = [token.translate(punctual_table) for token in returning_tokens]
    text_lower = [token.lower() for token in text_punctual if token]
    # remove stop_word
    text_rm_stopwords = [token for token in text_lower if token not in stop_words]
    # remove_java_keyword
    text_rm_javakeyword = [token for token in text_rm_stopwords if token not in java_keywords]

    listToStr = ' '.join([str(elem) for elem in text_rm_javakeyword])
    return listToStr


def semantic_simility(source_files):
    nlp = spacy.load('en_core_web_lg')
    # parser = Parser(DATASET)
    # src_prep = SrcPreprocessing(parser.src_parser())
    # src_prep.preprocess()
    # src_files = get_all_source_code(DATASET.src)
    # source_files = src_prep.src_files
    src_tokens = []
    for src in source_files.values():
        src_text = src.file_name['unstemmed'] + src.class_names['unstemmed'] + src.attributes['unstemmed'] + \
                   src.comments['unstemmed'] + src.method_names['unstemmed']
        src_token = (' '.join(src_text))
        src_tokens.append(src_token)

    # src_tfidf = tfidf.fit_transform(src_strings)
    bug_reports = tsv2dict2(DATASET.bug_repo)
    bug_tokens = []
    # report_strings = []
    for report in bug_reports:
        sum_re = process_text_not_stem(report["summary"])
        desc_re = report["description"]
        desc_tok = nltk.word_tokenize(desc_re)
        desc_pos = nltk.pos_tag(desc_tok)
        pos_tagged_desc = [token for token, pos in desc_pos if 'NN' in pos or 'VB' in pos]
        desc_pos_re = process_tex_not_stem_not_tokenize(pos_tagged_desc)
        report_text = sum_re + desc_pos_re
        # report_token = nltk.word_tokenize(report_text)
        bug_tokens.append(report_text)
    print("done pre data")

    # word embedding bug reports
    all_bug_words = []
    # for re_bug in bug_tokens:
    #     N = len(re_bug)
    #     re_word = np.zeros(300)
    #     for text in re_bug:
    #         w_i = nlp(text).vector
    #         re_word = re_word + w_i
    #     re_word1 = re_word/ N
    #     all_bug_words.append(re_word1)
    for re_bug in bug_tokens:
        w_i = nlp(re_bug).vector
        all_bug_words.append(w_i)
    print("done bug")

    # word embedding source files
    all_src_words = []
    # for src in src_tokens:
    #     N = len(src)
    #     src_word = np.zeros(300)
    #     for text in src:
    #         w_i = nlp(text).vector
    #         src_word = src_word + w_i
    #     src_word1 = src_word/N
    #     all_src_words.append(src_word1)
    for src in src_tokens:
        w_i = nlp(src).vector
        all_src_words.append(w_i)
    print("done src")
    print(len(all_bug_words))
    print(len(all_src_words))
    # src_docs = [nlp(process_text_not_stem(src)) for src in src_files.values()]

    return all_bug_words, all_src_words


def semantic_simility1(source_files):
    nlp = spacy.load('en_core_web_lg')
    min_max_scaler = MinMaxScaler()
    # parser = Parser(DATASET)
    # src_prep = SrcPreprocessing(parser.src_parser())
    # src_prep.preprocess()
    # src_files = get_all_source_code(DATASET.src)
    # source_files = src_prep.src_files
    src_docs = []
    for src in source_files.values():
        src_text = src.file_name['unstemmed'] + src.class_names['unstemmed'] + src.attributes['unstemmed'] + \
                   src.comments['unstemmed'] + src.method_names['unstemmed']
        src_token = (' '.join(src_text))
        src_doc = nlp(src_token)
        src_docs.append(src_doc)

    bug_reports = tsv2dict2(DATASET.bug_repo)
    bug_tokens = []
    all_semantic = []
    # report_strings = []
    for report in bug_reports:
        sum_re = process_text_not_stem(report["summary"])
        desc_re = report["description"]
        desc_tok = nltk.word_tokenize(desc_re)
        desc_pos = nltk.pos_tag(desc_tok)
        pos_tagged_desc = [token for token, pos in desc_pos if 'NN' in pos or 'VB' in pos]
        desc_pos_re = process_tex_not_stem_not_tokenize(pos_tagged_desc)
        report_text = sum_re + desc_pos_re
        report_doc = nlp(report_text)
        scores = []
        for src_doc in src_docs:
            simi = report_doc.similarity(src_doc)
            scores.append(simi)
        scores = np.array([float(count) for count in scores]).reshape(-1, 1)
        normalized_scores = np.concatenate(min_max_scaler.fit_transform(scores))
        all_semantic.append(normalized_scores.tolist())
    print("done semantic data")
    return all_semantic


# def tf_idf_all():
#     src_files = get_all_source_code(DATASET.src)
#     src_strings = []
#     for src in src_files.values():
#         src = process_text(src)
#         src_strings.append(src)
#     # src_strings = [' '.join(src) for src in src_files.values()]
#     bug_reports = tsv2dict(DATASET.bug_repo)
#     # report_strings = [' '.join(br['raw_text']) for br in bug_reports]
#     reports_strings = []
#     for br in bug_reports:
#         str = process_text(br['raw_text'])
#         reports_strings.append(str)
#     tfidf = TfidfVectorizer(sublinear_tf=True, smooth_idf=False)
#     src_tfidf = tfidf.fit_transform(src_strings)
#     reports_tfitf = tfidf.transform(reports_strings)
#     return src_tfidf, reports_tfitf


# def vsm_similarity_all(source_files, report_strings):
# def vsm_similarity_all(report_strings):
def vsm_similarity_all():
    src_files = get_all_source_code(DATASET.src)
    src_strings = []
    for src in src_files.values():
        src = process_text(src)
        src_strings.append(src)

    # src_strings = []
    # for src in source_files.values():
    #     src_text = src.file_name['stemmed'] + src.class_names['stemmed'] + src.method_names['stemmed'] + \
    #                src.pos_tagged_comments['stemmed'] + src.attributes['stemmed']
    #     src_tokens = ' '.join(src_text)
    #     src_strings.append(src_tokens)
    bug_reports = tsv2dict(DATASET.bug_repo)
    report_strings = []
    for br in bug_reports:
        str = process_text(br['raw_text'])
        report_strings.append(str)

    tfidf = TfidfVectorizer(sublinear_tf=True, smooth_idf=False)
    src_tfidf = tfidf.fit_transform(src_strings)

    reports_tfidf = tfidf.transform(report_strings)
    # src_tfidf, reports_tfidf = tf_idf_all()
    # normalizing the length of sources files
    src_lenghts = np.array([float(len(src_str.split())) for src_str in src_strings]).reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    normalized_src_len = min_max_scaler.fit_transform(src_lenghts)

    # Applying logistic length function
    src_len_score = 1 / (1 + np.exp(-12 * normalized_src_len))
    simis = []
    for report in reports_tfidf:
        s = cosine_similarity(src_tfidf, report)
        # revised VSM score caculation
        rvsm_score = s * src_len_score
        normalized_score = np.concatenate(min_max_scaler.fit_transform(rvsm_score))
        simis.append(normalized_score.tolist())
    # print(simis)
    return simis


# def get_traces_score(src_files, reports_bug):
# def get_traces_score():
#     parser = Parser(DATASET)
#     src_prep = SrcPreprocessing(parser.src_parser())
#     src_prep.preprocess()
#     source_files = src_prep.src_files
#     report_prep = ReportPreprocessing(parser.report_parser())
#     report_prep.preprocess()
#     reports_bug = report_prep.bug_reports
#     all_file_names = set(s.exact_file_name for s in source_files.values())
#     all_scores = []
#
#     for report in reports_bug.values():
#         scores = []
#         stack_traces = report.stack_traces
#
#         #Preprocessing stack_traces
#         final_st = []
#         for trace in stack_traces:
#             if trace[1] == 'Unkown Source':
#                 final_st.append((trace[0].split('.')[-2].split('$')[0], trace[0].strip()))
#             elif trace[1] != 'Native Method':
#                 final_st.append((trace[1].split('.')[0].replace(' ', ''), trace[0].strip()))
#         stack_traces = OrderedDict([(file, package) for file, package in final_st if file in all_file_names])
#         for src in source_files.values():
#             file_name = src.exact_file_name
#             #if source file has a package name
#             if src.package_name:
#                 if file_name in stack_traces and src.package_name in stack_traces[file_name]:
#                     scores.append(1 / (list(stack_traces).index(file_name)+1))
#                 else:
#                     scores.append(0)
#             elif file_name in stack_traces:
#                 scores.append(1 / (list(stack_traces).index(file_name)+ 1))
#             else:
#                 scores.append(0)
#
#         all_scores.append(scores)
#     print(all_scores)
#     return all_scores

class CodeTimer:
    """ Keeps time from the initalization, and print the elapsed time at the end.

        Example:

        with CodeTimer("Message"):
            foo()
    """

    def __init__(self, message=""):
        self.message = message

    def __enter__(self):
        print(self.message)
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = timeit.default_timer() - self.start
        print("Finished in {0:0.5f} secs.".format(self.took))


if __name__ == '__main__':
    # print(semantic_simility1())
    print(vsm_similarity_all())
    # print(get_traces_score())
