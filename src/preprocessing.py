import pickle
import re
import string

import inflection
import nltk
import glob
import javalang
import pygments
from nltk.stem.porter import PorterStemmer
import os
from assets import java_keywords, stop_words
import csv
# from util import *
from collections import OrderedDict
from pygments.lexers import JavaLexer
from pygments.token import Token
from datasets import DATASET
from datetime import datetime


class BugReport:
    """Class representing each bug report"""
    __slots__ = ['summary', 'description', 'fixed_files', 'report_time', 'pos_tagged_summary', 'pos_tagged_description',
                 'stack_traces', 'stack_traces_remove']

    def __init__(self, summary, description, fixed_files, report_time):
        self.summary = summary
        self.description = description
        self.fixed_files = fixed_files
        self.report_time = report_time
        self.pos_tagged_summary = None
        self.pos_tagged_description = None
        self.stack_traces = None
        self.stack_traces_remove = None


class SourceFile:
    """Class representing each source file"""
    __slots__ = ['all_content', 'comments', 'class_names', 'attributes', 'method_names', 'variables', 'file_name',
                 'pos_tagged_comments', 'exact_file_name', 'package_name']

    def __init__(self, all_content, comments, class_names, attributes, method_names, variables, file_name,
                 package_name):
        self.all_content = all_content
        self.comments = comments
        self.class_names = class_names
        self.attributes = attributes
        self.method_names = method_names
        self.variables = variables
        self.file_name = file_name
        self.exact_file_name = file_name[0]
        self.package_name = package_name
        self.pos_tagged_comments = None


class Parser:
    """Class containing different parsers"""
    __slots__ = ['name', 'src', 'bug_repo']

    def __init__(self, pro):
        self.name = pro.name
        self.src = pro.src
        self.bug_repo = pro.bug_repo

    def report_parser(self):
        reader = csv.DictReader(open(self.bug_repo, "r"), delimiter="\t")
        bug_reports = OrderedDict()
        # raw_texts = []
        # fixed_files = []
        for line in reader:
            # line["raw_text"] = line["summary"] + ' ' + line["description"]
            line["report_time"] = datetime.strptime(line["report_time"], "%Y-%m-%d %H:%M:%S")
            temp = line["files"].strip().split(".java")
            length = len(temp)
            x = []
            for i, f in enumerate(temp):
                if i == (length - 1):
                    x.append(os.path.normpath(f))
                    continue
                x.append(os.path.normpath(f + ".java"))
            line["files"] = x
            bug_reports[line["bug_id"]] = BugReport(line["summary"], line["description"], line["files"],
                                                    line["report_time"])
        # bug_reports = tsv2dict(self.bug_repo)

        return bug_reports

    def src_parser(self):
        """Parse source code directory of a program and colect its java files"""

        # Gettting the list of source files recursively from the source directory
        src_addresses = glob.glob(str(self.src) + '/**/*.java', recursive=True)
        print(src_addresses)
        # Creating a java lexer instance for pygments.lex() method
        java_lexer = JavaLexer()
        src_files = OrderedDict()
        # src_files = dict()
        # Looping to parse each source file
        for src_file in src_addresses:
            with open(src_file, encoding='latin-1') as file:
                src = file.read()

            # Placeholder for different parts of a source file
            comments = ''
            class_names = []
            attributes = []
            method_names = []
            variables = []

            # Source parsing
            parse_tree = None
            try:
                parse_tree = javalang.parse.parse(src)
                for path, node in parse_tree.filter(javalang.tree.VariableDeclarator):
                    if isinstance(path[-2], javalang.tree.FieldDeclaration):
                        attributes.append(node.name)
                    elif isinstance(path[-2], javalang.tree.VariableDeclaration):
                        variables.append(node.name)
            except:
                pass

            # Triming the source file
            ind = False
            if parse_tree:
                if parse_tree.imports:
                    last_imp_path = parse_tree.imports[-1].path
                    src = src[src.index(last_imp_path) + len(last_imp_path) + 1:]
                elif parse_tree.package:
                    package_name = parse_tree.package.name
                    src = src[src.index(package_name) + len(package_name) + 1:]
                else:  # no import and no package declaration
                    ind = True
            # javalang can't parse the source file
            else:
                ind = True

            # Lexically tokenize the source file
            lexed_src = pygments.lex(src, java_lexer)

            for i, token in enumerate(lexed_src):
                if token[0] in Token.Comment:
                    if ind and i == 0 and token[0] is Token.Comment.Multiline:
                        src = src[src.index(token[1]) + len(token[1]):]
                        continue
                    comments = comments + token[1]
                elif token[0] is Token.Name.Class:
                    class_names.append(token[1])
                elif token[0] is Token.Name.Function:
                    method_names.append(token[1])

            # get the package declaration if exists
            if parse_tree and parse_tree.package:
                package_name = parse_tree.package.name
            else:
                package_name = None

            if self.name == 'aspectj' or 'tomcat' or 'eclipse' or 'swt':
                src_files[os.path.relpath(src_file, start=self.src)] = SourceFile(src, comments, class_names,
                                                                                  attributes, method_names, variables, [
                                                                                      os.path.basename(src_file).split(
                                                                                          '.')[0]], package_name)
            else:
                # If source files has package declaration
                if package_name:
                    src_id = (package_name + '.' + os.path.basename(src_file))
                else:
                    src_id = os.path.basename(src_file)
                src_files[src_id] = SourceFile(src, comments, class_names, attributes, method_names, variables,
                                               [os.path.basename(src_file).split('.')[0]], package_name)
            # print(src_files)
            # print("===========")
        return src_files


class ReportPreprocessing:
    """Class preprocess bug reports"""
    __slots__ = ['bug_reports']

    def __init__(self, bug_reports):
        self.bug_reports = bug_reports

    def extract_stack_traces(self):
        """Extract stack traces from bug reports"""
        pattern = re.compile(r' at (.*?)\((.*?)\)')
        signs = ['.java', 'Unknown Source', 'Native Method']
        for report in self.bug_reports.values():
            st_canid = re.findall(pattern, report.description)
            st = [x for x in st_canid if any(s in x[1] for s in signs)]
            report.stack_traces = st

    def extract_stack_traces_remove(self):
        pattern = re.compile(r' at (.*?)\((.*?)\)')
        signs = ['.java', 'Unknown Source', 'Native Method']
        for report in self.bug_reports.values():
            st_canid = re.findall(pattern, report.description)
            st = [x for x in st_canid if any(s in x[1] for s in signs)]
            at = []
            for x in st:
                if x[1] == 'Unknown Source':
                    temp = 'Unknown'
                    y = x[0] + '(' + temp
                else:
                    y = x[0] + '(' + x[1] + ')'
                at.append(y)
            report.stack_traces_remove = at

    def pos_tagging(self):
        """Extracing specific pos tags from bug reports raw_text"""
        for report in self.bug_reports.values():
            # Tokenizing using word_tokeize for more accurate pos-tagging
            sum_tok = nltk.word_tokenize(report.summary)
            desc_tok = nltk.word_tokenize(report.description)
            sum_pos = nltk.pos_tag(sum_tok)
            desc_pos = nltk.pos_tag(desc_tok)
            report.pos_tagged_summary = [token for token, pos in sum_pos if 'NN' in pos or 'VB' in pos]
            report.pos_tagged_description = [token for token, pos in desc_pos if 'NN' in pos or 'VB' in pos]

    def tokenize(self):
        """Tokenize bug report intro tokens"""
        for report in self.bug_reports.values():
            report.summary = nltk.wordpunct_tokenize(report.summary)
            report.description = nltk.wordpunct_tokenize(report.description)

    def _split_camelcase(self, tokens):
        # copy tokens
        returning_tokens = tokens[:]
        for token in tokens:
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
        return returning_tokens

    def split_camelcase(self):
        """Split camelcase indentifiers"""
        for report in self.bug_reports.values():
            report.summary = self._split_camelcase(report.summary)
            report.description = self._split_camelcase(report.description)
            report.pos_tagged_summary = self._split_camelcase(report.pos_tagged_summary)
            report.pos_tagged_description = self._split_camelcase(report.pos_tagged_description)

    def normalize(self):
        """remove punctuation, numbers and lowecase conversion"""
        # build a translate table for punctuation and number removal
        punctnum_table = str.maketrans({c: None for c in string.punctuation + string.digits})

        for report in self.bug_reports.values():
            summary_punctnum_rem = [token.translate(punctnum_table) for token in report.summary]
            desc_punctnum_rem = [token.translate(punctnum_table) for token in report.description]
            pos_sum_punctnum_rem = [token.translate(punctnum_table) for token in report.pos_tagged_summary]
            pos_desc_punctnum_rem = [token.translate(punctnum_table) for token in report.pos_tagged_description]
            report.summary = [token.lower() for token in summary_punctnum_rem if token]
            report.description = [token.lower() for token in desc_punctnum_rem if token]
            report.pos_tagged_summary = [token.lower() for token in pos_sum_punctnum_rem if token]
            report.pos_tagged_description = [token.lower() for token in pos_desc_punctnum_rem if token]

    def remove_stopwords(self):
        """removing stop word from tokens"""
        for report in self.bug_reports.values():
            report.summary = [token for token in report.summary if token not in stop_words]
            report.description = [token for token in report.description if token not in stop_words]
            report.pos_tagged_summary = [token for token in report.pos_tagged_summary if token not in stop_words]
            report.pos_tagged_description = [token for token in report.pos_tagged_description if
                                             token not in stop_words]

    def remove_java_keywords(self):
        """removing java language keywords from tokens"""
        for report in self.bug_reports.values():
            report.summary = [token for token in report.summary if token not in java_keywords]
            report.description = [token for token in report.description if token not in java_keywords]
            report.pos_tagged_summary = [token for token in report.pos_tagged_summary if token not in java_keywords]
            report.pos_tagged_description = [token for token in report.pos_tagged_description if
                                             token not in java_keywords]

    def stem(self):
        # stemming tokens
        stemmer = PorterStemmer()
        for report in self.bug_reports.values():
            report.summary = dict(
                zip(['stemmed', 'unstemmed'], [[stemmer.stem(token) for token in report.summary], report.summary]))
            report.description = dict(
                zip(['stemmed', 'unstemmed'],
                    [[stemmer.stem(token) for token in report.description], report.description]))
            report.pos_tagged_summary = dict(
                zip(['stemmed', 'unstemmed'],
                    [[stemmer.stem(token) for token in report.pos_tagged_summary], report.pos_tagged_summary]))
            report.pos_tagged_description = dict(
                zip(['stemmed', 'unstemmed'],
                    [[stemmer.stem(token) for token in report.pos_tagged_description], report.pos_tagged_description]))

    def preprocess(self):
        self.extract_stack_traces()
        self.extract_stack_traces_remove()
        self.pos_tagging()
        self.tokenize()
        self.split_camelcase()
        self.normalize()
        self.remove_stopwords()
        self.remove_java_keywords()
        self.stem()


class SrcPreprocessing:
    """class to preprocess source code"""
    __slots__ = ['src_files']

    def __init__(self, src_files):
        self.src_files = src_files

    def pos_tagging(self):
        """Extracing specific pos tags from comments"""
        for src in self.src_files.values():
            # tokenize using word_tokenize
            comments_tok = nltk.word_tokenize(src.comments)
            comments_pos = nltk.pos_tag(comments_tok)
            src.pos_tagged_comments = [token for token, pos in comments_pos if 'NN' in pos or 'VB' in pos]

    def tokenize(self):
        """tokenize source code to tokens"""
        for src in self.src_files.values():
            src.all_content = nltk.wordpunct_tokenize(src.all_content)
            src.comments = nltk.wordpunct_tokenize(src.comments)

    def _split_camelcase(self, tokens):
        # copy token
        returning_tokens = tokens[:]
        for token in tokens:
            split_tokens = re.split(fr'[{string.punctuation}]+', token)
            # if token is split into some other tokens
            if len(split_tokens) > 1:
                returning_tokens.remove(token)
                # camelcase defect for new tokens
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
        return returning_tokens

    def split_camelcase(self):
        # Split camelcase indenti
        for src in self.src_files.values():
            src.all_content = self._split_camelcase(src.all_content)
            src.comments = self._split_camelcase(src.comments)
            src.class_names = self._split_camelcase(src.class_names)
            src.attributes = self._split_camelcase(src.attributes)
            src.method_names = self._split_camelcase(src.method_names)
            src.variables = self._split_camelcase(src.variables)
            src.pos_tagged_comments = self._split_camelcase(src.pos_tagged_comments)

    def normalize(self):
        "remove punctuation, number and lowercase conversion"
        # build a translate table for punctuation and number
        punctnum_table = str.maketrans({c: None for c in string.punctuation + string.digits})
        for src in self.src_files.values():
            content_punctnum_rem = [token.translate(punctnum_table) for token in src.all_content]
            comments_punctnum_rem = [token.translate(punctnum_table) for token in src.comments]
            classnames_punctnum_rem = [token.translate(punctnum_table) for token in src.class_names]
            attributes_punctnum_rem = [token.translate(punctnum_table) for token in src.attributes]
            methodnames_punctnum_rem = [token.translate(punctnum_table) for token in src.method_names]
            variables_punctnum_rem = [token.translate(punctnum_table) for token in src.variables]
            filename_punctnum_rem = [token.translate(punctnum_table) for token in src.file_name]
            pos_comments_punctnum_rem = [token.translate(punctnum_table) for token in src.pos_tagged_comments]

            src.all_content = [token.lower() for token in content_punctnum_rem if token]
            src.comments = [token.lower() for token in comments_punctnum_rem if token]
            src.class_names = [token.lower() for token in classnames_punctnum_rem if token]
            src.attributes = [token.lower() for token in attributes_punctnum_rem if token]
            src.method_names = [token.lower() for token in methodnames_punctnum_rem if token]
            src.variables = [token.lower() for token in variables_punctnum_rem if token]
            src.file_name = [token.lower() for token in filename_punctnum_rem if token]
            src.pos_tagged_comments = [token.lower() for token in pos_comments_punctnum_rem if token]

    def remove_stopwords(self):
        for src in self.src_files.values():
            src.all_content = [token for token in src.all_content if token not in stop_words]
            src.comments = [token for token in src.comments if token not in stop_words]
            src.class_names = [token for token in src.class_names if token not in stop_words]
            src.attributes = [token for token in src.attributes if token not in stop_words]
            src.method_names = [token for token in src.method_names if token not in stop_words]
            src.variables = [token for token in src.variables if token not in stop_words]
            src.file_name = [token for token in src.file_name if token not in stop_words]
            src.pos_tagged_comments = [token for token in src.pos_tagged_comments if token not in stop_words]

    def remove_javakeywords(self):
        for src in self.src_files.values():
            src.all_content = [token for token in src.all_content if token not in java_keywords]
            src.comments = [token for token in src.comments if token not in java_keywords]
            src.class_names = [token for token in src.class_names if token not in java_keywords]
            src.attributes = [token for token in src.attributes if token not in java_keywords]
            src.method_names = [token for token in src.method_names if token not in java_keywords]
            src.variables = [token for token in src.variables if token not in java_keywords]
            src.file_name = [token for token in src.file_name if token not in java_keywords]
            src.pos_tagged_comments = [token for token in src.pos_tagged_comments if token not in java_keywords]

    def stem(self):
        # stemming tokens
        stemmer = PorterStemmer()
        for src in self.src_files.values():
            src.all_content = dict(
                zip(['stemmed', 'unstemmed'], [[stemmer.stem(token) for token in src.all_content], src.all_content]))
            src.comments = dict(
                zip(['stemmed', 'unstemmed'], [[stemmer.stem(token) for token in src.comments], src.comments]))
            src.class_names = dict(
                zip(['stemmed', 'unstemmed'], [[stemmer.stem(token) for token in src.class_names], src.class_names]))
            src.attributes = dict(
                zip(['stemmed', 'unstemmed'], [[stemmer.stem(token) for token in src.attributes], src.attributes]))
            src.method_names = dict(
                zip(['stemmed', 'unstemmed'], [[stemmer.stem(token) for token in src.method_names], src.method_names]))
            src.variables = dict(
                zip(['stemmed', 'unstemmed'], [[stemmer.stem(token) for token in src.variables], src.variables]))
            src.file_name = dict(
                zip(['stemmed', 'unstemmed'], [[stemmer.stem(token) for token in src.file_name], src.file_name]))
            src.pos_tagged_comments = dict(zip(['stemmed', 'unstemmed'],
                                               [[stemmer.stem(token) for token in src.pos_tagged_comments],
                                                src.pos_tagged_comments]))

    def preprocess(self):
        self.pos_tagging()
        self.tokenize()
        self.split_camelcase()
        self.normalize()
        self.remove_stopwords()
        self.remove_javakeywords()
        self.stem()


def main():
    parser = Parser(DATASET)
    # src_prep = SrcPreprocessing(parser.src_parser())
    # src_prep.preprocess()
    # src_strings = src_prep.src_files
    # print(src_strings)
    report_prep = ReportPreprocessing(parser.report_parser())
    report_prep.preprocess()
    report_strings = report_prep.bug_reports
    print(report_strings)


if __name__ == '__main__':
    main()
