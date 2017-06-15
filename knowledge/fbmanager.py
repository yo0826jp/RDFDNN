#conding: utf-8
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from knowledge.evaluate import averaged_fvalue
import itertools
import random

class FBManager(object):
    """Freebase data manager class"""

    def __init__(self, train_file, test_file):
        self.load_rdf_file(train_file)
        self.load_test_file(test_file)
        self.mem_truth = None

    def load_rdf_file(self, rdf_file):
        rdf_df = pd.read_csv(rdf_file, sep="\t", header=None)
        rdf_df.columns = ["id1", "prop", "id2"]
        self.rdf_df = rdf_df[["id1","id2","prop"]]
        props = pd.unique(rdf_df["prop"])
        self.zip_pid = zip_pid = {p:i for i, p in enumerate(props)}
        self.unzip_pid = unzip_pid = {i:p for i, p in enumerate(props)}
        self.n_prop = props.size
        inst_ids = pd.unique(np.append(rdf_df.id1, rdf_df.id2))
        self.zip_id = zip_id = {v: i for i, v in enumerate(inst_ids)}
        self.unzip_id = unzip_id = {i: v for i, v in enumerate(inst_ids)}
        self.n_inst = len(zip_id)
        self.edges = {(zip_id[i1], zip_id[i2]): zip_pid[p]
                            for i1, i2, p in zip(rdf_df.id1,
                                                 rdf_df.id2,
                                                 rdf_df.prop)}

    def load_test_file(self, test_file):
        test_rdf = pd.read_csv(test_file, sep="\t", header=None)
        test_rdf.columns = ["id1", "prop", "id2"]
        self.test_rdf = test_rdf

    def get_train_data(self):
        train_rdf = self.rdf_df
        s_train = train_rdf["id1"].apply(lambda x: self.zip_id[x])
        o_train = train_rdf["id2"].apply(lambda x: self.zip_id[x])
        p_train = train_rdf["prop"].apply(lambda x: self.zip_pid[x])
        return s_train, o_train, p_train

    def get_train_triple_data(self, negative_ratio=1.0):
        return self.create_triple_sample_with_negative(self.rdf_df, negative_ratio)

    def get_test_triple_data(self, negative_ratio=1.0):
        return self.create_triple_sample_with_negative(self.test_rdf, negative_ratio)

    def get_test_data(self):
        test_rdf = self.test_rdf
        s_test = test_rdf["id1"].apply(lambda x: self.zip_id[x])
        o_test = test_rdf["id2"].apply(lambda x: self.zip_id[x])
        p_test = test_rdf["prop"].apply(lambda x: self.zip_pid[x])
        return s_test, o_test, p_test

    def get_pairwise_train_data(self):
        return self.create_pairwise_object_data(self.rdf_df)

    def create_triple_sample_with_negative(self, rdf_df, negative_ratio=1.0):
        """
        Create positive sample and negative sample (s,o,p, e=0or1)
        """
        s_positive = rdf_df["id1"].apply(lambda x: self.zip_id[x])
        o_positive = rdf_df["id2"].apply(lambda x: self.zip_id[x])
        p_positive = rdf_df["prop"].apply(lambda x: self.zip_pid[x])
        n_pos = s_positive.size
        n_neg = int(n_pos * negative_ratio)
        s_negative = np.random.randint(0, self.n_inst, n_neg)
        o_negative = np.random.randint(0, self.n_inst, n_neg)
        p_negative = np.random.randint(0, self.n_prop, n_neg)

        seq = np.arange(n_pos + n_neg)
        np.random.shuffle(seq)
        s_indices = np.append(s_positive, s_negative)[seq]
        o_indices = np.append(o_positive, o_negative)[seq]
        p_indices = np.append(p_positive, p_negative)[seq]
        exist_labels = np.array([1] * n_pos + [0] * n_neg)[seq]
        return s_indices, o_indices, p_indices, exist_labels

    def create_pairwise_object_data(self, rdf_df):
        """
        Create S-P pairwise data for object prediction
        """
        s_indices = rdf_df["id1"]
        o_indices = rdf_df["id2"]
        p_indices = rdf_df["prop"]
        sp_o = {}
        for s, o, p in zip(s_indices, o_indices, p_indices):
            sp_o.setdefault(o, [])
            sp_o[o].append((s, p))
        s_ind1 = []
        p_ind1 = []
        s_ind2 = []
        p_ind2 = []
        for o, pairs in sp_o.items():
            for pair1, pair2 in itertools.combinations(pairs, 2):
                s_ind1.append(pair1[0])
                p_ind1.append(pair1[1])
                s_ind2.append(pair2[0])
                p_ind2.append(pair2[1])
        seq = list(range(len(s_ind1)))
        random.shuffle(seq)
        s_ind1 = np.array(s_ind1)[seq]
        p_ind1 = np.array(p_ind1)[seq]
        s_ind2 = np.array(s_ind2)[seq]
        p_ind2 = np.array(p_ind2)[seq]
        return s_ind1, p_ind1, s_ind2, p_ind2

    def make_datasets(self, test_ratio=0.2):
        n_rdf = self.rdf_df.shape[0]
        test_size = int(n_rdf * test_ratio)
        indices = np.arange(n_rdf)
        np.random.shuffle(indices)
        test_rdf = rdf_df.ix[indices[:test_size]]
        train_rdf = rdf_df.ix[indices[test_size:]]

    def export_dataset(self, rdf_df):
        indices1 = rdf_df.id1.apply(lambda x: fbm.zip_id[x])
        indices2 = rdf_df.id2.apply(lambda x: fbm.zip_id[x])
        prop_indices = rdf_df.prop.apply(lambda x: fbm.zip_pid[x])
        enc_rdf = pd.DataFrame({"ind1":indices1, "ind2":indices2,
                                "prop_ind":prop_indices})

    @staticmethod
    def topk_accuracy(proba, true_labels, k):
        topk_labels = (-proba).argsort()[:,:k]
        total = proba.shape[0]
        n_correct = 0
        for p, topk in zip(true_labels, topk_labels):
            if p in topk: n_correct += 1
        score = n_correct / total
        print("accuracy@top{}: {}".format(k, score))
        return score

    @staticmethod
    def accuracy_with_threshold(proba, true_labels, t=0.5):
        pred_label = (proba >= t)
        correct = (pred_label == true_labels).sum()
        acc = correct / pred_label.size
        print("accuracy with t={}: {}".format(t, acc))
        return acc

