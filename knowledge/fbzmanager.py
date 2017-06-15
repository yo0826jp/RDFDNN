#conding: utf-8
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from knowledge.evaluate import averaged_fvalue

class FBManager(object):
    """Freebase data manager class"""

    def __init__(self, instance_def_file, rdf_file):
        self.load_instance_def_file(instance_def_file)
        self.load_rdf_file(rdf_file)
        self.mem_truth = None

    def load_instance_def_file(self, instance_def_file):
        inst_df = pd.read_csv(instance_def_file, sep=",")
        inst_df.columns = ["ctgy1", "ctgy2", "inst_id", "label"]
        self.n_inst = inst_df.shape[0]
        inst_df = inst_df[["inst_id","label","ctgy1","ctgy2"]]
        id_labels = inst_df[["inst_id","label"]].drop_duplicates()
        self.label_dict = {i:l for i, l in id_labels.values}
        #indices = inst_df.index
        #ids = inst_df["inst_id"]
        self.inst_master = inst_df.set_index("inst_id")
        self.categories = {}
        for c1, c2 in inst_df[["ctgy1","ctgy2"]].values:
            self.categories.setdefault(c1, set())
            self.categories[c1].add(c2)
        self.n_belong = inst_df.groupby("inst_id")["ctgy2"].count().to_dict()

    def load_rdf_file(self, rdf_file):
        rdf_df = pd.read_csv(rdf_file, header=None, sep=",")
        rdf_df.columns = ["id1", "prop", "id2"]
        self.rdf_df = rdf_df[["id1","id2","prop"]]
        props = pd.unique(rdf_df["prop"])
        self.zip_pid = zip_pid = {p:i for i, p in enumerate(props)}
        self.unzip_pid = unzip_pid = {i:p for i, p in enumerate(props)}
        self.n_prop = props.size
        inst_ids = pd.unique(np.append(rdf_df.id1, rdf_df.id2))
        self.zip_id = zip_id = {v: i for i, v in enumerate(inst_ids)}
        self.unzip_id = unzip_id = {i: v for i, v in enumerate(inst_ids)}
        self.edges = {(zip_id[i1], zip_id[i2]): zip_pid[p]
                            for i1, i2, p in zip(rdf_df.id1,
                                                 rdf_df.id2,
                                                 rdf_df.prop)}

    def generate_directed_graph(self, one_origin=False):
        """Generate unlabeled directed graph"""
        elist = list(self.edges.keys())
        if one_origin:
            for i in range(len(elist)):
                elist[i] = (elist[0][0]+1, elist[0][1]+1)
        return elist

    def set_community_labels(self, com_labels):
        """Set detected communiy label"""
        if not isinstance(com_labels, dict):
            com_labels = list(com_labels)
            com_labels = self._convert_comlist_into_comdict(com_labels)
        self.inst_com = {}
        for index, com in com_labels.items():
            inst_id = self.unzip_id[index]
            self.inst_com[inst_id] = com

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

    def _convert_comlist_into_comdict(self, com_labels):
        #if len(com_labels) != self.n_inst:
        #    raise ValueError("The length of com_labels must be equal to the number of instance")
        com_dict = {k:v for k, v in enumerate(com_labels)}
        return com_dict

    def set_overlapping_community(self, com_proba, how="n_truth"):
        assert how in ["n_truth"]
        com_cands = (-com_proba).argsort()
        overlap_com = {}
        for i, cands in enumerate(com_cands):
            inst_id = self.unzip_id[i]
            try:
                n_com = self.n_belong[inst_id]
                coms = cands[:n_com]
                for c in coms:
                    overlap_com.setdefault(c, set())
                    overlap_com[c].add(inst_id)
            except KeyError:
                pass
        self.overlap_com = overlap_com
        return overlap_com

    def calculate_fvalue(self, level="2"):
        rdf_df = self.rdf_df
        mem_clst = {}
        if self.mem_truth is None:
            mem_truth = {}
            inst_df = self.inst_master.reset_index()
            for inst_id, ctgy in inst_df[["inst_id","ctgy"+str(level)]].values:
                mem_truth.setdefault(ctgy, set())
                mem_truth[ctgy].add(inst_id)
            self.mem_truth = mem_truth
        #for inst_id, com in self.inst_com.items():
        #    mem_clst.setdefault(com, set())
        #    mem_clst[com].add(inst_id)
        self.mem_clst = self.overlap_com
        score = averaged_fvalue(self.mem_truth, self.mem_clst)
        print("Overlapping F-Value: {}".format(score))
        return score


    def calculate_nmi(self, com_labels=None, level=1):
        if com_labels:
            self.set_community_labels(com_labels)
        if level == 0:
            ctgy = "ctgy1"
        elif level == 1:
            ctgy = "ctgy2"
        else:
            raise ValueError("level must be 0 or 1")
        categories = self.inst_master[[ctgy]].reset_index()
        categories = categories.drop_duplicates("inst_id").set_index("inst_id")
        correct_labels = []
        com_labels = []
        for inst_id, com in self.inst_com.items():
            try:
                cor_label = categories.ix[inst_id][ctgy]
            except:
                continue
            com_labels.append(com)
            correct_labels.append(cor_label)
            assert(len(com_labels)==len(correct_labels))
        nmi = normalized_mutual_info_score(com_labels, correct_labels)
        print("NMI: " + str(nmi))
        return nmi



if __name__=="__main__":
    inst_file = "data/freebase_f15k_lines_name_id_gt_labels.csv"
    rdf_file = "data/freebase_mtr100_mte100-all.csv"

    fb = FBManager(inst_file, rdf_file)
    labels = np.random.randint(0, 10, fb.n_inst)
    fb.set_community_labels(labels)

    dgraph = fb.generate_directed_graph()
