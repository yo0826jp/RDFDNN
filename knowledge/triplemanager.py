import numpy as np

class TripleManager(object):
    def __init__(self, s, o, p, entity_num):
        self.entity_num = entity_num
        self.triplemap = {}
        
        for i in range(len(p)):
            if i == 0:
                self.triple = np.array([s[i], o[i], p[i]], ndmin=2)
            else:
                self.triple = np.append(self.triple, [[s[i], o[i], p[i]]], axis=0)
        
        for i in range(len(p)):   
            if p[i] not in self.triplemap:
                self.triplemap[p[i]] = np.array([s[i], o[i]], ndmin=2)
            else:
                temp = np.append(self.triplemap[p[i]], [[s[i], o[i]]], axis = 0)
                self.triplemap[p[i]] = temp
                
    def create_neg_triple(self,triple):
        s = triple[0]
        o = triple[1]
        p = triple[2]
        flag = True
        while flag:
            change = np.random.randint(2)
            neg = np.random.randint(self.entity_num)
            flag = False
            for t in self.triplemap[p]:
                if t[change] == neg:
                    flag = True
                    break
            
        if change == 0:
            return neg,o,p
        else:
            return s,neg,p