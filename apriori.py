import numpy as np


class Apriori:
    def __init__(self, baskets, min_supp=0.5, min_conf=0.8):
        """
        An apriori algorithm for association analysis

        :param baskets: list of sets, each set represents a basket of items
        :param min_supp: float between 0 and 1, denoting minimal support level
        :param min_conf: float between 0 and 1, denoting minimal confidence level
        """
        self.baskets = baskets
        self.n_baskets = len(baskets)
        self.min_supp = min_supp
        self.min_conf = min_conf

        # list of sets, each set contains all candidate itemsets of same size
        self.cand_list = list()

        # list of lists, each list contains all frequent itemsets of same size
        self.freq_list = list()

        # dictionary, storing support for each frequent itemset
        self.supp_dict = dict()

        # dictionary, storing consequences for each frequent itemset,
        # consequences grouped by size
        self.conseq_dict = dict()

        # list of tuples, each tuple represents an association rule;
        # that is, (antecedent, consequence, confidence)
        self.rule_list = list()

    def apriori(self):
        """
        Apriori algorithm, to iteratively generate frequent itemsets,
        from which association rules derived

        :return: all frequent itemsets satisfying given minimal support requirement,
        and all association rules satisfying given minimal confidence requirement
        """
        k = 0
        while True:
            self.gen_cands(k)
            k += 1
            if len(self.cand_list) < k:
                break
            self.gen_freqs(k)
            if len(self.freq_list) < k:
                break
        for i, freqs in enumerate(self.freq_list):
            if i > 0:
                for freq in freqs:
                    k = 0
                    self.conseq_dict[freq] = list()
                    while k < i:
                        self.gen_conseqs(freq, k)
                        k += 1
                        if len(self.conseq_dict[freq]) < k:
                            break

    def gen_cands(self, k):
        """
        Generate candidate itemsets of size k+1, either initialized from basket dataset,
        or derived from frequent itemsets of size k

        :param k: int, size of current candidate itemset - 1
        :return: set of frozensets, each frozenset represents a candidate itemset of size k+1
        """
        cands = set()
        if not k:
            for basket in self.baskets:
                for item in basket:
                    cands.add(frozenset([item]))
        else:
            freqs = self.freq_list[k - 1]
            for i, freq_i in enumerate(freqs):
                for j, freq_j in enumerate(freqs):
                    if i < j:
                        cand = freq_i | freq_j
                        if len(cand) == k + 1:
                            cands.add(cand)
        if cands:
            self.cand_list.append(cands)

    def gen_freqs(self, k):
        """
        Generate frequent itemsets of size k, by scanning candidate itemsets of same size
        w.r.t. basket dataset, and check minimal support requirement

        :param k: int, size of current frequent itemset
        :return: list of frozensets, each frozenset represents a frequent itemset of size k
        """
        cands = self.cand_list[k - 1]
        cand_cnt = dict([(k, 0) for k in cands])
        for basket in self.baskets:
            for cand in self.cand_list[k - 1]:
                if cand.issubset(basket):
                    cand_cnt[cand] += 1
        freqs = list()
        for cand, cnt in cand_cnt.items():
            supp = cnt / self.n_baskets
            if supp >= self.min_supp:
                freqs.append(cand)
                self.supp_dict[cand] = supp
        if freqs:
            self.freq_list.append(freqs)

    def gen_conseqs(self, freq, k):
        """
        Generate consequences of size k+1 w.r.t. given frequency itemset,
        either initialized from frequency itemset, or derived from consequences of size k;
        a consequence is added if associated rule satisfies minimal confidence requirement

        :param freq: frozenset, given frequency itemset
        :param k: int, size of current consequence - 1
        :return: list of frozensets, each frozenset represents a consequence of size k+1
        """
        conseqs = list()
        if not k:
            for item in freq:
                conseq = frozenset([item])
                if self.check_rule(freq, conseq):
                    conseqs.append(conseq)
        else:
            prev_conseqs = self.conseq_dict[freq][k - 1]
            for i, conseq_i in enumerate(prev_conseqs):
                for j, conseq_j in enumerate(prev_conseqs):
                    if i < j:
                        conseq = conseq_i | conseq_j
                        if len(conseq) == k + 1 and self.check_rule(freq, conseq):
                            conseqs.append(conseq)
        if conseqs:
            self.conseq_dict[freq].append(conseqs)

    def check_rule(self, freq, conseq):
        """
        Check minimal confidence requirement, and add to global association rules if satisfied

        :param freq: frozenset, representing frequency itemset the rule associates with
        :param conseq: frozenset, representing consequence of the rule
        :return: whether the rule satisfies minimal confidence requirement
        """
        ante = freq - conseq
        conf = self.supp_dict[freq] / self.supp_dict[ante]
        if conf >= self.min_conf:
            self.rule_list.append((ante, conseq, conf))
            return True
        return False


if __name__ == "__main__":
    np.random.seed(87)
    baskets = [set(np.random.choice(range(20), 20)) for _ in range(20)]

    apriori = Apriori(baskets)
    apriori.apriori()

    formulate = lambda s: "|".join([str(_) for _ in sorted(list(s))])

    print("%d frequent itemsets in total:" % len(apriori.supp_dict))
    for freq, supp in apriori.supp_dict.items():
        print("\tfrequent itemset %s with support %.2f" % (formulate(freq), supp))

    print("%d association rules in total:" % len(apriori.rule_list))
    for rule in apriori.rule_list:
        ante, conseq, conf = rule
        print("\tassociation rule %s -> %s with confidence %.2f" % (formulate(ante), formulate(conseq), conf))