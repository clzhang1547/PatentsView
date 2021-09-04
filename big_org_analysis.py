###################################
# Big Organization Analysis - try to reduce number of clusters
#
# chris zhang 8/26/2021
###################################
import pandas as pd
pd.set_option("max_colwidth", 100)
pd.set_option("display.max_columns", 999)
pd.set_option("display.width", 200)
import numpy as np

from aux_functions import *

# Read in data
d = pd.read_csv('./test_data/disamb_typo_manual_check.csv')
del d['correct_recluster']

# Get big orgs using original clusters
# - Assumption: big orgs already have largest cluster_size using original cluster ID
# TODO: try different n_big
n_big = 200
big_orgs = d[['cluster0','cluster_size']].drop_duplicates(subset='cluster0').\
    sort_values(by='cluster_size', ascending=False)
big_orgs = big_orgs[:n_big]
d['big_org'] = np.where(d['cluster0'].isin(big_orgs['cluster0']), 1, 0)
d_big = d[d['big_org']==1].drop_duplicates('cluster0').sort_values('cluster_size', ascending=False)

# Peel off the existing stopwords fro big org names
fp = './resources/stopwords_assignee_original/assignee-stopwords-lowercase.txt'
with open(fp) as f:
    sws = f.readlines()
sws = [x.strip() for x in sws]

d_big['org_name_no_sws'] = [remove_stopwords(x, sws) for x in d_big['organization']]

# Export for manual checking (to find proper 'core words')
d_big.to_csv('./test_data/big_orgs_for_manual_%s.csv' % n_big, index=False)

# Read into manually flagged file
# get all core words
d_big = pd.read_csv('./test_data/big_orgs_for_manual_corewords.csv')
d_big_cw = d_big[d_big['core_words'].notna()]
core_words = list(set(d_big_cw['core_words']))

# Get 50k sample ready for big-org analysis

# a function to check org name contains a core word, and return 1st core word found
def check_core_words(org_name, core_words):
    cw_match = None
    org_name = org_name.lower()
    org_name = re.sub(r'[^ \w+]', '', org_name)
    words = [x.strip() for x in org_name.split()]
    for cw in core_words:
        # cw may contain 2+ words e.g. 'toyoda jidoshokki'
        if len(cw.split()) == 1:
            if cw.lower() in words:
                cw_match = cw
                break
        elif len(cw.split()) > 1:  # e.g. 'toyoda jidoshokki'
            if all([w in words for w in cw.split()]):
                cw_match = cw
                break
    return cw_match
d['core_word'] = [check_core_words(x, core_words) for x in d['organization']]

# For obs with the same core_word, find most common cluster ID and assign to all
n_cluster_by_cw = d[d['core_word'].notna()].groupby('core_word')['cluster0'].value_counts()
n_cluster_by_cw.name = 'count'
n_cluster_by_cw = n_cluster_by_cw.reset_index()
n_cluster_by_cw = n_cluster_by_cw.sort_values(['core_word', 'count'], ascending=[True, False])
n_cluster_by_cw = n_cluster_by_cw.drop_duplicates('core_word', keep='first')
# Get dict from core word to most common cluster ID
dct_cw_cluster = dict(zip(n_cluster_by_cw['core_word'], n_cluster_by_cw['cluster0']))

# Update cluster ID for master
# TODO: code up below
# Idea: for the same core word, identify major-minor cluster0, use major-minor mapping to replace minor
# This will handle
# 1. the core word cases, and
# 2. cases without core word but have a minor cluster0 in the major-minor mapping
# e.g. cluster0 = 9b151b06-9c12-4260-bb4e-87440d939bb1, org = Bell Laboratories, Inc.
# No core word 'att' but it should be considered as the ATT entity (revealed by cluster0 in original analysis)

# TODO: now update from cluster0, NT combine with typo-analysis so update from cluster1


# TODO: replace below by major-minor approach
cluster2 = [dct_cw_cluster[x] for x in d[d['core_word'].notna()]['core_word']]
d['cluster2'] = np.nan
d.loc[d['core_word'].notna(), 'cluster2'] = cluster2

# If cluster2 = NA, fill in as cluster0
d['cluster2'] = np.where(d['cluster2'].isna(), d['cluster0'], d['cluster2'])
nc0, nc1 = len(set(d['cluster0'])), len(set(d['cluster2']))
print('Number of clusters before big-org core word analysis = %s' % nc0)
print('Number of clusters after big-org core word analysis = %s' % nc1)

# IDs that are successfully removed
id0_not_2 = set(d['cluster0']) - set(d['cluster2'])

# ids = 16 IDs in cluster/cw mapping df, 4 to keep, 12 to remove
ids = set(n_cluster_by_cw.cluster0)
# idx = IDs in set ids after removing those successfully removed
# so idx = IDs not removed (2) or to be kept (4)
idx = ids - id0_not_2

'''
att 50e2
epson 2f93
ibm d1e0
toyoda jidoshokki f73e

9bb1 ?
427b ?
9b151b06-9c12-4260-bb4e-87440d939bb1
890fa81a-0ce1-4951-8c39-09165cfb427b

'''
z = '9b151b06-9c12-4260-bb4e-87440d939bb1'
z = '890fa81a-0ce1-4951-8c39-09165cfb427b'
z in d['cluster2']

# TODO: replace 50k sample by all assignees, check 4 big-org perf
# TODO: if 4 big-org on universe works fine, expand to top-100/200 orgs
