###################################
# Assignee Cluster Analysis - functions
#
# chris zhang 8/20/2021
###################################
import pandas as pd
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 200)
import numpy as np
import re
import Levenshtein

# a function get short version of org name
def get_short_org_name(name):
    if name is None:
        return None
    else:
        name = name.lower()
        name = re.sub(r'[^ \w+]', '', name)
        words = name.split(' ')
        words = [x.strip() for x in words if x!='']
        if len(words) == 1:
            return words[0][:min(3, len(words[0]))]
        elif len(words) > 1:
            return '-'.join([x[:min(3, len(x))] for x in words])
        return None

def get_candidates(dct_check, org_name):
    org_name = org_name.lower()
    short = get_short_org_name(org_name)
    candidates = dct_check[short]
    candidates = [x for x in candidates if x[2]!=org_name]
    # sort candidates by org name (to avoid best candidate cycling among 3+ best candidate pairs)
    candidates.sort(key=lambda x: x[2])
    return candidates


# a function to get average edit distance between two strings
def get_edit_distance(str0, str1):
    str0, str1 = str0.lower(), str1.lower()
    if str0 == '' and str1 == '':
        return 0
    elif len(str0)==0 or len(str1)==0:
        return 1
    elif len(str0) > 0 and len(str1) > 0:
        dist = Levenshtein.distance(str0, str1)
        dist = 0.5 * (dist/len(str0) + dist/len(str1))
        return dist

# a function to get org name's best candidate and the minimum distance
def get_best_candidate(dct_check, org_name):
    candidates = get_candidates(dct_check, org_name)
    if len(candidates) > 0:
        # np.argmin returns first argmin if 2+ exist
        best_candidate = candidates[np.argmin([get_edit_distance(org_name, x[2]) for x in candidates])]
        assignee_id_j, cluster_size_j, org_j = best_candidate
        min_dist = get_edit_distance(org_name, best_candidate[2])
        return assignee_id_j, cluster_size_j, org_j, min_dist
    else:
        return np.nan, np.nan, np.nan, np.nan

# a function to get raw distance between strings
def get_raw_dist(str0, str1):
    str0, str1 = str0.lower(), str1.lower()
    if str0 == '' and str1 == '':
        return 0
    elif len(str0)==0 or len(str1)==0:
        return len(str0) + len(str1)
    elif len(str0) > 0 and len(str1) > 0:
        return Levenshtein.distance(str0, str1)

# a function to remove stopwords
def remove_stopwords(org_name, sws):
    org_name = org_name.lower()
    org_name = re.sub(r'[^ \w+]', '', org_name)
    org_words = [x.strip() for x in org_name.split()]
    org_words = [x for x in org_words if x not in sws]
    org_name_no_sws = ' '.join(org_words)
    return org_name_no_sws