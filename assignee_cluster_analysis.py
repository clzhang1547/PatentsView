###################################
# Assignee Cluster Analysis - try to reduce number of clusters
#
# chris zhang 8/19/2021
###################################

import pandas as pd
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 200)
import numpy as np
import pickle
import configparser

from pv.disambiguation.util.db import granted_table, pregranted_table
from aux_functions import *

# Read in assignee clustering output TSV (patent-cluster map)
fp = './exp_out/assignee/run_26/disambiguation_debug.tsv'
assignee_cluster_map = pd.read_csv(fp, sep='\t')
assignee_cluster_map.columns=['patent_id', 'assignee_id']

# Read in patent-uuid map
fp = './data/assignee/uuid.pkl'
granted_uuids, pgranted_uuids = pickle.load(open(fp, 'rb'))

# Generate uuid in assignee_cluster_map
assignee_cluster_map['uuid'] = [granted_uuids[x] if x in granted_uuids else pgranted_uuids[x]
                                for x in assignee_cluster_map['patent_id']]

# Get df of raw assignees
config = configparser.ConfigParser()
config.read(['config/database_config.ini', 'config/database_tables.ini', 'config/inventor/run_clustering.ini'])
granted = granted_table(config)
cursor = granted.cursor()
query = "SELECT * from disambiguation_testing_granted.rawassignee;"
cols = ['uuid', 'patent_id', 'assignee_id', 'rawlocation_id', 'type',
        'name_first', 'name_last', 'organization', 'sequence', 'version_indicator',
        'created_date', 'updated_date']
cursor.execute(query)
rawassignee = pd.DataFrame(cursor, columns=cols)
rawassignee = rawassignee.rename(columns={'assignee_id':'cluster0'})

# Merge assginee_cluster_map and rawassignee
# - for now, granted only
disamb = pd.merge(assignee_cluster_map[['uuid', 'assignee_id']], rawassignee[['uuid', 'cluster0', 'organization']],
                  on='uuid', how='right')
# Get cluster size
sizes = disamb['cluster0'].value_counts()
sizes = sizes.reset_index()
sizes.columns = ['cluster0', 'cluster_size']
disamb = pd.merge(disamb, sizes, on='cluster0', how='left')
disamb = disamb.sort_values(by=['cluster_size', 'organization'])

# Drop if organization is NA
disamb = disamb[disamb['organization'].notna()]
# Baseline (granted, 50k, main)
print('Baseline (granted, 50k, main): n_uuid = %s, n_cluster = %s'
      % (len(set(disamb['uuid'])), len(set(disamb['assignee_id']))))
print('Baseline (granted, full, main): n_uuid = %s, n_cluster = %s'
      % (len(set(disamb['uuid'])), len(set(disamb['cluster0']))))
#  Make a copy of baseline table (to add in updated cluster id)
disamb_orig = disamb.copy()
################################
# Fix typo
# - Aim to find distance of org name to similar names
# - Assumption 1: All current clusters are valid
# - Assumption 2: Typo uniquely occurs, so focus on cluster_size=1
# - Assumption 3: Define and get a short version of org name, get group size of short,
# - focus on group size > 1 (1+ candidates)
# - A3 rules out org names with cluster_size=1 that are typo-free but are real unique org names
# - so we focus on group size > 1 org names, which may come from good original org names and with-typo ones
################################

# Set-up disamb
disamb['organization'] = [x if x is None else x.lower() for x in disamb['organization']]

# Get short org name and group size by short
# short_group_size = number of patents with same short
disamb['short'] = [get_short_org_name(x) for x in disamb['organization']]
short_group_size = disamb[['short']].groupby(['short']).size().reset_index()
short_group_size.columns = ['short', 'short_group_size']
disamb = pd.merge(disamb, short_group_size, how='left', on='short')

# Keep unique short-cluster-org combinations (remove different patents under the same short-cluster-org)
disamb = disamb.sort_values(by=['short', 'cluster0', 'organization'])
disamb = disamb.drop_duplicates(subset=['short', 'cluster0', 'organization'], keep='first')

# Get short groups size for unique short-cluster-org cells
# short_group_size2 = number of cluster-org names with same short
short_group_size2 = disamb[['short']].groupby(['short']).size().reset_index()
short_group_size2.columns = ['short', 'short_group_size2']
disamb = pd.merge(disamb, short_group_size2, how='left', on='short')
disamb = disamb.sort_values(by='short_group_size2', ascending=False)
print('The max size of short-group for checking edit distance = %s'
      % (disamb['short_group_size2'].max()))

# Create short-groups to be checked
# These are dict from short to cluster-org
dct_check = {}
cols = ['cluster0', 'cluster_size', 'organization']
for s in set(disamb['short']):
    dct_check[s] = disamb[disamb['short']==s][cols].values.tolist()

# For each org i with cluster_size = 1, get candidate buddies using dct_check, get distance to each candidate j
# By construction, assignee_id is different among candidates,
# so if distance < thre, set assginee_id of i = assignee_id of j (cluster_size of j always >= that of i = 1)
# if multiple such j exist, use assignee_id of the j with smallest distance


# Set thre (for qualifying for cluster merging) to 0.15 to 0.2 based on example below
str0 = 'Abbott Laboraties'
str1 = 'ABBOTT LABORATORIES'
print(get_edit_distance(str0, str1))

# Get best candidate profile (cols) for disamb rows with cluster_size=1
cols = ['cluster_j', 'cluster_size_j', 'org_j', 'min_dist']
disamb[cols] = np.nan
disamb.loc[disamb['cluster_size'] == 1, cols] = [get_best_candidate(dct_check, x) for x in
                                               disamb.loc[disamb['cluster_size']==1, 'organization']]

# Get raw distance to best candidate org_j
disamb['organization'] = disamb['organization'].fillna('')
disamb['org_j'] = disamb['org_j'].fillna('')
disamb['raw_dist_j'] = [get_raw_dist(i, j) for i, j in disamb[['organization', 'org_j']].values]

# Flag rows that will be assigned to a different cluster
# If dist below thre and cluster size j > 1 - assign to j's cluster id
# TODO: test thre here. Consider stripping stopwords to org name then apply thre
thre_dist, thre_raw_dist = 0.15, 2
cond_thre = (disamb['min_dist'] <= thre_dist) & (disamb['raw_dist_j'] <= thre_raw_dist)
disamb['cluster1'] = np.nan
disamb.loc[cond_thre & (disamb['cluster_size_j'] > 1), 'cluster1'] = disamb['cluster_j']
# If dist below thre and cluster size j = 1 - find ordered set of org names for pairs
# - RULE: for these pairs, use cluster id of the first one in sorted pair
# - if org name ('foo') is the 1st element in org name pair sequence string ('foo-abc'), do not change cluster id
# - if org name ('abc') is the 2nd element in org name pair sequence string ('foo-abc'), change cluster id to foo's
join_marker = '###'
disamb['org_pair_set'] = np.nan
cond = (disamb['cluster_size'] == 1) & (disamb['cluster_size_j'] == 1)
disamb.loc[cond, 'org_pair_set'] = \
    [join_marker.join(sorted(list(x))) for x in disamb.loc[cond, ['organization', 'org_j']].values]
set_size = disamb[['org_pair_set']].groupby('org_pair_set').size().reset_index()
set_size.columns = ['org_pair_set', 'set_size']
disamb['org_pair_elt2'] = np.nan
disamb.loc[disamb['org_pair_set'].notna(), 'org_pair_elt2'] = \
    [x.split(join_marker)[1] for x in disamb.loc[disamb['org_pair_set'].notna(), 'org_pair_set']]

disamb = pd.merge(disamb, set_size, on='org_pair_set', how='left')
disamb.loc[cond_thre & (disamb['set_size']==2) & (disamb['organization']==disamb['org_pair_elt2']), 'cluster1'] = \
    disamb['cluster_j']

# Create mapping from old to new assignee (cluster id)
dct_typo_cluster_remap = dict(zip(disamb[disamb['cluster1'].notna()]['cluster0'],
                                  disamb[disamb['cluster1'].notna()][['cluster1', 'org_j']].values.tolist()))

# Update assignee_id in master clustering table
disamb_orig['cluster1'] = [dct_typo_cluster_remap[x][0] if x in dct_typo_cluster_remap else x
                                      for x in disamb_orig['cluster0']]
disamb_orig['org_j'] = [dct_typo_cluster_remap[x][1] if x in dct_typo_cluster_remap else np.nan
                                      for x in disamb_orig['cluster0']]
# Print summary
p = len(disamb_orig)
nc0, nc1 = len(set(disamb_orig['cluster0'])), len(set(disamb_orig['cluster1']))
p01 = (disamb_orig[disamb_orig['cluster0'].notna()]['cluster0']==
       disamb_orig[disamb_orig['cluster0'].notna()]['cluster1']).value_counts(())[False]
print('-'*50)
print('---- threshold setting -------')
print('---- Max Normalized Distance = %s ---- '
      '\n---- Max Raw Distance = %s' % (thre_dist, thre_raw_dist))
print('-'*50)
print('Before fixing typo: %s patents are classified into %s clusters' % (p, nc0))
print('After fixing typo: %s patents are classified into %s clusters' % (p, nc1))
print('After fixing typo: n_clusters reduced = %s \nn_patents associated = %s' % (nc0 - nc1, p01))

# Save
disamb_orig.to_csv('./test_data/disamb_typo.csv', index=False)

#####################
# Manual check of reclustering by comparing org name and org_j
# post-check file: disamb_typo_manual_check.csv
#####################

disamb_manual = pd.read_csv('./test_data/disamb_typo_manual_check.csv')
rc = int(disamb_manual['correct_recluster'].sum())
print('Number of correctly reduced clusters = %s' % rc)