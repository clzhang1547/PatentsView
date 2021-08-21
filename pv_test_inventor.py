################################################################
# Test Nick's PatentsView code - inventor
# https://github.com/PatentsView/PatentsView-Disambiguation/tree/dev/docs
#
# chris zhang 6/2/2021
################################################################

import pandas as pd
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 200)
from pprint import pprint

from pv.disambiguation.core import InventorMention
# methods for grabbing data from SQL
from pv.disambiguation.inventor.load_mysql import get_granted, get_pregrants
# connection to database
from pv.disambiguation.util.db import granted_table, pregranted_table
# database configuration
import configparser

config = configparser.ConfigParser()
config.read(['config/database_config.ini', 'config/database_tables.ini', 'config/inventor/run_clustering.ini'])

# create a connection to the table
granted = granted_table(config)

# get test rows
cursor = granted.cursor()
#query = "SELECT * from disambiguation_testing_granted.patent;"
query = "SELECT * from disambiguation_testing_granted.rawinventor limit 10;"
cols = ['uuid', 'patent_id', 'inventor_id', 'rawlocation_id', 'name_first', 'name_last', 'sequence', 'rule_47', 'deceased', 'version_indicator', 'created_date', 'updated_date']

# query for disambiguation_testing_granted
# query = "SELECT * from disambiguation_testing_granted.rawlocation;"
# cols for rawassignee table
# cols = ['uuid', 'patent_id', 'assignee_id', 'rawlocation_id', 'type', 'name_first', 'name_last', 'organization', 'sequence', 'version_indicator', 'created_date', 'updated_date']
# cols for patent table
# cols = ['id', 'type', 'number', 'country', 'date', 'abstract', 'title', 'kind', 'num_claims', 'filename', 'withdrawn', 'version_indicator', 'created_date', 'updated_date'])
# cols for rawinventor table
# cols = ['uuid', 'patent_id', 'inventor_id', 'rawlocation_id', 'name_first', 'name_last', 'sequence', 'rule_47', 'deceased', 'version_indicator', 'created_date', 'updated_date']
# cols for rawlocation table
# cols = ['id', 'location_id', 'city', 'state', 'country', 'country_transformed', 'location_id_transformed', 'version_indicator', 'created_date', 'updated_date', 'location_new_id']

# query for disambiguation_testing_pregranted
# query = "SELECT * from disambiguation_testing_pregranted.application;"
# cols for disambiguation_testing_pregranted.application
# cols = ['id', 'document_number', 'type', 'application_number', 'date', 'country', 'series_code', 'invention_title', 'invention_abstract', 'rule_47_flag', 'filename', 'created_date', 'updated_date', 'version_indicator']

# cols for disambiguation_testing_pregranted.rawassignee
# cols = ['id', 'document_number', 'assignee_id', 'sequence', 'name_first', 'name_last', 'organization', 'type', 'rawlocation_id', 'city', 'state', 'country', 'filename', 'created_date', 'updated_date', 'version_indicator']
# cols for disambiguation_testing_pregranted.rawinventor
# cols = ['id', 'document_number', 'inventor_id', 'name_first', 'name_last', 'sequence', 'designation', 'deceased', 'rawlocation_id', 'city', 'state', 'country', 'filename', 'created_date', 'updated_date', 'version_indicator']
# cols for disambiguation_testing_pregranted.rawlocation
# cols = ['id', 'location_id', 'city', 'state', 'country', 'latitude', 'longitude', 'filename', 'created_date', 'updated_date', 'version_indicator']
cursor.execute(query)
df = pd.DataFrame(cursor, columns=cols)

'''
id_set = []
for i in cursor:
    print(i)
    print(i[0])
    id_set.append(i[0])
'''

'''
rawassignee = ['uuid', 'patent_id', 'assignee_id', 'rawlocation_id', 'type', 'name_first', 'name_last', 'organization', 'sequence', 'version_indicator', 'created_date', 'updated_date']
rawinventor = ['uuid', 'patent_id', 'inventor_id', 'rawlocation_id', 'name_first', 'name_last', 'sequence', 'rule_47', 'deceased', 'version_indicator', 'created_date', 'updated_date']

'''

# build inventor mentions
id_set = df['uuid'].values
# internally this calls mentions = [InventorMention.from_granted_sql_record(r) for r in records]
mentions = get_granted(id_set, granted)
pprint(mentions[0].__dict__)

######################
# Canopies - alternative way (subgroup) to load data?
# Note: config[inventor] needs to access [inventor] section in run_clustering.ini in inventor folder
######################
from pv.disambiguation.inventor.load_mysql import Loader
# this object will load from two files of canopies:
config['inventor']['pregranted_canopies'] = 'data/inventor/archive/canopies.pregranted.pkl'
config['inventor']['granted_canopies'] = 'data/inventor/archive/canopies.granted.pkl'
loader = Loader.from_config(config, 'inventor')

######################
# Featurizing Data
# Note: config[inventor] needs to access [inventor] section in run_clustering.ini in inventor folder
######################
from pv.disambiguation.inventor.model import InventorModel
encoding_model = InventorModel.from_config(config)
features = encoding_model.encode(mentions)

######################
# Clustering model
######################
import torch
model = torch.load(config['inventor']['model']).eval()
# Run the clustering
from grinch.agglom import Agglom
grinch = Agglom(model, features, num_points=len(mentions))
grinch.build_dendrogram_hac()
fc = grinch.flat_clustering(model.aux['threshold'])

