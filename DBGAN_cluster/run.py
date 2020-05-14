import settings

from clustering import Clustering_Runner

dataname = 'cora'       # 'cora' or 'citeseer' or 'pubmed'
model = 'DBGAN'          # 'arga_ae' or 'DBGAN'
task = 'clustering'        

settings = settings.get_settings(dataname, model, task)

if task == 'clustering':
    runner = Clustering_Runner(settings)

runner.erun()

