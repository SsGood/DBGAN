import settings

from link_prediction import Link_pred_Runner

dataname = 'cora'       # 'cora' or 'citeseer' or 'pubmed'
model = 'DBGAN'          # 'arga_ae' or 'DBGAN'
task = 'link_prediction'    

settings = settings.get_settings(dataname, model, task)

if task == 'link_prediction':
    runner = Link_pred_Runner(settings)

runner.erun()

