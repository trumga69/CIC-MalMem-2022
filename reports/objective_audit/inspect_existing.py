import json, hashlib, platform, importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
from preprocessing import master_split, checksum
root=Path(__file__).resolve().parents[2]
X,b,m,t=master_split()
d=pd.read_csv(root/'data/raw/Obfuscated-MalMem2022.csv')
e=X.groupby(list(X.columns),dropna=False,sort=False).ngroup()
lc=pd.DataFrame({'g':e,'b':b,'m':m}).groupby('g')[['b','m']].nunique()
token=d.Category.str.extract(r'^[^-]+-[^-]+-([0-9a-fA-F]+)-[0-9]+\.raw$')[0].str.lower()
files={}
fullhash=pd.util.hash_pandas_object(d,index=False)
for p in sorted((root/'data/raw').glob('*.csv')):
 z=pd.read_csv(p)
 files[p.name]={'sha256':checksum(p),'bytes':p.stat().st_size,'shape':list(z.shape),'same_schema':list(z.columns)==list(d.columns)}
 if list(z.columns)==list(d.columns): files[p.name]['rows_matching_full_dataset']=int(pd.util.hash_pandas_object(z,index=False).isin(fullhash).sum())
metric=json.loads((root/'reports/evaluation/binary/random_forest/validation_metrics.json').read_text())
pred=pd.read_csv(root/'reports/evaluation/binary/random_forest/validation_predictions.csv')
r={'python':platform.python_version(),'raw_files':files,'pdf_sha256':checksum(root/'Final term.pdf'),'shape':list(d.shape),'feature_count':len(X.columns),'feature_columns':list(X.columns),'dtypes':d.dtypes.astype(str).to_dict(),'binary_counts':b.value_counts().to_dict(),'multiclass_counts':m.value_counts().to_dict(),'missing_cells':int(d.isna().sum().sum()),'infinite_cells':int(np.isinf(X).sum().sum()),'constant_columns':X.columns[X.nunique()==1].tolist(),'near_constant_99pct':{c:float(X[c].value_counts(normalize=True).iloc[0]) for c in X if X[c].value_counts(normalize=True).iloc[0]>=.99},'identical_column_pairs':[[a,c] for i,a in enumerate(X) for c in X.columns[i+1:] if X[a].equals(X[c])],'negative_cells':int(X.lt(0).sum().sum()),'duplicate_full_excess':int(d.duplicated().sum()),'duplicate_features_excess':int(X.duplicated().sum()),'rows_in_duplicate_feature_groups':int(X.duplicated(keep=False).sum()),'binary_conflict_groups':int(lc.b.gt(1).sum()),'multiclass_conflict_groups':int(lc.m.gt(1).sum()),'connected_groups':int(t.group_id.nunique()),'groups_crossing_splits':int(t.groupby('group_id').split.nunique().gt(1).sum()),'exact_features_crossing_splits':int(pd.DataFrame({'e':e,'s':t.split}).groupby('e').s.nunique().gt(1).sum()),'tokens_crossing_splits':int(pd.DataFrame({'g':token,'s':t.split}).dropna().groupby('g').s.nunique().gt(1).sum()),'unparsed_malware_tokens':int((m.ne('Benign') & token.isna()).sum()),'token_lengths':token.str.len().value_counts().to_dict(),'split_counts':pd.crosstab(t.split,m).to_dict(),'split_sha256':checksum(root/'data/splits/master_split.csv'),'metric_provenance_matches':{k:metric.get(k)==checksum(root/p) for k,p in {'raw_sha256':'data/raw/Obfuscated-MalMem2022.csv','split_sha256':'data/splits/master_split.csv','preprocessing_sha256':'preprocessing.py'}.items()},'validation_prediction_columns':list(pred.columns),'validation_prediction_rows':len(pred),'notebooks_present':list(map(str,(root/'notebooks').glob('*.ipynb'))),'model_files_present':[str(p.relative_to(root)) for pattern in ['*.joblib','*.pkl'] for p in root.rglob(pattern)],'lightgbm_available':importlib.util.find_spec('lightgbm') is not None}
out=root/'reports/objective_audit/observations.json'
out.write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in r.items() if k not in ['feature_columns','dtypes','near_constant_99pct']},indent=2))

# Additional consistency checks; no classifier training or test performance evaluation.
from sklearn.metrics import accuracy_score, f1_score
extra={}
for p in sorted((root/'data/raw').glob('*.csv')):
 z=pd.read_csv(p)
 if p.name== 'Obfuscated-MalMem2022.csv': continue
 common=[c for c in d if c in z]
 extra[p.name]={'missing_columns':list(d.columns.difference(z.columns)),'extra_columns':list(z.columns.difference(d.columns)),'shared_columns':len(common),'rows_matching_full_on_shared_columns':int(pd.util.hash_pandas_object(z[common],index=False).isin(pd.util.hash_pandas_object(d[common],index=False)).sum())}
r['auxiliary_csv_checks']=extra
r['stored_prediction_checks']={'validation_row_ids_match':set(pred.row_index)==set(t.loc[t.split.eq('validation'),'row_index']),'unique_row_ids':bool(pred.row_index.is_unique),'actual_labels_match_raw':bool(pred.actual.equals(d.Class.iloc[pred.row_index].reset_index(drop=True))),'recomputed_accuracy':accuracy_score(pred.actual,pred.predicted),'recomputed_macro_f1':f1_score(pred.actual,pred.predicted,average='macro')}
r['local_git_directory_present']=(root/'.git').exists()
out.write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'auxiliary_csv_checks':extra,'stored_prediction_checks':r['stored_prediction_checks'],'local_git_directory_present':r['local_git_directory_present']},indent=2))
