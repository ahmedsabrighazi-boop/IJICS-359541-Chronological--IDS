#!/usr/bin/env python3
"""R1-5, IJICS-359541. V2 metadata layout correction. No training: refit calibration mappings from saved RF
raw scores, reselect thresholds, audit flow-end cutoff. Requires V5/RECORDED
per-flow support/test Parquet and V5/csv/Final_All_Evaluation_Metrics.csv.

IMPORTANT: strict unpurged control gates interpretation. This tool does not
recreate or fit the original RF model; scores are from the V5 repeat run.
Historical label delays are NOT simulated. The emitted rows are exploratory if
any unpurged control gate fails. Never modify V5 or August results.
"""
from __future__ import annotations
import argparse, hashlib, json, warnings, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _SigmoidCalibration
from sklearn.metrics import confusion_matrix
import importlib.util, sys

ORIGINAL='nf_unsw_nb15_v3_temporal_reliability_18W_DUAL_FINAL_READY_BOOTSTRAP_FIX.py'
ORIGINAL_SHA='48883b3ec91a63e09e98a74705d160e5ac1819b6f72b19bd78a67a280877d058'
VARIANTS=('Strict-Clean','Strict-Clean-RFE-TopK')
COLS=['row_id','y','p_uncalibrated','p_sigmoid','p_isotonic']

def hashfile(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()

def load_original(path):
 if not path.is_file() or hashfile(path)!=ORIGINAL_SHA:raise RuntimeError('Original source SHA256 mismatch')
 spec=importlib.util.spec_from_file_location('ijics_r1_5_orig',path)
 mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 return mod

def as_time(values):
 a=pd.to_numeric(values,errors='coerce').to_numpy(np.float64)
 v=a[np.isfinite(a)]
 if not len(v):raise ValueError('No valid numeric start/end timestamps')
 mid=np.median(np.abs(v));unit='us' if mid>1e14 else ('ms' if mid>1e11 else ('s' if mid>1e9 else None))
 if unit is None:raise ValueError(f'Cannot determine timestamp unit; median={mid}')
 res=pd.to_datetime(values,errors='coerce',unit=unit).to_numpy(dtype='datetime64[ns]')
 if np.isnat(res).any():raise ValueError('NaT in timing values')
 return res

def read_metadata(fd,role):
 """Read the EXACT V5 recorder layout; test metadata and labels are separate.

 V5 writes calibration/raw_threshold/historical_pool_Support_Metadata.parquet,
 but deliberately writes Test_Flow_Metadata.parquet plus Test_Labels.parquet
 for the test role. Never invent a test_Support_Metadata filename.
 """
 if role=='test':
  # The original recorder maintains identical test row_id order in both files.
  pmeta=fd/'Test_Flow_Metadata.parquet'
  plabel=fd/'Test_Labels.parquet'
  if not pmeta.is_file() or not plabel.is_file():
   raise FileNotFoundError(f'Incomplete V5 test recorder for {fd}: {pmeta.name} and {plabel.name} required')
  tm=pd.read_parquet(pmeta,columns=['row_id','FLOW_START_MILLISECONDS','FLOW_END_MILLISECONDS'])
  yl=pd.read_parquet(plabel,columns=['row_id','y'])
  if len(tm)!=len(yl) or not np.array_equal(tm.row_id.to_numpy(),yl.row_id.to_numpy()):
   raise ValueError(f'V5 test flow metadata and labels row_id/order differ in {fd}')
  x=tm.copy()
  x.insert(1,'y',yl.y.to_numpy(dtype=np.int8))
 else:
  p=fd/f'{role}_Support_Metadata.parquet'
  if not p.is_file():raise FileNotFoundError(f'Missing V5 {role} metadata: {p}')
  x=pd.read_parquet(p,columns=['row_id','y','FLOW_START_MILLISECONDS','FLOW_END_MILLISECONDS'])
 if x.row_id.duplicated().any():raise ValueError(f'Duplicate {role} metadata row_id in {fd}')
 if x[['FLOW_START_MILLISECONDS','FLOW_END_MILLISECONDS']].isna().any().any():
  raise ValueError(f'Missing V5 start/end timestamps: {fd} role={role}')
 return x

def read_scores(fd,variant,role):
 p=fd/f'{variant}_RF_{role}_Scores.parquet' if role!='test' else fd/f'{variant}_RF_PerFlow.parquet'
 x=pd.read_parquet(p,columns=COLS)
 if x.row_id.duplicated().any():raise ValueError('Duplicate score row_id')
 return x

def align(meta,score):
 if len(meta)!=len(score):raise ValueError('Metadata / scores lengths disagree')
 if not np.array_equal(meta.row_id.to_numpy(),score.row_id.to_numpy()):raise ValueError('Score/metatada row_id order changed')
 if not np.array_equal(meta.y.to_numpy(),score.y.to_numpy()):raise ValueError('Score/metadata labels changed')
 return score

def fitmap(raw_cal,y_cal,method):
 if len(np.unique(y_cal))!=2:
  raise ValueError('Calibration pool single class; calibrated inference is not eligible')
 if method=='sigmoid':
  return _SigmoidCalibration().fit(raw_cal, y_cal)
 if method=='isotonic':
  return IsotonicRegression(out_of_bounds='clip').fit(raw_cal,y_cal)
 if method=='uncalibrated':return None
 raise ValueError(method)

def applymap(obj,raw):
 return np.asarray(raw if obj is None else obj.predict(raw),dtype=np.float64)

def cm(y,prob,th):
 pred=(np.asarray(prob)>=float(th)).astype(np.int8)
 tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return {'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def replay_variant(fd,variant,mod,reference, meta,cut1,cut2):
 cal,thr,hist,test=[read_scores(fd,variant,role) for role in ('calibration','raw_threshold','historical_pool','test')]
 for role,x in (('calibration',cal),('raw_threshold',thr),('historical_pool',hist),('test',test)):
  align(meta[role],x)
 y_cal=cal.y.to_numpy(np.int8);y_thr=thr.y.to_numpy(np.int8)
 y_hist=hist.y.to_numpy(np.int8);y_test=test.y.to_numpy(np.int8)
 ends_cal=as_time(meta['calibration'].FLOW_END_MILLISECONDS)
 ends_thr=as_time(meta['raw_threshold'].FLOW_END_MILLISECONDS)
 ends_hist=as_time(meta['historical_pool'].FLOW_END_MILLISECONDS)
 keep_cal=ends_cal<=cut1
 keep_thr=ends_thr<=cut2
 keep_hist=ends_hist<=cut2
 if np.any(ends_cal>cut2):raise ValueError('Calibration flows end after test onset; staged review needed')
 if np.any(ends_cal<as_time(meta['calibration'].FLOW_START_MILLISECONDS)) or np.any(ends_thr<as_time(meta['raw_threshold'].FLOW_START_MILLISECONDS)):
  raise ValueError('Flow end precedes start')
 summary={'calibration_n':len(cal),'calibration_purged':int((~keep_cal).sum()),
          'raw_threshold_n':len(thr),'raw_threshold_purged':int((~keep_thr).sum()),
          'historical_pool_n':len(hist),'historical_pool_purged':int((~keep_hist).sum()),
          'calibration_max_excess_s':float(np.max((ends_cal[~keep_cal]-cut1)/np.timedelta64(1,'s'))) if (~keep_cal).any() else 0.,
          'raw_threshold_max_excess_s':float(np.max((ends_thr[~keep_thr]-cut2)/np.timedelta64(1,'s'))) if (~keep_thr).any() else 0.,
          'calibration_median_excess_s':float(np.median((ends_cal[~keep_cal]-cut1)/np.timedelta64(1,'s'))) if (~keep_cal).any() else 0.,
          'raw_threshold_median_excess_s':float(np.median((ends_thr[~keep_thr]-cut2)/np.timedelta64(1,'s'))) if (~keep_thr).any() else 0.}
 rows=[];issues=[];tol=1e-12
 raw_keys=['uncalibrated','sigmoid','isotonic']
 for source in raw_keys:
  for scenario,purged in [('unpurged_control',False),('flow_end_purged',True)]:
   mc=keep_cal if purged else np.ones(len(cal),dtype=bool)
   mt=keep_thr if purged else np.ones(len(thr),dtype=bool)
   mh=keep_hist if purged else np.ones(len(hist),dtype=bool)
   if not mc.any() or not mt.any():raise RuntimeError('Purging left empty calibration / raw threshold pool')
   if source!='uncalibrated' and np.unique(y_cal[mc]).size!=2:
    issues.append({'variant':variant,'source':source,'scenario':scenario,'issue':'purged_calibration_single_class'});continue
   mapper=fitmap(cal.p_uncalibrated.to_numpy()[mc],y_cal[mc],source)
   vals=applymap(mapper,thr.p_uncalibrated.to_numpy())
   pools=applymap(mapper,hist.p_uncalibrated.to_numpy())
   tests=applymap(mapper,test.p_uncalibrated.to_numpy())
   if not purged:
    for role, predicted, observed in [('raw_threshold',vals,thr[f'p_{source}'].to_numpy()),
         ('historical_pool',pools,hist[f'p_{source}'].to_numpy()),
         ('test',tests,test[f'p_{source}'].to_numpy())]:
     delta=np.abs(predicted-observed)
     if np.max(delta)>tol:
      issues.append({'variant':variant,'source':source,'scenario':scenario,
          'role':role,'issue':'score_replay_gt_1e-12','max_abs_diff':float(np.max(delta)),'n_gt_tol':int((delta>tol).sum())})
   rules=mod.threshold_rules(y_thr[mt],vals[mt],y_hist[mh],pools[mh],mod.CFG)
   for rule in rules:
    name=rule['threshold_name'];threshold=float(rule['threshold_value'])
    out={'variant':variant,'probability_source':source,'scenario':scenario,'threshold_name':name,
         'threshold_value':threshold,'calibration_n':int(mc.sum()),'raw_threshold_n':int(mt.sum()),'historical_pool_n':int(mh.sum()),
         'selection_status':rule['threshold_selection_status'],**cm(y_test,tests,threshold)}
    rows.append(out)
    if not purged:
     expected=reference[(reference.variant==variant)&(reference.probability_source==source)&(reference.threshold_name==name)]
     if len(expected)!=1:
      issues.append({'variant':variant,'source':source,'threshold_name':name,'issue':'missing_or_duplicate_v5_reference','matches':len(expected)})
     else:
      orig=expected.iloc[0]
      for key in ('tn','fp','fn','tp'):
       if int(orig[key])!=out[key]:issues.append({'variant':variant,'source':source,'threshold_name':name,
              'issue':'unpurged_confusion_mismatch','metric':key,'v5':int(orig[key]),'recomputed':out[key]})
      tdiff=abs(float(orig.threshold_value)-threshold)
      if tdiff>tol:issues.append({'variant':variant,'source':source,'threshold_name':name,
                'issue':'unpurged_threshold_mismatch','v5':float(orig.threshold_value),'recomputed':threshold,'abs_diff':tdiff})
 return rows,summary,issues

def selftest():
 y=np.array([0,0,0,1,1,1,0,1],dtype=np.int8)
 raw=np.array([.01,.02,.05,.7,.8,.99,.1,.9],dtype=float)
 a=fitmap(raw,y,'sigmoid');b=fitmap(raw,y,'isotonic')
 assert np.isfinite(applymap(a,raw)).all() and np.isfinite(applymap(b,raw)).all()
 assert cm(y,raw,.5)=={'tn':4,'fp':0,'fn':0,'tp':4}
 t=as_time(pd.Series([1420070400000,1420070401000]));assert (t[1]-t[0])/np.timedelta64(1,'s')==1
 # Compare direct calibrator fitting with the actual original CalibratedClassifierCV path.
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.calibration import CalibratedClassifierCV
 from sklearn.frozen import FrozenEstimator
 x=np.arange(300,dtype=float).reshape(-1,1);labels=(x.ravel()%11>4).astype(int)
 rf=RandomForestClassifier(n_estimators=10,n_jobs=1,random_state=37).fit(x,labels)
 expected_raw=rf.predict_proba(x)[:,1]
 for method in ['sigmoid','isotonic']:
  orig=CalibratedClassifierCV(estimator=FrozenEstimator(rf),method=method).fit(x,labels)
  score=orig.predict_proba(x)[:,1]
  manual=applymap(fitmap(expected_raw,labels,method),expected_raw)
  assert np.max(np.abs(score-manual))<1e-12,(method,np.max(np.abs(score-manual)))
 # No real Parquet IO: assert the test-role layout by intercepting the reader.
 from unittest.mock import patch
 from tempfile import TemporaryDirectory
 with TemporaryDirectory() as td:
  d=Path(td)
  (d/'Test_Flow_Metadata.parquet').touch();(d/'Test_Labels.parquet').touch()
  tests=pd.DataFrame({'row_id':[4,9], 'FLOW_START_MILLISECONDS':[1420070400000,1420070402000],
                      'FLOW_END_MILLISECONDS':[1420070401000,1420070403000]})
  labels=pd.DataFrame({'row_id':[4,9],'y':[0,1]})
  def loader(path,columns=None):
   name=Path(path).name
   if name=='Test_Flow_Metadata.parquet':return tests[list(columns)].copy()
   if name=='Test_Labels.parquet':return labels[list(columns)].copy()
   raise AssertionError(name)
  with patch.object(pd,'read_parquet',side_effect=loader):
   got=read_metadata(d,'test')
   assert list(got.columns)==['row_id','y','FLOW_START_MILLISECONDS','FLOW_END_MILLISECONDS']
   assert got.row_id.tolist()==[4,9] and got.y.tolist()==[0,1]
   labels.loc[1,'row_id']=10
   try:read_metadata(d,'test')
   except ValueError as e:assert 'row_id/order differ' in str(e)
   else:raise AssertionError('Out-of-order test labels silently accepted')
 print('SELF-TEST PASS: V5 test metadata+labels split, row_id validation, sklearn frozen RF, timestamps, confusion; no training')

def run(args):
 root=args.root.resolve(); recorded=root/'RECORDED'; src=args.original.resolve()
 mod=load_original(src)
 metrics=root/'csv'/'Final_All_Evaluation_Metrics.csv'
 if not metrics.is_file():raise FileNotFoundError(f'Missing repeat-run reference {metrics}')
 ref=pd.read_csv(metrics,low_memory=False)
 required={'protocol','fold_id','variant','model','probability_source','threshold_name','threshold_value','tp','fp','tn','fn'}
 if required-set(ref.columns):raise ValueError('Missing metrics columns: '+str(required-set(ref.columns)))
 fold_dirs=[d for d in sorted(recorded.glob('*/*')) if d.is_dir() and d.parent.name in ('sliding_temporal','expanding_temporal')]
 if not fold_dirs:raise FileNotFoundError('No Sliding/Expanding fold directories found')
 if len(fold_dirs)!=22:raise RuntimeError(f'Expected 22 train-valid temporal folds from V5, found {len(fold_dirs)}; do not interpret incomplete replay')
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
 all_rows=[];all_limits=[];issues=[];fold_health=[]
 for i,fd in enumerate(fold_dirs,1):
  meta={}
  for role in ('calibration','raw_threshold','historical_pool','test'):
   meta[role]=read_metadata(fd,role)
  cut1=np.min(as_time(meta['raw_threshold'].FLOW_START_MILLISECONDS))
  cut2=np.min(as_time(meta['test'].FLOW_START_MILLISECONDS))
  foldid=fd.name;protocol=fd.parent.name
  print(f'[{i}/{len(fold_dirs)}] {protocol}/{foldid}',flush=True)
  fref=ref[(ref.protocol==protocol)&(ref.fold_id==foldid)&(ref.model=='RF')]
  for var in VARIANTS:
   if fref[fref.variant==var].empty:
    issues.append({'protocol':protocol,'fold_id':foldid,'variant':var,'issue':'reference_missing_variant'});continue
   try:
    rows,lims,problems=replay_variant(fd,var,mod,fref[fref.variant==var],meta,cut1,cut2)
    for x in rows:x.update({'protocol':protocol,'fold_id':foldid})
    all_rows.extend(rows);all_limits.append({'protocol':protocol,'fold_id':foldid,'variant':var,**lims})
    for x in problems:x.update({'protocol':protocol,'fold_id':foldid})
    issues.extend(problems)
    fold_health.append({'protocol':protocol,'fold_id':foldid,'variant':var,'status':'CONTROL_PASS' if not problems else 'CONTROL_FAIL','n_control_issues':len(problems)})
   except Exception as e:
    issues.append({'protocol':protocol,'fold_id':foldid,'variant':var,'issue':'exception','detail':repr(e)})
    fold_health.append({'protocol':protocol,'fold_id':foldid,'variant':var,'status':'ERROR'})
 rows=pd.DataFrame(all_rows);lim=pd.DataFrame(all_limits)
 issue_frame=pd.DataFrame(issues);health=pd.DataFrame(fold_health)
 rows.to_csv(out/'R1_5_ScoreReplay_Baseline_and_Purged.csv',index=False)
 lim.to_csv(out/'R1_5_Cutoff_Removed_Rows_and_Overhang_Seconds.csv',index=False)
 health.to_csv(out/'R1_5_Control_Gates.csv',index=False)
 issue_frame.to_csv(out/'R1_5_Control_Issues.csv',index=False)
 comparisons=[]
 if len(rows):
  keys=['protocol','fold_id','variant','probability_source','threshold_name']
  a=rows[rows.scenario=='unpurged_control'].set_index(keys)
  b=rows[rows.scenario=='flow_end_purged'].set_index(keys)
  common=a.index.intersection(b.index)
  for k in common:
   aa=a.loc[k];bb=b.loc[k]
   if isinstance(aa,pd.DataFrame) or isinstance(bb,pd.DataFrame):raise ValueError('Duplicate scenario key')
   rec=dict(zip(keys,k))
   rec.update({f'before_{x}':float(aa[x]) if x=='threshold_value' else int(aa[x]) for x in ['threshold_value','tn','fp','fn','tp']})
   rec.update({f'after_{x}':float(bb[x]) if x=='threshold_value' else int(bb[x]) for x in ['threshold_value','tn','fp','fn','tp']})
   rec.update({f'delta_{x}':rec[f'after_{x}']-rec[f'before_{x}'] for x in ['threshold_value','tn','fp','fn','tp']})
   ctrl=health[(health.protocol==rec['protocol'])&(health.fold_id==rec['fold_id'])&(health.variant==rec['variant'])]
   rec['inference_eligible']=bool(len(ctrl)==1 and ctrl.iloc[0]['status']=='CONTROL_PASS')
   comparisons.append(rec)
 pd.DataFrame(comparisons).to_csv(out/'R1_5_Purged_Changes_ONLY_IF_Control_Passes.csv',index=False)
 summary={'status':'PURGE_SENSITIVITY_INTERPRETABLE' if len(health)>0 and health.status.eq('CONTROL_PASS').all() else 'CONTROL_GATE_NOT_FULLY_PASSED',
          'fold_variant_controls':len(health),'control_pass':int(health.status.eq('CONTROL_PASS').sum()),
          'control_fail':int((health.status!='CONTROL_PASS').sum()),'n_issues':len(issues),
          'original_source_sha256':ORIGINAL_SHA,
          'probability_scoring':'Saved raw RF score vectors from V5, no RF inference or fitting',
          'calibrator_fit':'RF score only; sklearn _SigmoidCalibration and IsotonicRegression(out_of_bounds=clip)',
          'cutoff':'FLOW_END_MILLISECONDS <= first FLOW_START_MILLISECONDS in successor role',
          'caveats':['Uses V5 (repeat-run) scores, not lost August per-flow data',
                     'No label delay simulated', 'No test flows removed; test score timing is end-of-flow',
                     'Baseline control must reproduce V5 test scores, thresholds and confusion before purge changes interpreted',
                     'This script does NOT assess change to RF model if its training rows were unavailable'],
          'no_rf_training':True,'no_original_outputs_modified':True,
          'tool_version':'V2',
          'metadata_layout':'test from Test_Flow_Metadata + Test_Labels (validated row_id); support roles from *_Support_Metadata'}
 (out/'R1_5_Score_Replay_ReadMe.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
 archive=out.parent/(out.name+'_Results.zip')
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
  for f in sorted(out.iterdir()):
   if f.is_file(): zf.write(f,arcname=f.name)
 with zipfile.ZipFile(archive) as zf:
  if zf.testzip() is not None:raise RuntimeError('Compressed V2 output did not pass ZIP integrity test')
 print('R1-5 OFFLINE REPLAY COMPLETE:',out, 'control:',summary['status'],'passes:',summary['control_pass'],'/',len(health))
 print('UPLOAD THIS ZIP:',archive)

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--self-test',action='store_true')
 ap.add_argument('--root',type=Path,required=not ('--self-test' in sys.argv))
 ap.add_argument('--output',type=Path,default=Path('C:/IJ5_R1_5_Purged_Score_Replay_V2'))
 ap.add_argument('--original',type=Path,default=Path(ORIGINAL))
 args=ap.parse_args()
 if args.self_test:selftest()
 else:run(args)