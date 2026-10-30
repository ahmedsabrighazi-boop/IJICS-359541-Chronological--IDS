#!/usr/bin/env python3
"""IJICS-359541 R1-5: read-only flow-end/chronological-availability audit.

Reuses original August CSV loading, exact deduplication, label mapping, stable
chronological order and sliding/expanding split construction. No model fitting.
Measures flows whose recorded END is later than a *later* dedicated role
window's first START. Reports temporal overlap; does NOT automatically infer
that a model was contaminated, or that those flows were available online.

Memory warning: original 2.36m x 55 CSV parsing/dedup is sizable on 8 GiB RAM.
Run with unrelated applications closed; do NOT run concurrently with training.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

ORIGINAL='nf_unsw_nb15_v3_temporal_reliability_18W_DUAL_FINAL_READY_BOOTSTRAP_FIX.py'
SOURCE_SHA='48883b3ec91a63e09e98a74705d160e5ac1819b6f72b19bd78a67a280877d058'
DATA_SHA='4ebb97bd74412d566137d95a6fc3ffd8f374f1cf8cfe204d007848e7a668f9b5'

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()

def original_module(path):
 spec=importlib.util.spec_from_file_location('ijics_frozen_r1_5_module',str(path));m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def parse_end(column):
    """Convert original FLOW_END_MILLISECONDS to datetime64[ns], no imputation."""
    numeric=pd.to_numeric(column,errors='coerce'); vals=numeric.to_numpy(dtype='float64')
    finite=np.isfinite(vals)
    if not finite.any():return pd.to_datetime(column,errors='coerce',utc=True).dt.tz_localize(None).to_numpy(dtype='datetime64[ns]'),'datetime_parse'
    mid=np.nanmedian(np.abs(vals[finite]));unit='us' if mid>1e14 else ('ms' if mid>1e11 else ('s' if mid>1e9 else None))
    if unit is None:raise ValueError('FLOW_END_MILLISECONDS does not resemble Unix timestamps. Manual check necessary.')
    return pd.to_datetime(numeric,unit=unit,errors='coerce').to_numpy(dtype='datetime64[ns]'),unit

def crossing(end, boundary, idx, labels):
    idx=np.asarray(idx,dtype=np.int64); y=labels[idx]; e=end[idx]
    valid=~np.isnat(e); crossed=valid&(e>boundary)
    return {'rows':int(idx.size),'valid_end':int(valid.sum()),'end_missing':int((~valid).sum()),
      'end_after_boundary':int(crossed.sum()),'benign_crossing':int((crossed&(y==0)).sum()),'attack_crossing':int((crossed&(y==1)).sum()),
      'crossing_pct_valid':float(crossed.sum()/valid.sum()*100) if valid.any() else None}

def test():
    end=np.array(['2015-01-01T10:05:00','2015-01-01T10:00:00','NaT','2015-01-01T10:01:00'],dtype='datetime64[ns]')
    labels=np.array([0,1,0,1]);r=crossing(end,np.datetime64('2015-01-01T10:00:00'),np.array([0,1,2,3]),labels)
    assert r['end_after_boundary']==2 and r['benign_crossing']==1 and r['attack_crossing']==1 and r['end_missing']==1
    print('SELF-TEST PASS: strict end-after-start crossing, class counts, missing end; no training')

def run(args):
    original=args.original.resolve(); dataset=args.dataset.resolve();out=args.output.resolve()
    if not original.is_file() or not dataset.is_file():raise FileNotFoundError('Original source and full dataset required')
    source_digest=sha256(original);dataset_digest=sha256(dataset)
    if (source_digest,dataset_digest)!=(SOURCE_SHA,DATA_SHA):raise ValueError('SHA-256 mismatch. Do not run on an unverified source or dataset')
    mod=original_module(original);cfg=replace(mod.CFG, DATA_PATHS=(str(dataset),))
    out.mkdir(parents=True,exist_ok=True); logfile=out/'FlowEnd_Order_and_Split_Log.txt'
    df=mod.load_dataset(cfg,logfile)
    found=mod.detect_columns(df,cfg);df=mod.derive_labels(df,found)
    if 'FLOW_END_MILLISECONDS' not in df.columns:raise KeyError('FLOW_END_MILLISECONDS absent from raw CSV')
    df,sort_reason=mod.prepare_order(df,found['time'],logfile)
    if sort_reason!='time_column:FLOW_START_MILLISECONDS' or len(df)!=2350609:raise AssertionError('Original ordering or deduplicated row count inconsistent')
    ends,unit=parse_end(df['FLOW_END_MILLISECONDS'])
    starts=pd.to_datetime(df['__time_order__'],errors='coerce').to_numpy(dtype='datetime64[ns]')
    y=df['__binary_label__'].to_numpy(dtype=np.int8)
    bad=(~np.isnat(ends))&(~np.isnat(starts))&(ends<starts)
    # After exact provenance-preserving sort, the split constructors need only the labels.
    thin=pd.DataFrame({'__binary_label__':y})
    del df
    win=np.array_split(np.arange(len(thin),dtype=np.int64),cfg.N_TEMPORAL_WINDOWS)
    win_rows=[]
    for w,idx in enumerate(win):
      ref=starts[win[w+1][0]] if w+1<len(win) else np.datetime64('NaT')
      r=crossing(ends,ref,idx,y) if w+1<len(win) else {'rows':int(len(idx)),'valid_end':int((~np.isnat(ends[idx])).sum()),'end_missing':int(np.isnat(ends[idx]).sum()),'end_after_boundary':None,'benign_crossing':None,'attack_crossing':None,'crossing_pct_valid':None}
      win_rows.append({'window':f'W{w}','next_window_start':str(ref),**r})
    rows=[]
    for protocol in ('sliding','expanding'):
      for sp in mod.rolling_splits(thin,cfg,protocol,out/f'{protocol}_split_construction.log'):
        cal_start=starts[win[sp.calibration_window_id][0]]
        thr_start=starts[win[sp.threshold_window_id][0]]
        test_start=starts[win[sp.test_window_id][0]]
        role_tests=[('model_train','calibration_start',sp.train_idx,cal_start),
          ('model_train','raw_threshold_start',sp.train_idx,thr_start),
          ('model_train','test_start',sp.train_idx,test_start),
          ('effective_calibration_pool','raw_threshold_start',sp.calibration_idx,thr_start),
          ('effective_calibration_pool','test_start',sp.calibration_idx,test_start),
          ('raw_threshold_window','test_start',sp.threshold_idx,test_start),
          ('historical_threshold_pool','test_start',sp.threshold_pool_idx,test_start)]
        for role,bn,idx,boundary in role_tests:
          rows.append({'protocol':sp.protocol,'fold':sp.fold_id,'training_window_ids':','.join(map(str,sp.train_window_ids)),
                       'calibration_window':sp.calibration_window_id,'threshold_window':sp.threshold_window_id,
                       'test_window':sp.test_window_id,'role':role,'compared_with':bn,
                       'boundary_start':str(boundary),**crossing(ends,boundary,idx,y)})
    if not rows:raise RuntimeError('No valid temporal folds')
    pd.DataFrame(rows).to_csv(out/'R1_5_FlowEnd_Crossing_By_Fold_Role.csv',index=False)
    pd.DataFrame(win_rows).to_csv(out/'R1_5_FlowEnd_Crossing_By_Window.csv',index=False)
    info={'status':'PASS_READ_ONLY','source_sha256':source_digest,'dataset_sha256':dataset_digest,
       'n_dedup_rows':int(len(y)),'n_valid_flow_end':int(np.count_nonzero(~np.isnat(ends))),
       'n_missing_flow_end':int(np.count_nonzero(np.isnat(ends))),
       'n_end_before_start':int(bad.sum()),'parsed_end_unit':unit,
       'n_temporal_fold_role_comparisons':len(rows),
       'definition':'A row is counted if recorded FLOW_END_MILLISECONDS is strictly later than the first FLOW_START_MILLISECONDS in a later role window. This is a potential availability conflict, not proof of actual look-ahead; results are not purged or used for training.',
       'source_code_modified':False,'training_performed':False}
    (out/'R1_5_FlowEnd_Audit_Summary.json').write_text(json.dumps(info,indent=2),encoding='utf8')
    print('R1-5 READ-ONLY FLOW-END AUDIT COMPLETE:',out)
    print('Valid ends:',info['n_valid_flow_end'],'missing:',info['n_missing_flow_end'],'end-before-start:',info['n_end_before_start'])

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--self-test',action='store_true')
 ap.add_argument('--dataset',type=Path,default=Path('NF-UNSW-NB15-v3.csv'))
 ap.add_argument('--original',type=Path,default=Path(ORIGINAL))
 ap.add_argument('--output',type=Path,default=Path('C:/IJ5_R1_5_FlowEnd'))
 args=ap.parse_args()
 if args.self_test:test()
 else:run(args)