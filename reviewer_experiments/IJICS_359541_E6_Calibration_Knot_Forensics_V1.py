#!/usr/bin/env python3
"""Read-only IJICS-359541 E6 isotonic flow-completion boundary investigation.

Reads saved August-source V5 scores and support metadata; NO Random Forest/RFE fit,
NO dataset load, NO modifications to LEGACY results. The only fitting is two
IsotonicRegression maps on previously recorded RAW RF calibration scores.
This reproduces the before/after eligibility-controlled fixed-.5 decision.

Run:
  python IJICS_359541_E6_Calibration_Knot_Forensics_V1.py --self-test
  python IJICS_359541_E6_Calibration_Knot_Forensics_V1.py --root "C:\\Users\\...\\IJ5"

Review the JSON gate before attributing changes to purging.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

FOLD = "expanding_06"
PROTOCOL = "expanding_temporal"
VARIANT = "Strict-Clean-RFE-TopK"
EXPECTED_CM_BEFORE = (120100, 26, 6, 10457)
EXPECTED_CM_AFTER = (120125, 1, 24, 10439)
THRESHOLD = 0.5


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def datetime_ms(series:pd.Series)->np.ndarray:
    ts=pd.to_datetime(pd.to_numeric(series, errors='raise'), unit='ms', errors='raise')
    if ts.isna().any():raise ValueError('Missing flow timestamps')
    return ts.to_numpy(dtype='datetime64[ns]')


def aligned(meta, scores, role):
    for frame in (meta,scores):
        if frame.row_id.isna().any() or frame.row_id.duplicated().any():
            raise ValueError(f'Missing/duplicate row_id in {role}')
    if len(meta)!=len(scores):raise ValueError(f'{role} lengths disagree')
    if not np.array_equal(meta.row_id.to_numpy(),scores.row_id.to_numpy()):
        raise ValueError(f'{role} row_id/order mismatch')
    if not np.array_equal(meta.y.to_numpy(dtype=np.int8),scores.y.to_numpy(dtype=np.int8)):
        raise ValueError(f'{role} binary label mismatch')


def confusion(y,p):
    positive=np.asarray(p,dtype=np.float64)>=THRESHOLD
    lab=np.asarray(y,dtype=np.int8)
    return (int(np.count_nonzero((lab==0)&(~positive))),
            int(np.count_nonzero((lab==0)&positive)),
            int(np.count_nonzero((lab==1)&(~positive))),
            int(np.count_nonzero((lab==1)&positive)))


def interval_details(model, x):
    """Local *geometric* brackets, not a claim that endpoints alone fit the plateau."""
    xx=np.asarray(model.X_thresholds_,dtype=np.float64)
    yy=np.asarray(model.y_thresholds_,dtype=np.float64)
    if not len(xx) or len(xx)!=len(yy):raise ValueError('Invalid isotonic knots')
    # Exact knot is allowed to belong to both adjacent segments.
    seg=[]
    for j in range(len(xx)-1):
        if xx[j]<=x<=xx[j+1]:
            seg.append(j)
    if not seg:
        j=0 if x<xx[0] else len(xx)-2
        seg=[max(0,j)]
    out=[]
    for j in seg:
        out.append(dict(left_x=float(xx[j]),right_x=float(xx[j+1]),
                        left_y=float(yy[j]),right_y=float(yy[j+1]),
                        segment=('flat' if yy[j]==yy[j+1] else 'sloped'),
                        score_at_endpoint=bool(x==xx[j] or x==xx[j+1])))
    return out


def flat_plateaus(model,x):
    """Return maximal adjacent horizontal *knot segments* containing x."""
    xx=np.asarray(model.X_thresholds_,dtype=np.float64)
    yy=np.asarray(model.y_thresholds_,dtype=np.float64)
    found=[];i=0
    while i < len(xx)-1:
        if yy[i] != yy[i+1]:i+=1;continue
        start=i
        while i<len(xx)-1 and yy[i] == yy[i+1]:i+=1
        if xx[start]<=x<=xx[i]:
            found.append(dict(left_x=float(xx[start]),right_x=float(xx[i]),
                              plateau_y=float(yy[start])))
    return found


def summarize_support(xs,ys,keep,left,right):
    inside=(xs>=left)&(xs<=right)
    removed=inside & (~keep)
    return dict(n=int(inside.sum()),benign=int(np.count_nonzero(inside &(ys==0))),
                attack=int(np.count_nonzero(inside &(ys==1))),
                removed_total=int(removed.sum()),
                removed_benign=int(np.count_nonzero(removed &(ys==0))),
                removed_attack=int(np.count_nonzero(removed &(ys==1))))


def selftest():
    y=np.array([0,1,0,1,0],dtype=np.int8)
    a=np.array([0.1,0.5,0.5,0.8,0.5])
    assert confusion(y,a)==(1,2,0,2)
    xs=np.array([0,.1,.2,.2,.4,.7]);ys=np.array([0,0,1,1,0,1]);keep=np.array([1,1,0,1,1,1],bool)
    z=summarize_support(xs,ys,keep,.1,.2)
    assert z==dict(n=3,benign=1,attack=2,removed_total=1,removed_benign=0,removed_attack=1)
    model=IsotonicRegression(out_of_bounds='clip').fit(np.array([0,.1,.2,.3,.4,.7,.8]),
                                                       np.array([0,0,0,1,1,1,1]))
    assert len(interval_details(model,.15))>=1
    assert flat_plateaus(model,0.0)
    # The comparison is >=; a score exactly at .5 must be a positive prediction.
    assert confusion(np.array([0]),np.array([.5]))==(0,1,0,0)
    print('SELF-TEST PASS: confusion, support counts, local isotonic knots, plateau interval, inclusive threshold; no data reads or fitting RF')


def run(args):
    root=args.root.expanduser().resolve();fd=root/'RECORDED'/PROTOCOL/FOLD
    if not fd.is_dir():raise FileNotFoundError(f'Missing V5 E6 fold directory: {fd}')
    out=(args.output or (root.parent/'IJ5_E6_Knot_Forensics')).expanduser().resolve()
    if out==fd or out==root or (root in out.parents):
        raise ValueError('Output must be separate from and outside original LEGACY root')
    prefix=f'{VARIANT}_RF'
    paths={'calibration_scores':fd/f'{prefix}_calibration_Scores.parquet',
           'calibration_metadata':fd/'calibration_Support_Metadata.parquet',
           'threshold_metadata':fd/'raw_threshold_Support_Metadata.parquet',
           'test_scores':fd/f'{prefix}_PerFlow.parquet',
           'test_metadata':fd/'Test_Flow_Metadata.parquet',
           'test_labels':fd/'Test_Labels.parquet',
           'knots':fd/f'{VARIANT}_RF_Isotonic_Knots.json',
           'metric':root/'csv'/'Final_All_Evaluation_Metrics.csv'}
    missing=[name for name,path in paths.items() if not path.is_file()]
    if missing:raise FileNotFoundError('Required original V5 files missing: '+', '.join(missing))
    print('Reading saved E6 calibration and test scores, without RF/RFE fitting...',flush=True)
    score_cols=['row_id','y','p_uncalibrated','p_isotonic']
    cal=pd.read_parquet(paths['calibration_scores'],columns=score_cols)
    cm=pd.read_parquet(paths['calibration_metadata'],columns=['row_id','y','FLOW_START_MILLISECONDS','FLOW_END_MILLISECONDS'])
    aligned(cm,cal,'calibration')
    tm=pd.read_parquet(paths['threshold_metadata'],columns=['row_id','FLOW_START_MILLISECONDS'])
    test=pd.read_parquet(paths['test_scores'],columns=score_cols)
    tmeta=pd.read_parquet(paths['test_metadata'],columns=['row_id'])
    ty=pd.read_parquet(paths['test_labels'],columns=['row_id','y'])
    if not np.array_equal(tmeta.row_id.to_numpy(),ty.row_id.to_numpy()):raise ValueError('Test metadata and label order mismatch')
    aligned(ty,test,'test')
    if len(test)!=130589:raise ValueError(f'E6 expected 130589 test flows; got {len(test)}')
    if len(cal)!=182828:raise ValueError(f'E6 expected 182828 calibration support; got {len(cal)}')
    cutoff=np.min(datetime_ms(tm.FLOW_START_MILLISECONDS))
    end_times=datetime_ms(cm.FLOW_END_MILLISECONDS)
    keep=end_times<=cutoff
    raw_cal=cal.p_uncalibrated.to_numpy(dtype=np.float64)
    lab_cal=cal.y.to_numpy(dtype=np.int8)
    raw_test=test.p_uncalibrated.to_numpy(dtype=np.float64)
    lab_test=test.y.to_numpy(dtype=np.int8)
    if not np.isfinite(raw_cal).all() or not np.isfinite(raw_test).all():raise ValueError('Nonfinite scores')
    if len(np.unique(lab_cal[keep]))!=2:raise ValueError('Purging left calibration pool one-class')
    before=IsotonicRegression(out_of_bounds='clip').fit(raw_cal,lab_cal)
    after=IsotonicRegression(out_of_bounds='clip').fit(raw_cal[keep],lab_cal[keep])
    computed_before=before.predict(raw_test)
    recorded_before=test.p_isotonic.to_numpy(dtype=np.float64)
    computed_after=after.predict(raw_test)
    score_error=float(np.max(np.abs(computed_before-recorded_before)))
    knots_stored=json.loads(paths['knots'].read_text(encoding='utf8'))
    stored_x=np.asarray(knots_stored['x_thresholds'],float)
    stored_y=np.asarray(knots_stored['y_thresholds'],float)
    knot_match=bool(len(stored_x)==len(before.X_thresholds_) and
                    np.max(np.abs(stored_x-before.X_thresholds_))<=1e-12 and
                    np.max(np.abs(stored_y-before.y_thresholds_))<=1e-12)
    before_cm=confusion(lab_test,recorded_before)
    before_refit_cm=confusion(lab_test,computed_before)
    after_cm=confusion(lab_test,computed_after)
    removed=int(np.count_nonzero(~keep))
    pre_prediction=(recorded_before>=THRESHOLD)
    post_prediction=(computed_after>=THRESHOLD)
    flip=np.flatnonzero(pre_prediction!=post_prediction)
    expected_flips=int(sum(abs(a-b) for a,b in zip(before_cm,after_cm))//2)
    test_ref=pd.read_csv(paths['metric'],low_memory=False)
    one=test_ref.loc[(test_ref.protocol==PROTOCOL)&(test_ref.fold_id==FOLD)&
                     (test_ref.variant==VARIANT)&(test_ref.model=='RF')&
                     (test_ref.probability_source=='isotonic')&(test_ref.threshold_name=='fixed_0.5')]
    if len(one)!=1:raise ValueError('E6 original V5 fixed-.5 result missing / duplicated')
    frozen_tuple=tuple(int(one.iloc[0][c]) for c in ('tn','fp','fn','tp'))
    gates={'source_E6_metric_match':frozen_tuple==before_cm==EXPECTED_CM_BEFORE,
           'unpurged_probability_within_1e-12':score_error<=1e-12,
           'recorded_isotonic_knots_within_1e-12':knot_match,
           'refit_control_confusion_match':before_refit_cm==before_cm,
           'expected_purged_32':removed==32,
           'purged_cm_matches_V2':after_cm==EXPECTED_CM_AFTER,
           'expected_43_flips':len(flip)==43,
           'no_model_fit':True,
           'read_only_original':True}
    out.mkdir(parents=True,exist_ok=True)
    gate={'status':'PASS' if all(gates.values()) else 'FAIL_DO_NOT_ATTRIBUTE_TO_PURGING',
          'gates':gates,'original_root_does_not_change':True,
          'calibration_total':len(cal),'calibration_removed':removed,
          'calibration_removed_by_label':{'benign':int(np.count_nonzero((~keep)&(lab_cal==0))),
                                          'attack':int(np.count_nonzero((~keep)&(lab_cal==1)))},
          'cutoff_at_first_threshold_window_start':str(cutoff),
          'test_n':len(test),'before_cm_tn_fp_fn_tp':before_cm,
          'before_refit_cm_tn_fp_fn_tp':before_refit_cm,
          'after_cm_tn_fp_fn_tp':after_cm,'flip_count':len(flip),
          'new_tn_from_fp':int(np.count_nonzero((lab_test==0)&pre_prediction&(~post_prediction))),
          'new_fn_from_tp':int(np.count_nonzero((lab_test==1)&pre_prediction&(~post_prediction))),
          'max_unpurged_probability_discrepancy':score_error,
          'knot_count_before':len(before.X_thresholds_),
          'knot_count_after':len(after.X_thresholds_),
          'reference_files_sha256':{name:sha256_file(path) for name,path in paths.items()}}
    (out/'E6_Forensic_Control_Gate.json').write_text(json.dumps(gate,indent=2,ensure_ascii=False),encoding='utf8')
    # Raw point evidence is saved even on FAIL, but cannot support causal attribution on FAIL.
    xx=pd.DataFrame({'row_id':test.row_id.to_numpy()[flip],
                     'y':lab_test[flip],
                     'raw_rf_score':raw_test[flip],
                     'raw_rf_score_hex':[float(x).hex() for x in raw_test[flip]],
                     'before_isotonic':recorded_before[flip],
                     'after_isotonic':computed_after[flip],
                     'before_pred':pre_prediction[flip].astype(np.int8),
                     'after_pred':post_prediction[flip].astype(np.int8)})
    xx.to_csv(out/'E6_Changed_43_Test_Flows.csv',index=False,float_format='%.17g')
    deleted=pd.DataFrame({'row_id':cal.row_id.to_numpy()[~keep],
                          'y':lab_cal[~keep],
                          'raw_rf_score':raw_cal[~keep],
                          'FLOW_END_MILLISECONDS':cm.FLOW_END_MILLISECONDS.to_numpy()[~keep],
                          'overhang_seconds':((end_times[~keep]-cutoff)/np.timedelta64(1,'s')).astype(float)})
    deleted.to_csv(out/'E6_Removed_32_Calibration_Flows.csv',index=False,float_format='%.17g')
    for name,obj in [('Before',before),('After',after)]:
        pd.DataFrame({'knot_index':range(len(obj.X_thresholds_)),
                      'raw_score_x':obj.X_thresholds_,
                      'isotonic_p':obj.y_thresholds_}).to_csv(out/f'E6_Isotonic_Knots_{name}.csv',index=False,float_format='%.17g')
    group_rows=[];seg_rows=[]
    for raw in np.unique(raw_test[flip]):
        which=flip[raw_test[flip]==raw]
        raw_support=(raw_cal==raw)
        extra={'all_calibration_at_exact_raw':int(raw_support.sum()),
               'all_calibration_exact_raw_attack':int(np.count_nonzero(raw_support &(lab_cal==1))),
               'removed_calibration_exact_raw':int(np.count_nonzero(raw_support &(~keep))),
               'removed_attack_exact_raw':int(np.count_nonzero(raw_support &(~keep)&(lab_cal==1)))}
        unique_meta=dict(raw_score=float(raw),raw_score_hex=float(raw).hex(),
                         n_changed_test=int(len(which)),
                         n_benign_fp_to_tn=int(np.count_nonzero(lab_test[which]==0)),
                         n_attack_tp_to_fn=int(np.count_nonzero(lab_test[which]==1)),
                         before_probability=float(before.predict([raw])[0]),
                         after_probability=float(after.predict([raw])[0]),**extra)
        for label,obj in [('before',before),('after',after)]:
            brackets=interval_details(obj,raw)
            flats=flat_plateaus(obj,raw)
            unique_meta[f'{label}_local_segments']=json.dumps(brackets,ensure_ascii=False)
            unique_meta[f'{label}_flat_intervals']=json.dumps(flats,ensure_ascii=False)
            for kind,items in [('adjacent_knots',brackets),('flat_plateau',flats)]:
                for piece in items:
                    a,b=piece['left_x'],piece['right_x']
                    seg_rows.append({'raw_score':raw,'scenario':label,'support_region_type':kind,
                                     **piece,**summarize_support(raw_cal,lab_cal,keep,a,b)})
        group_rows.append(unique_meta)
    pd.DataFrame(group_rows).to_csv(out/'E6_Changed_RawScore_Levels.csv',index=False,float_format='%.17g')
    pd.DataFrame(seg_rows).to_csv(out/'E6_Local_Knot_Segment_Support.csv',index=False,float_format='%.17g')
    markdown=f'''# E6 targeted calibration investigation (read-only)\n\nControl gate: **{gate['status']}**.\n\n- Repeated RF fitting: **none**. Isotonic only was refitted from logged raw scores.\n- Original E6 calibration rows: **{len(cal):,}**; removed **{removed}** ({gate['calibration_removed_by_label']['benign']} benign, {gate['calibration_removed_by_label']['attack']} attack).\n- Unpurged V5 CM (TN, FP, FN, TP): **{before_cm}**.\n- Purged re-fit CM (TN, FP, FN, TP): **{after_cm}**.\n- Changed test flows: **{len(flip)}**, spread over **{len(group_rows)}** distinct exact binary64 raw-RF score levels.\n- Largest difference between logged and re-fitted unpurged probability: **{score_error:.9g}**.\n- Original knot array matches unpurged refit to 1e-12: **{knot_match}**.\n\n**Interpretation caveat:** The adjacent knot interval counts and flat-plateau interval counts are observational *support sizes* within specified raw-score ranges, not proof that exactly those observations individually determined the PAV isotonic estimates. Mapping changes can be influenced by globally pooled adjacent isotonic blocks. Attribution is conditional on every gate passing.\n'''
    (out/'E6_One_Page_ReadMe.md').write_text(markdown,encoding='utf8')
    archive=out.parent/(out.name+'_Review.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in sorted(out.iterdir()):
            if file.is_file():z.write(file,arcname=file.name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise RuntimeError('Archive corrupted')
    print('E6 READ-ONLY FORENSIC COMPLETE:',out)
    print('Gate:',gate['status'],'removed calibration:',removed,'changed test decisions:',len(flip),'raw levels:',len(group_rows))
    print('Unpurged:',before_cm,'purged:',after_cm)
    print('UPLOAD:',archive)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:selftest();return
    if args.root is None:p.error('--root is required')
    run(args)
if __name__=='__main__':main()