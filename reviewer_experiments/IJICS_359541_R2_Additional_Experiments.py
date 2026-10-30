"""Targeted reviewer experiments running through the frozen study pipeline."""
from __future__ import annotations
import argparse
import gc
import json
import time
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
import nf_unsw_nb15_v3_temporal_reliability_18W_DUAL_FINAL_READY_BOOTSTRAP_FIX as core


def save_csv(items, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(items).to_csv(path, index=False)
    print(f"Saved: {path}", flush=True)


def prepare(dataset, output):
    cfg = replace(core.CFG, DATA_PATHS=(str(dataset),), EXECUTION_MODE='preflight',
                  MAX_ROWS_PER_FILE=None, DROP_GLOBAL_EXACT_DUPLICATES=True)
    output.mkdir(parents=True, exist_ok=True)
    df = core.load_dataset(cfg, output/'data_load.log')
    detected = core.detect_columns(df, cfg)
    df = core.derive_labels(df, detected)
    df, ordering = core.prepare_order(df, detected['time'], output/'data_load.log')
    clean, ttl, strict, audit = core.leakage_audit(df, detected, cfg)
    prohibited = {core.normalize_name(x) for x in cfg.STRICT_DROP_FEATURES}
    cols = [c for c in clean if core.normalize_name(c) not in prohibited]
    if len(df) != 2350609 or int(df['__binary_label__'].sum()) != 127679 or len(cols) != 46:
        raise ValueError('Frozen 18-window data and strict-clean cohort control failed')
    return cfg, df, cols


def transform(df, sp, cols):
    idx = (sp.train_idx, sp.calibration_idx, sp.threshold_idx,
           sp.threshold_pool_idx, sp.test_idx)
    data = [df.iloc[i] for i in idx]
    y = [x['__binary_label__'].to_numpy(dtype=np.int8) for x in data]
    x = core.fit_transform(*data, cols)
    return data, y, x[:5], list(x[5])


def frozen_select(frozen, sp, names):
    fold = frozen[(frozen['protocol']==sp.protocol) & (frozen['fold_id']==sp.fold_id)
                  & (frozen['base_variant']=='Strict-Clean')]
    if fold.empty:
        raise ValueError(f'Frozen RFE features absent for {sp.protocol}/{sp.fold_id}')
    top = fold[fold['selected'].astype(str).str.lower()=='true']['feature'].astype(str).tolist()
    if len(top) != 15 or any(feature not in names for feature in top):
        raise ValueError(f'Frozen RFE feature control failed {sp.fold_id}')
    return [names.index(t) for t in top]


def rf(cfg, max_features=None, seed=10200):
    return RandomForestClassifier(n_estimators=cfg.RF_N_ESTIMATORS,
                                  max_depth=cfg.RF_MAX_DEPTH,
                                  min_samples_leaf=cfg.RF_MIN_SAMPLES_LEAF,
                                  class_weight='balanced_subsample',n_jobs=-1,
                                  max_features=max_features or 'sqrt',
                                  random_state=int(cfg.RANDOM_STATE+seed))


def confusion(y, pred):
    tn, fp, fn, tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
                fpr=float(fp/(fp+tn)) if fp+tn else float('nan'),
                macro_f1=float(f1_score(y,pred,average='macro',labels=[0,1],zero_division=0)))


def label_delay(cfg,df,cols,out):
    windows=list(np.array_split(np.arange(len(df),dtype=np.int64), cfg.N_TEMPORAL_WINDOWS))
    originals=core.rolling_splits(df,cfg,training_mode='sliding')
    result=[]
    for original in originals:
        if original.candidate_number not in range(7,14):
            continue
        for lag in (0,1,2):
            cal_w=original.calibration_window_id-lag
            thr_w=original.threshold_window_id-lag
            train_ws=list(range(cal_w-3,cal_w))
            row=dict(protocol='sliding_temporal',fold_id=original.fold_id,
                     test_window=f'W{original.test_window_id}',lag_windows=lag,
                     train_windows=';'.join('W'+str(i) for i in train_ws),
                     calibration_window=f'W{cal_w}',threshold_window=f'W{thr_w}',
                     test_rows=len(original.test_idx))
            roles=core._build_past_only_support_pools(df,[windows[i] for i in train_ws],
                                                      windows[cal_w],windows[thr_w],cfg)
            train,cal,pool,calmode,thmode,calres,thres=roles
            row.update(train_rows=len(train),train_attacks=int(df.iloc[train]['__binary_label__'].sum()),
                       train_classes=int(core._class_count(df,train)))
            if core._class_count(df,train)<2:
                row['status']='INFEASIBLE_NO_TWO_CLASS_MODEL_HISTORY'
                for rule in ('fixed_0.5','youden'):
                    result.append({**row,'threshold_name':rule})
                continue
            sp=replace(original,train_idx=train,calibration_idx=cal,threshold_idx=windows[thr_w],
                       threshold_pool_idx=pool,train_window_ids=tuple(train_ws),
                       calibration_window_id=cal_w,threshold_window_id=thr_w)
            _,ys,mats,names=transform(df,sp,cols)
            rfe=core.run_rfe(mats[0],ys[0],names,cfg,
                                 sample_seed=cfg.RFE_FIXED_SAMPLE_SEED+2,
                                 model_seed=cfg.RFE_FIXED_MODEL_SEED+2)
            model=rf(cfg,seed=10210).fit(mats[0][:,rfe.selected],ys[0])
            calibrator=core.fit_calibrator(model,mats[1][:,rfe.selected],ys[1],'isotonic')
            if calibrator is None:
                for rule in ('fixed_0.5','youden'):
                    result.append({**row,'threshold_name':rule,'status':'INFEASIBLE_CALIBRATION_ONE_CLASS'})
                continue
            score_thr=core.positive_probability(calibrator,mats[2][:,rfe.selected])
            score_test=core.positive_probability(calibrator,mats[4][:,rfe.selected])
            for rule in ('fixed_0.5','youden'):
                r=row.copy();r['threshold_name']=rule
                if rule=='youden':
                    if len(np.unique(ys[2]))<2:
                        r['status']='INFEASIBLE_THRESHOLD_ONE_CLASS';result.append(r);continue
                    t=float(core.youden_threshold(ys[2],score_thr))
                else:t=0.5
                pred=(score_test>=t).astype(np.int8)
                r.update(threshold_value=t,status='COMPLETED',
                         test_attacks=int(ys[4].sum()),**confusion(ys[4],pred))
                result.append(r)
            del mats,model,rfe;gc.collect()
    save_csv(result,out/'R2_1_LabelDelay_Reexecuted.csv')


def matched_cost(cfg,df,cols,frozen,out):
    rows=[]
    folds={s.candidate_number:s for s in core.rolling_splits(df,cfg,training_mode='expanding')}
    for k in range(4,14):
        sp=folds.get(k)
        if sp is None: raise ValueError(f'Expanding E{k} unavailable')
        _,ys,mats,names=transform(df,sp,cols)
        idx=frozen_select(frozen,sp,names)
        start=time.perf_counter()
        rfe=core.run_rfe(mats[0],ys[0],names,cfg,
                                 sample_seed=cfg.RFE_FIXED_SAMPLE_SEED+2,
                                 model_seed=cfg.RFE_FIXED_MODEL_SEED+2)
        rfe_sec=time.perf_counter()-start
        if set(rfe.selected.tolist())!=set(idx):
            raise ValueError(f'RFE selected-set control mismatch E{k}; stopping timing comparison')
        sec={}
        for arm,ix in [('RF_15',idx),('RF_46',list(range(46)))]:
            model=rf(cfg,max_features=6,seed=10210)
            start=time.perf_counter()
            model.fit(mats[0][:,ix],ys[0]);sec[arm]=time.perf_counter()-start
            del model;gc.collect()
        saving=100*(sec['RF_46']-(rfe_sec+sec['RF_15']))/sec['RF_46']
        row=dict(fold=sp.fold_id,n_train=len(ys[0]),n_features_selected=15,
                 n_features_reference=46,max_features_both=6,
                 rfe_seconds=rfe_sec,rf15_fitting_seconds=sec['RF_15'],
                 rf46_fitting_seconds=sec['RF_46'],
                 feature_selection_plus_rf_fitting_seconds=rfe_sec+sec['RF_15'],
                 saving_percent=saving)
        rows.append(row)
        save_csv(rows,out/'R2_2_Matched_Selector_Plus_RF_Fitting.csv')
        del mats,rfe;gc.collect()
    summary={'median_saving_percent':float(np.median([r['saving_percent'] for r in rows])),
             'median_rf15_fitting_seconds':float(np.median([r['rf15_fitting_seconds'] for r in rows])),
             'median_rf46_fitting_seconds':float(np.median([r['rf46_fitting_seconds'] for r in rows]))}
    (out/'R2_2_Matched_Summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')


def field(df, possibilities):
    mapping={core.normalize_name(c):c for c in df.columns}
    for value in possibilities:
        key=core.normalize_name(value)
        if key in mapping:return mapping[key]
    return None


def alert_groups(frame, binary, correct_y, src, dst, stamp_ms):
    # Chronological runs of alerts for each ordered source/destination pair.
    mask=np.flatnonzero(binary)
    if len(mask)==0:return (0,0,0)
    pairs=list(zip(frame.iloc[mask][src].astype(str).to_numpy(),
                   frame.iloc[mask][dst].astype(str).to_numpy()))
    ordered=sorted(zip(pairs,stamp_ms[mask]),key=lambda z:(z[0],z[1]))
    previous={};groups=0
    for pair,timestamp in ordered:
        if pair not in previous or timestamp-previous[pair]>60000:
            groups+=1
        previous[pair]=timestamp
    return len(mask),groups,len(set(pairs))


def predictions_stability_and_ties(cfg,df,cols,frozen,out,run_stability,run_alerts,run_ties):
    stability=[];alerts=[];ties=[]
    src=field(df,('IPV4_SRC_ADDR','SRC_IP','SRC_ADDR','SOURCE_IP'))
    dst=field(df,('IPV4_DST_ADDR','DST_IP','DST_ADDR','DESTINATION_IP'))
    tm=field(df,('FLOW_START_MILLISECONDS',))
    if run_alerts and (src is None or dst is None or tm is None):
        raise ValueError('R2-7 requires original directed host fields and FLOW_START_MILLISECONDS')
    for mode in ('sliding','expanding'):
        splits=core.rolling_splits(df,cfg,training_mode=mode)
        for sp in splits:
            _,ys,mats,names=transform(df,sp,cols)
            selected=frozen_select(frozen,sp,names)
            trained={}
            for arm,ix,offset in [('Strict-Clean',list(range(46)),10200),
                                  ('Strict-Clean-RFE-TopK',selected,10210)]:
                model=rf(cfg,seed=offset)
                model.fit(mats[0][:,ix],ys[0])
                trained[arm]=(model,ix,offset)
            predictions={};bunches={}
            for arm,(model,ix,offset) in trained.items():
                bunches[arm]=core.probability_bundles(
                    'RF',model,mats[1][:,ix],ys[1],mats[2][:,ix],
                    mats[3][:,ix],mats[4][:,ix])
                iso=next((b for b in bunches[arm] if b.name=='isotonic'),None)
                if iso is None:
                    raise ValueError(f'Isotonic calibration not eligible {sp.fold_id}')
                predictions[arm]=(np.asarray(iso.test_prob)>=0.5).astype(np.int8)
            a=predictions['Strict-Clean'];b=predictions['Strict-Clean-RFE-TopK']
            same=int(np.sum(a==b))
            row=dict(protocol=sp.protocol,fold_id=sp.fold_id,n_test=len(ys[4]),
                     n_attack=int(ys[4].sum()),same_predictions=same,
                     agreement_fraction=same/len(ys[4]),different_predictions=len(ys[4])-same,
                     changed_benign=int(np.sum((a!=b)&(ys[4]==0))),
                     changed_attack=int(np.sum((a!=b)&(ys[4]==1))))
            row.update({'full_'+k:v for k,v in confusion(ys[4],a).items()})
            row.update({'rfe_'+k:v for k,v in confusion(ys[4],b).items()})
            stability.append(row)
            if run_alerts and mode=='sliding' and sp.candidate_number in range(7,14):
                testframe=df.iloc[sp.test_idx]
                times=pd.to_numeric(testframe[tm],errors='raise').to_numpy(dtype=float)
                iso=next(q for q in bunches['Strict-Clean-RFE-TopK'] if q.name=='isotonic')
                rules=core.threshold_rules(ys[2],iso.val_prob,ys[3],iso.threshold_pool_prob,cfg)
                for rule_name in ('fixed_0.5','fpr_0.5pct'):
                    rule=next(q for q in rules if q['threshold_name']==rule_name)
                    prediction=np.asarray(iso.test_prob)>=float(rule['threshold_value'])
                    cm=confusion(ys[4],prediction.astype(np.int8))
                    for scope,which in [('all_alert_flows',prediction),
                                        ('false_positive_flows',prediction & (ys[4]==0))]:
                        count,groups,n_pairs=alert_groups(testframe,which,ys[4],src,dst,times)
                        alerts.append(dict(fold=sp.fold_id,rule=rule_name,scope=scope,
                                           n_alert_flows=count,n_proxy_groups=groups,
                                           n_directed_host_pairs=n_pairs,
                                           **{k:cm[k] for k in ('tn','fp','fn','tp')}))
            if run_ties:
                for arm in ('Strict-Clean','Strict-Clean-RFE-TopK'):
                    for bundle in bunches[arm]:
                        rules=core.threshold_rules(ys[2],bundle.val_prob,ys[3],
                                                   bundle.threshold_pool_prob,cfg)
                        for rule in rules:
                            threshold=float(rule['threshold_value'])
                            scores=np.asarray(bundle.test_prob)
                            eq=int(np.count_nonzero(scores==threshold))
                            ge=(scores>=threshold).astype(np.int8)
                            gt=(scores>threshold).astype(np.int8)
                            first=confusion(ys[4],ge);second=confusion(ys[4],gt)
                            ties.append(dict(protocol=sp.protocol,fold_id=sp.fold_id,
                                variant=arm,model='RF',probability_source=bundle.name,
                                threshold_name=rule['threshold_name'],threshold_value=threshold,
                                threshold_method_eligible=rule['threshold_method_eligible'],
                                n_test=len(scores),boundary_n=eq,
                                **{'ge_'+k:v for k,v in first.items()},
                                **{'gt_'+k:v for k,v in second.items()}))
                    # LinearSVM uses the original study's sigmoid-only calibration.
                    model=core.make_model('LinearSVM',cfg,
                                          10200 if arm=='Strict-Clean' else 10210)
                    ix=trained[arm][1]
                    model.fit(mats[0][:,ix],ys[0])
                    for bundle in core.probability_bundles('LinearSVM',model,
                            mats[1][:,ix],ys[1],mats[2][:,ix],mats[3][:,ix],mats[4][:,ix]):
                        for rule in core.threshold_rules(ys[2],bundle.val_prob,
                                                         ys[3],bundle.threshold_pool_prob,cfg):
                            threshold=float(rule['threshold_value'])
                            scores=np.asarray(bundle.test_prob)
                            eq=int(np.count_nonzero(scores==threshold))
                            a=confusion(ys[4],(scores>=threshold).astype(np.int8))
                            b=confusion(ys[4],(scores>threshold).astype(np.int8))
                            ties.append(dict(protocol=sp.protocol,fold_id=sp.fold_id,
                                variant=arm,model='LinearSVM',probability_source=bundle.name,
                                threshold_name=rule['threshold_name'],threshold_value=threshold,
                                threshold_method_eligible=rule['threshold_method_eligible'],
                                n_test=len(scores),boundary_n=eq,
                                **{'ge_'+k:v for k,v in a.items()},
                                **{'gt_'+k:v for k,v in b.items()}))
                    del model
            if run_stability:save_csv(stability,out/'R2_5_Paired_Prediction_Agreement.csv')
            if run_alerts:save_csv(alerts,out/'R2_7_Directed_Host_60s_Alert_Proxy.csv')
            if run_ties:save_csv(ties,out/'R2_8_Threshold_Equality_Sensitivity.csv')
            del mats,trained,predictions,bunches;gc.collect()

def feature_group_stability(frozen, feature_map_path, out):
    grouping=pd.read_csv(feature_map_path,encoding='utf-8-sig')
    groups=dict(zip(grouping['feature'].astype(str),grouping['functional_group'].astype(str)))
    rows=[]
    for mode in ('sliding_temporal','expanding_temporal'):
        previous=None
        frame=frozen[(frozen['protocol']==mode)&
                     (frozen['base_variant']=='Strict-Clean')]
        keys=frame[['fold_id','candidate_number']].drop_duplicates().sort_values('candidate_number')
        for record in keys.itertuples(index=False):
            f=frame[frame['fold_id']==record.fold_id]
            selected=set(f.loc[f['selected'].astype(str).str.lower()=='true','feature'].astype(str))
            if len(selected)!=15:raise ValueError('Frozen RFE selection cardinality mismatch')
            row=dict(protocol=mode,fold=record.fold_id,selected_n=len(selected),
                     jaccard_previous=(len(selected & previous)/len(selected|previous)
                                       if previous is not None else float('nan')),
                     newly_selected_features=(len(selected-previous) if previous is not None else float('nan')))
            for key in sorted(set(groups.values())):row['group_'+key]=sum(groups.get(name)==key for name in selected)
            if any(feature not in groups for feature in selected):
                raise ValueError('A retained feature does not have a functional group')
            rows.append(row);previous=selected
    save_csv(rows,out/'R2_5_Functional_Feature_Group_Stability.csv')


def run(dataset, output, frozen_path, experiments, feature_map_path=None):
    cfg,df,cols=prepare(dataset,output)
    frozen=pd.read_csv(frozen_path)
    if 'stability' in experiments:
        if feature_map_path is None:raise ValueError('Functional feature taxonomy required')
        feature_group_stability(frozen,feature_map_path,output)
    if 'label-delay' in experiments:label_delay(cfg,df,cols,output)
    if 'matched-cost' in experiments:matched_cost(cfg,df,cols,frozen,output)
    selected=bool({'stability','alert-groups','boundary-mass'}&set(experiments))
    if selected:
        predictions_stability_and_ties(cfg,df,cols,frozen,output,
            'stability' in experiments,'alert-groups' in experiments,
            'boundary-mass' in experiments)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--feature-map',type=Path)
    p.add_argument('--experiments',nargs='+',required=True,
                   choices=('label-delay','matched-cost','stability','alert-groups','boundary-mass'))
    a=p.parse_args()
    run(a.data,a.output,a.frozen,a.experiments,a.feature_map)

if __name__=='__main__':main()
