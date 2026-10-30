#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IJICS-359541: strictly bounded reviewer follow-up experiments.

Run the explicit --task one at a time. FROZEN source and data hashes are mandatory.
No changed dataset, model search, alternative partition sweep, or test-set k tuning.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import gc, hashlib, json, os, sys, time, traceback
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, confusion_matrix, average_precision_score,
                             roc_auc_score, brier_score_loss)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# This is the exact auditor distributed in the previously verified read-only kit.
import IJICS_359541_R1_11_R2_9_ReadOnly_Audits as audit

ORIGINAL_SOURCE = audit.SOURCE_SHA
ORIGINAL_DATA = audit.DATA_SHA
NROWS=2350609
NATTACK=127679
W8A=87190
W8B=43399
MAX_FEATURES=6  # identical candidate count for original RF-46 / RF-15 arms
ALLOWED_TASKS=("plan", "r1-6", "r1-11", "r2-3", "r2-9", "self-test", "pack")

def jwrite(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str,allow_nan=False),encoding="utf-8")

def sha_array(ix):
    ix=np.asarray(ix,dtype=np.int64)
    return hashlib.sha256(ix.tobytes()).hexdigest()

def roles(split):
    return {name:sha_array(getattr(split,name)) for name in (
        "train_idx","calibration_idx","threshold_idx","threshold_pool_idx","test_idx")}

def same_roles(a,b):
    return roles(a)==roles(b)

def get_windows(n,cut):
    old=[np.asarray(x,dtype=np.int64) for x in np.array_split(np.arange(n,dtype=np.int64),18)]
    assert len(old)==18 and len(old[8])==130589 and old[8][0]<cut<=old[8][-1]
    new=old[:8]+[old[8][old[8]<cut],old[8][old[8]>=cut]]+old[9:]
    names=[f"W{x}" for x in range(8)]+["W8a","W8b"]+[f"W{x}" for x in range(9,18)]
    assert len(new)==len(names)==19
    assert len(new[8])==W8A and len(new[9])==W8B
    assert sum(map(len,new))==n
    assert np.array_equal(np.concatenate(new),np.arange(n,dtype=np.int64))
    return old,new,names

def make_new_splits(df,mod,cfg,new,names,mode):
    """Use frozen past-only support function; Expanding retains W8 block when whole in history.
    A W8a/W8b split must not change support-reserve slices of complete W8 history.
    """
    out=[]
    for cal_w in range(cfg.ROLLING_TRAIN_WINDOWS,len(new)-2):
        thr_w=cal_w+1; test_w=cal_w+2
        train_ids=list(range(cal_w)) if mode=="expanding" else list(range(cal_w-3,cal_w))
        history=[];j=0
        while j<len(train_ids):
            idx=train_ids[j]
            if mode=="expanding" and idx==8 and j+1<len(train_ids) and train_ids[j+1]==9:
                history.append(np.concatenate((new[8],new[9])))
                j+=2
            else:
                history.append(new[idx]);j+=1
        (train,cal,pool,cal_strategy,thr_strategy,cal_reserve,thr_reserve) = (
            mod._build_past_only_support_pools(df,history,new[cal_w],new[thr_w],cfg))
        if mod._class_count(df,train)<2:
            continue
        cand=cal_w-2
        out.append(mod.DataSplit(
            protocol=f"{mode}_session_aligned",fold_id=f"{mode}_alt_{cand:02d}",
            candidate_number=cand,train_idx=train,calibration_idx=cal,
            threshold_idx=new[thr_w],threshold_pool_idx=pool,test_idx=new[test_w],
            calibration_strategy=cal_strategy,threshold_pool_strategy=thr_strategy,
            calibration_reserve_rows=cal_reserve,threshold_reserve_rows=thr_reserve,
            train_window_ids=tuple(train_ids),calibration_window_id=cal_w,
            threshold_window_id=thr_w,test_window_id=test_w,training_mode=mode))
    return out

def original_window_name(w):return f"W{w}"

def plan_partition(df,mod,cfg,old,new,names):
    output=[]; map_old={};map_new={}
    for mode in ("sliding","expanding"):
        o=mod.rolling_splits(df,cfg,training_mode=mode)
        n=make_new_splits(df,mod,cfg,new,names,mode)
        # Testing row IDs, not fold sequence number, define matching.
        oldby={sha_array(z.test_idx):z for z in o}
        newby={sha_array(z.test_idx):z for z in n}
        map_old[mode]=oldby;map_new[mode]=newby
        for sp in n:
            tid=names[sp.test_window_id]
            oldmatch=oldby.get(sha_array(sp.test_idx))
            eq=bool(oldmatch is not None and same_roles(sp,oldmatch))
            if mode=="expanding" and eq is False and oldmatch is not None:
                # Complete W8 in historical model support should remain exactly unchanged
                # UNLESS calibration/threshold/test uses a separate W8 half.
                original_hist=set(range(oldmatch.calibration_window_id))
                new_hist=set(range(sp.calibration_window_id))
                if 8 in original_hist and {8,9}.issubset(new_hist):
                    assert sha_array(sp.train_idx)==sha_array(oldmatch.train_idx), (
                        f"Expanding complete-W8 train history unexpectedly changed for {tid}")
            yy=df.iloc[sp.test_idx]["__binary_label__"].to_numpy(dtype=np.int8)
            caly=df.iloc[sp.calibration_idx]["__binary_label__"].to_numpy(dtype=np.int8)
            output.append(dict(mode=mode,partition="session_aligned",test_window=tid,
                fold_id=sp.fold_id,original_fold_id=oldmatch.fold_id if oldmatch else "",
                same_test_as_original=bool(oldmatch is not None),
                all_roles_identical=eq,affected=not eq,
                old_test_sha256=sha_array(oldmatch.test_idx) if oldmatch else "",
                new_test_sha256=sha_array(sp.test_idx),
                n_train=int(len(sp.train_idx)),n_calibration=int(len(sp.calibration_idx)),
                n_threshold=int(len(sp.threshold_idx)),n_threshold_pool=int(len(sp.threshold_pool_idx)),
                n_test=int(len(sp.test_idx)),n_test_attack=int(yy.sum()),
                eligible_discrimination=bool(np.unique(yy).size==2),
                eligible_isotonic=bool(np.unique(caly).size==2 and np.unique(yy).size==2),
                original_published_seven=bool(oldmatch is not None and
                        oldmatch.fold_id in ([f"sliding_{i:02d}" for i in range(7,14)]
                                              if mode=="sliding" else
                                              [f"expanding_{i:02d}" for i in range(7,14)]))))
    return pd.DataFrame(output),map_old,map_new

def get_base_features(mod,df,cfg):
    columns=mod.detect_columns(df,cfg)
    clean,ttl,strict_present,leak=mod.leakage_audit(df,columns,cfg)
    bad={mod.normalize_name(c) for c in cfg.STRICT_DROP_FEATURES}
    cols=[c for c in clean if mod.normalize_name(c) not in bad]
    if len(cols)!=46:
        raise AssertionError(f"Strict-clean feature count {len(cols)} !=46; STOP")
    return cols

def load_all(dataset,source,out):
    for label,path,expected in (("dataset",dataset,ORIGINAL_DATA),("source",source,ORIGINAL_SOURCE)):
        if not path.is_file():raise FileNotFoundError(f"Missing {label} {path}")
        got=audit.digest(path)
        if got!=expected:raise AssertionError(f"{label} SHA256 mismatch: {got}")
        print(f"{label} SHA-256 verified",flush=True)
    mod=audit.import_source(source)
    cfg=replace(mod.CFG, DATA_PATHS=(str(dataset.resolve()),),EXECUTION_MODE="preflight",
                N_TEMPORAL_WINDOWS=18,MAX_ROWS_PER_FILE=None, DROP_GLOBAL_EXACT_DUPLICATES=True,
                ENFORCE_TEMPORAL_DUPLICATE_ISOLATION=False, SAVE_DETAILED_PLOTS=False)
    out.mkdir(parents=True,exist_ok=True)
    df=mod.load_dataset(cfg,out/"load.log")
    assert len(df)==NROWS
    detected=mod.detect_columns(df,cfg)
    df=mod.derive_labels(df,detected)
    df,reason=mod.prepare_order(df,detected["time"],out/"load.log")
    assert reason=="time_column:FLOW_START_MILLISECONDS"
    assert int(df["__binary_label__"].sum())==NATTACK
    win,seg,names,sessions,timev,cut,gap,span = audit.windows_and_sessions(df)
    old,new,names2=get_windows(len(df),cut)
    assert names==names2
    cols=get_base_features(mod,df,cfg)
    return mod,cfg,df,old,new,names,cols,cut

def train_arrays(mod,df,sp,cols):
    t0=time.perf_counter()
    idxs=(sp.train_idx,sp.calibration_idx,sp.threshold_idx,sp.threshold_pool_idx,sp.test_idx)
    raw=[df.iloc[idx] for idx in idxs]
    labels=[d["__binary_label__"].to_numpy(dtype=np.int8) for d in raw]
    xx=mod.fit_transform(*raw,cols)
    mats=tuple(xx[:5]);names=list(xx[5]);prep_sec=time.perf_counter()-t0
    return raw,labels,mats,names,prep_sec

def choose_features(mod,xtr,ytr,names,cfg):
    t0=time.perf_counter()
    rfe=mod.run_rfe(xtr,ytr,names,cfg,
        sampling_mode=cfg.FULL_RFE_SAMPLING_MODE,
        sample_seed=cfg.RFE_FIXED_SAMPLE_SEED+2,
        model_seed=cfg.RFE_FIXED_MODEL_SEED+2)
    return rfe,time.perf_counter()-t0

def model_offset(which):return 10200 if which==46 else 10210

def new_rf(mod,cfg,nfeatures,which,max_candidates=MAX_FEATURES):
    # For candidate-matched 46-vs-15 controls, hold the RF estimator seed
    # identical. Use the original Strict-Clean-RFE-TopK RF seed in every arm.
    model=mod.make_model("RF",cfg,offset=10210)
    model.set_params(max_features=max_candidates,random_state=cfg.RANDOM_STATE+10210)
    if nfeatures<max_candidates:raise AssertionError("Fewer features than matched candidates")
    return model

def calc_metrics(mod,y,prob,tau):
    pred=(prob>=tau).astype(np.int8)
    cm=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=map(int,cm.ravel())
    ans=dict(macro_f1=float(f1_score(y,pred,average="macro",zero_division=0)),
        fp=fp,fn=fn,tp=tp,tn=tn,fpr=float(fp/(fp+tn)) if (fp+tn) else None,
        pr_auc=float(average_precision_score(y,prob)) if np.unique(y).size==2 else None,
        roc_auc=float(roc_auc_score(y,prob)) if np.unique(y).size==2 else None,
        brier=float(brier_score_loss(y,prob)))
    ans["ece_15"]=float(mod.ece_mce(y,prob,bins=15)[0])
    return ans,pred

def calibrated(mod,model,xcal,ycal,xval,xpool,xtest):
    cal=mod.fit_calibrator(model,xcal,ycal,"isotonic")
    if cal is None:raise AssertionError("No two-class calibration support")
    out=[]
    for x in (xval,xpool,xtest):
        a=mod.positive_probability(cal,x)
        if not np.isfinite(a).all():raise AssertionError("Nonfinite probabilities")
        out.append(a)
    return cal,out

def evaluate_prepared(mod,cfg,mats,labels,select,which,protocol,fold_id,window,arm="base",
                      extra=None,max_candidates=MAX_FEATURES):
    # Shared exact rows/labels and frozen RFE feature list; a 3 vs 3 RF split count not silently assumed.
    if extra is None: extra={}
    if select is None:select=np.arange(mats[0].shape[1])
    x=[np.ascontiguousarray(a[:,select]) for a in mats]
    if len(select)<max_candidates:raise AssertionError("RF candidate count exceeds feature count")
    ytr,ycal,yval,ypool,ytest=labels
    model=new_rf(mod,cfg,len(select),which,max_candidates=max_candidates)
    t0=time.perf_counter();model.fit(x[0],ytr);fit_sec=time.perf_counter()-t0
    cal,pp=calibrated(mod,model,x[1],ycal,x[2],x[3],x[4])
    pv,pool,test=pp
    rules=mod.threshold_rules(yval,pv,ypool,pool,cfg)
    required={"fixed_0.5","youden","mu_sigma_raw_k_0.10","fpr_0.5pct",
              "fpr_1.0pct","fpr_2.0pct","fpr_histpool_0.5pct"}
    out=[];fixed_pred=None
    for rule in rules:
        if rule["threshold_name"] not in required:continue
        tau=float(rule["threshold_value"])
        metrics,pred=calc_metrics(mod,ytest,test,tau)
        out.append(dict(partition=protocol,fold_id=fold_id,test_window=window,
            arm=arm,feature_set=which,max_features=max_candidates,
            rf_random_state_common=int(cfg.RANDOM_STATE+10210),n_train=len(ytr),
            n_test=len(ytest),n_attack_test=int(ytest.sum()),n_selected=len(select),
            rf_fitting_seconds=fit_sec,threshold_name=rule["threshold_name"],
            threshold_value=tau,threshold_selection_status=rule["threshold_selection_status"],
            threshold_method_eligible=bool(rule["threshold_method_eligible"]),
            **metrics,**extra))
        if rule["threshold_name"]=="fixed_0.5":fixed_pred=pred
    return out,model,cal,fixed_pred,test

def one_fold(df,mod,cfg,sp,cols,arm_prefix,window,out_dir,plan_signature):
    """Matched six-candidate control: Strict-Clean 46 vs frozen-source RFE 15.
    A separate structured file is written only when both have finished.
    """
    raw,labels,xx,names,prep_sec=train_arrays(mod,df,sp,cols)
    if xx[0].shape[1]!=46:raise AssertionError("Expected 46 transformed features")
    rfe,selection_sec=choose_features(mod,xx[0],labels[0],names,cfg)
    if len(rfe.selected)!=15:raise AssertionError("RFE did not select 15 features")
    results=[]
    for size,select in ((46,None),(15,rfe.selected)):
        rr,model,cal,pred,p = evaluate_prepared(
            mod,cfg,xx,labels,select,size,arm_prefix,sp.fold_id,window,
            extra=dict(feature_selection_plus_rf_fitting_seconds=None,
                       preprocessing_seconds=prep_sec,rfe_seconds=selection_sec,
                       role_signature=plan_signature))
        for z in rr:
            z["feature_selection_plus_rf_fitting_seconds"] = z["rf_fitting_seconds"]+(selection_sec if size==15 else 0.0)
        results+=rr
        del model,cal,pred,p
        gc.collect()
    del raw,xx,labels
    gc.collect()
    return pd.DataFrame(results),list(rfe.selected_names)


def do_r16(df,mod,cfg,old,new,names,cols,out):
    plan,oldmap,newmap=plan_partition(df,mod,cfg,old,new,names)
    plan.to_csv(out/"R1_6_Exact_Role_and_Eligibility_Plan.csv",index=False)
    rows=[]
    for mode in ("sliding","expanding"):
        o=oldmap[mode];nn=newmap[mode]
        new_folds={sp.fold_id:sp for sp in nn.values()}
        sub=plan[plan["mode"]==mode]
        for idx,pr in sub.iterrows():
            if not bool(pr.eligible_isotonic):
                print("[R1-6] recorded in eligibility manifest, skipped non-eligible",mode,str(pr.test_window),flush=True)
                continue
            s=new_folds[str(pr.fold_id)]
            w=str(pr.test_window)
            original=o.get(sha_array(s.test_idx))
            cases=[("session_aligned",s)]
            if original is not None and not pr.all_roles_identical:
                # Run the original matched-control only if the exact roles changed.
                cases=[("original_equalcount",original),("session_aligned",s)]
            flds={}
            for partition,sp in cases:
                sig="|".join(roles(sp).values())
                run_id=hashlib.sha256(sig.encode()).hexdigest()[:16]
                stem=f"R1_6_{mode}_{w}_{partition}_{run_id}"
                target=out/f"{stem}.csv"
                if target.is_file():
                    current=pd.read_csv(target)
                    if not len(current) or not (current.role_signature==sig).all():
                        raise AssertionError("Cached role signature mismatch; STOP")
                    print("[R1-6] verified existing",target.name,flush=True)
                else:
                    print(f"[R1-6] fitting {mode} {w} {partition} 46 and 15",flush=True)
                    current,chosen=one_fold(df,mod,cfg,sp,cols,partition,w,out,sig)
                    current["rfe_names"]="|".join(chosen)
                    tmp=target.with_suffix(".partial")
                    current.to_csv(tmp,index=False)
                    tmp.replace(target)
                flds[partition]=current
            if original is not None and pr.all_roles_identical:
                # Exact role reuse: change the display label, not data, prediction or RF fitting result.
                flds["original_equalcount"]=flds["session_aligned"].copy()
                flds["original_equalcount"]["partition"]="original_equalcount"
                flds["original_equalcount"]["reused_exact_roles"]=True
            for partition,x in flds.items():
                x=x.copy()
                x["mode"]=mode;x["matched_test_window"]=bool(original is not None)
                x["roles_identical_across_partitions"]=bool(pr.all_roles_identical)
                x["original_published_seven"]=bool(pr.original_published_seven)
                x["eligible_isotonic"] = bool(pr.eligible_isotonic)
                x["eligible_discrimination"] = bool(pr.eligible_discrimination)
                rows.append(x)
            if len(rows):
                pd.concat(rows,ignore_index=True).to_csv(out/"R1_6_All_Fold_Results.csv",index=False)
    if not rows:raise AssertionError("No eligible folds after session split")
    res=pd.concat(rows,ignore_index=True)
    # Explicit three-set eligibility and same-test support on BOTH protocols.
    fixed=res[(res.threshold_name=="fixed_0.5") & res.eligible_isotonic]
    csets={}
    for partition in ("original_equalcount","session_aligned"):
        q=fixed[fixed.partition==partition]
        s=set(q[q["mode"]=="sliding"].test_window.astype(str))
        e=set(q[q["mode"]=="expanding"].test_window.astype(str))
        csets[partition]=s&e
    c_matched=csets["original_equalcount"] & csets["session_aligned"]
    # Record the full sets, not only their sizes; W8a/b cannot be treated as W8.
    all_w=set(csets["original_equalcount"])|set(csets["session_aligned"])
    comparability=pd.DataFrame([dict(
        test_window=w,in_original_common=w in csets["original_equalcount"],
        in_new_common=w in csets["session_aligned"],in_cross_partition_common=w in c_matched,
        eligible_for_published_seven_comparison=(w in c_matched and w in
            {"W11","W12","W13","W14","W15","W16","W17"}))
        for w in sorted(all_w,key=lambda x:(int(x[1:].rstrip("ab")),x))])
    comparability.to_csv(out/"R1_6_Original_New_and_Matched_Eligible_Test_Sets.csv",index=False)
    paired=fixed[fixed.test_window.isin(c_matched)].copy()
    paired.to_csv(out/"R1_6_Only_CrossPartition_Common_Eligible_PerFold.csv",index=False)
    # Aggregate descriptive means only over an identical window set.
    # Inferential independence of chronologically overlapping folds is NOT assumed.
    summary=paired.groupby(["partition","mode","feature_set"],as_index=False).agg(
        n_unique_test_windows=("test_window","nunique"),
        mean_macro_f1=("macro_f1","mean"),
        sum_fp=("fp","sum"),sum_fn=("fn","sum"),
        total_benign=("tn","sum"),
        sum_rf_fitting_seconds=("rf_fitting_seconds","sum"),
        sum_rfe_plus_rf_fitting_seconds=("feature_selection_plus_rf_fitting_seconds","sum"),
        mean_model_train_rows=("n_train","mean"))
    summary.to_csv(out/"R1_6_Descriptive_Comparison_Common_Test_Only.csv",index=False)
    old7=paired[paired.test_window.isin({"W11","W12","W13","W14","W15","W16","W17"})]
    old7.to_csv(out/"R1_6_Published_Seven_Intersection_Only.csv",index=False)
    jwrite(out/"R1_6_Eligibility_Sets.json",dict(
        original_common=sorted(csets["original_equalcount"]),
        session_aligned_common=sorted(csets["session_aligned"]),
        cross_partition_common=sorted(c_matched),
        published_seven_intersection=sorted(c_matched&{"W11","W12","W13","W14","W15","W16","W17"}),
        note="Original RF-46 vs RF-15 defaults were UNEQUAL sqrt candidates; these sensitivity controls instead use 6 vs 6 and may only be compared against other matched-candidate controls."))
    jwrite(out/"R1_6_Completion.json",dict(status="completed" if len(c_matched) else "no_cross_partition_common_eligible",n_fit_result_rows=int(len(res)),
        tested_alt_partitions=1,original_windows=18,new_windows=19,
        note="Comparisons require identical test row IDs. Model baseline uses matched 6/6 split candidates, not default sqrt. RF fitting and RFE fitting times are separate. Within-history availability reserves preserved when W8 halves both fall in Expanding training."))
    return res


def do_r111(df,mod,cfg,old,cols,out,k):
    if k!=8:
        raise AssertionError("Prespecified difficulty drop-k=8; do not optimize on final test")
    folds=mod.rolling_splits(df,cfg,"sliding")
    fold=next(z for z in folds if z.fold_id=="sliding_09")
    assert fold.test_window_id==13
    raw,labels,xx,names,prep=train_arrays(mod,df,fold,cols)
    rfe,rfe_sec=choose_features(mod,xx[0],labels[0],names,cfg)
    base_indices=np.asarray(rfe.selected,dtype=int)
    # RFE selection is training-only; importance ranking is also training-only.
    order=sorted(base_indices,key=lambda i:(-float(rfe.importances[i]),str(names[i])))
    removed=order[:k]
    retained=[i for i in base_indices if i not in set(removed)]
    assert len(retained)==7
    freeze=dict(k=k,fold="S9",train_test_roles=roles(fold),
         selected_features=[names[i] for i in base_indices],
         removed_features=[names[i] for i in removed],
         retained_features=[names[i] for i in retained],
         selection_source="Frozen-source RFE on historical model training rows and RF importances",
         criterion="Fixed k=8, predeclared; actual test difficulty reported without k search",
         baseline_max_features=3,ablation_max_features=3)
    jwrite(out/"R1_11_Frozen_Difficulty_Design_BEFORE_MODEL_TEST.json",freeze)
    rows=[];preds={}
    for arm,indices in (("original_15",base_indices),("remove_top8",retained)):
        z,model,cal,pr,piso=evaluate_prepared(mod,cfg,xx,labels,np.asarray(indices),15,
                    "original_18window",fold.fold_id,"W13",arm=arm,
                    extra=dict(k_dropped=(0 if arm=="original_15" else k)),max_candidates=3)
        p_raw=mod.positive_probability(model,xx[4][:,indices])
        pm_raw,_=calc_metrics(mod,labels[4],p_raw,0.5)
        pm_iso,_=calc_metrics(mod,labels[4],piso,0.5)
        for row in z:
            row["pr_auc_raw"]=pm_raw["pr_auc"]
            row["ece_raw"]=pm_raw["ece_15"]
            row["brier_raw"]=pm_raw["brier"]
            row["n_calibration_changed_fixed05"]=int(np.count_nonzero((p_raw>=0.5)!=(piso>=0.5)))
            row["pr_auc_isotonic"]=pm_iso["pr_auc"]
            row["observed_difficulty_not_target_tuned"]=True
        rows+=z;preds[arm]=np.asarray(pr,dtype=np.int8)
        del model,cal
        gc.collect()
    # pr from evaluate_prepared fixed0.5, not last method.
    pd.DataFrame(rows).to_csv(out/"R1_11_Difficulty_Threshold_and_Calibration.csv",index=False)
    jwrite(out/"R1_11_Completion.json",dict(status="completed",k_frozen_before_test=8,
        actual_removed_features=freeze["removed_features"],
        n_original_fixed_05_vs_ablated_flips=int(np.count_nonzero(preds["original_15"]!=preds["remove_top8"])),
        note="Single prespecified ablation; target PR-AUC~0.95 is not guaranteed and no test-set tuning used. Feature removal does not by itself prove attack-class overlap."))


def value_masks_on_raw(rows):
    """Fixed one indicator for recorded-zero duration / undefined rates, no model labels used."""
    v=rows["FLOW_DURATION_MILLISECONDS"]
    return v.eq(0).to_numpy(dtype=np.float32)

def impute_sentinel(dfpart):
    # Uses the same physical negative sentinel for rate columns whose finite values
    # on training support must all be >=0. No fitting on future test data.
    result=dfpart.copy()
    for col in ("SRC_TO_DST_SECOND_BYTES","DST_TO_SRC_SECOND_BYTES"):
        raw=pd.to_numeric(result[col],errors="coerce").to_numpy(dtype=np.float64)
        mask=~np.isfinite(raw) | (np.abs(raw)>1e30)
        raw[mask]=-1.0
        result[col]=raw
    return result

def do_r29(df,mod,cfg,old,cols,out):
    fold=next(x for x in mod.rolling_splits(df,cfg,"sliding") if x.fold_id=="sliding_09")
    assert fold.test_window_id==13
    raw,labels,xx,names,prep=train_arrays(mod,df,fold,cols)
    rfe,rfe_sec=choose_features(mod,xx[0],labels[0],names,cfg)
    chosen=np.asarray(rfe.selected,dtype=int)
    if "SRC_TO_DST_SECOND_BYTES" not in [names[i] for i in chosen]:
        raise AssertionError("Critical affected rate not selected in frozen S9 RFE; STOP")
    for c in ("SRC_TO_DST_SECOND_BYTES","DST_TO_SRC_SECOND_BYTES"):
        tr=raw[0][c].to_numpy(dtype=np.float64)
        valid=tr[np.isfinite(tr)&(np.abs(tr)<=1e30)]
        if np.any(valid<0):
            raise AssertionError("Sentinel -1 not guaranteed outside training valid rate range")
    orig=tuple(np.ascontiguousarray(x[:,chosen]) for x in xx)
    indicator=tuple(value_masks_on_raw(r) for r in raw)
    # A single shared bit, because masks were proved exactly coincident for both rates.
    with_indicator=tuple(np.column_stack((x,indicator[i])).astype(np.float32)
                         for i,x in enumerate(orig))
    alter_raw=[impute_sentinel(r) for r in raw]
    x_alt=mod.fit_transform(*alter_raw,cols)
    if list(x_alt[5])!=names:
        raise AssertionError("C preprocessing altered frozen feature names/order; STOP")
    alt=tuple(np.ascontiguousarray(x[:,chosen]) for x in x_alt[:5])
    jwrite(out/"R2_9_PreRegistered_ThreeArm_Config.json",dict(
        fold="S9",test_window="W13",role_signatures=roles(fold),
        selected_features=[names[i] for i in chosen],
        arms=["A_original_median","B_original_plus_one_indicator","C_sentinel_minus1"],
        nonfinite_handling="Frozen original sanitizer for A/B; replace undefined rate with -1 before preprocessing for C",
        matched_max_features=3,indicator_definition="FLOW_DURATION_MILLISECONDS==0 recorded strictly before fitting, true on both originally nonfinite rate masks",
        sentinel_value=-1.0,indicator_append_after_frozen_RFE=True,
        note="No reuse of test labels for imputation or feature choice"))
    rows=[];preds={}
    for arm,matrices in (("A_original_median",orig),
                         ("B_original_plus_one_indicator",with_indicator),
                         ("C_sentinel_minus1",alt)):
        print("[R2-9] fitting",arm,flush=True)
        z,model,cal,pred,p=evaluate_prepared(mod,cfg,matrices,labels,None,15,
                            "original_18window",fold.fold_id,"W13",arm=arm,
                            extra=dict(n_original_selected_features=len(chosen),
                                       n_model_features=matrices[0].shape[1]),max_candidates=3)
        rows+=z;preds[arm]=pred
        del model,cal,pred,p
        gc.collect()
        pd.DataFrame(rows).to_csv(out/"R2_9_Three_Arm_Results.csv",index=False)
    flip={}
    for arm in ("B_original_plus_one_indicator","C_sentinel_minus1"):
        flip[arm]=int(np.count_nonzero(preds[arm]!=preds["A_original_median"]))
    jwrite(out/"R2_9_Completion.json",dict(status="completed",test_flips_vs_original_fixed_05=flip,
        processed_fold="S9",selected_15_frozen_once=True,
        caveat="Feature additions change input dimension; matched RF candidate count is fixed at six. Inference not proof of cause."))
    return pd.DataFrame(rows)

def get_peak_rss_bytes():
    """Process peak working set (Windows) or ru_maxrss (Linux); scoped to this process."""
    if os.name=="nt":
        import ctypes
        from ctypes import wintypes
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD),
                ("PeakWorkingSetSize",ctypes.c_size_t),("WorkingSetSize",ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage",ctypes.c_size_t),
                ("QuotaPagedPoolUsage",ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage",ctypes.c_size_t),
                ("PagefileUsage",ctypes.c_size_t),("PeakPagefileUsage",ctypes.c_size_t)]
        proc=ctypes.WinDLL("kernel32",use_last_error=True).GetCurrentProcess
        proc.restype=wintypes.HANDLE
        psapi=ctypes.WinDLL("psapi",use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
        pm=PROCESS_MEMORY_COUNTERS();pm.cb=ctypes.sizeof(pm)
        if not psapi.GetProcessMemoryInfo(proc(),ctypes.byref(pm),pm.cb):
            raise OSError("GetProcessMemoryInfo failed; do not infer peak RSS")
        return int(pm.PeakWorkingSetSize)
    if sys.platform.startswith("linux"):
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    raise RuntimeError("Peak RSS measurement not supported on this OS")

def preprocessor_with_retained_state(mod,raw,cols):
    # Exact structure used by the original `fit_transform`, retaining the
    # training-fitted ColumnTransformer so its footprint can be measured.
    tr=raw[0]
    numeric=[c for c in cols if pd.api.types.is_numeric_dtype(tr[c])]
    categorical=[c for c in cols if c not in numeric]
    transformers=[]
    if numeric:
        numeric_pipe=Pipeline([
            ("sanitize",FunctionTransformer(mod.sanitize_numeric_block,validate=False,
                                              feature_names_out="one-to-one")),
            ("imputer",SimpleImputer(strategy="median",keep_empty_features=True)),
            ("scaler",StandardScaler()),
        ])
        transformers.append(("num",numeric_pipe,numeric))
    if categorical:
        cat_pipe=Pipeline([
            ("imputer",SimpleImputer(strategy="most_frequent",keep_empty_features=True)),
            ("encoder",OrdinalEncoder(handle_unknown="use_encoded_value",
               unknown_value=-1,encoded_missing_value=-1)),
            ("scaler",StandardScaler())
        ])
        transformers.append(("cat",cat_pipe,categorical))
    pre=ColumnTransformer(transformers,remainder="drop",sparse_threshold=0.0,
                          verbose_feature_names_out=False)
    t0=time.perf_counter()
    mats=[mod.finite_float32(pre.fit_transform(raw[0].loc[:,cols]))]
    for part in raw[1:]:
        mats.append(mod.finite_float32(pre.transform(part.loc[:,cols])))
    return pre,mats,list(pre.get_feature_names_out()),time.perf_counter()-t0

def do_r23_resource(df,mod,cfg,old,cols,out,arm):
    import joblib
    if arm not in (15,46):raise AssertionError("Only 15 or 46 arms permitted")
    fold=next(x for x in mod.rolling_splits(df,cfg,"sliding") if x.fold_id=="sliding_09")
    assert fold.test_window_id==13
    peak_after_load=get_peak_rss_bytes()
    raw=[df.iloc[getattr(fold,k)] for k in
         ("train_idx","calibration_idx","threshold_idx","threshold_pool_idx","test_idx")]
    labels=[r["__binary_label__"].to_numpy(dtype=np.int8) for r in raw]
    pre,mats,names,prepsec=preprocessor_with_retained_state(mod,raw,cols)
    assert len(names)==46
    peak_after_pre=get_peak_rss_bytes()
    rfe_seconds=0.0;rfe=None
    if arm==15:
        rfe,rfe_seconds=choose_features(mod,mats[0],labels[0],names,cfg)
        select=np.asarray(rfe.selected)
    else:select=np.arange(46)
    peak_after_rfe=get_peak_rss_bytes()
    x=[np.ascontiguousarray(m[:,select]) for m in mats]
    model=new_rf(mod,cfg,len(select),arm)
    t0=time.perf_counter()
    model.fit(x[0],labels[0]); fit_seconds=time.perf_counter()-t0
    peak_after_fit=get_peak_rss_bytes()
    cal=mod.fit_calibrator(model,x[1],labels[1],"isotonic")
    if cal is None:raise AssertionError("No valid two-class calibration support")
    ptest=mod.positive_probability(cal,x[4])
    met,_=calc_metrics(mod,labels[4],ptest,0.5)
    peak_after_cal=get_peak_rss_bytes()
    artifacts=out/"serialized_artifacts"
    artifacts.mkdir(parents=True,exist_ok=True)
    # Preserve fitted preprocessing, RF and calibrator file footprints separately.
    prepath=artifacts/f"S9_RF{arm}_preprocessor.joblib"
    modelpath=artifacts/f"S9_RF{arm}_model.joblib"
    calpath=artifacts/f"S9_RF{arm}_calibrator.joblib"
    joblib.dump(pre,prepath,compress=0)
    joblib.dump(model,modelpath,compress=0)
    joblib.dump(cal,calpath,compress=0)
    stored={p.name:int(p.stat().st_size) for p in (prepath,modelpath,calpath)}
    if rfe is not None:
        rankpath=artifacts/"S9_RF15_RFE_selected_and_rankings.json"
        jwrite(rankpath,dict(names=names,ranking=rfe.ranking.tolist(),
                selected_feature_names=list(rfe.selected_names),
                importances=rfe.importances.tolist()))
        stored[rankpath.name]=int(rankpath.stat().st_size)
    peak_after_artifacts=get_peak_rss_bytes()
    result={"status":"completed_resource_arm","arm":arm,"fold":"S9",
        "dataset_sha256":ORIGINAL_DATA,"source_sha256":ORIGINAL_SOURCE,
        "n_original_dataset_rows":len(df),"n_model_training_rows":len(fold.train_idx),
        "n_calibration_rows":len(fold.calibration_idx),"n_threshold_rows":len(fold.threshold_idx),
        "n_historical_threshold_pool_rows":len(fold.threshold_pool_idx),
        "n_test_rows":len(fold.test_idx),"n_features_model":len(select),
        "matched_split_candidate_count":MAX_FEATURES,
        "rf_random_state_common":int(cfg.RANDOM_STATE+10210),
        "peak_rss_bytes":int(peak_after_artifacts),
        "peak_rss_by_stage":dict(after_load=peak_after_load,after_preprocessing=peak_after_pre,
            after_rfe=peak_after_rfe,after_rf_fitting=peak_after_fit,
            after_calibration=peak_after_cal,after_serialized_artifacts=peak_after_artifacts),
        "original_dataframe_retained_bytes":int(df.memory_usage(deep=True).sum()),
        "five_role_dataframe_retained_bytes":sum(int(z.memory_usage(deep=True).sum()) for z in raw),
        "role_dataframe_retained_bytes":dict(zip(("train","calibration","raw_threshold",
           "historical_threshold_pool","test"),
           (int(z.memory_usage(deep=True).sum()) for z in raw))),
        "original_transformed_matrix_bytes":sum(int(z.nbytes) for z in mats),
        "role_original_transformed_bytes":dict(zip(("train","calibration","raw_threshold",
           "historical_threshold_pool","test"), (int(z.nbytes) for z in mats))),
        "selected_transformed_matrix_bytes":sum(int(z.nbytes) for z in x),
        "role_selected_transformed_bytes":dict(zip(("train","calibration","raw_threshold",
           "historical_threshold_pool","test"), (int(z.nbytes) for z in x))),
        "support_index_bytes":sum(int(getattr(fold,k).nbytes) for k in
             ("train_idx","calibration_idx","threshold_idx","threshold_pool_idx","test_idx")),
        "stored_artifact_bytes_total":sum(stored.values()),
        "artifact_bytes_by_name":stored,"preprocessing_seconds":prepsec,
        "rfe_seconds":rfe_seconds,"rf_fitting_seconds":fit_seconds,
        "feature_selection_plus_rf_fitting_seconds":rfe_seconds+fit_seconds,
        "model_metrics_fixed05":met,
        "limits":"Peak RSS is process-wide incl. CSV dataframe and OS allocation, and not a sum of retained component sizes. Each arm must run in a FRESH independent process on same machine/environment. Source RFE and RF hyperparameters retained except explicit 6-vs-6 matched max_features. Full end-to-end execution wall time is NOT inferred from partial timings."}
    jwrite(out/f"R2_3_RF{arm}_Resource_Manifest.json",result)
    print(json.dumps(result,indent=2),flush=True)


def selftest():
    # Test immutable role hashing and exact reuse without loading the ~577 MB source CSV.
    from types import SimpleNamespace
    a=np.array([0,1,2,3,4]);b=np.array([0,1,2,3,4]);c=np.array([0,1,2,4,3])
    assert sha_array(a)==sha_array(b) and sha_array(a)!=sha_array(c)
    arrs=[np.array([3,4],dtype=np.int64),np.array([5]),np.array([6]),np.array([7]),np.array([8])]
    xx=SimpleNamespace(**dict(zip(("train_idx","calibration_idx","threshold_idx",
               "threshold_pool_idx","test_idx"),arrs)))
    yy=SimpleNamespace(**dict(zip(("train_idx","calibration_idx","threshold_idx",
               "threshold_pool_idx","test_idx"),[np.array(z) for z in arrs])))
    assert same_roles(xx,yy)
    yy.test_idx=np.array([9]);assert not same_roles(xx,yy)
    rate=np.array([np.nan,np.inf,0.5]);mask=~np.isfinite(rate)|(np.abs(rate)>1e30)
    assert mask.tolist()==[True,True,False]
    print("SYNTHETIC CONTRACT SELF-TEST PASS (no scientific evaluation)")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",choices=ALLOWED_TASKS,required=True)
    p.add_argument("--dataset",type=Path)
    p.add_argument("--source",type=Path)
    p.add_argument("--out",type=Path,default=Path("IJICS_Remaining4_Actual_Results"))
    p.add_argument("--resource-arm",type=int,choices=(15,46))
    p.add_argument("--difficulty-drop-k",type=int,default=8)
    a=p.parse_args()
    if a.task=="self-test":selftest();return
    if a.task=="pack":
        root=a.out.resolve()
        files=[f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in
               (".csv",".json",".txt",".log")]
        if not files:raise FileNotFoundError("No results found to pack")
        zip_path=root.parent/"IJICS_359541_Remaining4_Results_FOR_REVIEW.zip"
        with ZipFile(zip_path,"w",ZIP_DEFLATED) as z:
            for f in files:z.write(f,arcname=str(f.relative_to(root)))
        with ZipFile(zip_path) as z:
            assert z.testzip() is None
        print(f"PACKED {len(files)} small result files (excluded large .joblib): {zip_path}")
        return
    if a.dataset is None or a.source is None:
        p.error("--dataset and --source mandatory for real tasks")
    out=a.out.resolve()/a.task
    mod,cfg,df,old,new,names,cols,cut=load_all(a.dataset,a.source,out)
    if a.task=="plan":
        plan,_,_=plan_partition(df,mod,cfg,old,new,names)
        plan.to_csv(out/"R1_6_Exact_Role_and_Eligibility_Plan.csv",index=False)
        print(plan[["mode","test_window","original_fold_id","affected","all_roles_identical",
                    "n_train","n_test","eligible_isotonic"]].to_string(index=False),flush=True)
        jwrite(out/"Plan_Completion.json",dict(status="completed",n_rows=len(plan),n_affected=int(plan.affected.sum())))
    elif a.task=="r1-6":
        do_r16(df,mod,cfg,old,new,names,cols,out)
    elif a.task=="r1-11":
        do_r111(df,mod,cfg,old,cols,out,a.difficulty_drop_k)
    elif a.task=="r2-9":
        do_r29(df,mod,cfg,old,cols,out)
    elif a.task=="r2-3":
        if a.resource_arm is None:
            p.error("--resource-arm 46 or --resource-arm 15 required; run separately")
        do_r23_resource(df,mod,cfg,old,cols,out,a.resource_arm)

if __name__=="__main__":
    main()
