#!/usr/bin/env python
from penquins import Kowalski
import pandas as pd
import numpy as np
import os
import pathlib
from scope.utils import sort_lightcurve, parse_load_config
from tools.featureGeneration import periodsearch
import time

BASE_DIR = pathlib.Path.cwd()
config = parse_load_config()

# use tokens specified as env vars (if exist)
kowalski_token_env = os.environ.get("KOWALSKI_INSTANCE_TOKEN")
gloria_token_env = os.environ.get("GLORIA_INSTANCE_TOKEN")
melman_token_env = os.environ.get("MELMAN_INSTANCE_TOKEN")

if kowalski_token_env is not None:
    config["kowalski"]["hosts"]["kowalski"]["token"] = kowalski_token_env
if gloria_token_env is not None:
    config["kowalski"]["hosts"]["gloria"]["token"] = gloria_token_env
if melman_token_env is not None:
    config["kowalski"]["hosts"]["melman"]["token"] = melman_token_env

timeout = config['kowalski']['timeout']

hosts = [
    x
    for x in config['kowalski']['hosts']
    if config['kowalski']['hosts'][x]['token'] is not None
]
instances = {
    host: {
        'protocol': config['kowalski']['protocol'],
        'port': config['kowalski']['port'],
        'host': f'{host}.caltech.edu',
        'token': config['kowalski']['hosts'][host]['token'],
    }
    for host in hosts
}

kowalski_instances = Kowalski(timeout=timeout, instances=instances)


def prep_for_period_finding(lcs, max_freq=288):  # freq units per day 5 min
    tme_collection = []
    labels = []
    baseline = 0
    if len(list(lcs.keys())) == 0:
        print('No light curves')
        return False
    for alert in lcs.keys():
        labels.append(alert)
        loop_lc = np.array(lcs[alert])
        t, m, e = loop_lc.T
        tt, mm, ee = sort_lightcurve(t, m, e)
        new_baseline = max(tt) - min(tt)
        if new_baseline > baseline:
            baseline = new_baseline
        new_tme_arr = np.array([tt, mm, ee])
        tme_collection += [new_tme_arr]
    fmin, fmax = 2 / baseline, max_freq

    df = 1.0 / (10 * baseline)
    nf = int(np.ceil((fmax - fmin) / df))
    freqs = fmin + df * np.arange(nf)
    freqs_to_remove = [
        [0.0025, 0.003],  # 1y
        [0.00125, 0.0015],  # 2 y
        [0.000833, 0.001],  # 3 y
        [0.000625, 0.00075],  # 4 y
        [0.0005, 0.0006],  # 5 y
        [0.005, 0.006],  # 0.5 y
        [3e-2, 4e-2],  # 30 d
        [3.95, 4.05],  # 0.25 d
        [2.95, 3.05],  # 0.33 d
        [1.95, 2.05],  # 0.5 d
        [0.95, 1.05],  # 1 d
        [0.48, 0.52],  # 2 d
        [0.32, 0.34],  # 3 d
    ]
    freqs_copy = freqs.copy()
    if freqs_to_remove is not None:
        for pair in freqs_to_remove:
            idx = np.where((freqs_copy < pair[0]) | (freqs_copy > pair[1]))[0]
            freqs_copy = freqs_copy[idx]
    freqs_no_terrestrial = freqs_copy
    return labels, tme_collection, freqs_no_terrestrial


def extract_relavant(response, min_jd, max_jd, min_obs=20):
    lcs = {}
    # get only relavant information
    for name in response.keys():  # max 3 loops db name
        if len(response[name]) > 0:
            r_list = response[name]
            for r in r_list:  # loop over number of batches
                if r.get("status") == "success":
                    light_curves = r.get("data")
                    for light_curve in light_curves:  # number of alerts
                        _id = light_curve['_id']
                        prv_candidates = light_curve['prv_candidates']
                        for point in prv_candidates:  # number of points
                            keys = point.keys()
                            if (
                                'jd' in keys
                                and 'magpsf' in keys
                                and 'sigmapsf' in keys
                                and 'fid' in keys
                            ):  # make sure relavant info is present
                                if point['jd'] < min_jd or point['jd'] > max_jd:
                                    continue  # remove points that do not meet time criterion
                                add_point = np.array(
                                    [point['jd'], point['magpsf'], point['sigmapsf']]
                                )
                                fid = point['fid']
                            else:
                                continue
                            if f'{_id}_{fid}' not in lcs.keys():  # first time
                                lcs[f'{_id}_{fid}'] = [add_point]
                            elif f'{_id}_{fid}' in lcs.keys():
                                lcs[f'{_id}_{fid}'].append(add_point)

    for key in list(lcs.keys()):  # satisfy min obs criterion
        if len(lcs[key]) < min_obs:
            lcs.pop(key)
    return lcs


def get_alert_ids(kowalski_instances, fields=None, min_jd=0.0, max_jd=None):

    match = {"candidate.jd": {'$gte': min_jd, '$lte': max_jd}}
    if fields is not None:
        match["candidate.field"] = {'$in': fields}

    pipeline = [
        {'$match': match},
        {'$project': {'objectId': 1}},
        {'$group': {'_id': "$objectId", 'nb_alerts': {'$sum': 1}}},
    ]
    query = {
        "query_type": "aggregate",
        "query": {"catalog": "ZTF_alerts", "pipeline": pipeline},
    }
    response = kowalski_instances.query(query=query)
    alerts = [a['_id'] for a in response['kowalski']['data']]
    return alerts


def get_alert_lcs_by_ids(kowalski_instances, ids, limit=1000, Ncore=4):
    size = len(ids)
    batches = size // limit
    qs = [
        {
            "query_type": "find",
            "query": {
                "catalog": 'ZTF_alerts_aux',
                "filter": {"_id": {"$in": ids[i * limit : (i + 1) * limit]}},
                "projection": {},
            },
            "kwargs": {"limit": limit, "skip": 0},
        }
        for i in range(batches)
    ]
    # the remainder
    last_q = {
        "query_type": "find",
        "query": {
            "catalog": 'ZTF_alerts_aux',
            "filter": {"_id": {"$in": ids[batches * limit : -1]}},
            "projection": {},
        },
        "kwargs": {"limit": limit, "skip": 0},
    }
    qs.append(last_q)
    r = kowalski_instances.query(queries=qs, use_batch_query=True, max_n_threads=Ncore)
    return r


if __name__ == '__main__':
    # fields=list(np.array(args.field.split(',')).astype(int))
    start_jd, end_jd = 2460491.69692, 2460493.95787
    fields = [590]
    path_to_save = '/expanse/lustre/projects/umn141/dwarshofsky/devlop_scope/Zach_experiment/day1.csv'
    alerts = get_alert_ids(
        kowalski_instances, fields=fields, min_jd=start_jd, max_jd=end_jd
    )
    r = get_alert_lcs_by_ids(kowalski_instances, alerts, limit=5000, Ncore=4)
    lcs = extract_relavant(r, min_jd=start_jd, max_jd=end_jd)
    labels, tme_collection, freqs = prep_for_period_finding(lcs, max_freq=288)

    period_algorithms = ['ELS_periodogram', 'ECE_periodogram', 'EAOV_periodogram']

    all_periods = {algorithm: [] for algorithm in period_algorithms}
    all_significances = {algorithm: [] for algorithm in period_algorithms}
    all_pdots = {algorithm: [] for algorithm in period_algorithms}
    for algorithm in period_algorithms:
        t0 = time.time()
        print(f'Running {algorithm} algorithm:')
        # Iterate over algorithms
        periods, significances, pdots = periodsearch.find_periods(
            algorithm,
            tme_collection,
            freqs,
            doGPU=True,
            doCPU=False,
            doUsePDot=False,
            doSingleTimeSegment=False,
            # freqs_to_remove=freqs_to_remove,
            phase_bins=20,
            mag_bins=10,
            Ncore=4,
        )
        print(f'finished {algorithm} in {time.time()-t0}')
        if algorithm == 'ECE_periodogram':
            all_periods[algorithm] = [1 / freqs[np.argmin(p['data'])] for p in periods]
            all_significances[algorithm] = [np.min(p['data']) for p in periods]
        else:
            all_periods[algorithm] = [1 / freqs[np.argmax(p['data'])] for p in periods]
            all_significances[algorithm] = [np.max(p['data']) for p in periods]
    df = pd.DataFrame(labels, columns=['labels'])
    df['LS_p'] = all_periods['ELS_periodogram']
    df['LS_s'] = all_significances['ELS_periodogram']
    df['CE_p'] = all_periods['ECE_periodogram']
    df['CE_s'] = all_significances['ECE_periodogram']
    df['AOV_p'] = all_periods['EAOV_periodogram']
    df['AOV_s'] = all_significances['EAOV_periodogram']
    df.to_csv(path_to_save, index=False)
