"""Evaluation protocols. Each returns a list of dicts: {protocol, fold, train (row_ids), test (row_ids), grouped_tuning}.

row_split          legacy random 80/20 split of development rows (optimistic; kept for comparison)
lobo_dev           leave-one-bed-out inside the 15-45 m development domain (new bed, seen pressures)
loso_dev           leave-one-soil-out inside the development domain (new grain size)
transfer55         train on all development rows, test at 55 m (original design; same beds, new pressure)
lobo_transfer55    train on development rows of the other beds, test the left-out bed at 55 m (new bed AND new pressure)
pressure_ahead     train on P <= p0, predict all higher pressures (rolling origin; p0 = 25, 35, 45)
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split
from . import config as C


def row_split(df, seeds=range(C.N_ROW_SPLIT_SEEDS)):
    d = df[df.domain == "dev"]
    out = []
    for s in seeds:
        tr, te = train_test_split(d.row_id.values, test_size=C.LEGACY_TEST_SIZE, random_state=s)
        out.append(dict(protocol="row_split", fold=f"seed{s}", train=tr, test=te, grouped_tuning=False))
    return out


def lobo_dev(df):
    d = df[df.domain == "dev"]
    return [dict(protocol="lobo_dev", fold=b, train=d[d.bed != b].row_id.values,
                 test=d[d.bed == b].row_id.values, grouped_tuning=True) for b in sorted(d.bed.unique())]


def loso_dev(df):
    d = df[df.domain == "dev"]
    return [dict(protocol="loso_dev", fold=s, train=d[d.soil != s].row_id.values,
                 test=d[d.soil == s].row_id.values, grouped_tuning=True) for s in ("fine", "medium", "coarse")]


def transfer55(df):
    d, t = df[df.domain == "dev"], df[df.domain == "transfer"]
    return [dict(protocol="transfer55", fold="all", train=d.row_id.values, test=t.row_id.values, grouped_tuning=True)]


def lobo_transfer55(df):
    d, t = df[df.domain == "dev"], df[df.domain == "transfer"]
    return [dict(protocol="lobo_transfer55", fold=b, train=d[d.bed != b].row_id.values,
                 test=t[t.bed == b].row_id.values, grouped_tuning=True) for b in sorted(d.bed.unique())]


def pressure_ahead(df, p0s=(25, 35, 45)):
    return [dict(protocol="pressure_ahead", fold=f"trainmax{p0}", train=df[df.P <= p0].row_id.values,
                 test=df[df.P > p0].row_id.values, grouped_tuning=True) for p0 in p0s]


ALL = dict(row_split=row_split, lobo_dev=lobo_dev, loso_dev=loso_dev, transfer55=transfer55,
           lobo_transfer55=lobo_transfer55, pressure_ahead=pressure_ahead)
