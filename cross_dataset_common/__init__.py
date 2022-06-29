import requests
import yaml
import anndata
import pandas as pd
from typing import List, Dict, Iterable
from pathlib import Path
from os import walk, fspath
import scanpy as sc
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import hashlib
from scipy.sparse import coo_matrix
import subprocess
import json
import awswrangler as wr
import boto3

SERVERS = ["https://cells.test.hubmapconsortium.org/api/", "https://cells.dev.hubmapconsortium.org/api/", "3.236.187.179/api/"]
CHUNK_SIZE = 1000
modality_ranges_dict = {"rna": [-3, 2], "atac": [-4, 1], "codex": [-1, 4]}
min_percentages = [0, 1, 2, 5, 10]

def hash_cell_id(semantic_cell_ids: pd.Series):
    hash_list = [hashlib.sha256(semantic_cell_id.encode('UTF-8')).hexdigest() for semantic_cell_id in semantic_cell_ids]
    return pd.Series(hash_list, index=semantic_cell_ids.index)

def get_tissue_type(dataset: str, token: str = None) -> str:

    organ_dict = yaml.load(open('/opt/organ_types.yaml'), Loader=yaml.BaseLoader)

    dataset_query_dict = {
        "query": {
            "bool": {
                "must": [],
                "filter": [
                    {
                        "match_all": {}
                    },
                    {
                        "exists": {
                            "field": "files.rel_path"
                        }
                    },
                    {
                        "match_phrase": {
                            "uuid": {
                                "query": dataset
                            },
                        }

                    }
                ],
                "should": [],
                "must_not": [
                    {
                        "match_phrase": {
                            "status": {
                                "query": "Error"
                            }
                        }
                    }
                ]
            }
        }
    }

    if token is not None:
        dataset_response = requests.post(
            'https://search.api.hubmapconsortium.org/search',
            json=dataset_query_dict,
            headers={'Authorization': 'Bearer ' + token})
    else:
        dataset_response = requests.post(
            'https://search.api.hubmapconsortium.org/search',
            json=dataset_query_dict)

    hits = dataset_response.json()['hits']['hits']

    for hit in hits:
        for ancestor in hit['_source']['ancestors']:
            if 'organ' in ancestor.keys():
                raw_organ_name = organ_dict[ancestor['organ']]['description']
                if 'Kidney' in raw_organ_name:
                    return 'Kidney'
                elif 'Bronchus' in raw_organ_name:
                    return 'Bronchus'
                elif 'Lung' in raw_organ_name:
                    return 'Lung'
                elif 'Lymph' in raw_organ_name:
                    return 'Lymph Node'
                else:
                    return raw_organ_name


def get_gene_response(ensembl_ids: List[str]):
    request_url = 'https://mygene.info/v3/gene?fields=symbol'

    chunk_size = 1000
    chunks = (len(ensembl_ids) // chunk_size) + 1

    base_list = []

    for i in range(chunks):
        if i < chunks - 1:
            ensembl_slice = ensembl_ids[i * chunk_size: (i + 1) * chunk_size]
        else:
            ensembl_slice = ensembl_ids[i * chunk_size:]
        request_body = {'ids': ', '.join(ensembl_slice)}
        base_list.extend(requests.post(request_url, request_body).json())

    return base_list


def get_gene_dicts(ensembl_ids: List[str]) -> (Dict, Dict):
    #    temp_forwards_dict = {ensembl_id:ensembl_id.split('.')[0] for ensembl_id in ensembl_ids}
    temp_backwards_dict = {ensembl_id.split('.')[0]: ensembl_id for ensembl_id in ensembl_ids}
    ensembl_ids = [ensembl_id.split('.')[0] for ensembl_id in ensembl_ids]

    json_response = get_gene_response(ensembl_ids)

    forwards_dict = {temp_backwards_dict[item['query']]: item['symbol'] for item in json_response if
                     'symbol' in item.keys()}
    backwards_dict = {item['symbol']: temp_backwards_dict[item['query']] for item in json_response if
                      'symbol' in item.keys()}

    return forwards_dict, backwards_dict


def find_files(directory: Path, pattern: str) -> Iterable[Path]:
    for dirpath_str, dirnames, filenames in walk(directory):
        dirpath = Path(dirpath_str)
        for filename in filenames:
            filepath = dirpath / filename
            if filepath.match(pattern):
                yield filepath

def get_dataset_cluster_df(adata: anndata.AnnData)->pd.DataFrame:
    num_genes = len(adata.var_names)

    data_frames = []

    for dataset in adata.obs["dataset"].unique():
        print(dataset)

        dataset_adata = adata[adata.obs["dataset"] == dataset]
        sc.tl.rank_genes_groups(dataset_adata, "dataset_leiden", method='t-test', rankby_abs=True, n_genes=num_genes)

        pval_dict_list = []

        for group_id in adata.obs["dataset_leiden"].unique():

            if type(group_id) == float and np.isnan(group_id):
                continue

            gene_names = adata.uns['rank_genes_groups']['names'][group_id]

            pvals = adata.uns['rank_genes_groups']['pvals_adj'][group_id]

            names_and_pvals = zip(gene_names, pvals)

            pval_dict_list.extend([{'grouping_name': group_id, 'gene_id': n_p[0], 'value': n_p[1]} for n_p in names_and_pvals])

        data_frames.append(pd.DataFrame(pval_dict_list))

    df = pd.concat(data_frames)

    return df

def get_cell_type_df(adata: anndata.AnnData)->pd.DataFrame:
    num_genes = len(adata.var_names)

    adata = adata[adata.obs["cell_type"] != "unknown"]

    sc.tl.rank_genes_groups(adata, "cell_type", method='t-test', rankby_abs=True, n_genes=num_genes)

    pval_dict_list = []

    for group_id in adata.obs["cell_type"].unique():

        if type(group_id) == float and np.isnan(group_id):
            continue

        gene_names = adata.uns['rank_genes_groups']['names'][group_id]

        pvals = adata.uns['rank_genes_groups']['pvals_adj'][group_id]

        names_and_pvals = zip(gene_names, pvals)

        pval_dict_list.extend([{'grouping_name': group_id, 'gene_id': n_p[0], 'value': n_p[1]} for n_p in names_and_pvals])

    df = pd.DataFrame(pval_dict_list)

    return df

def get_pval_dfs(adata: anndata.AnnData)->List[pd.DataFrame]:

    groupings_list = ['organ', 'leiden']

    num_genes = len(adata.var_names)

    data_frames = []

    for grouping in groupings_list:

        sc.tl.rank_genes_groups(adata, grouping, method='t-test', rankby_abs=True, n_genes=num_genes)

        cell_df = adata.obs.copy()

        pval_dict_list = []

        for group_id in cell_df[grouping].unique():

            if type(group_id) == float and np.isnan(group_id):
                continue

            gene_names = adata.uns['rank_genes_groups']['names'][group_id]

            pvals = adata.uns['rank_genes_groups']['pvals_adj'][group_id]

            names_and_pvals = zip(gene_names, pvals)

            pval_dict_list.extend([{'grouping_name': group_id, 'gene_id': n_p[0], 'value': n_p[1]} for n_p in names_and_pvals])

        data_frames.append(pd.DataFrame(pval_dict_list))

    for df in data_frames:
        print(len(df['grouping_name'].unique()))

    if "cell_type" in adata.obs.columns and len(adata.obs["cell_type"].unique()) > 1:
        data_frames.append(get_cell_type_df(adata))
    else:
        data_frames.append(pd.DataFrame())

    return data_frames

def get_cluster_df(adata:anndata.AnnData, dataset)->pd.DataFrame:
    cell_df = adata.obs.copy()
#    dataset = cell_df['dataset'][0]

    pval_dict_list = []

    sc.tl.rank_genes_groups(adata, 'leiden', method='t-test', rankby_abs=True, n_genes=num_genes)

    for group_id in cell_df['leiden'].unique():

        if type(group_id) == float and np.isnan(group_id):
            continue

        gene_names = adata.uns['rank_genes_groups']['names'][group_id]

        pvals = adata.uns['rank_genes_groups']['pvals_adj'][group_id]

        names_and_pvals = zip(gene_names, pvals)

        grouping_name = f"leiden-UMAP-{dataset}-{group_id}"

        pval_dict_list.extend([{'grouping_name': grouping_name, 'gene_id': n_p[0], 'value': n_p[1]} for n_p in names_and_pvals])

    return pd.DataFrame(pval_dict_list)

def precompute_gene(params_tuple):
    dataset_df = params_tuple[0]
    modality = params_tuple[1]
    uuid = params_tuple[2]
    var_id = params_tuple[3]

    exponents = list(
        range(modality_ranges_dict[modality][0], modality_ranges_dict[modality][1] + 1)
    )

    num_cells_in_dataset = len(dataset_df.index)

    kwargs_list = []

    zero = False
    for exponent in exponents:
        if zero:
            percentage = 0.0
        else:
            cutoff = 10 ** exponent
            subset_df = dataset_df[dataset_df[var_id] > cutoff]
            num_matching_cells = len(subset_df.index)
            percentage = num_matching_cells / num_cells_in_dataset * 100.0
            if percentage == 0.0:
                print("Hit a zero")
                zero = True

        kwargs = {
            "modality": modality,
            "dataset": uuid,
            "var_id": var_id,
            "cutoff": cutoff,
            "percentage": percentage,
        }
        kwargs_list.append(kwargs)

    return kwargs_list

def precompute_dataset_percentages(dataset_adata):

    kwargs_list = []
    modality = list(dataset_adata.obs['modality'])[0]

    uuid = list(dataset_adata.obs['dataset'])[0]
    dataset_df = dataset_adata.to_df()

    params_tuples = [(dataset_df, modality, uuid, var_id) for var_id in dataset_df.columns]
    with ThreadPoolExecutor(max_workers=10) as e:
        kwargs_lists = e.map(precompute_gene, params_tuples)

    for kl in kwargs_lists:
        kwargs_list.extend(kl)

    return pd.DataFrame(kwargs_list)

def precompute_dataset_values_series(dataset_df, dataset_adata):
    param_tuples_list = [(dataset_df, dataset_adata, var) for var in adata.index]
    with ThreadPoolExecutor(max_workers=20) as e:
        values_series_dicts = e.map(precompute_dataset_single_values_series, param_tuples_list)
    values_series_dict = {}
    for vsd in values_series_dicts:
        values_series_dict.update(vsd)

    return values_series_dict

def precompute_dataset_single_values_series(cell_df_subset, dataset_adata, var):
    values_series_dict = {}
    values_list = [json.dumps({'var': dataset_adata[cell, var]} for cell in cell_df_subset.index)]
    values_series = pd.Series(values_list, index=cell_df_subset.index)
    values_series_dict[f"{dataset}+{var}"] = values_series
    return values_series_dict

def precompute_values_series(cell_df, adata):
    dataset_dfs = [cell_df[cell_df["dataset"] == dataset] for dataset in cell_df["dataset"].unique()]
    cell_ids_lists = [list(dataset_df["cell_id"].unique()) for dataset_df in dataset_dfs]
    dataset_adatas = [adata[cell_id_list] for cell_id_list in cell_ids_lists]

    param_tuples_list = [(dataset_dfs[i], dataset_adatas[i]) for i in range(len(dataset_dfs))]

    with ThreadPoolExecutor(max_workers=10) as e:
        values_series_dicts = e.map(precompute_dataset_values_series, param_tuples_list)
    values_series_dict = {}
    for vsd in values_series_dicts:
        values_series_dict.update(vsd)

    return values_series_dict

def make_minimal_adata(adata, modality):

    X = csr_matrix(adata.X)
    obs = pd.DataFrame(index=adata.obs["cell_id"])
    var = pd.DataFrame(index=adata.var.index)

    min_adata = anndata.AnnData(X=X, obs=obs, var=var)
    min_adata.write_h5ad(f"{modality}.h5ad")

def upload_file_to_s3(path_to_file, boto_session):
    remote_path = f"s3://cells-api-index-assets/{path_to_file.name}"
    with open(local_path, "rb") as local_f:
        wr.s3.upload(local_file=local_f, path=remote_path,
                     boto3_session=boto_session)

def upload_files_to_s3(paths_to_files, access_key_id, secret_access_key):
    boto_session = boto3.Session(access_key_id, secret_access_key)
    for path in paths_to_files:
        upload_file_to_s3(path, boto_session)


def find_files(directory, patterns):
    for dirpath_str, dirnames, filenames in walk(directory):
        dirpath = Path(dirpath_str)
        for filename in filenames:
            filepath = dirpath / filename
            for pattern in patterns:
                if filepath.match(pattern):
                    return filepath

def find_file_pairs(directory):
    filtered_patterns = ['cluster_marker_genes.h5ad', 'secondary_analysis.h5ad']
    unfiltered_patterns = ['out.h5ad', 'expr.h5ad']
    filtered_file = find_files(directory, filtered_patterns)
    unfiltered_file = find_files(directory, unfiltered_patterns)
    return filtered_file, unfiltered_file