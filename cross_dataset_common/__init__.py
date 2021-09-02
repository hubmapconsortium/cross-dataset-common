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

SERVERS = ["https://cells.test.hubmapconsortium.org/api/", "https://cells.dev.hubmapconsortium.org/api/", "3.236.187.179/api/"]
CHUNK_SIZE = 1000
modality_ranges_dict = {"rna": [0, 5], "atac": [-4, 1], "codex": [-1, 4]}
min_percentages = [10 * i for i in range(0, 11)]

def hash_cell_id(semantic_cell_ids: pd.Series):
    hash_list = [hashlib.sha256(semantic_cell_id.encode('UTF-8')).hexdigest() for semantic_cell_id in semantic_cell_ids]
    return pd.Series(hash_list, index=semantic_cell_ids.index)

def make_quant_df(adata: anndata.AnnData):

    adata.obs = adata.obs.set_index('cell_id', drop=False, inplace=False)

    genes = list(adata.var.index)
    cells = list(adata.obs.index)
    coo = coo_matrix(adata.X)

    triples = [(coo.row[i], coo.col[i], coo.data[i]) for i in range(len(coo.row))]

    dict_list = [{'q_cell_id':cells[row], 'q_var_id':genes[col], 'value':val} for row, col, val in triples]
    quant_df = pd.DataFrame(dict_list)
    return quant_df

def process_quant_column(quant_df_and_column):
    quant_df = quant_df_and_column[0]
    column = quant_df_and_column[1]

    dict_list =  [{'cell_id': i, 'gene_id': column, 'value': quant_df.at[i, column]} for i in
                 quant_df.index if quant_df.at[i, column] > 0.0]

    return dict_list

def get_zero_cells_column(quant_df_and_column):
    quant_df = quant_df_and_column[0]
    column = quant_df_and_column[1]

    zero_cells = [i for i in quant_df.index if quant_df.at[i, column] > 0.0]

    return zero_cells

def flatten_quant_df(quant_df:pd.DataFrame):

    dict_list = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:

        df_and_columns = [(quant_df, column) for column in quant_df.columns]

        for column_list in executor.map(process_quant_column, df_and_columns):
            dict_list.extend(column_list)

    return pd.DataFrame(dict_list)

def get_zero_cells(quant_df:pd.DataFrame):

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        df_and_columns = [(quant_df, column) for column in quant_df.columns]
        zero_cells = {column: cell_list for column, cell_list in zip(quant_df.columns, executor.map(get_zero_cells_column, df_and_columns))}

    return zero_cells


def get_dataset_uuids(modality: str, token: str = None) -> List[str]:
    hits = []
    for i in range(50):
        dataset_query_dict = {
            "from": 10 * i,
            "size": 10,
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
                                "immediate_ancestors.entity_type": {
                                    "query": "Dataset"
                                }
                            }
                        },
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
                json=dataset_query_dict, headers={'Authorization': 'Bearer ' + token})
        else:
            dataset_response = requests.post(
                'https://search.api.hubmapconsortium.org/search',
                json=dataset_query_dict)
        hits.extend(dataset_response.json()['hits']['hits'])

    uuids = []
    for hit in hits:
        for ancestor in hit['_source']['ancestors']:
            if 'data_types' in ancestor.keys():
                if modality in ancestor['data_types'][0] and 'bulk' not in ancestor['data_types'][0]:
                    uuids.append(hit['_source']['uuid'])

    return uuids

def get_tissue_type(dataset: str, token: str = None) -> str:

    print(dataset)

    special_cases = {'ucsd-snareseq':'Kidney', 'caltech-sciseq':'Heart'}

    #Hacky handling of datasets not yet exposed to search-api
    if dataset in special_cases.keys():
        return special_cases[dataset]


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

    return data_frames

def get_cluster_df(adata:anndata.AnnData)->pd.DataFrame:
    cell_df = adata.obs.copy()
    dataset = cell_df['dataset'][0]

    pval_dict_list = []

    for group_id in cell_df['leiden'].unique():

        if type(group_id) == float and np.isnan(group_id):
            continue

        gene_names = adata.uns['rank_genes_groups']['names'][group_id]

        pvals = adata.uns['rank_genes_groups']['pvals_adj'][group_id]

        names_and_pvals = zip(gene_names, pvals)

        pval_dict_list.extend([{'grouping_name': group_id, 'gene_id': n_p[0], 'value': n_p[1]} for n_p in names_and_pvals])

    return pd.DataFrame(pval_dict_list)

def make_mini_cell_df(cell_df:pd.DataFrame, modality:str):
    mini_cell_df = cell_df.head(10).copy()
    if "cell_id" not in mini_cell_df.columns:
        mini_cell_df["cell_id"] = mini_cell_df.index
    cell_ids = mini_cell_df["cell_id"].to_list()

    new_file = "mini_" + modality + ".hdf5"
    if modality == 'codex':
        with pd.HDFStore(new_file) as store:
            store.put("cell", mini_cell_df)
    else:
        with pd.HDFStore(new_file) as store:
            store.put("cell", mini_cell_df, format='t')
    return cell_ids


def make_mini_quant_df(quant_df:pd.DataFrame, modality:str, cell_ids):

    print(f"Length of quant_df index: ")

    csv_file = modality + '.csv'
    quant_df = quant_df[quant_df['q_cell_id'].isin(cell_ids)]

    genes = []

    if modality in ['rna', 'atac']:
        genes = list(quant_df['q_var_id'].unique())[:10]
        gene_filter = quant_df['q_var_id'].isin(genes)
        quant_df = quant_df[gene_filter]

    quant_df.to_csv('mini_' + csv_file)

    return genes


def make_mini_pval_dfs(pval_dfs, keys, modality, gene_ids):
    new_file = "mini_" + modality + ".hdf5"

    for i, pval_df in enumerate(pval_dfs):
        dict_list = pval_df.to_dict(orient='records')
        print(keys[i])
        print(len(pval_df.index))
        print(len(pval_df["grouping_name"].unique()))
        dict_list = [record for record in dict_list if record['gene_id'] in gene_ids]
        grouping_name_set = set([record["grouping_name"] for record in dict_list])
        print(len(grouping_name_set))
        filtered_pval_df = pd.DataFrame(dict_list)
        print(len(filtered_pval_df["grouping_name"].unique()))

        filtered_pval_df.to_hdf(new_file, keys[i])

    return


def create_minimal_dataset(cell_df, quant_df, organ_df=None, cluster_df=None, modality=None):

    cell_ids = make_mini_cell_df(cell_df, modality)
    gene_ids = make_mini_quant_df(quant_df, modality, cell_ids)
    if modality in ["atac", "rna"]:
        make_mini_pval_dfs([organ_df, cluster_df],['organ', 'cluster'], modality, gene_ids)

def delete_data_from_servers(modality):
    request_dict = {"modality":modality}
    for server in SERVERS:
        request_url = server + "delete/"
        response = requests.post(request_url, request_dict).json()
        if "error" in response.keys():
            print(response['error'])
        else:
            print(response['message'])
            print(response['time'])

def make_post_request(params_tuple):
    server = params_tuple[0]
    request_dict = params_tuple[1]
    request_url = server + "insert/"
    response = requests.post(request_url, request_dict).json()
    if "error" in response.keys():
        print(response['error'])
    else:
        print(response['message'])
        print(response['time'])

def add_data_to_server(kwargs_list, model_name):
    for i in range(len(kwargs_list) // CHUNK_SIZE + 1):
        print(f"Sending chunk {i} of {len(kwargs_list) // CHUNK_SIZE}")
        kwargs_subset = kwargs_list[i * CHUNK_SIZE:(i+1)*CHUNK_SIZE]
        request_dict = {"model_name":model_name, "kwargs_list":json.dumps(kwargs_subset)}
        params_tuples = [(server, request_dict) for server in SERVERS]
        with ThreadPoolExecutor(max_workers=len(params_tuples)) as e:
            e.map(make_post_request, params_tuples)

def create_modality(modality):
    add_data_to_server([{"modality_name":modality}], "modality")

def create_datasets(datasets_list, modality):
    kwargs_list = [{"modality":modality, "uuid":dataset} for dataset in datasets_list]
    add_data_to_server(kwargs_list, "dataset")

def create_organs(cell_df):
    organs_list = list(cell_df["organ"].unique())
    kwargs_list = [{"grouping_name":organ} for organ in organs_list]
    add_data_to_server(kwargs_list, "organ")

def create_clusters(cell_df):
    clusters_set = set({})
    unique_cluster_lists = [string.split(",") for string in cell_df["clusters"].unique()]
    for cluster_list in unique_cluster_lists:
        for cluster in cluster_list:
            clusters_set.add(cluster)

    cluster_splits = [cluster.split("-") for cluster in clusters_set]
    for cluster_split in cluster_splits:
        if len(cluster_split) < 4:
            print(cluster_split)
    kwargs_list = [{"cluster_method":cs[0], "cluster_data":cs[1], "dataset":cs[2], "grouping_name":"-".join(cs)}
                   for cs in cluster_splits]
    add_data_to_server(kwargs_list, "cluster")

    return list(clusters_set)

def create_genes(quant_df):
    gene_symbols = list(quant_df["q_var_id"].unique())
    kwargs_list = [{"gene_symbol":gene_symbol} for gene_symbol in gene_symbols]
    add_data_to_server(kwargs_list, "gene")

def create_proteins(quant_df):
    protein_ids = list(quant_df["q_var_id"].unique())
    kwargs_list = [{"protein_id":protein_id} for protein_id in protein_ids]
    add_data_to_server(kwargs_list, "protein")

def create_cells(cell_df):
    cell_df = cell_df[["cell_id", "modality", "dataset", "organ"]]
    kwargs_list = cell_df.to_dict("records")
    add_data_to_server(kwargs_list, "cell")

def create_quants(quant_df, modality):
    model_name = modality + "quant"
    kwargs_list = quant_df.to_dict("records")
    add_data_to_server(kwargs_list, model_name)

def create_pvals(grouping_df, modality):
    kwargs_list = grouping_df.to_dict("records")
    for kwargs in kwargs_list:
        kwargs["modality"] = modality
    add_data_to_server(kwargs_list, "pvalue")

def set_up_relationships(cell_df, clusters):
    cell_clusters_dict = {}
    for cluster in clusters:
        cell_clusters_dict[cluster] = []
        for i in cell_df.index:
            if cluster in cell_df.at[i, "clusters"].split(","):
                cell_clusters_dict[cluster].append(cell_df.at[i, "cell_id"])

    for server in SERVERS:
        request_url = server + "setuprelationships/"
        response = requests.post(request_url, cell_clusters_dict).json()
        if "error" in response.keys():
            print(response['error'])
        else:
            print(response['message'])
            print(response['time'])

def load_data_to_vms(modality, cell_df, quant_df, organ_df = None, cluster_df = None):
    delete_data_from_servers(modality)
    create_modality(modality)
    datasets_list = list(cell_df["dataset"].unique())
    create_datasets(datasets_list, modality)
    clusters = create_clusters(cell_df)
    create_organs(cell_df)
    if modality in ["rna", "atac"]:
        create_genes(quant_df)
    elif modality in ["codex"]:
        create_proteins(quant_df)

    #Do things up to here first because later things depend on them
    with ThreadPoolExecutor(max_workers=5) as e:
        e.submit(create_cells, cell_df)
        e.submit(create_quants, quant_df, modality)
        if modality in ["rna", "atac"]:
            e.submit(create_pvals, organ_df, modality)
            e.submit(create_pvals,cluster_df, modality)
        e.submit(set_up_relationships, cell_df, clusters)

def precompute_dataset_percentages(dataset_adata, modality):

    kwargs_list = []
    exponents = list(
        range(modality_ranges_dict[modality][0], modality_ranges_dict[modality][1] + 1)
    )

    num_cells_in_dataset = len(dataset_adata.obs.index)
    uuid = dataset_adata.obs['dataset'][0]

    for var_id in dataset_adata.var.index:
        zero = False
        for exponent in exponents:
            if zero:
                percentage = 0.0
            else:
                cutoff = 10 ** exponent
                subset_adata = dataset_adata[dataset_adata[var_id] > cutoff]
                num_matching_cells = len(subset_adata.obs.index)
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

    return pd.DataFrame(kwargs_list)