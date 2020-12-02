import requests
import yaml
import anndata
import pandas as pd
from typing import List, Dict, Iterable
from pathlib import Path
from os import walk
import scanpy as sc
import numpy as np
import concurrent.futures
import hashlib
from scipy.sparse import coo_matrix

def hash_cell_id(semantic_cell_ids: pd.Series):
    hash_list = [hashlib.sha256(semantic_cell_id.encode('UTF-8')).hexdigest() for semantic_cell_id in semantic_cell_ids]
    hash_list

def make_quant_df(adata: anndata.AnnData):

    adata.obs.index = adata.obs['cell_id']

    genes = list(adata.var.index)
    cells = list(adata.obs.index)
    coo = coo_matrix(adata.X)

    triples = [(coo.row[i], coo.col[i], coo.data[i]) for i in range(len(coo.row))]

    dict_list = [{'q_cell_id':cells[row], 'q_gene_id':genes[col], 'value':val} for row, col, val in triples]
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

def get_tissue_type(dataset: str, token: str) -> str:

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

    dataset_response = requests.post(
        'https://search.api.hubmapconsortium.org/search',
        json=dataset_query_dict,
        headers={'Authorization': 'Bearer ' + token})
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

    groupings_dict = {'tissue_type':'organ_name', 'leiden':'cluster'}

    num_genes = len(adata.var_names)

    data_frames = []

    for grouping in groupings_dict:

        print(grouping)
        print(adata.obs[grouping].unique())

        sc.tl.rank_genes_groups(adata, grouping, method='t-test', rankby_abs=True, n_genes=num_genes)

        cell_df = adata.obs.copy()
        if 'cell_id' not in cell_df.columns:
            cell_df['cell_id'] = cell_df.index

        pval_dict_list = []

        group_descriptor = groupings_dict[grouping]

        for group_id in cell_df[grouping].unique():

            if type(group_id) == float and np.isnan(group_id):
                continue

            gene_names = adata.uns['rank_genes_groups']['names'][group_id]
            pvals = adata.uns['rank_genes_groups']['pvals'][group_id]
            names_and_pvals = zip(gene_names, pvals)

            pval_dict_list.extend([{group_descriptor: group_id, 'gene_id': n_p[0], 'value': n_p[1]} for n_p in names_and_pvals])

        data_frames.append(pd.DataFrame(pval_dict_list))

    for df in data_frames:
        print(df.columns)

    return data_frames

def get_cluster_df(adata:anndata.AnnData)->pd.DataFrame:
    cell_df = adata.obs.copy()
    dataset = cell_df['dataset'][0]

    pval_dict_list = []

    for group_id in cell_df['leiden'].unique():

        if type(group_id) == float and np.isnan(group_id):
            continue

        gene_names = adata.uns['rank_genes_groups']['names'][group_id]
        pvals = adata.uns['rank_genes_groups']['pvals'][group_id]
        names_and_pvals = zip(gene_names, pvals)

        pval_dict_list.extend([{'leiden': group_id, 'dataset':dataset, 'gene_id': n_p[0], 'value': n_p[1]} for n_p in names_and_pvals])

    return pd.DataFrame(pval_dict_list)

def make_mini_cell_df(cell_df:pd.DataFrame, modality:str):

    print('Original df index')
    print(cell_df.index)

    mini_cell_df = cell_df.head(1000).copy()
    if "cell_id" not in mini_cell_df.columns:
        mini_cell_df["cell_id"] = mini_cell_df.index
    cell_ids = mini_cell_df.index.to_list()

    print('Mini df index')
    print(mini_cell_df.index)
    print('Mini df cell_ids')
    print(mini_cell_df['cell_id'])
    print('Cell id list')
    print(cell_ids)

    new_file = "mini_" + modality + ".hdf5"
    with pd.HDFStore(new_file) as store:
        store.put("cell", mini_cell_df)
    return cell_ids


def make_mini_quant_df(quant_df:pd.DataFrame, modality:str, cell_ids):

    csv_file = modality + '.csv'
    genes = list(quant_df['q_gene_id'].unique())[:1000]
    quant_df.set_index('q_gene_id', inplace=True, drop=False)
    quant_df = quant_df.loc[genes]
    cell_ids = [cell_id for cell_id in cell_ids if cell_id in quant_df['q_cell_id'].unique()]
    print('cell ids')
    print(cell_ids)
    quant_df.set_index('q_cell_id', inplace=True, drop=False)
    quant_df = quant_df.loc[cell_ids]
    quant_df = quant_df.reset_index(drop=True, inplace=False)
    print(quant_df.columns)

    quant_df.to_csv('mini_' + csv_file)

    return genes


def make_mini_pval_dfs(pval_dfs, keys, modality, gene_ids):
    new_file = "mini_" + modality + ".hdf5"

    for i, pval_df in enumerate(pval_dfs):
        pval_df = pval_df.set_index("gene_id", drop=False)
        filtered_pval_df = pval_df.loc[gene_ids]

        with pd.HDFStore(new_file) as store:
            store.put(keys[i], filtered_pval_df)

    return


def create_minimal_dataset(cell_df, quant_df, organ_df, cluster_df, modality):

    cell_ids = make_mini_cell_df(cell_df, modality)
    print(cell_ids)
    gene_ids = make_mini_quant_df(quant_df, modality, cell_ids)
    if modality in ["atac", "rna"]:
        make_mini_pval_dfs([organ_df, cluster_df],['organ', 'cluster'], modality, gene_ids)
