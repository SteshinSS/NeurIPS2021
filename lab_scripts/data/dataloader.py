import anndata as ad
import numpy as np


def load_custom_je_data(task_type, train_batches, test_batches, val_size=None):
    if task_type in ["cite_pre", "cite"]:
        result = load_custom_mp_data(
            "gex_to_adt", train_batches, test_batches, val_size
        )
        solution = ad.read_h5ad(JE_CITE_SOLUTION)
        result["train_solution"] = solution[result['train_mod1'].obs.index]
        result["test_solution"] = solution[result['test_mod1'].obs.index]
    elif task_type == "atac":
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    return result


def load_custom_mm_data(task_type, train_batches, test_batches, val_size=None):
    result = load_custom_mp_data(task_type, train_batches, test_batches, val_size)
    train_n = result["train_mod1"].shape[0]
    train_pairing_ix = np.arange(train_n)
    result["train_sol"] = ad.AnnData(
        np.eye(train_n), uns={"pairing_ix": train_pairing_ix}
    )

    test_n = result["test_mod1"].shape[0]
    test_pairing_ix = np.arange(test_n)
    result["test_sol"] = ad.AnnData(np.eye(test_n), uns={"pairing_ix": test_pairing_ix})
    if val_size:
        val_pairing_ix = np.arange(val_size)
        result["val_sol"] = ad.AnnData(
            np.eye(val_size), uns={"pairing_ix": val_pairing_ix}
        )
    return result


def load_custom_mp_data(task_type, train_batches, test_batches, val_size=None):
    if task_type == "gex_to_adt":
        first = ad.read_h5ad(COMMON_GEX_ADT)
        second = ad.read_h5ad(COMMON_ADT)
    elif task_type == "adt_to_gex":
        first = ad.read_h5ad(COMMON_ADT)
        second = ad.read_h5ad(COMMON_GEX_ADT)
    elif task_type == 'atac_to_gex':
        first = ad.read_h5ad(COMMON_ATAC)
        second = ad.read_h5ad(COMMON_GEX_ATAC)
    elif task_type == 'gex_to_atac':
        first = ad.read_h5ad(COMMON_GEX_ATAC)
        second = ad.read_h5ad(COMMON_ATAC)
    else:
        raise NotImplementedError()

    train_first = select_batches(first, train_batches)
    test_first = select_batches(first, test_batches)

    train_second = select_batches(second, train_batches)
    test_second = select_batches(second, test_batches)

    result = {
        "test_mod1": test_first,
        "test_mod2": test_second,
    }

    if val_size:
        all_idx = np.arange(train_second.shape[0])
        np.random.shuffle(all_idx)
        val_idx = all_idx[:val_size]
        train_idx = all_idx[val_size:]
        result["train_mod1"] = train_first[train_idx]
        result["val_mod1"] = train_first[val_idx]
        result["train_mod2"] = train_second[train_idx]
        result["val_mod2"] = train_second[val_idx]
    else:
        result["train_mod1"] = train_first
        result["train_mod2"] = train_second
    return result



def select_batches(dataset, batches):
    dataset_batches = dataset.obs["batch"].astype("string")
    selected_idx = dataset_batches.apply(lambda batch: batch in batches).values
    return dataset[selected_idx]


JE_CITE_SOLUTION = "data/official/joint_embedding/openproblems_bmmc_cite_phase1/openproblems_bmmc_cite_phase1.censor_dataset.output_solution.h5ad"
COMMON_GEX_ADT = "data/official/common/openproblems_bmmc_cite_phase1/openproblems_bmmc_cite_phase1.manual_formatting.output_rna.h5ad"
COMMON_GEX_ATAC = "data/official/common/openproblems_bmmc_multiome_phase1/openproblems_bmmc_multiome_phase1.manual_formatting.output_rna.h5ad"
COMMON_ADT = "data/official/common/openproblems_bmmc_cite_phase1/openproblems_bmmc_cite_phase1.manual_formatting.output_mod2.h5ad"
COMMON_ATAC = "data/official/common/openproblems_bmmc_multiome_phase1/openproblems_bmmc_multiome_phase1.manual_formatting.output_mod2.h5ad"
