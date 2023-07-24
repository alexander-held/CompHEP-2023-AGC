import asyncio
import json

import awkward as ak
import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import uproot

from func_adl_servicex import ServiceXSourceUpROOT
from func_adl import ObjectStream
from coffea.processor import servicex
from servicex import ServiceXDataset


def get_client(af="coffea-casa"):
    if af == "coffea-casa":
        from dask.distributed import Client

        client = Client("tls://localhost:8786")

    elif af == "EAF":
        from lpcdaskgateway import LPCGateway

        gateway = LPCGateway()
        cluster = gateway.new_cluster()
        cluster.scale(10)
        print("Please allow up to 60 seconds for HTCondor worker jobs to start")
        print(f"Cluster dashboard: {str(cluster.dashboard_link)}")

        client = cluster.get_client()

    elif af == "local":
        from dask.distributed import Client

        client = Client()

    else:
        raise NotImplementedError(f"unknown analysis facility: {af}")

    return client


def set_style():
    mpl.style.use("ggplot")
    plt.rcParams["axes.facecolor"] = "none"
    plt.rcParams["axes.edgecolor"] = "222222"
    plt.rcParams["axes.labelcolor"] = "222222"
    plt.rcParams["xtick.color"] = "222222"
    plt.rcParams["ytick.color"] = "222222"
    plt.rcParams["font.size"] = 10
    plt.rcParams["text.color"] = "222222"
    plt.rcParams["axes.grid"] = False  # for cabinetry modifier_grid


def construct_fileset(n_files_max_per_sample, use_xcache=False):
    # using https://atlas-groupdata.web.cern.ch/atlas-groupdata/dev/AnalysisTop/TopDataPreparation/XSection-MC15-13TeV.data
    # for reference
    # x-secs are in pb
    xsec_info = {
        "ttbar": 396.87 + 332.97, # nonallhad + allhad, keep same x-sec for all
        "single_top_s_chan": 2.0268 + 1.2676,
        "single_top_t_chan": (36.993 + 22.175)/0.252,  # scale from lepton filter to inclusive
        "single_top_tW": 37.936 + 37.906,
        "wjets": 61457 * 0.252,  # e/mu+nu final states
        "data": None
    }

    # list of files
    with open("nanoaod_inputs.json") as f:
        file_info = json.load(f)

    # process into "fileset" summarizing all info
    fileset = {}
    for process in file_info.keys():
        if process == "data":
            continue  # skip data

        for variation in file_info[process].keys():
            file_list = file_info[process][variation]["files"]
            if n_files_max_per_sample != -1:
                file_list = file_list[:n_files_max_per_sample]  # use partial set of samples

            file_paths = [f["path"] for f in file_list]
            if use_xcache:
                file_paths = [f.replace("https://xrootd-local.unl.edu:1094", "root://red-xcache1.unl.edu") for f in file_paths]
            nevts_total = sum([f["nevts"] for f in file_list])
            metadata = {"process": process, "variation": variation, "nevts": nevts_total, "xsec": xsec_info[process]}
            fileset.update({f"{process}__{variation}": {"files": file_paths, "metadata": metadata}})

    return fileset


def save_histograms(all_histograms, fileset, filename):
    nominal_samples = [sample for sample in fileset.keys() if "nominal" in sample]

    pseudo_data = (8*all_histograms[:, :, "ttbar", "nominal"] + all_histograms[:, :, "ttbar", "ME_var"] + all_histograms[:, :, "ttbar", "PS_var"]) / 10  +\
        all_histograms[:, :, "wjets", "nominal"] +\
        all_histograms[:, :, "single_top_s_chan", "nominal"] +\
        all_histograms[:, :, "single_top_t_chan", "nominal"] +\
        all_histograms[:, :, "single_top_tW", "nominal"]

    with uproot.recreate(filename) as f:
        for region in ["4j1b", "4j2b"]:
            f[f"{region}_pseudodata"] = pseudo_data[110j::hist.rebin(2), region]
            for sample in nominal_samples:
                sample_name = sample.split("__")[0]
                f[f"{region}_{sample_name}"] = all_histograms[110j::hist.rebin(2), region, sample_name, "nominal"]

                # b-tagging variations
                for i in range(4):
                    for direction in ["up", "down"]:
                        variation_name = f"btag_var_{i}_{direction}"
                        f[f"{region}_{sample_name}_{variation_name}"] = all_histograms[110j::hist.rebin(2), region, sample_name, variation_name]

                # jet energy scale variations
                for variation_name in ["pt_scale_up", "pt_res_up"]:
                    f[f"{region}_{sample_name}_{variation_name}"] = all_histograms[110j::hist.rebin(2), region, sample_name, variation_name]

            # ttbar modeling
            f[f"{region}_ttbar_ME_var"] = all_histograms[110j::hist.rebin(2), region, "ttbar", "ME_var"]
            f[f"{region}_ttbar_PS_var"] = all_histograms[110j::hist.rebin(2), region, "ttbar", "PS_var"]

            f[f"{region}_ttbar_scaledown"] = all_histograms[110j :: hist.rebin(2), region, "ttbar", "scaledown"]
            f[f"{region}_ttbar_scaleup"] = all_histograms[110j :: hist.rebin(2), region, "ttbar", "scaleup"]

            # W+jets scale
            f[f"{region}_wjets_scale_var_down"] = all_histograms[110j :: hist.rebin(2), region, "wjets", "scale_var_down"]
            f[f"{region}_wjets_scale_var_up"] = all_histograms[110j :: hist.rebin(2), region, "wjets", "scale_var_up"]


def make_datasource(fileset:dict, name: str, query: ObjectStream, ignore_cache: bool, backend_name: str = "uproot"):
    """Creates a ServiceX datasource for a particular Open Data file."""
    datasets = [ServiceXDataset(fileset[name]["files"], backend_name=backend_name, ignore_cache=ignore_cache)]
    return servicex.DataSource(
        query=query, metadata=fileset[name]["metadata"], datasets=datasets
    )


# functions creating systematic variations
def jet_pt_resolution(pt):
    # normal distribution with 1% variations, shape matches jets
    counts = ak.num(pt)
    pt_flat = ak.flatten(pt)
    resolution_variation = np.random.normal(np.ones_like(pt_flat), 0.01)
    return ak.unflatten(resolution_variation, counts)


class ServiceXDatasetGroup():
    def __init__(self, fileset, backend_name="uproot", ignore_cache=False):
        self.fileset = fileset

        # create list of files (& associated processes)
        filelist = []
        for i, process in enumerate(fileset):
            filelist += [[filename, process] for filename in fileset[process]["files"]]

        filelist = np.array(filelist)
        self.filelist = filelist
        self.ds = ServiceXDataset(filelist[:,0].tolist(), backend_name=backend_name, ignore_cache=ignore_cache)

    def get_data_rootfiles_uri(self, query, as_signed_url=True, title="Untitled"):

        all_files = np.array(self.ds.get_data_rootfiles_uri(query, as_signed_url=as_signed_url, title=title))
        parent_file_urls = np.asarray([f.replace(":", "/").split("1094//")[-1] for f in np.array([f.file for f in all_files])])

        # order is not retained after transform, so we can match files to their parent files using the filename
        # (replacing / with : to mitigate servicex filename convention )
        parent_key = np.array([np.where(parent_file_urls==self.filelist[i][0].split("1094//")[-1])[0][0]
                               for i in range(len(self.filelist))])

        files_per_process = {}
        for i, process in enumerate(self.fileset):
            # update files for each process
            files_per_process.update({process: all_files[parent_key[self.filelist[:,1]==process]]})

        return files_per_process
