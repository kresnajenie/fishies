# import matplotlib as plt
import os
import anndata
import json
import numpy as np
from scipy.stats import zscore
from scipy.stats import pearsonr
import pandas as pd
from datetime import datetime
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import linregress
import os
import scipy.stats as stats
from shapely.geometry import Point, Polygon




class Jenie:
    def get_pallete(name, color_dict_file='color_dicts.json'):
        with open(f"/Users/kjenie/DataspellProjects/ptb_timepoints/Class/{color_dict_file}", 'r') as f:
            color_dicts = json.load(f)
        if name in color_dicts:
            return color_dicts[name]
        else:
            raise ValueError(f"Color dictionary '{name}' not found in {color_dict_file}")
        
    immnw4edu_celltype = {
        "0": "Stage 2",
        "1": "Choroid plexus",
        "2": "Oligos",
        "3": "Stage 1",
        "4": "Ependymal",
        "5": "Stage 3",
        "6": "Lateral ependymal",
        "7": "GABAergic neurons",
        "8": "Medial ependymal",
        "9": "Oligos",
    }
    
    immnasoedu_celltype = {
        "0": "Stage 1",
        "1": "Stage 3",
        "2": "Stage 2",
        "3": "Medial ependymal",
        "4": "Immature OB",
        "5": "Lateral ependymal",
        "6": "Ependymal",
        "7": "Microglia",
        "8": "Oligos",
    }
    controledu_celltype = {
        "0": "Olfactory bulb neurons",
        "1": "Olfactory bulb neurons",
        "2": "Striatal neurons",
        "3": "Sparse endothelial",
        "4": "Corpus collosum oligodendrocytes",
        "5": "Cortical neurons",
        "6": "Midbrain oligodendrocytes",
        "7": "Olfactory bulb neurons",
        "8": "Sparse astrocytes",
        "9": "Olfactory bulb outer plexiform neurons",
        "10": "Subpial astrocytes",
        "11": "Cortical oligodendrocytes",
        "12": "Midbrain neurons",
        "13": "Cortical neurons 2",
        "14": "Midbrain astrocytes",
        "15": "Sparse oligodendrocytes",
        "16": "Cortical L4/5 neurons",
        "17": "Sparse GABAergic neurons",
        "18": "Sparse microglia",
        "19": "Midbrain neurons",
        "20": "Thalamic neurons",
        "21": "Olfactory bulb neurons",
        "22": "Dentate gyrus",
        "23": "Immature cells!!!",
        "24": "Striatal oligodendrocytes",
        "25": "Choroid plexus",
        "26": "CA1/3",
        "27": "Olfactory bulb fibertract endothelial",
        "28": "Olfactory bulb glumelar layer neurons",
        "29": "Cortical neurons",
        "30": "Olfactory bulb neurons",
        "31": "Olfactory bulb neurons",
        "32": "Nuclear thalamic neurons",
        "33": "Olfactory tubercel neurons",
        "34": "Cortical neurons",
        "35": "Olfactory bulb neurons",
    }

    asoedu_celltype = {
        "0": "Olfactory bulb neurons (immature)",
        "1": "Striatal neurons",
        "2": "Sparse oligodendrocytes",
        "3": "Cortical striatal astrocytes",
        "4": "Thalamic midbrain oligodendrocytes",
        "5": "Cortical neurons",
        "6": "Olfactory bulb outer lexiform layer neurons",
        "7": "Subpial ventrical astrocytes",
        "8": "Midbrain neurons",
        "9": "Cortical L4/5 neurons",
        "10": "Sparse endothelial (microglia)",
        "11": "Cortical oligodendrocytes",
        "12": "Sparse endothelial (microglia) 2",
        "13": "Cortical L2/3 neurons",
        "14": "Sparse activated microglia/astrocytes",
        "15": "Sparse oligodendrocytes 2",
        "16": "Olfactory nucleus anterior neurons",
        "17": "Coritcal and stiratal neurons",
        "18": "Olfactory bulb astrocytes",
        "19": "Thalamic midbrain astrocytes",
        "20": "Thalamic neurons",
        "21": "Olfactory bulb mitral layer neurons",
        "22": "Striatal neurons",
        "23": "Olfactory bulb neurons",
        "24": "Immature cells!!!",
        "25": "Dentate gyrus neurons",
        "26": "Cortical neurons 2",
        "27": "Choroid plexus",
        "28": "CA1",
        "29": "Olfactory tuberkel neurons",
        "30": "Reticular thalamus neurons",
        "31": "CA3",
        "32": "Hybrid cells",
        "33": "Olfactory bulb neurons",
        "34": "Olfactory bulb neurons",
        "35": "Hippocampal astoryctes",
        "36": "Olfactory bulb neurons",
    }
    edu_celltype = {
        "0": "GABAergic olfactory bulb neurons (immature)",
        "1": "GABAergic striatal neurons",
        "2": "Sparse oligodendrocytes",
        "3": "GABAergic olfactory bulb immature neurons",
        "4": "Cortical neurons",
        "5": "Sparse astrocytes",
        "6": "Sparse endothelial",
        "7": "Subpial ventricular astrocytes",
        "8": "Cortical L5 neurons",
        "9": "Sparse pvm",
        "10": "GABAergic olfactury bulb immature outer neurons",
        "11": "Sparse oligodendrocytes 2",
        "12": "Thalamic astrocytes",
        "13": "Midbrain neurons",
        "14": "Anterior olfactory nucleus neurons",
        "15": "Sparse endothelial 2",
        "16": "Cortical oligodendrocytes",
        "17": "Thalamic neurons",
        "18": "Sparse GABAergic neurons",
        "19": "Glomerular layer olfactory bulb neurons",
        "20": "Immature cells!!!",
        "21": "Olfactur bulb astrocytes (path)",
        "22": "GABAergic mitral layer olfactory bulb imamture neurons",
        "23": "Dentate gyrus neurons",
        "24": "Glomerular layer astrocytes",
        "25": "Striatum neurons (oligos)",
        "26": "Cortical L2/3 neurons",
        "27": "GABAergic neurons",
        "28": "Cortical L2/3 neurons 2",
        "29": "CA1 neurons",
        "30": "CA3 neurons",
        "31": "GABAergic striatal neurons 2",
        "32": "GABAergic olfactory bulb neurons",
    }

    edu_immn = {
        "0": "Chroid plexus",
        "1": "Immature GABAergic striatal neurons",
        "2": "Stage 1",
        "3": "Ependymal cells",
        "4": "Oligodendrocytes",
        "5": "Stage 3",
        "6": "Lateral ependymal",
        "7": "Immature OB neurons",
        "8": "Medial ependymal",
        "9": "GABAergic striatal neurons",
        "10": "Oligodendrocytes 2",
        "8": "Medial ependymal",

    }


    merfish_celltype_general = {
        'CA1 neurons': "Neurons",
        'CA2 neurons': "Neurons",
        'CA3 neurons': "Neurons",
        'Cortical GABAergic neuron': "Neurons",
        'Cortical sparse neurons': "Neurons",
        'GABAergic neurons': "Neurons",
        'Hypothalamic neurons': "Neurons",
        'Immature DG neurons': "Neurons",
        'Immature neurons': "Neurons",
        'Interneuron': "Neurons",
        'L1 neurons': "Neurons",
        'L1/2 neurons': "Neurons",
        'L2/3 neurons': "Neurons",
        'L3 neurons': "Neurons",
        'L4 neurons': "Neurons",
        'L5 neurons': "Neurons",
        'L6 neurons': "Neurons",
        'Mature DG neurons': "Neurons",
        'Medial L1 Glutamatergic neurons': "Neurons",
        'Medial L1 neurons': "Neurons",
        'Medial L2 neurons': "Neurons",
        'Nucleus of thalamus neurons': "Neurons",
        'Outer thalamic neurons': "Neurons",
        'Striatal neurons': "Neurons",
        'Thalamic neuron': "Neurons",
        'Upper thalamic neurons': "Neurons",
        'New Cell Type': "Neurons",

        'Astrocyte': "Astrocytes",
        'Cortical astrocyte': "Astrocytes",
        'Hippocampal astrocytes': "Astrocytes",
        'Outer thalamic astrocyte': "Astrocytes",
        'Striatal astrocyte': "Astrocytes",
        'Subventricular immature astrocytes': "Astrocytes",
        'Upper thalamic astrocyte': "Astrocytes",
        'Radial glia': 'Astrocytes',
        'Subpial mature astrocytes': "Ependymal cells",


        'Corpus callosum oligo': "Oligodendrocytes",
        'Cortical oligos': "Oligodendrocytes",
        'Oligo sparse': "Oligodendrocytes",
        'Oligodendrocytes': "Oligodendrocytes",
        'Outer thalamic oligo': "Oligodendrocytes",
        'Sparse oligo': "Oligodendrocytes",
        'Striatal oligo': "Oligodendrocytes",
        'Thalamic oligo': "Oligodendrocytes",

        'Corpus callosum endothelial': "Endothelial cells",
        'Cortical endothelial': "Endothelial cells",
        'Endothelial sparse': "Endothelial cells",
        'Hippocampal/striatum endothelial': "Endothelial cells",
        'Outer thalamic endothelial': "Endothelial cells",
        'Thalamic endothelial': "Endothelial cells",
        'Pericyte': 'Endothelial cells',
        'Subpial endothelial cells': "Ependymal cells",

        'Sparse microglia': "Microglia",
        'Thalamic microglia': "Microglia",
        'Microglia corpus callosum': "Microglia",
        'Microglia sparse': "Microglia",

        'Ependymal cells': 'Endothelial cells',
        'Progenitor cells': 'Ependymal cells',
        'Thalamic OPC': 'Oligodendrocytes',
    }

    scrna_celltype_general = {
        'Astrocytes': "Astrocytes", 
        'Thalamic glia': "Astrocytes",

        'Dentate gyrus': "Neurons",
        'Immature neuron': "Neurons",
        'CA1 neurons': "Neurons",
        'CA2 neurons': "Neurons",
        'CA3 neurons': "Neurons",
        'Choroid plexus neurons': "Neurons",
        'Cortical neurons': "Neurons",
        'GABAergic neuron 1': "Neurons",
        'GABAergic neuron 2': "Neurons",
        'L1/2 neurons': "Neurons",
        'L3 neurons': "Neurons",
        'L5/6 neurons': "Neurons",
        'L6 neurons': "Neurons",
        'Thalamic neurons': "Neurons",

        'Endothelial': "Endothelial cells",
        'Pial perictyes': "Endothelial cells",

        'OPC': "Oligodendrocytes",
        'Oligodendrocytes': "Oligodendrocytes",

        'Ependymal cells': "Ependymal cells",
        'Immune cells': "Immune cells",

    }


    neuro_genes = ['Sox11', 'Dcx', 'Dlx1', 'Vim', 'Mki67', 'Sox2', 'Ptbp2', 'Gfap', 'Aldh1a1', 'Pdgfra', 'Sox9', 'Aqp4', 'Ascl1', 'Bsg', 'S100b', 'Olig1', 'Olig2', 'Dlx2', 'Pax6', 'Rtn4', 'Rest', 
    'Ptbp1'
    ]

    svz_cells = [
    "Immature neurons",
    "Progenitor cells",
    # "Ependymal cells",
    "Immature DG neurons",
    # "Mature DG neurons"
    ]

    svz_4week = [
        "SVZ ependymal cells",
        "SVZ astrocytes",
        "Choroid plexus",
        "Sparse astrocytes",
        "Sparse neurons (immature)",
        "Fibertracts oligodendrocytes",
    ]

    dg_4week = [
        "Dentate gyrus neurons",
        "Dentate gyrus immature neurons",
        "Hippocampal astrocytes",
        "Hippocampal endothelial",
    ]

    week4_clusters = {
        '0': 'Cerebellar granular neurons',
        '1': 'Olfactory bulb GABAergic granule neurons',
        '2': 'Brain stem oligodendrocytes',
        '3': 'L6 glutamatergic neurons',
        '4': 'Fibertracts oligodendrocytes',
        '5': 'Sparse astrocytes',
        '6': 'Cerebellar molecular layer neurons',
        '7': 'Brain stem endothelial',
        '8': 'L2/3 visual cortex neurons',
        '9': 'Brain stem oligodendrocytes (astrocytic)',
        '10': 'Inferior colliculus neurons',
        '11': 'Fibertracts oligodendrocytes 2',
        '12': 'Cortical neurons',
        '13': 'Pia matter',
        '14': 'Oligodendrocytes (artifact)',
        '15': 'Striatal neurons',
        '16': 'Thalamic neurons',
        '17': 'Cortical endothelial',
        '18': 'Striatal neurons 2',
        '19': 'L5/6 cortical neurons',
        '20': 'Hippocampal astrocytes',
        '21': 'Fibertracts oligodendrocytes 3',
        '22': 'L6 oligodendrocytes',
        '23': 'Olfactory bulb plexiform neurons',
        '24': 'Sparse endothlial',
        '25': 'Olfactory bulb glomerular neurons',
        '26': 'L2/3 somatomotor neurons',
        '27': 'Thalamic neurons 2',
        '28': 'Medulla neurons',
        '29': 'Nucleus accumbens neurons',
        '30': 'Superior colliculus neurons',
        '31': 'Brain stem astrocytes',
        '32': 'Dentate gyrus neurons',
        '33': 'Astrocytes',
        '34': 'Brain stem astrocytes 2',
        '35': 'Thalamic oligodendrocytes',
        '36': 'Cerebellar granular neurons 2',
        '37': 'Striatal astrocytes',
        '38': 'Striatal endothelial',
        '39': 'Thalamic oligodendrocytes 2',
        '40': 'Sparse oligodendrocytes',
        '41': 'Medullary reticular nucleus astrocytes',
        '42': 'Pallidum neurons',
        '43': 'Cortical neurons',
        '44': 'Sparse neurons',
        '45': 'Striatal neurons 3',
        '46': 'Cerebellar fibertract oligodendrocytes',
        '47': 'Thalamic oligodendrocytes 3',
        '48': 'L1 anterior olfactory nucleus astrocytes',
        '49': 'L2 anterior olfactory nucleus neurons',
        '50': 'SVZ ependymal cells',
        '51': 'Sparse oligodendrocytes 2',
        '52': 'Olfactory bulb fibertract astrocytes',
        '53': 'Hippocampal CA1 neurons',
        '54': 'Striatal oligodendrocytes',
        '55': 'Thalamic endothelial',
        '56': 'Sparse neurons',
        '57': 'Cortical oligodendrocytes',
        '58': 'SVZ astrocytes',
        '59': 'Cortical oligodendrocytes',
        '60': 'Striatal oligodendrocytes',
        '61': 'Medulla fibertract astrocytes',
        '62': 'Pons neurons',
        '63': 'Olfactory tubercle neurons',
        '64': 'Hippocampal endothelial',
        '65': 'Substantia nigra astrocytes',
        '66': 'Hippocampal CA3 neurons',
        '67': 'Retrospenial area neurons',
        '68': 'Sparse neurons',
        '69': 'Olfactory immature neurons',
        '70': 'Reticular nucleus neurons',
        '71': 'Retrospenial area neurons',
        '72': 'Sparse micro PVM',
        '73': 'Sparse neurons (immature)',
        '74': 'Cortical & Hippocampal neurons (immature)',
        '75': 'Cortical micro PVM',
        '76': 'L2/3 cortical neurons',
        '77': 'Dorsal stream endothelial',
        '78': 'Choroid plexus',
        '79': 'Artifact',
        '80': 'L1 neurons',
        '81': 'Dentate gyrus immature neurons',
        '82': 'Artifact',
        '83': 'Thalamic oligodendrocytes',
        '84': 'Artifact',
        '85': 'Artifact',
        '86': 'Ventral posteromedial nucleus oligodendrocytes',
        '87': 'Lateral preoptic oligodendrocytes',
        '88': 'Artifact',
        '89': 'Artifact',
        '90': 'Artifact',
        '91': 'Superior vestibular nucleus',
        '92': 'L5 neurons',
        '93': 'Endopiriform nucleus, dorsal-part neurons',
        '94': 'Striatal neurons 4',
        '95': 'Artifact',
        '96': 'Artifact',
        '97': 'Artifact',
        '98': 'Artifact',
        '99': 'Artifact',
        '100': 'Artifact',
        '101': 'Artifact',
        '102': 'Artifact',
        '103': 'Artifact',
        '104': 'Artifact',
        '105': 'Artifact',
        '106': 'Olfactory bulb immature neurons',
        '107': 'Artifact',
        '108': 'Subiculum neurons',
        '109': 'Artifact',
        '110': 'Artifact',
        '111': 'Sst neurons',
        '112': 'Artifact',
        '113': 'Artifact',
        '114': 'Artifact',
        '115': 'Artifact',
        '116': 'Artifact',
        '117': 'Artifact',
        '118': 'Artifact'
    }

    def plot_gene_scdataa(adata, gene='Sox10',sz_min=5,sz_max=30,transpose=1,flipx=1,flipy=1, percentile=95, fig=(15,15), umap=False, ref=None, ssm=5, bg=False):
        plt.style.use("dark_background")
        plt.figure(figsize=fig, facecolor="black")
        ign = list(ref.var.index).index(gene) if ref else list(adata.var.index).index(gene)

        group = "X_umap" if umap else "X_spatial"

        if ref and bg:
            x2,y2 = (ref.obsm[group]*[flipx,flipy])[:,::transpose].T
            plt.scatter(x2, y2, c='gray', s=ssm, marker='o')

        # nmax = np.percentile(adata.obsm["X_raw"][:, ref.var.index.get_loc(gene)], percentile) if ref else np.percentile(adata.obsm["X_raw"][:, adata.var.index.get_loc(gene)], percentile)

        # nmax = np.percentile(adata.X[:, ign], percentile)
        nmax= percentile
        

        # a = self.data.obsm["X_raw"][:, ref.var.index.get_loc(gene)]
        # if ref else np.percentile(self.data.obsm["X_raw"][:, self.data.var.index.get_loc(gene)], percentile)
        
        # Xcells = self.data.obsm['X_spatial'][:,::transpose]*[flipx,flipy]
        # self.data.obsm['X_spatial']
        if 'X_raw' not in adata.obsm:
            Xnorm = (np.exp(adata.X)-1)
            ncts = np.sum(Xnorm,axis=1)[0]
            adata.obsm['X_raw']=np.round(Xnorm/ncts*np.array(adata.obs['total_counts'])[:,np.newaxis])
        cts = adata.obsm['X_raw'][:,ign]

        cts[np.isnan(cts)]=0
        # ncts = np.clip(cts,0,1)
        ncts = np.clip(cts/nmax,0,1)
        size = sz_min+ncts*(sz_max-sz_min)
        from matplotlib import cm as cmap
        cols = cmap.coolwarm(ncts)

        
        x,y = (adata.obsm[group]*[flipx,flipy])[:,::transpose].T

        indices = np.argsort(size)
        x_sorted = x[indices]
        y_sorted = y[indices]
        cols_sorted = cols[indices]
        size_sorted = size[indices]

        # plt.scatter(x, y, c=cols, s=size, marker='.')
        plt.scatter(x_sorted, y_sorted, c=cols_sorted, s=size_sorted, marker='.')

        # Add color scale
        norm = plt.Normalize(0, nmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap.coolwarm)
        sm.set_array([])
        cbar = plt.colorbar(sm, shrink=0.5)
        cbar.set_label("Counts", rotation=270, labelpad=15)

        # Set the maximum value as a tick on the color bar
        # a = cbar.get_ticks()
        # print(a)
        cbar.set_ticks(np.append(cbar.get_ticks()[:-1], nmax))

        plt.title(gene)
        
        # plt.grid(b=False)
        plt.axis("off")
        plt.axis("equal")
        plt.legend()
        plt.tight_layout()

        # plt.savefig(f"{self.folder_path}/brainproj/brainproj_{gene}_{self.get_filename()}", transparent=True, dpi=self.dpi_save)

    

    def ptb_boxplot(adata, cells, length, group="MERFISH celltype", vis=True, y=None, gene="Ptbp1", mean=True, title="", save=None, fig=(10,6), outlier=True, cbm5=None, log=False, order=None, facecolor=None):
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")

        # Assuming you have a list of predetermined index values
        predetermined_index = adata.obs.index.tolist()
        

        # Assuming you have some data for "Ptbp1" and "leiden"
        # mat = adata.raw.X[:,position]
        # dense_mat = mat.toarray()
        # dense_list = dense_mat.flatten().tolist()
        print(cbm5)
        if cbm5 != None:
            if gene == "Ptbp1":
                data = {
                    # "Ptbp1": adata.obsm["X_raw"][:,position], 
                    "Ptbp1": adata.obs["New_PTB"], 
                    "celltype": adata.obs[group],  
                }
            else:
                position = adata.var.index.get_loc(gene)
                data = {
                    "Ptbp1": adata.layers["Raw counts"][:, position], 
                    "celltype": adata.obs[group],  
                }
        else:
            position = adata.var.index.get_loc(gene)
            data = {
                "Ptbp1": adata.obsm["X_raw"][:,position], 
                "celltype": adata.obs[group],  
            }

        # # Creating the new DataFrame
        df = pd.DataFrame(data, index=predetermined_index)

        # Set the style for dark background
        # sns.set_style("dark")
        # astro  = df[df["celltype"].isin(result_list)]


        d = df[df["celltype"].isin(cells)]

        # df = pd.DataFrame(data)
        if log:
            d["Ptbp1"] = np.log1p(d["Ptbp1"])


        # Calculate the mean Ptbp1 values for each cell type
        mean_values = d.groupby('celltype')['Ptbp1'].mean().round(10).sort_values(ascending=False) if mean else d.groupby('celltype')['Ptbp1'].median().round(10).sort_values(ascending=False)

        # print(mean_values)
        # mean_values = d.groupby('celltype')['ptbp1_log'].mean().sort_values(ascending=False)

        # Reorder the cell types based on the mean values
        sorted_celltypes = mean_values.index.tolist()

        # Creating the box plot
        plt.figure(figsize=fig, dpi=300)
        # sns.boxplot(x='celltype', y='ptbp1_log', data=d, order=sorted_celltypes)
        if facecolor:
            boxprops = dict(edgecolor='white', facecolor=facecolor)
        else:
            boxprops = dict(edgecolor='white')


        sns.boxplot(x='celltype', y='Ptbp1', data=d, order=order if order else sorted_celltypes[:length], showfliers=outlier, boxprops=boxprops, medianprops=dict(color='white'), flierprops=dict(markerfacecolor='white', markeredgecolor='white'), whiskerprops={'color':'white'}, capprops={'color':'white'}, )
        sns.despine()
        plt.xlabel('celltype')
        plt.ylabel(gene)
        plt.title(f'{title} Box Plot: {gene} vs. celltype (sorted by {"mean" if mean else "median"})')
        plt.xticks(rotation=90, visible=vis)  # Rotate x-axis labels for better readability
        plt.grid(False)
        y_min, y_max = plt.ylim()

        if y:
            plt.ylim(y[0], y[1])

        print("Y-axis minimum value:", y_min)
        print("Y-axis maximum value:", y_max)

        if save:
            plt.figure(figsize=fig)
            plt.savefig(save, transparent=True, dpi=150)

        # plt.show()
        # return d


    def cut_data(adata):
        polygon = Polygon([[123.07344484064697, 1732.2878504558776], [-821.5553340318229, 1032.3520685202511], [-1914.1380180288963, 764.8969323334677], [-3171.746211588028, -26.087406601913244], [-3444.891882587296, -600.8314226628736], [-2881.528936151305, -1431.649505285648], [-1134.5347487184845, -1670.6519674100082], [1238.4182680876593, -1687.7235718474622], [2621.2182275214554, -1454.4116445355876], [3304.082405019627, -1243.8618564736516], [2678.123575646304, 1265.6639958321266], [1426.2059168996566, 2796.4178603905257], [458.81499877724764, 2699.6787685782856]])
        keep = [Point(x,y).within(polygon) for x,y in adata.obsm['X_spatial']]
        adata_cut = adata[keep]
        adata_s2 = adata[adata.obs["tag"]=='DCBBL1_4week_6_2_2023_set2']
        adata_s3 = adata[adata.obs["tag"]=='DCBBL1_4week_6_2_2023_set3']
        adata_cut_comb = sc.concat([adata_cut,adata_s2,adata_s3])

        return adata_cut_comb


    def __init__(self, data, name, folder,  pallete="color_dict",dpi_save=300, save=False):
        if not isinstance(data, anndata.AnnData):
            raise ValueError("Provided data is not an anndata structure.")
        
        self.dpi_save = dpi_save
        self.data = data
        self.name = name
        self.folder = folder
        self.figures_dir = "figures"
        self.folder_path = os.path.join(self.figures_dir, self.folder)
        current_path = os.getcwd()
        parent_directory = os.path.dirname(current_path)
        self.path = parent_directory
        self.pallete = self.get_colors(name=pallete)
        self.save = save
        self.create_folder = False
        
        if save:
            # Check if folder already exists
            if os.path.exists(self.folder_path):
                num_files = len(os.listdir(self.folder_path))
                if num_files == 0:
                    print(f"WARNING: A folder with the same name already exists in the figures directory, but it is empty: {self.folder_path}")
                else:
                    print(f"WARNING: A folder with the same name already exists in the figures directory and contains {num_files} files: {self.folder_path}")
            else:
                # Create the figures folder
                os.makedirs(self.folder_path)
            
            # Create the figures subdirectories if they don't exist
            subdirs = ["umap", "linnarson", "brainproj","trackplot","cebrian", "gene_corr"]
            for subdir in subdirs:
                subdir_path = os.path.join(self.folder_path, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path)
            
            self.create_folder = True

        
    def __str__(self):
        return str(self.data)
    
    def __repr__(self):
        return str(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def change_save(self, save):
        if save == True and self.create_folder == False:
            # Check if folder already exists
            if os.path.exists(self.folder_path):
                num_files = len(os.listdir(self.folder_path))
                if num_files == 0:
                    print(f"WARNING: A folder with the same name already exists in the figures directory, but it is empty: {self.folder_path}")
                else:
                    print(f"WARNING: A folder with the same name already exists in the figures directory and contains {num_files} files: {self.folder_path}")
            else:
                # Create the figures folder
                os.makedirs(self.folder_path)
            
            # Create the figures subdirectories if they don't exist
            subdirs = ["umap", "linnarson", "brainproj","trackplot","cebrian", "gene_corr"]
            for subdir in subdirs:
                subdir_path = os.path.join(self.folder_path, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path)
            
            self.create_folder = True

    
    def get_folder(self):
        print(self.folder)
    
    
    def get_cluster(self, celltypes, group="MERFISH celltype"):
        a = set(self.data.obs[group])
        b = set(celltypes)
        if type(celltypes) == str:
            celltypes = [celltypes]
        if not set(celltypes).issubset(set(self.data.obs[group])):
            raise ValueError("The specified cell types were not found in the group column.")
        return self.data[self.data.obs[group].isin(celltypes)]
    
    def change_style(self, style):
        plt.style.use(style)
    
    def get_figures_path(self):
        return self.figures_dir
    
    def get_folder_path(self):
        return self.folder_path
    
    def change_pallete(self, pallete):
        self.pallete = self.get_colors(name=pallete)

    def change_dpi(self, dpi):
        self.dpi_save = dpi

    def all_umap(self, group, treatment=None):
        for c in tqdm(set(self.data.obs[group])):
            self.umap(group=group, groups=[c], treatment=treatment)

    def all_brainproj(self, group, treatment=None, s=1, ref=None):
        for c in tqdm(set(self.data.obs[group])):
            self.brainproj(group=group, groups=[c], treatment=treatment, s=s, ref=ref)

    def gene_corr_ptb(self, genex, geney, groups=None, group=None, data =None, color=True, xmax=None, ymax=None, log=False):
        subset = data if data else self.data
        title = f"{self.name}"

        if groups != None:
            subset = self.get_cluster(groups, group=group)
            title = f"{self.name} {groups}"

        if genex == "Ptbp1":
            x = subset.obs["New_PTB"]
        else:
            position_x = self.data.var.index.get_loc(genex)
            x = subset.layers["Raw counts"][:,position_x]
            
        if geney == "Ptbp1":
            y = subset.obs["New_PTB"]
        else:
            position_y = self.data.var.index.get_loc(geney)
            y = subset.layers["Raw counts"][:,position_y]
            # y = subset[:,geney].layers["Raw counts"].flatten()

        if log == True:
            x += 1
            x = np.log(x)
            y += 1
            y = np.log(y)
            print(x)

        if color:
            colors = subset.obs[group].map(self.pallete)



        # Set the style to dark background
        plt.style.use('dark_background')

        # Calculate the Pearson correlation coefficient
        pearson_corr, _ = pearsonr(x, y)

        # Scatter plot
        x = x + np.random.rand(len(x))
        y = y + np.random.rand(len(y))

        plt.scatter(x, y, c=colors if color else None, s=2)
        

        # Calculate the line of best fit
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        line_fit = intercept + slope * x

        

        # Add the Pearson correlation coefficient to the plot
        plt.text(0.7, 0.9, f"Pearson correlation: {pearson_corr:.2f}", transform=plt.gca().transAxes, color='white')

        # Set the x-axis and y-axis labels
        plt.xlabel(genex)
        plt.ylabel(geney)

        plt.xlim(0)
        plt.ylim(0)
        if xmax is not None:
            plt.xlim(0, xmax)  # Set x-axis limits from 0 to xmax
        if ymax is not None:
            plt.ylim(0, ymax)  # Set y-axis limits from 0 to ymax
        # Plot the dashed line of best fit
        plt.plot(x, line_fit, color='red', linewidth=2)
        # Remove the grid lines
        plt.grid(False)

        # Set the title
        plt.title(title)

        # Get the maximum value on the x-axis
        x_max = plt.xlim()[1]
        y_max = plt.ylim()[1]

        if self.save:
            plt.savefig(f"{self.folder_path}/gene_corr/gene_corr_{genex}{geney}_{groups}_{self.get_filename()}", transparent=True, dpi=self.dpi_save)

        # Display the plot
        plt.show()

        return (x_max, y_max)
    
    def gene_corr_normal_anndata(self, gene, groups=None, group=None, data =None, color=True, xmax=None, ymax=None, raw=False):
        subset = data if data else self.data
        title = f"{self.name}"

        if groups != None:
            subset = self.get_cluster(groups, group=group)
            title = f"{self.name} {groups}"

        if raw:
            x = subset.raw[:, "Ptbp1"].X.flatten()
            y = subset.raw[:, gene].X.flatten()
        
        else:
            try:
                x = subset[:, "Ptbp1"].X.toarray().flatten()
            except KeyError:
                x = subset.obs["Ptbp1"].to_numpy().flatten()

            try:
                y = subset[:, gene].X.toarray().flatten()
            except KeyError:
                # Handle the KeyError for `gene` as needed
                y = subset.obs[gene].to_numpy().flatten()

        # Set the style to dark background
        plt.style.use('dark_background')


        # Calculate the Pearson correlation coefficient
        pearson_corr, _ = pearsonr(x, y)
        # Scatter plot
        x = x + np.random.rand(len(x))
        y = y + np.random.rand(len(y))

        if color:
            colors = subset.obs.new_clusters.map(self.pallete)

        plt.scatter(x, y, c=colors if color else None, s=2)

        # Calculate the line of best fit
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        line_fit = intercept + slope * x

        # Add the Pearson correlation coefficient to the plot
        plt.text(0.7, 0.9, f"Pearson correlation: {pearson_corr:.2f}", transform=plt.gca().transAxes, color='white')

        # Set the x-axis and y-axis labels
        plt.xlabel("Ptbp1")
        plt.ylabel(gene)

        plt.xlim(0)
        plt.ylim(0)

        if xmax is not None:
            plt.xlim(0, xmax)  # Set x-axis limits from 0 to xmax
        if ymax is not None:
            plt.ylim(0, ymax)  # Set y-axis limits from 0 to ymax
        # Plot the dashed line of best fit
        plt.plot(x, line_fit, color='red', linewidth=2)

        # Add p-value, F-statistic, and R-squared to the plot
        plt.text(0.7, 0.85, f"p-value: {p_value:.2e}", transform=plt.gca().transAxes, color='white')
        plt.text(0.7, 0.8, f"R-squared: {r_value**2:.2f}", transform=plt.gca().transAxes, color='white')
    
        # Remove the grid lines
        plt.grid(False)

        # Set the title
        plt.title(title)

        # Get the maximum value on the x-axis
        x_max = plt.xlim()[1]
        y_max = plt.ylim()[1]

        if self.save:
            plt.savefig(f"{self.folder_path}/gene_corr/gene_corr_{gene}_{groups}_{self.get_filename()}", transparent=True, dpi=self.dpi_save)

        plt.show()

        return (x_max, y_max)
    
    def gene_corr_other_anndata(self, geney, genex="Ptbp1", groups=None, group=None, data =None, color=True, xmax=None, ymax=None, raw=False):
        subset = data if data else self.data
        title = f"{self.name}"

        if groups != None:
            subset = self.get_cluster(groups, group=group)
            title = f"{self.name} {groups}"

        # Assuming you have a list of predetermined index values
        # predetermined_index_ptb = self.data.obs.index.tolist()
        position_ptb = self.data.var.index.get_loc(genex)

        # predetermined_index_gene = self.data.obs.index.tolist()
        position_gene = self.data.var.index.get_loc(geney)

        x = subset.obsm["X_raw"][:,position_ptb]
        y = subset.obsm["X_raw"][:,position_gene]
        

        # Set the style to dark background
        plt.style.use('dark_background')


        # Calculate the Pearson correlation coefficient
        pearson_corr, _ = pearsonr(x, y)
        # Scatter plot
        x = x + np.random.rand(len(x))
        y = y + np.random.rand(len(y))

        if color:
            colors = subset.obs["MERFISH celltype"].map(self.pallete)

        plt.scatter(x, y, c=colors if color else None, s=2)

        # Calculate the line of best fit
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        line_fit = intercept + slope * x

        # Add the Pearson correlation coefficient to the plot
        plt.text(0.7, 0.9, f"Pearson correlation: {pearson_corr:.2f}", transform=plt.gca().transAxes, color='white')

        # Set the x-axis and y-axis labels
        plt.xlabel(genex)
        plt.ylabel(geney)

        plt.xlim(0)
        plt.ylim(0)

        if xmax is not None:
            plt.xlim(0, xmax)  # Set x-axis limits from 0 to xmax
        if ymax is not None:
            plt.ylim(0, ymax)  # Set y-axis limits from 0 to ymax
        # Plot the dashed line of best fit
        plt.plot(x, line_fit, color='red', linewidth=2)

        # Add p-value, F-statistic, and R-squared to the plot
        plt.text(0.7, 0.85, f"p-value: {p_value:.2e}", transform=plt.gca().transAxes, color='white')
        plt.text(0.7, 0.8, f"R-squared: {r_value**2:.2f}", transform=plt.gca().transAxes, color='white')
    
        # Remove the grid lines
        plt.grid(False)

        # Set the title
        plt.title(title)

        # Get the maximum value on the x-axis
        x_max = plt.xlim()[1]
        y_max = plt.ylim()[1]

        if self.save:
            plt.savefig(f"{self.folder_path}/gene_corr/gene_corr_{genex}{geney}_{groups}_{self.get_filename()}", transparent=True, dpi=self.dpi_save)

        plt.show()

        return (x_max, y_max)

    
    def umap(self, group="leiden", subset=None, fig=(15,15), size=20, treatment=None, pallete=True, groups=None, style="default", embed=False, ref=None, legend="dfasd", bg_color="white",
    add_outline=True, legend_fontsize=5, legend_fontoutline=2,**kwargs):
        plt.style.use(style)

        sc.set_figure_params(figsize=fig)

        d = self.data
        title = f"{self.name} ({len(self.data):,d} cells)"

        if treatment != None:
            d = self.get_cluster([treatment[0]], group=treatment[1])
            title = f"{self.name} {treatment[0]}"

        if subset:
            d = self.get_cluster(subset, group)

        self.data.obs["x"] = "x"
        if ref:
            ref.obs["x"] = "x"
        ax = sc.pl.umap(ref if ref else self.data, show=False, return_fig=False, color="x", palette=[bg_color], legend_loc="dfad", size=int(0.75*size))
        
        fig = None

        a = sc.pl.umap(
            d[d.obs[group].isin(groups)] if groups else d,
            ax=ax if ax else None,
            size=size,
            color=[group],
            add_outline=add_outline,
            legend_loc=legend,
            legend_fontsize=legend_fontsize,
            legend_fontoutline=legend_fontoutline,
            frameon=False,
            title=title,
            palette=self.pallete if pallete else None,
            cmap="coolwarm",
            show=False,
            # return_fig=True,
            **kwargs
        )

        fig = a.get_figure()

        if self.save:
            fig.savefig(f"{self.folder_path}/umap/umap_{self.get_filename()}", transparent=True, dpi=self.dpi_save)

    def get_num_cells(self):
        return self.data.obs.groupby("new_clusters").count()["n_genes"]


    def reprocess(self, umap=True, n_neighbors=15):
        sc.pp.pca(self.data)
        sc.pp.neighbors(self.data, use_rep="X", n_neighbors=n_neighbors, metric="cosine")
        if umap:
            sc.tl.umap(self.data)

    def reumap(self, min_dist=0.5, random_state=0):
        sc.tl.umap(self.data, min_dist=min_dist, random_state=random_state)

    def recluster(self, res=1):
        sc.tl.leiden(self.data, resolution=res)


    def get_colors(self, name="color_dict", color_dict_file='color_dicts.json'):
        with open(f"{self.path}/Class/{color_dict_file}", 'r') as f:
            color_dicts = json.load(f)
        if name in color_dicts:
            return color_dicts[name]
        else:
            raise ValueError(f"Color dictionary '{name}' not found in {color_dict_file}")
        
    def brainproj(self, ref=None, s=1, group="leiden", groups=None, treatment=None, vertical_flip=False, horizontal_flip=False, rotate=False):
        size_x=15
        size_y=15

        # create a subgroup from major celltype
        # subgroup = adata[adata.obs["major_celltype"] == group]

        plt.figure(figsize=(size_x, size_y), facecolor="black")

        y_translate=1
        x_translate=1

        if vertical_flip:
            y_translate = -1

        if horizontal_flip:
            x_translate = -1

        if ref:
            scdata = ref
            x = -scdata.obsm['X_spatial'][:, 0]*x_translate
            y = -scdata.obsm['X_spatial'][:, 1]*y_translate
            plt.scatter(y if rotate else x, x if rotate else y, c='gray', s=1, marker='.')

        scdata = self.data

        if treatment != None:
            scdata = self.get_cluster([treatment[0]], group=treatment[1])
            # title = f"{self.name} {treatment[0]}"

        if groups:
            scdata = self.get_cluster(groups, group)

        

        cell_types = np.unique(scdata.obs[group]).tolist()
        for cell in cell_types:#np.unique(scdata.obs[group]):
            # print(cell)
            inds = scdata.obs[scdata.obs[group] == cell].index
            # print(inds, "inds")
            x = -scdata[inds].obsm['X_spatial'][:, 0]*x_translate
            y = -scdata[inds].obsm['X_spatial'][:, 1]*y_translate
            c= self.pallete[cell]
            plt.scatter(y if rotate else x, x if rotate else y, c=c, s=s, marker='.')

        # plt.text(np.median(x),np.median(y),cluster,color='w',fontsize=20)
        #plt.text(np.median(x),np.median(y),cluster,color='w',fontsize=20)
        # plt.grid(b=False)
        plt.axis("off")
        plt.axis("equal")
        plt.tight_layout()

        if self.save:
            plt.savefig(f"{self.folder_path}/brainproj/brainproj_{self.get_filename()}", transparent=True, dpi=self.dpi_save)

    def plot_cluster_scdata(self, group,clusters=None,transpose=1,flipx=1,flipy=1,ssm=5,sbig=30, fig=(15,15), ref=None, marker="."):
        plt.figure(figsize=fig, facecolor="black")

        x,y = (self.data.obsm['X_spatial']*[flipx,flipy])[:,::transpose].T
        #np.unique(self.data.obs["leiden"].astype(np.int))[::-1]
        if ref:
            x2,y2 = (ref.obsm['X_spatial']*[flipx,flipy])[:,::transpose].T
            plt.scatter(x2, y2, c='gray', s=ssm, marker=marker)

        for cluster in clusters if clusters else np.unique(self.data.obs[group]):
            cluster_ = str(cluster)
            inds = self.data.obs[group] == cluster_
            x_ = x[inds]
            y_ = y[inds]
            # col = cmap[int(cluster) % len(cmap)]
            col = self.pallete[cluster_]
            plt.scatter(x_, y_, c=col, s=sbig, marker=marker)
        
        # plt.grid(b=False)
        plt.axis("off")
        plt.axis("equal")
        plt.legend()
        plt.tight_layout()

        if self.save:
            plt.savefig(f"{self.folder_path}/brainproj/brainproj_{self.get_filename()}", transparent=True, dpi=self.dpi_save)

    def plot_gene_scdata(self, gene='Sox10',sz_min=5,sz_max=30,transpose=1,flipx=1,flipy=1, percentile=95, fig=(15,15), umap=False, ref=None, ssm=5, bg=False, cbm5=False, buylla=False):
        plt.style.use("dark_background")
        plt.figure(figsize=fig, facecolor="black")
        ign = list(ref.var.index).index(gene) if ref else list(self.data.var.index).index(gene)

        group = "X_umap" if umap else "X_spatial"

        if ref and bg:
            x2,y2 = (ref.obsm[group]*[flipx,flipy])[:,::transpose].T
            plt.scatter(x2, y2, c='gray', s=ssm, marker='o')

        if cbm5:
            nmax = np.percentile(self.data.layers["Raw counts"][:, self.data.var.index.get_loc(gene)], percentile)
        elif buylla:
            nmax = np.percentile(self.data.X[:, self.data.var.index.get_loc(gene)].toarray(), percentile)
        else:
            nmax = np.percentile(self.data.obsm["X_raw"][:, ref.var.index.get_loc(gene)], percentile) if ref else np.percentile(self.data.obsm["X_raw"][:, self.data.var.index.get_loc(gene)], percentile)
        

        # a = self.data.obsm["X_raw"][:, ref.var.index.get_loc(gene)]
        # if ref else np.percentile(self.data.obsm["X_raw"][:, self.data.var.index.get_loc(gene)], percentile)
        
        # Xcells = self.data.obsm['X_spatial'][:,::transpose]*[flipx,flipy]
        # self.data.obsm['X_spatial']
        if 'X_raw' not in self.data.obsm:
            Xnorm = (np.exp(self.data.X)-1)
            ncts = np.sum(Xnorm,axis=1)[0]
            self.data.obsm['X_raw']=np.round(Xnorm/ncts*np.array(self.data.obs['total_counts'])[:,np.newaxis])

        cts = self.data.obsm['X_raw'][:,ign]

        cts[np.isnan(cts)]=0
        # ncts = np.clip(cts,0,1)
        ncts = np.clip(cts/nmax,0,1)
        size = sz_min+ncts*(sz_max-sz_min)
        from matplotlib import cm as cmap
        cols = cmap.coolwarm(ncts)

        
        x,y = (self.data.obsm[group]*[flipx,flipy])[:,::transpose].T

        indices = np.argsort(size)
        x_sorted = x[indices]
        y_sorted = y[indices]
        cols_sorted = cols[indices]
        size_sorted = size[indices]

        # plt.scatter(x, y, c=cols, s=size, marker='.')
        plt.scatter(x_sorted, y_sorted, c=cols_sorted, s=size_sorted, marker='.')

        # Add color scale
        norm = plt.Normalize(0, nmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap.coolwarm)
        sm.set_array([])
        cbar = plt.colorbar(sm, shrink=0.5)
        cbar.set_label("Counts", rotation=270, labelpad=15)

        # Set the maximum value as a tick on the color bar
        # a = cbar.get_ticks()
        # print(a)
        cbar.set_ticks(np.append(cbar.get_ticks()[:-1], nmax))

        plt.title(gene)
        
        # plt.grid(b=False)
        plt.axis("off")
        plt.axis("equal")
        plt.legend()
        plt.tight_layout()

        if self.save:
            plt.savefig(f"{self.folder_path}/brainproj/brainproj_{gene}_{self.get_filename()}", transparent=True, dpi=self.dpi_save)

    def pearson(self, ref, group_ref, group="leiden", treatment=("",""), order_genes=[]):
        plt.style.use("dark_background")
        refdata = ref

        sc.pp.normalize_total(refdata)
        sc.pp.log1p(refdata)

        d = self.data

        if treatment != ("",""):
            d = self.get_cluster([treatment[0]], group=treatment[1])
            # title = f"{self.name} {treatment[0]}"


        common_genes = refdata.var_names.intersection(d.var_names)
        cbmtemp = d[:,common_genes]
        refdata = refdata[:,common_genes]

        refdf = refdata.to_df()
        refdf["cluster"] = refdata.obs[group_ref]
        refmeans = refdf.groupby("cluster").mean()
        cbmdf = cbmtemp.to_df()
        cbmdf["cluster"] = cbmtemp.obs[group]
        cbmmeans = cbmdf.groupby("cluster").mean()

        refmeans = zscore(refmeans, axis=0)
        cbmmeans = zscore(cbmmeans, axis=0)

        # Get all the correlations
        ps = []
        for name1, row1 in cbmmeans.iterrows():
            ps_ = []
            for name2, row2 in refmeans.iterrows():
                ps_.append(pearsonr(row1, row2)[0])
            ps.append(ps_)
        cordf = pd.DataFrame(ps, index=cbmmeans.index, columns=refmeans.index)

        # if order_genes != []:
        #     cordf = cordf.reindex(columns=order_genes)

        celltypes = ["Radial_Glia-like", "Granule-immature", "Granule-mature", "Neuroblast_2", "nIPC", "Neuroblast_1"]

        cordf = cordf[celltypes]
        # cordf = cordf.drop("7")

        plt.figure(figsize=(15,15), dpi=150)
        
        row_colors = cordf.index.map(self.pallete)
        fig = sns.clustermap(cordf, cmap='coolwarm', vmax=1, vmin=-1, figsize=(7,8), yticklabels=True, xticklabels=True, row_colors=row_colors)

        
        # for tick_label in fig.ax_heatmap.axes.get_yticklabels():
        #     tick_text = tick_label.get_text()
        #     species_name = cordf.index.loc[int(tick_text)]
        #     tick_label.set_color(self.pallete[species_name])

        # fig = swarm_plot.get_figure()
        if self.save:
            fig.savefig(f"{self.folder_path}/linnarson/linnarson_{treatment[0]}_{self.get_filename()}", transparent=True, dpi=self.dpi_save)

    def corr_linnarson(self, group="leiden", treatment=("",""), order_x=["Granule-immature", "Granule-mature", "Neuroblast_2","Neuroblast_1", "Radial_Glia-like", "nIPC"], order_genes = [], annot=False):
        plt.style.use("dark_background")
        refdata = sc.read("../DataF/supfig5/datasetA.h5ad")

        sc.pp.normalize_total(refdata)
        sc.pp.log1p(refdata)

        d = self.data

        if treatment != ("",""):
            d = self.get_cluster([treatment[0]], group=treatment[1])
            # title = f"{self.name} {treatment[0]}"


        common_genes = refdata.var_names.intersection(d.var_names)
        cbmtemp = d[:,common_genes]
        refdata = refdata[:,common_genes]

        refdf = refdata.to_df()
        refdf["cluster"] = refdata.obs["cluster_name"]
        refmeans = refdf.groupby("cluster").mean()
        cbmdf = cbmtemp.to_df()
        cbmdf["cluster"] = cbmtemp.obs[group]
        cbmmeans = cbmdf.groupby("cluster").mean()

        refmeans = zscore(refmeans, axis=0)
        cbmmeans = zscore(cbmmeans, axis=0)

        # Get all the correlations
        ps = []
        for name1, row1 in cbmmeans.iterrows():
            ps_ = []
            for name2, row2 in refmeans.iterrows():
                ps_.append(pearsonr(row1, row2)[0])
            ps.append(ps_)
        cordf = pd.DataFrame(ps, index=cbmmeans.index, columns=refmeans.index)

        # if order_genes != []:
        #     cordf = cordf.reindex(columns=order_genes)

        celltypes = order_x

        cordf = cordf[celltypes]
        if order_genes != []:
            cordf = cordf.loc[order_genes]
        cordf = cordf[order_x]
        # cordf = cordf.drop("7")

        plt.figure(figsize=(15,15), dpi=150)
        
        row_colors = cordf.index.map(self.pallete)
        fig = sns.clustermap(cordf, cmap='coolwarm', vmax=1, vmin=-1, figsize=(7,8), yticklabels=True, xticklabels=True, row_colors=row_colors, row_cluster=False, col_cluster=False, annot=annot)

        
        # for tick_label in fig.ax_heatmap.axes.get_yticklabels():
        #     tick_text = tick_label.get_text()
        #     species_name = cordf.index.loc[int(tick_text)]
        #     tick_label.set_color(self.pallete[species_name])

        # fig = swarm_plot.get_figure()
        if self.save:
            fig.savefig(f"{self.folder_path}/linnarson/linnarson_{treatment[0]}_{self.get_filename()}", transparent=True, dpi=self.dpi_save)

        # return cordf
    
    def rank_genes_groups(self, group="leiden", groups=None, dotplot=False, style="dark_background", n_genes=20):
        try:
            sc.tl.rank_genes_groups(self.data, groupby=group, n_genes=n_genes)
        except KeyError:
            self.data.uns["log1p"]["base"] = None
            sc.tl.rank_genes_groups(self.data, groupby=group, n_genes=n_genes)
        plt.style.use("default")

        if dotplot:
            plt.style.use(style)
            sc.pl.rank_genes_groups_dotplot(self.data, n_genes=2, standard_scale='var')
        else:
            sc.pl.rank_genes_groups(self.data, groups=groups, n_genes=n_genes)

    def ingest(self, ref, group="leiden"):
        # left_out_genes = self.data.var_names.difference(ref.var_names)

        # Assuming `day3` is your AnnData object and `left_out_genes` is the list of genes
        # left_out_data = pd.concat([self.data[:, gene].to_df().squeeze() for gene in left_out_genes], axis=1)

        var_names = ref.var_names.intersection(self.data.var_names)
        ref2 = ref[:, var_names]
        d_ing = self.data[:, var_names]

        sc.pp.pca(ref2)
        sc.pp.neighbors(ref2, n_neighbors=15)
        # sc.tl.umap(ref2)

        # sc.pp.pca(d_ing)
        # sc.pp.neighbors(d_ing, n_neighbors=15)
        # sc.tl.umap(d_ing)

        sc.tl.ingest(d_ing, ref2, obs=group)

        # self.data.obs[group] = d_ing.obs[group]
        self.data = d_ing

        # self.data.obs = pd.concat([self.data.obs, left_out_data], axis=1)

        # return left_out_data

    def pca(self, color):
        plt.style.use('default')

        # Assuming `adata` is your AnnData object
        sc.tl.pca(self.data, svd_solver='arpack')

        # Plot the PCA results
        # sc.pl.pca(self.data, color=color)
        # sc.pl.pca_variance_ratio(self.data, log=True)
        sc.pl.pca_overview(self.data, color=color)
    
    def corr_cebrian(self, group="leiden", treatment=("",""), order_genes=[], order_x = ["Ependymal cells", "B cells", "Mitosis","A cells"], annot=False):
        refdata = sc.read("../DataF/supfig5/data.h5ad")

        plt.style.use('dark_background')

        sc.pp.normalize_total(refdata)
        sc.pp.log1p(refdata)

        d = self.data

        if treatment != ("",""):
            d = self.get_cluster([treatment[0]], group=treatment[1])
            # title = f"{self.name} {treatment[0]}"


        common_genes = refdata.var_names.intersection(d.var_names)
        cbmtemp = d[:,common_genes]
        refdata = refdata[:,common_genes]

        celltypes = ["Ependymal cells", "A cells", "C cells", "Mitosis", "B cells","Endothelial cells"]

        if order_genes != []:
            cbmtemp = cbmtemp[cbmtemp.obs[group].isin(order_genes)]

        refdf = refdata.to_df()
        refdf["cluster"] = refdata.obs["Cell_Type"]
        refmeans = refdf.groupby("cluster").mean()
        cbmdf = cbmtemp.to_df()
        cbmdf["cluster"] = cbmtemp.obs[group]
        cbmmeans = cbmdf.groupby("cluster").mean()

        refmeans = zscore(refmeans, axis=0)
        cbmmeans = zscore(cbmmeans, axis=0)

        # Get all the correlations
        ps = []
        for name1, row1 in cbmmeans.iterrows():
            ps_ = []
            for name2, row2 in refmeans.iterrows():
                row1_clean = np.nan_to_num(row1)
                row2_clean = np.nan_to_num(row2)
                ps_.append(pearsonr(row1_clean, row2_clean)[0])
            ps.append(ps_)
        cordf = pd.DataFrame(ps, index=cbmmeans.index, columns=refmeans.index)

        cordf = cordf[celltypes]
        cordf = cordf.loc[order_genes]
        cordf = cordf[order_x]
        # cordf = cordf.drop("11")


        # if order_genes != []:
        #     cordf = cordf.reindex(columns=order_genes)

        plt.figure(figsize=(15,15), dpi=150)
        
        row_colors = cordf.index.map(self.pallete)
        fig = sns.clustermap(cordf, cmap='coolwarm', vmax=1, vmin=-1, figsize=(7,8), yticklabels=True, xticklabels=True, row_colors=row_colors, row_cluster=False, col_cluster=False, annot=annot)

        # y = ["A cells", "Neuron", "C cells", "Mitosis", "Astrocytes", "B cells", "E"]
        # fig = sns.heatmap(cordf.sort_index(axis=1), cmap='coolwarm', vmax=1, vmin=-1, yticklabels=True, xticklabels="auto")

        
        # for tick_label in fig.ax_heatmap.axes.get_yticklabels():
        #     tick_text = tick_label.get_text()
        #     species_name = cordf.index.loc[int(tick_text)]
        #     tick_label.set_color(self.pallete[species_name])

        # fig = swarm_plot.get_figure()
        if self.save:
            fig.savefig(f"{self.folder_path}/cebrian/cebrian_{treatment[0]}_{self.get_filename()}", transparent=True, dpi=self.dpi_save)

        return cordf
    
    def subset_analysis(self, group, clusters, umap=True):
        subset = self.get_cluster(clusters, group=group)

        # if umap:d
        #     self.umap()
        self.pca()
    
    def corr_genes(self):
        plt.style.use('default')

        cbmtemp = self.data

        cbmdf = cbmtemp.to_df()

        cbmmeans = zscore(cbmdf, axis=0)

        # Get all the correlations
        ps = []
        for name1, row1 in cbmmeans.iterrows():
            ps_ = []
            for name2, row2 in cbmmeans.iterrows():
                row1_clean = np.nan_to_num(row1)
                row2_clean = np.nan_to_num(row2)
                ps_.append(pearsonr(row1_clean, row2_clean)[0])
            ps.append(ps_)
        cordf = pd.DataFrame(ps, index=cbmmeans.index, columns=cbmmeans.index)



        # if order_genes != []:
        #     cordf = cordf.reindex(columns=order_genes)

        plt.figure(figsize=(15,15), dpi=150)
        
        row_colors = cordf.index.map(self.pallete)
        fig = sns.clustermap(cordf, cmap='coolwarm', vmax=1, vmin=-1, figsize=(7,8), yticklabels=True, xticklabels=True, row_colors=row_colors)

        # y = ["A cells", "Neuron", "C cells", "Mitosis", "Astrocytes", "B cells", "E"]
        # fig = sns.heatmap(cordf.sort_index(axis=1), cmap='coolwarm', vmax=1, vmin=-1, yticklabels=True, xticklabels="auto")

        
        # for tick_label in fig.ax_heatmap.axes.get_yticklabels():
        #     tick_text = tick_label.get_text()
        #     species_name = cordf.index.loc[int(tick_text)]
        #     tick_label.set_color(self.pallete[species_name])

        # fig = swarm_plot.get_figure()
        # fig.savefig(f"{self.folder_path}/cebrian/cebrian_{treatment[0]}_{self.get_filename()}", transparent=True, dpi=self.dpi_save)

        return cordf


    def get_genes(self, x, n):
        return sc.get.rank_genes_groups_df(self.data, group=str(x)).names[0:n].to_list() 
    
    def trackplot(self, groupby="leiden", add_genes=neuro_genes, n=5, ptb=True):
        try:
            sc.tl.rank_genes_groups(self.data, groupby=groupby)
        except KeyError:
            self.data.uns["log1p"]["base"] = None
            sc.tl.rank_genes_groups(self.data, groupby=groupby)
        genes_top5ea = []
        for k in np.unique(self.data.obs[groupby].to_list()):
            genes_top5ea += self.get_genes(k, n)

        genes_top5ea += add_genes

        genes = []
        for element in genes_top5ea:
            if element not in genes:
                genes.append(element)
        # print(len(genes))
        # len(genes_top5ea)
        if ptb == False:
            genes.remove("Ptbp1")

        # df = data[:, genes].to_df()
        # df["group"] = data.obs[groupby]
        # means = df.groupby("group").mean()
        # # df.shape
        

        
        # dists = pdist(means.T, metric="correlation")
        
        # gene_order = data[:,genes].var.iloc[leaves_list(ward(dists))].index

        fig = sc.pl.tracksplot(self.data, genes, groupby=groupby, return_fig=True, dendrogram=True)
        print(f'tracksplot_{self.get_filename()}'[:-4])
        # save the figure
        # fig.savefig(f'{self.folder_path}/tracksplot/tracksplot_{self.get_filename()}', dpi=self.dpi_save)
        # sc.pl.tracksplot(self.data, genes, groupby=groupby, dendrogram=True)

    def get_filename(self):
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        return f"{self.name.lower().replace(' ', '_')}_{now}.png"
    
    def get_data(self):
        return self.data
    
    @property
    def obs(self):
        return self.data.obs
    
    @property
    def var(self):
        return self.data.var
    
    @property
    def obsm(self):
        return self.data.obsm
    
    @property
    def uns(self):
        return self.data.uns
    
    @property
    def layers(self):
        return self.data.layers
    
    @property
    def X(self):
        return self.data.X



