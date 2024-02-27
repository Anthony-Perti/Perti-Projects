import sys

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    !pip install scikit-learn==1.4.0 > /dev/null 2>&1

import numpy as np
import pandas as pd
import joblib

ISOLATE_CSV_PATH = "https://gitlab.com/oasci/courses/pitt/biosc1540-2024s/-/raw/main/biosc1540/files/csv/checkpoints/genomics/ecoli-amr-isolates.csv"
df_isolates = pd.read_csv(ISOLATE_CSV_PATH)
n_isolates = len(df_isolates)
print(f"There are {n_isolates} isolates in the dataset.")
print(df_isolates.head(n=5))

antibiotic_sel = "levofloxacin"

def encode_labels(df, antibiotic_sel):
    labels = df[antibiotic_sel].to_numpy()
    resistance_mapping = {"S": 0, "I": 1, "R": 2}
    labels = np.vectorize(resistance_mapping.get)(labels)
    labels = labels
    return labels


labels = encode_labels(df_isolates, antibiotic_sel)
print(labels.shape)

!wget https://gitlab.com/oasci/courses/pitt/biosc1540-2024s/-/raw/main/large-files/genomics-checkpoint-genes.zip > /dev/null 2>&1

!unzip genomics-checkpoint-genes.zip > /dev/null 2>&1
!mv genes-array genes

def load_gene_data(gene_name):
    """Loads all variants of gene into a NumPy array."""
    return np.load(f"genes/{gene_name}.npy")

gene_fadL = load_gene_data("fadL")
print(gene_fadL)
print(gene_fadL.shape)

import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def numberizer(msa_array):
    nucleotide_to_number = {"A": 0.25, "C": 0.5, "G": 0.75, "T": 1.0}
    msa_array = np.vectorize(nucleotide_to_number.get)(msa_array)
    return msa_array


def plot_alignment(msa_array, start_seq, n_seq, nuc_per_ax=200):
    msa_array = numberizer(msa_array)
    seq_stop = start_seq + n_seq

    msa_array = msa_array[start_seq:seq_stop]

    n_nuc = msa_array.shape[1]
    n_axes = math.ceil(n_nuc / nuc_per_ax)

    custom_cmap = ListedColormap(["#264653", "#f94144", "#f9c74f", "#43aa8b"])

    fig, axs = plt.subplots(n_axes, 1, figsize=(10, 3 * n_axes), sharex=True)
    axs = np.atleast_1d(axs)  # Ensure axs is always an array for consistency

    nuc_start = 0
    for ax in axs:
        nuc_stop = min(nuc_start + nuc_per_ax, n_nuc)
        seq_sliced = msa_array[:, nuc_start:nuc_stop]  # Correct slicing

        im = ax.imshow(
            seq_sliced,  # Transpose for correct orientation
            cmap=custom_cmap,
            aspect="auto",
            interpolation="none",
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(f"Seq {start_seq} to {seq_stop}")
        ax.set_title(f"Positions {nuc_start + 1} to {nuc_stop}")
        nuc_start += nuc_per_ax

    plt.tight_layout()
    plt.show()

gene_argE = load_gene_data("argE")
plot_alignment(gene_argE, 0, 50)
gene_argE = load_gene_data("argE")
gene_argE_feat = numberizer(gene_argE)

print(gene_argE)
print(gene_argE_feat)

import io
from urllib import request

GENE_NAMES_NPY_PATH = "https://gitlab.com/oasci/courses/pitt/biosc1540-2024s/-/raw/main/biosc1540/assessments/checkpoints/genomics/gene-names.npy"

response = request.urlopen(GENE_NAMES_NPY_PATH)
content = response.read()
gene_names = np.load(io.BytesIO(content))
print(gene_names)

desc_all = []
for gene_name in gene_names:
    gene_data = load_gene_data(gene_name)
    gene_data = numberizer(gene_data)
    gene_desc = np.mean(np.var(gene_data, axis=1), axis=0)

    desc_all.append(gene_desc)

desc_all = np.array(desc_all)
print(desc_all)

threshold = 0.093

gene_idxs = np.argwhere(desc_all > threshold).ravel()
gene_selection = gene_names[gene_idxs].tolist()
print(gene_selection)
print(len(gene_selection))

gene_selection.append("fadL")
gene_selection.append("bfr")

n_genes = len(gene_selection)
print(f"There are {n_genes} genes selected: {gene_selection}")

"""tnaC: I chose tnAc due to: 1) The high variability of the gene will be a good factor for the machine learning algorithm. 2) The fact it partially codes for the enzymes that use tryptophan, an important amino acid.

shoB: I chose shoB due to: 1) The high variability it displays will be beneficial for the machine learning algorithm. 2) It codes for small toxic proteins which help the E. coli defend from outside bacteria its host may have.

fadL: I chose fadL due to: 1) Its important function is metabolism and stress response for the E. coli. 2) Margalit et al. (2020) implicates genes used in the stress response of E. coli can be useful for  AMR resistence when the E. coli is under cefotaxime or ciprofloxacin, thus supporting its relevance to my study.

bfr: I chose bfr due to: 1) Its use in storing iron, an essential mineral for the E. coli 2) The same study used above also implicates genes that help with oxidative stress. Since iron is oxidized by what is coded in the bfr gene, it can be relevant to AMR when exposed to different drugs that increase the oxidative stress on the E. coli.

Margalit, A., Carolan, J. C., & Walsh, F. (2021, December 15). Global protein responses of multidrug resistance plasmid-containing escherichia coli to ampicillin, cefotaxime, imipenem and ciprofloxacin. Journal of Global Antimicrobial Resistance. https://www.sciencedirect.com/science/article/pii/S2213716521002770
"""
k = 3

"""I chose a k=3 since it is the length of a codon. Since the 3 genes I am looking at all code for different protiens, looking at their codons breaks them down into a large enough block to do analysis on while still making the model less computationally complex.

The following cells define the functions to efficiently compute k-mers of multiple genetic sequences.
Algorithm explanations are also provided.
"""
import itertools
from multiprocessing import Pool

def generate_all_kmers(k):
  
    nucleotides = ["A", "C", "G", "T"]
    return ["".join(p) for p in itertools.product(nucleotides, repeat=k)]

def create_kmer_mapping(all_kmers):

    return {kmer: i for i, kmer in enumerate(all_kmers)}

def count_kmers_seq(sequence, k, kmer_mapping):
    if not isinstance(sequence, str):
        sequence = "".join(sequence)
    kmer_counts = np.zeros(len(kmer_mapping), dtype=np.int32)
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i : i + k]
        index = kmer_mapping.get(kmer)
        if index is not None:
            kmer_counts[index] += 1

def parallel_count_kmers(sequences, k, kmer_mapping):
   
    args_list = [(sequence, k, kmer_mapping) for sequence in sequences]

    with Pool() as pool:
        result = pool.starmap(count_kmers_seq, args_list)

    return np.array(result)

all_kmers = generate_all_kmers(k)
kmer_mapping = create_kmer_mapping(all_kmers)
n_kmers = len(kmer_mapping)
print(f"There are {n_kmers} unique k-mers in the k-mer mapping")
print(kmer_mapping)


features = np.empty((n_genes, n_isolates, n_kmers))
for i, gene_name in enumerate(gene_selection):
    gene_data = load_gene_data(gene_name)
    features[i] = parallel_count_kmers(gene_data, k, kmer_mapping)
print(features.shape)

from sklearn.model_selection import train_test_split

RANDOM_STATE = 472929478

features_train = []
features_test = []

for i in range(len(features)):
    f_train, f_test = train_test_split(
        features[i], test_size=0.4, random_state=RANDOM_STATE
    )
    features_train.append(f_train)
    features_test.append(f_test)

features_train = np.array(features_train)
features_test = np.array(features_test)

print(features_train.shape)
print(features_test.shape)

# DO NOT MODIFY CODE BELOW THIS LINE.

labels_train, labels_test = train_test_split(
    labels, test_size=0.4, random_state=RANDOM_STATE
)

print(labels_train.shape)
print(labels_test.shape)

desc_kmers = []
for feat in features_train:
    desc = np.var(feat, axis=0)
    if desc.ndim != 1:
        raise ValueError("desc should be a 1D array")
    if desc.shape[0] != n_kmers:
        raise ValueError(
            "Make sure you are computing your descriptor using the correct axis"
        )
    desc_kmers.append(desc)

desc_kmers = np.array(desc_kmers)
print(desc_kmers.shape)
print(desc_kmers)

mask = desc_kmers > 2

# DO NOT MODIFY CODE BELOW THIS LINE.
kmer_idxs = np.argwhere(mask)
print(f"You would have {len(kmer_idxs)} k-mers selected")

"""

I decided to chose variance as a descriptor as I believe that it would give me more insight than the mean. The variance lets me see what genes display more difference and thus might hold more information as to why they are influincing AMR. This makes sense to me as if a gene undergoes positive mutation in one strand, it could cause a better chance at gaining AMR and the higher variance means the higher chance of this happening.

The first value I chose was 10, I feel like it narrows it down to high variance k-mers, but not ones that are irrelative of the entire set. At 17.6% of total k-mers, I believe that I maintain a good balance of meanigful and inclusive data.

The second value I chose was 15, this led me to a test score that was very low and I believe this was due to me cherry-picking the data by only having a limited amount of k-mers chosen.

The third value I chose was 6, this gave me a much better score as testing for a higher amount of k-mers gave the model sufficiently good and large enough data to predict with better accuracy.

I chose the above values when the amount of genes I was testing for was 3. When I appended fadL to the list, the larger amount of k-mers became more favorable to test for.

The fourth value I chose was 4, which gave me a much more expansive list of k-mers, which I felt would be good to test for as I added another gene. This decision was simply based on seeing that 6 gave me a higher score than 10.

The fifth value I chose was 2, this was just based off of the back of 4, and it proved to be the sweet spot since it gave my model the highest score. This means that I gave it a sufficient amount of k-mers to learn off of, and they were of good enough quality (they had a good variance), to let the model make a decently accurate prediction.
"""

kmer_selections = [[] for _ in range(n_genes)]
for sel in kmer_idxs:
    gene_idx, kmer_idx = sel
    kmer_selections[gene_idx].append(kmer_idx)
print(kmer_selections)

def select_kmers(features, kmer_selection):
    features_selected = []
    if not isinstance(kmer_selection, list):
        raise TypeError("kmer_selection must be a list")
    for feature, selection in zip(features, kmer_selection):
        if len(selection) == 0:
            continue
        feature_sel = feature[:, selection].T
        features_selected.extend(feature_sel.tolist())
    print(features_selected)
    features_selected = np.array(features_selected).T
    return features_selected


features_train_selected = select_kmers(features_train, kmer_selections)
features_test_selected = select_kmers(features_test, kmer_selections)
print(features_train_selected)
print(features_train_selected.shape)

from sklearn import preprocessing

select_scaler = preprocessing.StandardScaler

def process_features_train(features, scaler_selection):
    scaler = select_scaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features, scaler


features_train_scaled, scaler_fit = process_features_train(
    features_train_selected, select_scaler
)
features_test_scaled = scaler_fit.transform(features_test_selected)
print(features_train_scaled)
print(features_train_scaled.shape)
print(features_test_scaled)
print(features_test_scaled.shape)

from sklearn.linear_model import RidgeClassifier

model_selection = RidgeClassifier
model_kwargs = {"alpha": 1.0,
                "fit_intercept": False,
                "tol": 1e-4,
                "solver":"saga",
                "max_iter":10000,
                "class_weight":None,
                "random_state": RANDOM_STATE}

model = model_selection(**model_kwargs)
model.fit(X=features_train_scaled, y=labels_train)

from sklearn.metrics import balanced_accuracy_score

labels_pred = model.predict(features_test_scaled)
score = balanced_accuracy_score(labels_test, labels_pred)
print("True: ", labels_test)
print("Pred: ", labels_pred)
print(score)

joblib.dump(model, "model.joblib")
np.save("features_test.npy", features_test_scaled)

- To download your files, locate the file icon on the left sidebar of the Colab interface. This opens the file browser.
- Navigate to find the files you've just saved: `"model.joblib"` and `"features_test.npy"`.
- Click on the three dots (`...`) next to each file name to open the context menu, and select "Download" to save the file to your computer.
"""
