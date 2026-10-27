import os
from multiprocessing import Pool
import pandas as pd
import aleph_classes
import ALEPH.aleph.core.Bioinformatics as Bioinformatics
import subprocess
import tempfile
import json

three_to_one = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "UNK": "-"
}

skipped = list()

def create_annotations(database, name_folder, max_size=None):
    database = os.path.realpath(database)
    name_folder = os.path.realpath(name_folder)
    if not os.path.isdir(database):
        raise ValueError(f"Database folder does not exist: {database}")
    if database == name_folder:
        raise ValueError("Input and output folders must be different")
    if max_size is not None and max_size < 0:
        raise ValueError("max_size must be zero or greater")

    results = {"saved": [], "skipped": [], "failed": []}
    for root, directories, filenames in os.walk(database):
        directories[:] = sorted(directory for directory in directories
                                if os.path.realpath(os.path.join(root, directory)) != name_folder)
        output_folder = os.path.join(name_folder, os.path.relpath(root, database))
        os.makedirs(output_folder, exist_ok=True)
        for filename in sorted(filenames):
            if os.path.splitext(filename)[1].lower() not in {".pdb", ".ent", ".cif", ".mmcif"}:
                continue
            pdb_path = os.path.join(root, filename)
            if max_size is not None and os.path.getsize(pdb_path) > max_size:
                results["skipped"].append(pdb_path)
                print(f"Skipping {pdb_path}: exceeds {max_size} bytes.")
                continue
            try:
                print(f"Annotating {pdb_path}.")
                previous_directory = os.getcwd()
                with tempfile.TemporaryDirectory(prefix="aleph-annotation-") as temporary:
                    try:
                        os.chdir(temporary)
                        annotation = aleph_classes.AlephPDB(pdb_path, output_folder)
                    finally:
                        os.chdir(previous_directory)
                annotation_path = os.path.join(output_folder, filename + ".pkl")
                annotation.save_annotation(annotation_path)
                results["saved"].append(annotation_path)
            except Exception as error:
                results["failed"].append(pdb_path)
                print(f"Failed {pdb_path}: {error}")

    print(f"Annotations: {len(results['saved'])} saved, "
          f"{len(results['skipped'])} skipped, {len(results['failed'])} failed.")
    return results

def process_single_pdb(pdb_path, name_folder):
    print(f"Processing {pdb_path}.")
    try:
        # Annotate object with ALEPH
        aleph_annotation = aleph_classes.AlephPDB(pdb_path, name_folder)
        # Find NN and store modules, distances and angles of reference and NN nodes.
        #aleph_annotation.findNN_sidechain(pdb_path, long=False)
        #aleph_annotation.findNN_VC() # !! -> change distance range limit to 12 (7-12)
        #aleph_annotation.findNN()  # !! -> change distance range limit to 10
        #aleph_annotation.findNN_long(pdb_path)
        aleph_annotation.compute_alef()
        aleph_annotation.check_thresholds()
        # FoldSeek module on AlephPDB class. Annotate FS labels.
        #aleph_annotation.associate_FStoken()
        # Save embeddings and extra annotations of the AlephPDB object
        aleph_annotation.save_annotation(os.path.join(os.getcwd(), "all.pkl"))
        test = aleph_annotation.load_annotation(os.path.join(os.getcwd(), "all.pkl"))
        test_cvs = test.get_cvs()
        test_fr = test.cvs_to_fragments(test_cvs)
        test_comm = test.fragments_to_communities(test_fr)
        final = test.hierarchy_membership(test_cvs, test_fr, test_comm)
        cv_to_comm = final["cv_to_communities"]
        for e in aleph_annotation.embeddings:
            try:
                e.community = cv_to_comm[e.pair[0].cvid][0]
            except:
                pass
        aleph_annotation.save()
        # Optional: plot pairwise distances between NN
        #aleph_annotation.generate_pymol_session()
        # Optional: extract fasta files from structure
        #aleph_annotation.extract_fasta()
        # Obtain the graph form of the object
        #aleph_annotation.save_graph()
        return f"Success: {aleph_annotation.id}"
    except Exception as e:
        #errors.write(f"Crashed: {os.path.basename(pdb_path)} - Error: {e}\n")
        return f"Crashed: {pdb_path} - Error: {e}"

def superpose_aleph(reference_path: str, target_path: str):
    print(f"Annotating Reference: {os.path.basename(reference_path)}")
    aleph_ref = aleph_classes.AlephPDB(reference_path, 'super')
    print(f"Annotating Target: {os.path.basename(target_path)}")
    aleph_tgt = aleph_classes.AlephPDB(target_path, 'super')
    best_matches = aleph_ref.match_clusters_against_target(target_aleph=aleph_tgt)
    if best_matches:
        print(f"Ref Cluster {best_matches['ref_anchor_idx']} "
              f"Score: {best_matches['score']}")
        R = best_matches["transformation_R"]
        t = best_matches["transformation_t"]
        print(f"Loading target structure: {aleph_tgt.pdb_path}")
        target_struct = Bioinformatics.get_structure(aleph_tgt.id, aleph_tgt.pdb_path)
        target_atoms = Bioinformatics.get_list_of_atoms(target_struct)
        Bioinformatics.transform_atoms(target_atoms, R, t, invert=False)
        pdb_string, _ = Bioinformatics.get_pdb_from_list_of_atoms(target_atoms, renumber=False, uniqueChain=False)
        with open(f"{aleph_tgt.id}_superposed.pdb", "w") as f:
            f.write(pdb_string)

def run_database_aleph(database, name_folder, max_workers=6):
    os.makedirs(name_folder, exist_ok=True)
    pdb_paths = []
    for pdb in os.listdir(database):
        pdb_path = os.path.join(database, pdb)
        if os.path.isfile(pdb_path):
            size = os.path.getsize(pdb_path)
            if size <= 999999:
                pdb_paths.append((pdb_path, name_folder))
            else:
                print(f"Skipping {pdb} due to large size.")
                skipped.append(pdb)

    print(f"Found {len(pdb_paths)} PDB files. Starting Pool with {max_workers} workers...")

    with Pool(processes=max_workers) as pool:
        results = pool.starmap(process_single_pdb, pdb_paths)
    crashed = 0
    success = 0
    for res in results:
        if "Success" in res:
            success += 1
        else:
            crashed += 1
    print(f"\n\nFinal results of ALEPH: processed {len(results)} structures -> {success} success / {crashed} crashed"
          f"\n========================================================================================================")
    for res in results:
        print(res)

    print(f"\n\nSkipped files due to large size: {len(skipped)}"
          f"\n========================================================================================================")
    for skip in skipped:
        print(skip)

# CODE TO CREATE GRAPHS
def create_single_graph(pdb_info, input_path, output_path):
    pdb, fam = pdb_info.split(" ")
    pdb_path = os.path.join(input_path, fam, pdb+".pdb")
    try:
        # Annotate object with ALEPH
        aleph_annotation = aleph_classes.AlephPDB(pdb_path, output_path)
        # here annotate
        graph_name = f"{pdb}--{fam}" # Separator: --
        # Obtain the graph form of the object
        aleph_annotation.save_graph(graph_name)
        return f"Success: {aleph_annotation.id}"

    except Exception as e:
        return f"Crashed: {pdb_path} - Error: {e}"

def worker_wrapper(args):
    return create_single_graph(*args)

def create_graphs_pdb(selection_path, input_path, output_path, max_workers=8):
    os.makedirs(output_path, exist_ok=True)
    selected_pdbs = []
    with open(selection_path, "r") as f:
        print(f"Reading {selection_path}")
        for line in f:
            selected_pdbs.append(line.strip())

    unique_families = set([pdb_info.split(" ")[1] for pdb_info in selected_pdbs])
    print(f"Loaded {len(unique_families)} families from {len(selected_pdbs)} PDBs. Starting Pool with {max_workers} workers...")

    tasks = [(pdb_info, input_path, output_path) for pdb_info in selected_pdbs]
    with Pool(processes=max_workers, maxtasksperchild=50) as pool:
        crashed = 0
        success = 0
        for res in pool.imap_unordered(worker_wrapper, tasks, chunksize=1):
            if "Success" in res:
                success += 1
            else:
                crashed += 1
                skipped.append(res)
            print(res)

    print(f"\n\nFinal results of ALEPH: processed {len(selected_pdbs)} structures -> {success} success / {crashed} crashed"
          f"\n========================================================================================================")


    print(f"\n\nSkipped files due to crash: {len(skipped)}"
          f"\n========================================================================================================")
    for skip in skipped:
        print(skip)

# CODE TO COMPUTE TMSCORE
def foldseek_superposition(database_path_1: str,
                           database_path_2: str,
                           selection_path: str,
                           output_csv: str = "canon_data_pairs.csv",
                           family_check: bool = True,
                           output_family: str = "canon_family_check.json",
                           threads: int = 8,
                           canon: bool = False):
    """
    Compute tmscore for every pair on a given set of PDBs. Baseline to assess similarity.
    'selection_path' contains the path to the JSON with information about PDBs.
    'family_check' groups pairs of the same family and computes their mean tmscore.
    """
    if canon:
        split_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets", "scope95_full")
        pdb_paths = []
        with open(os.path.join(split_dir, "train_sid.txt"), "r") as f:
            for line in f:
                path, fam = line.strip().split(" ")
                line = fam+"/"+path
                pdb_paths.append(line)
        with open(os.path.join(split_dir, "val_unseen_sid.txt"), "r") as f:
            for line in f:
                path, fam = line.strip().split(" ")
                line = fam + "/" + path
                pdb_paths.append(line)
        with open(os.path.join(split_dir, "test_unseen_sid.txt"), "r") as f:
            for line in f:
                path, fam = line.strip().split(" ")
                line = fam + "/" + path
                pdb_paths.append(line)
        print(pdb_paths)

    else:
        with open(selection_path, "r") as f:
            data = json.load(f)

        pdb_paths = []
        for key in data.keys():
            pdb_paths += data[key]['pdbs']

    # Prepare Foldseek output
    output_tsv = "foldseek_out.tsv"
    format_cols = ["query", "target", "alntmscore"]
    format_str = ",".join(format_cols)

    # Create ghost folder (symlinks) with all PDBs
    with tempfile.TemporaryDirectory() as symlink_dir:
        for path in pdb_paths:
            filename = os.path.basename(path)
            symlink_target = os.path.join(symlink_dir, filename)
            if not os.path.exists(symlink_target):
                possible_path = os.path.join(database_path_1, path)
                if os.path.exists(possible_path):
                    os.symlink(os.path.abspath(possible_path), symlink_target)
                else:
                    possible_path = os.path.join(database_path_2, path)
                    os.symlink(os.path.abspath(possible_path), symlink_target)

        # Execute Foldseek
        cmd = ["foldseek",
               "easy-search", symlink_dir, symlink_dir, output_tsv, "tmp_foldseek",
               "--exhaustive-search", "1",
               "--alignment-type", "1",
               "--format-output", format_str,
               "--threads", str(threads),
               "-e", "100000"]

        print(f"Running Foldseek on {len(pdb_paths)} structures with {threads} threads...")

        try:
            subprocess.run(cmd, check=True, capture_output=False)

            # Create dataframe with results
            df = pd.read_csv(output_tsv, sep="\t", names=format_cols)
            df = df[df["query"] < df["target"]].copy()  # Remove self-alignments and duplicates

            if family_check:
                family_dict = {os.path.splitext(path.split("/")[-1])[0]:path.split("/")[0] for path in pdb_paths}

                df["PDB1"] = df["query"].str[:7]
                df["PDB2"] = df["target"].str[:7]

                df["family1"] = df["PDB1"].map(family_dict)
                df["family2"] = df["PDB2"].map(family_dict)

                sharefam_df = df[df["family1"] == df["family2"]]
                family_scores = sharefam_df.groupby("family1")["alntmscore"].mean().to_dict()
                family_scores['not_same_family'] = df[df["family1"] != df["family2"]]["alntmscore"].mean()

                with open(output_family, "w") as f:
                    json.dump(family_scores, f, indent=4)

            df = df.rename(columns={"alntmscore": "tmscore",})
            df.sort_values(by=["PDB1", "PDB2"], inplace=True)
            df.to_csv(output_csv, index=False)
            print(f"Saved {len(df)} structural pairs to {os.path.abspath(output_csv)}")

        finally:
            # Remove temporal files
            if os.path.exists(output_tsv):
                os.remove(output_tsv)
            if os.path.exists("tmp_foldseek"):
                subprocess.run(["rm", "-rf", "tmp_foldseek"])

'''
# Not efficient but it could be useful someday
import sys
from itertools import combinations
import pymol
def run_tmalign(pdb1: str, pdb2: str):
    try:
        result = subprocess.run(
            ['TMalign', str(pdb1), str(pdb2)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        output = result.stdout

    except Exception as e:
        print("Error running TM-align:", e)
        return os.path.basename(pdb1), os.path.basename(pdb2), None, None
    tm_scores = []
    for line in output.splitlines():
        line = line.strip()
        if line.startswith('TM-score=') and 'normalized by length' in line:
            tm_scores.append(float(line.split()[1].strip()))
    symmetric_tm = sum(tm_scores) / 2.0

    pymol_rmsd = None
    try:
        pymol.cmd.reinitialize()
        pymol.cmd.load(pdb1, "objA")
        pymol.cmd.load(pdb2, "objB")
        super_result = pymol.cmd.super("objA", "objB")
        pymol_rmsd = super_result[0]


    except Exception as e:
        print(f"CRITICAL ERROR calculating RMSD: {e}", file=sys.stderr)
    finally:
        pymol.cmd.delete("all")

    pdb1 = os.path.splitext(os.path.basename(pdb1))[0]
    pdb2 = os.path.splitext(os.path.basename(pdb2))[0]
    return pdb1, pdb2, symmetric_tm, pymol_rmsd

def run_database_superposition(selection_path, max_workers=8):
    with open(selection_path, "r") as f:
        data = json.load(f)

    pdb_paths = []
    for key in data.keys():
        pdb_paths += data[key]['pdbs']

    print(f"Found {len(pdb_paths)} PDB files. Starting Pool with {max_workers} workers...")

    # Generate all combinations of length 2
    res = list(combinations(pdb_paths, 2))

    with Pool(processes=max_workers) as pool:
        results = pool.starmap(run_tmalign, res)
    results = pd.DataFrame(results, columns = ["PDB1", "PDB2", "tmscore", "rmsd"])
    results = results.sort_values(by="PDB1")  # Sort alphabetically
    results.to_csv("data.csv", index=False)
'''
