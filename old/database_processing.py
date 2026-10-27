import os
import random
import json
import sys
import math
from Bio import PDB

# Code to create a duplicate of all the elements in the database with the correct extension (.pdb)
# Single use code
'''
import shutil
from pathlib import Path

base_dir = Path("/localdata2/arnau/scope/scope95_fam") #scope40_fam/scope95_fam

for root, _, files in os.walk(base_dir):
    for file in files:
        source = Path(root) / file
        destination = source.with_suffix(".pdb")
        if source==destination:
            continue
        shutil.copy(source, destination)
'''

MAX_PDB_RESIDUES = 500
MIN_PDB_RESIDUES = 20

def count_residues(path: str):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', path)
    residues = list(structure.get_residues())
    return len(residues)

class DataSelection:
    """
    Class designed to manage a database of PDB files. In charge of creating sets of data and sorting families.
    Set creation in general should be done through this class, previously to anything else.
    """
    def __init__(self, database):
        self.database = database # Path to the database
        self.full_database = []

    def locate_pdbs(self,
                    force: bool = False):
        if not os.path.exists(self.database):
            raise FileNotFoundError(f"Database in location {self.database} does not exist.")

        good_path = os.path.join(self.database, "_good.txt")
        n = 0

        if force or not os.path.exists(good_path):
            if force and input("Are you sure you want to overwrite the previous filtering? (y/N)\n>>>").lower() != "y":
                raise Exception("Program interrupted")
            print(f"Filtering {self.database}")
            open(os.path.join(self.database, "_bad.txt"), "w")
            with open (good_path, "w") as gf:
                for element in os.scandir(self.database):
                    if element.is_dir():
                        for path in os.scandir(element.path):
                            if path.is_file() and path.name.endswith(".pdb"):
                                print(f"{n} PDBs processed.", end="\r")
                                n += 1

                                rel_path = os.path.relpath(path.path, self.database)
                                pdb_name = os.path.splitext(rel_path.split("/")[-1])[0]
                                pdb_family = rel_path.split("/")[0]

                                size = count_residues(path.path)
                                if MIN_PDB_RESIDUES <= size <= MAX_PDB_RESIDUES: # Ensure acceptable number of residues (20-500)
                                    gf.write(pdb_name + " " + pdb_family + "\n")
                                else:
                                    with open (os.path.join(self.database, "_bad.txt"), "a") as bf:
                                        bf.write(pdb_name + " " + pdb_family + f" #{size} residues\n")

        self.full_database = []
        with open(good_path, "r") as gf:
            print(f"Reading {good_path}")
            for line in gf:
                self.full_database.append(line.strip())

        unique_families = set([path.split(" ")[1] for path in self.full_database])

        print(f"\nFound {len(unique_families)} families, with a total of {len(self.full_database)} PDBs.")

    def make_selection(self,
                       n: int,
                       blacklist: list | None = None,
                       force_family: bool = False,
                       seed: int = 42):
        """
        Select a portion of the found PDBs and store them in a list. Keep their families in a second list as labels.
        It can be used as a standalone function, but it is usually used to create multiple sets through 'create_sets'.
        'out_path' saves the selection on a given path. Only needed for standalone use.
        'force_family' when True ensures the PDBs are clustered by family and, if multiple sets are being created,
        that families are not on more than one of them.
        'not_alone_data' allows the function to receive a dictionary with the same structure as the one it returns.
        It can be used to skip parts of data already sampled in previous selections.
        """
        random.seed(seed)
        possible_pdbs = []

        # Filter PDBs
        if blacklist:
            prev_pdbs = [pdb.split(" ")[0] for pdb in blacklist]
            prev_fams = [fam.split(" ")[1] for fam in blacklist]

            for pdb_info in self.full_database:
                pdb, fam = pdb_info.split(" ")
                if pdb in prev_pdbs:
                    continue
                if fam in prev_fams and force_family:
                    continue
                possible_pdbs.append(pdb + " " + fam)
        else:
            possible_pdbs = self.full_database

        if n > len(possible_pdbs):
            raise ValueError(f"Not enough elements to sample {n} PDBs. There are only {len(possible_pdbs)} available PDBs.")

        if not force_family:
            selected_pdbs = random.sample(possible_pdbs, n)

        else:
            # Group PDBs by family
            fams_pdbs = {}
            for pdb_info in possible_pdbs:
                _, fam = pdb_info.split(" ")
                fams_pdbs.setdefault(fam, []).append(pdb_info)

            # Choose families randomly
            possible_fams = list(fams_pdbs.keys())
            random.shuffle(possible_fams)

            # Select until completion
            selected_pdbs = []
            for family in possible_fams:
                if len(selected_pdbs) + len(fams_pdbs[family]) <= n:
                    selected_pdbs.extend(fams_pdbs[family])
                else:
                    selected_pdbs.extend(random.sample(fams_pdbs[family], n - len(selected_pdbs)))
                    break

        unique_families = set([pdb_info.split(" ")[1] for pdb_info in selected_pdbs])
        print(f"Selection with {len(selected_pdbs)} PDBs from {len(unique_families)} different families successfully created.")
        return selected_pdbs

    def create_sets(self,
                    proportion: list[float],
                    size: int = 0,
                    output: str = ".",
                    force_family: bool = True): # about families and size
        """
        Create len(proportion) sets using 'make_selection'. Similar to a 'train_test_split' function.
        'proportion' defines the proportion of data each set should contain. An 80-20 split should be [0.8, 0.2].
        'size' defines the number of elements the dataset will have. If zero or negative, all possible PDBs are used.
        'out_path' saves the sets on a given path. The JSON can later be used to compute the similarity between pairs
        of PDBs and to create their respective graphs.
        'force_family' when True ensures the PDBs are clustered by family and, if multiple sets are being created,
        that families are not on more than one of them.
        """
        sets = {}
        if size <= 0:
            size = len(self.full_database)

        # Normalize the proportions given. Largest Remainder Method.
        proportion = [prop/sum(proportion) for prop in proportion]
        prop_exact = [size*prop for prop in proportion]
        prop_grounded = [math.floor(x) for x in prop_exact]
        remainder = sorted(range(len(proportion)), key=lambda i: prop_exact[i] - prop_grounded[i], reverse=True)
        for i in remainder[:size-sum(prop_grounded)]:
            prop_grounded[i] += 1

        cumulative_set = []

        # Create sets without overlapping
        for i, n in enumerate(prop_grounded):
            actual_set = self.make_selection(n=n, blacklist=cumulative_set, force_family=force_family)
            sets[f"set_{i+1}"] = actual_set
            cumulative_set.extend(actual_set)

            with open(output+f"/set_{i+1}.txt", "w") as f:
                f.write("\n".join(actual_set))

        return sets

'''
MAIN
'''
if __name__ == "__main__":
    selection = DataSelection("/localdata1/arnau/scope/scope95_fam_circular_permutation/scope95_fam_cp")
    selection.locate_pdbs()
    sets = selection.create_sets(proportion=[0.8, 0.2], force_family=False)
