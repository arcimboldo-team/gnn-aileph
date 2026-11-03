from dataclasses import dataclass, field, replace
from pathlib import Path
import pickle
import ALEPH.aleph.core.ALEPH as ALEPH


@dataclass(frozen=True)
class Residue:
    res_id: int
    chain: str
    amino_acid: str | None

    @classmethod
    def from_residue_data(cls, record):
        return cls(record[3][1], record[2], record[4] if len(record) > 4 else None)


@dataclass
class CV:
    cvid: int
    secstr: int
    secondary_structure: str
    reslist: list
    residues: list
    nearsecstr: int | None = None
    unique_cv: object = None
    previous_cvid: int | None = None
    next_cvid: int | None = None
    sequence: str = ""
    sequence_secstr: str = ""
    isStart: bool = False
    isMiddle: bool = False
    isEnd: bool = False
    isSpecial: bool = False
    posandsec: int | None = None
    pos: int | None = None
    name: str = ""

    @property
    def chain(self):
        return self.residues[0].chain if self.residues else ""


@dataclass
class CVPairData:
    pair: tuple
    dist: float
    angle1: float
    angle2: float
    weight: float
    sstype1: int  # Native codes: helix=1, strand=2, coil=3.
    sstype2: int
    fragment_angle1: float
    fragment_dist: float
    fragment_angle2: float
    cv1_fragment2_dist: float
    cv2_fragment1_dist: float
    fragment_count: int  # 1: same fragment; 2: different fragments.


@dataclass
class FragmentAleph:
    fragid: int
    sstype: str
    sequence: str
    reslist: list
    residues_fragment: list
    cvids: list
    cvls: list
    unique_cv: object = None
    vecLength: object = None
    cv_list: list = field(default_factory=list)

    @property
    def chain(self):
        return self.residues_fragment[0].chain if self.residues_fragment else ""


@dataclass
class FragmentData:
    dist: float
    angle1: float
    angle2: float
    min_distance: float
    max_distance: float | None = None
    weight: float | None = None
    spantree_weight: float | None = None
    mean_cv_uniques: object = None


class AlephPDB:
    """ALEPH fragment/CV graphs and their structured node/pair views."""

    def __init__(self, pdb_path, name_folder=""):
        self.pdb_path = str(Path(pdb_path).resolve())
        self.id = Path(pdb_path).stem
        self.folder = str(name_folder)
        graph, _, matrix, cvs, _ = ALEPH.annotate_pdb_model_with_aleph(
            pdb_model=self.pdb_path, weight="distance_avg", strictness_ah=0.50,
            strictness_bs=0.30, peptide_length=3, write_pdb=False, verbose=False)
        self.graph = graph
        self.cv_graph = (ALEPH.generate_graph_for_cvs(graph, matrix, cvs) if cvs else ALEPH.igraph.Graph())
        self.raw_cvs = cvs

    def read_graphs(self):
        self.fragment_dict = {}
        self.cv_dict = {}
        self.distances_between_fragments = {}
        self.distances_between_cv_dict = {}
        for vertex in self.graph.vs:
            data = vertex.attributes()
            fragment = FragmentAleph(
                fragid=vertex.index, sstype=data['sstype'], sequence=data['sequence'],
                reslist=data['reslist'],
                residues_fragment=[Residue.from_residue_data(r) for r in data['reslist']],
                cvids=data['cvids'], cvls=data['cvls'],
                unique_cv=data.get('unique_cv'), vecLength=data.get('vecLength'))
            self.fragment_dict[vertex.index] = fragment
        for edge in self.graph.es:
            data = edge.attributes()
            angle1, dist, angle2 = data['mean'][:3]
            pair = FragmentData(
                dist=dist, angle1=angle1, angle2=angle2, min_distance=data['min'],
                max_distance=max((row[3] for row in data['alls']), default=None),
                weight=data.get('weight'), spantree_weight=data.get('spantree_weight'),
                mean_cv_uniques=data.get('mean_cv_uniques'))
            self.distances_between_fragments.setdefault(edge.source, {})[edge.target] = pair
            if not self.graph.is_directed():
                self.distances_between_fragments.setdefault(edge.target, {})[edge.source] = pair
        self.fragment_list = list(self.fragment_dict.values())
        if self.cv_graph is not None:
            for vertex in self.cv_graph.vs:
                data = vertex.attributes()
                cv = CV(
                    cvid=data['cvs'], secstr=data['secstr'], secondary_structure=data['sstype'],
                    reslist=data['reslist'],
                    residues=[Residue.from_residue_data(r) for r in data['reslist']],
                    nearsecstr=data.get('nearsecstr'), unique_cv=data.get('unique_cv'),
                    previous_cvid=data.get('-1'), next_cvid=data.get('+1'),
                    **{name: data[name] for name in ('sequence', 'sequence_secstr', 'isStart',
                    'isMiddle', 'isEnd', 'isSpecial', 'posandsec', 'pos', 'name') if name in data})
                self.cv_dict[cv.cvid] = cv
                self.fragment_dict[cv.secstr].cv_list.append(cv)
            for edge in self.cv_graph.es:
                a = self.cv_dict[self.cv_graph.vs[edge.source]['cvs']]
                b = self.cv_dict[self.cv_graph.vs[edge.target]['cvs']]
                data = edge.attributes()
                metric = data['metric']
                pair = CVPairData(
                    pair=(a, b), dist=metric[1], angle1=metric[0], angle2=metric[2],
                    weight=data['weight'], sstype1=metric[3], sstype2=metric[4],
                    fragment_angle1=metric[5], fragment_dist=metric[6],
                    fragment_angle2=metric[7], cv1_fragment2_dist=metric[8],
                    cv2_fragment1_dist=metric[9], fragment_count=metric[10])
                self.distances_between_cv_dict.setdefault(a.cvid, {})[b.cvid] = pair
                if not self.cv_graph.is_directed():
                    reverse = replace(
                        pair, pair=(b, a), sstype1=pair.sstype2, sstype2=pair.sstype1,
                        cv1_fragment2_dist=pair.cv2_fragment1_dist,
                        cv2_fragment1_dist=pair.cv1_fragment2_dist)
                    self.distances_between_cv_dict.setdefault(b.cvid, {})[a.cvid] = reverse
        self.cv_list = list(self.cv_dict.values())


    def get_cvs(self, include_coil=False):
        return [cv for cv in self.cv_list if include_coil or cv.secondary_structure != 'coil']

    def get_sse_fragments(self):
        return [fragment for fragment in self.fragment_list if fragment.sstype in ('ah', 'bs')]

    def get_cv_geometry(self, cvid1, cvid2):
        return self.distances_between_cv_dict.get(cvid1, {}).get(cvid2)

    def get_distances_between_fragments(self, fragid1, fragid2):
        return self.distances_between_fragments.get(fragid1, {}).get(fragid2)

    def save_annotation(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load_annotation(cls, path):
        """Load an annotation saved with this standalone class."""
        with Path(path).open('rb') as handle:
            annotation = pickle.load(handle)
        if not isinstance(annotation, cls):
            raise TypeError("Use a pickle saved with aleph_classes_graph.AlephPDB")
        annotation.read_graphs()
        return annotation
