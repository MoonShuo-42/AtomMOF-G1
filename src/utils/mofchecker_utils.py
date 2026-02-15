from mofchecker import MOFChecker
from pymatgen.core import Structure


EXPECTED_CHECK_VALUES = {
    "has_carbon": True,
    "has_hydrogen": True,
    "has_atomic_overlaps": False,
    "has_overcoordinated_c": False,
    "has_overcoordinated_n": False,
    "has_overcoordinated_h": False,
    "has_undercoordinated_c": False,
    "has_undercoordinated_n": False,
    "has_undercoordinated_rare_earth": False,
    "has_metal": True,
    "has_lone_molecule": False,
    "has_high_charges": False,
    "is_porous": True,
    "has_suspicicious_terminal_oxo": False,
    "has_undercoordinated_alkali_alkaline": False,
    "has_geometrically_exposed_metal": False,
    "has_3d_connected_graph": True # Ignored
}

TRUE_CHECKS = [k for k, v in EXPECTED_CHECK_VALUES.items() if v]

def is_mof_valid(structure: Structure, descriptors=None):
    """
    Only thing not checked is has_lone_molecule
    """
    checker = MOFChecker(structure)
    desc = checker.get_mof_descriptors(descriptors)
    all_check = []
    for k, v in desc.items():
        if type(v) == bool:
            if k == "has_3d_connected_graph":
                continue
            if k in ["has_carbon", "has_hydrogen", "has_metal", "is_porous"]:
                all_check.append(int(v))
            else:
                all_check.append(int(not v))
    return all(all_check), dict(desc)

def get_failed_checks(desc: dict):
    """
    Args:
    - desc: dict, output from check_mof
    Returns:
    - a list of all failed keys
    """
    failed_checks = []

    for k, v in desc.items():
        if not isinstance(v, bool):
            continue
        if k in TRUE_CHECKS:
            if not v:
                failed_checks.append(k)
        else:
            if v:
                failed_checks.append(k)
    
    return failed_checks

def has_failed_check(desc: dict, key: str) -> bool:
    return desc[key] != EXPECTED_CHECK_VALUES[key]