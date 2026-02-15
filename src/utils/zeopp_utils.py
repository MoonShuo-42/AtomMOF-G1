"""
Utility functions for running zeo++ calculations (adapted from MOFDiff)
"""
import os
import re
import json
import subprocess
from pathlib import Path

# Set ZEO_PATH
zeo_path_env = os.environ.get("ZEO_PATH")

if not zeo_path_env:
    raise EnvironmentError(
        "ZEO_PATH environment variable not set. "
        "Please export ZEO_PATH=/path/to/zeo++/network"
    )

ZEO_PATH = Path(zeo_path_env)


class ZeoSimulation:
    def __init__(self, cif_file):
        self.sorbent_file = cif_file

        # make temp directory for Zeo++ output
        if not os.path.exists("temp"):
            os.mkdir("temp")

        # Zeo++ related parameters
        self.pld = float  # pore-limiting diameter of a He probe, A
        self.lcd = float  # largest cavity diameter, A
        self.gravimetric_surface_area = (
            None  # gravimetric surface area as measured with a He probe, m2/g
        )
        self.volumetric_surface_area = (
            None  # volumetric surface area as measured with a He probe, m2/mL
        )
        self.unitcell_volume = None  # volume of unit cell, A3
        self.density = None  # density, g/mL
        self.accessible_volume = None  # volume accessible to a He probe, A3
        self.void_fraction = None  # ratio of accessible volume to total volume

    def write_out(self, output_path):
        with open(output_path, "w") as log_file:
            log_file.write("input file: " + self.sorbent_file + "\n")
            log_file.write("pore limiting diameter: " + str(self.pld) + " A\n")
            log_file.write("largest cavity diameter: " + str(self.lcd) + " A\n")
            log_file.write(
                "gravimetric surface area: "
                + str(self.gravimetric_surface_area)
                + " m2/g\n"
            )
            log_file.write(
                "volumetric surface area: "
                + str(self.volumetric_surface_area)
                + " m2/mL\n"
            )
            log_file.write("unitcell_volume: " + str(self.unitcell_volume) + " A3\n")
            log_file.write("density: " + str(self.density) + " g/mL\n")
            log_file.write(
                "accessible_volume: " + str(self.accessible_volume) + " A3\n"
            )
            log_file.write("void_fraction: " + str(self.void_fraction))


# calculate pore limiting diameter of a porous solid
def calculate_pld(simulation, zeo_output_path=None):
    # update zeo++ output path
    if zeo_output_path is None:
        zeo_output_path = os.getcwd() + "/temp/zeo_output.cssr"

    # call Zeo++ to calculate pore dimensions
    subprocess.run([ZEO_PATH, "-ha", "-res", zeo_output_path, simulation.sorbent_file])

    # retrieve PLD from Zeo++ output
    with open(zeo_output_path, "r") as zeo_output:
        zeo_output = zeo_output.read()

    pore_diameters = zeo_output.split("\n")[0].split()[1:]
    # store pld in simulation object
    # Zeo++ returns three diameters: largest included sphere, largest free sphere (pld),
    # and largest included sphere along free sphere path.
    if len(pore_diameters) > 0:
        simulation.pld = float(pore_diameters[1])
        simulation.lcd = float(pore_diameters[0])
    else:
        # fail-state for nonporous materials
        simulation.pld = 0
        simulation.lcd = 0


# calculate accessible surface area for a helium atom
def calculate_accessible_sa(
    simulation, probe_radius=1.2755, num_samples=10000, zeo_output_path=None
):
    # update zeo++ output path
    if zeo_output_path is None:
        zeo_output_path = os.getcwd() + "/temp/zeo_output.cssr"

    # call Zeo++ to find accessible surface area
    subprocess.run(
        [
            ZEO_PATH,
            "-ha",
            "-sa",
            str(probe_radius),
            str(probe_radius),
            str(num_samples),
            zeo_output_path,
            simulation.sorbent_file,
        ]
    )

    # retrieve Zeo++ output
    with open(zeo_output_path, "r") as zeo_output_file:
        zeo_output = zeo_output_file.read()

    # pull surface area estimates
    gravimetric_surface_area = re.findall(r"\sASA_m\^2/g:\s\d+.\d+", zeo_output)
    if len(gravimetric_surface_area) > 0:
        gravimetric_surface_area = float(
            re.findall(r"\d+.\d+", gravimetric_surface_area[0])[0]
        )
    else:
        # fail-state for nonporous materials
        gravimetric_surface_area = 0

    volumetric_surface_area = re.findall(r"\sASA_m\^2/cm\^3:\s\d+.\d+", zeo_output)
    if len(volumetric_surface_area) > 0:
        volumetric_surface_area = float(
            re.findall(r"\d+.\d+", volumetric_surface_area[0])[0]
        )
    else:
        # fail-state for nonporous materials
        volumetric_surface_area = 0

    # store pld in simulation object
    simulation.gravimetric_surface_area = gravimetric_surface_area
    simulation.volumetric_surface_area = volumetric_surface_area


# calculate volume-related properties
def calculate_volume(
    simulation, probe_radius=1.2755, num_samples=100000, zeo_output_path=None
):
    # update zeo++ output path
    if zeo_output_path is None:
        zeo_output_path = os.getcwd() + "/temp/zeo_output.cssr"

    # call Zeo++
    subprocess.run(
        [
            ZEO_PATH,
            "-vol",
            str(probe_radius),
            str(probe_radius),
            str(num_samples),
            zeo_output_path,
            simulation.sorbent_file,
        ]
    )

    # retrieve Zeo++ output
    with open(zeo_output_path, "r") as zeo_output_file:
        zeo_output = zeo_output_file.read()

    # pull from log file
    volume = re.findall(r"\sUnitcell_volume:\s\d+.\d+", zeo_output)
    if len(volume) > 0:
        volume = float(re.findall(r"\d+.\d+", volume[0])[0])
    else:
        volume = 0

    density = re.findall(r"\sDensity:\s\d+.\d+", zeo_output)
    if len(density) > 0:
        density = float(re.findall(r"\d+.\d+", density[0])[0])
    else:
        density = 0

    accessible_volume = re.findall(r"\sAV_A\^3:\s\d+.\d+", zeo_output)
    if len(accessible_volume) > 0:
        accessible_volume = float(re.findall(r"\d+.\d+", accessible_volume[0])[0])
    else:
        accessible_volume = 0

    void_fraction = re.findall(r"AV_Volume_fraction:\s\d+.\d+", zeo_output)
    if len(void_fraction) > 0:
        void_fraction = float(re.findall(r"\d+.\d+", void_fraction[0])[0])
    else:
        void_fraction = 0

    # store in simulation object
    simulation.unitcell_volume = volume
    simulation.density = density
    simulation.accessible_volume = accessible_volume
    simulation.void_fraction = void_fraction


def mof_properties(cif_file, zeo_store_path="./", cleanup=False, save_file=False):
    """
    gather Zeo++ properties
    """
    try:
        uid = Path(cif_file).parts[-1].split(".")[0]
        if isinstance(zeo_store_path, str):
            zeo_store_path = Path(zeo_store_path)

        cssr_file = str(Path(zeo_store_path) / f"{uid}_zeo_output.cssr")
        zeosim = ZeoSimulation(cif_file)
        calculate_pld(zeosim, zeo_output_path=cssr_file)
        calculate_accessible_sa(zeosim, zeo_output_path=cssr_file)
        calculate_volume(zeosim, zeo_output_path=cssr_file)

        results = {
            "pld": zeosim.pld,
            "lcd": zeosim.lcd,
            "gravimetric_sa": zeosim.gravimetric_surface_area,
            "volumetric_sa": zeosim.volumetric_surface_area,
            "unitcell_volume": zeosim.unitcell_volume,
            "density": zeosim.density,
            "accessible_volume": zeosim.accessible_volume,
            "void_fraction": zeosim.void_fraction,
            "uid": uid,
            "is_porous": zeosim.pld >= 2.4,  # criterion used by MOFChecker
        }

        if cleanup:
            os.remove(cssr_file)

        if save_file:
            (Path(zeo_store_path) / "results").mkdir(exist_ok=True, parents=True)
            with open(Path(zeo_store_path) / "results" / f"{uid}.json", "w") as f:
                json.dump(results, f)

        return results
    except Exception as e:
        print(e)
        return None