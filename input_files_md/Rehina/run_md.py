import multiprocessing as mp
from openmm import *
from openmm.app import *
from openmm.unit import *
from pathlib import Path
import numpy as np
from functools import partial


def run_md(pdbfile:Path, n_steps):

    pdbfile_struc = PDBFile(str(pdbfile))

    modeller = Modeller(pdbfile_struc.topology, pdbfile_struc.positions)

    forcefield = ForceField("amber14-all.xml", "amber14/tip3p.xml")
    modeller.addHydrogens(forcefield)
    modeller.addSolvent(forcefield, model="tip3p", padding=0.5 * nanometer, boxShape='dodecahedron')

    system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, constraints=HBonds)
    temperature = 300 * kelvin
    pressure = 1 * bar
    integrator = LangevinMiddleIntegrator(temperature, 1 / picosecond, 2 * femtoseconds)
    system.addForce(MonteCarloBarostat(pressure, temperature))

    platform = Platform.getPlatformByName('CUDA')
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=1000)
    positions = simulation.context.getState(getPositions=True).getPositions()
    with open(pdbfile.parent / 'topology.pdb', "w") as f:
        PDBFile.writeFile(simulation.topology, positions, f)

    simulation.reporters = []
    simulation.reporters.append(DCDReporter(str(pdbfile.parent / "traj.dcd"), 1000))
    simulation.reporters.append(
        StateDataReporter(
            str(pdbfile.parent / "metrics.csv"),
            100,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            remainingTime=True,
            totalSteps=n_steps
        )
    )
    print(f"starting simulation for {pdbfile}")
    simulation.step(n_steps)

    # The last line is only needed for Windows users,
    # to close the DCD file before it can be opened by nglview.
    del simulation


def main(dirpath='.', batch_size=8, sim_duration=0.5*nanosecond):

    current_path = Path(dirpath)

    pdb_files = current_path.rglob('*.pdb')

    nsteps = int(np.ceil((sim_duration) / (2*femtosecond)))

    run_md_fn = partial(run_md, n_steps=nsteps)

    ctx = mp.get_context('spawn')

    with ctx.Pool(processes=batch_size) as pool:
        results = pool.map(run_md_fn, pdb_files)


if __name__ == '__main__':
    main(dirpath='.', batch_size=4, sim_duration=5*nanosecond)

