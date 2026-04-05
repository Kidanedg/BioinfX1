import numpy as np

def distance(a, b):
    return np.linalg.norm(a - b)

def docking_score(protein_coords, ligand_coords):

    vdw = 0
    elec = 0
    hbond = 0
    hydrophobic = 0

    epsilon = 0.2
    sigma = 3.5

    for p in protein_coords:
        for l in ligand_coords:

            r = distance(p, l)

            if r < 8:
                # Lennard-Jones
                vdw += 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

                # Electrostatic (toy)
                elec += -1 / r

                # Hydrogen bond
                if 2.5 < r < 3.5:
                    hbond += -1

                # Hydrophobic
                if r < 5:
                    hydrophobic += -0.1

    total = vdw + elec + hbond + hydrophobic

    return {
        "vdw": vdw,
        "electrostatic": elec,
        "hbond": hbond,
        "hydrophobic": hydrophobic,
        "total": total
    }
