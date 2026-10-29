
import csv
import numpy as np
import matplotlib.pyplot as plt

from nupack import Strand, Complex, Tube, SetSpec, Model, complex_analysis, tube_analysis
from nupack.constants import water_molarity

# ============================================================
# INPUT SEQUENCES
# Entered as 5' -> 3' sequences.
# T is converted to U for the RNA thermodynamic model.
# ============================================================

seq1 = """
AACTCTCGATCAGCGGACAGCGCTAATACGACTCACTATAGGGAGATACCTGGGAATGTGCTCACGTCTCGAGCATATTCGGCCTGCTTCGGCAGGCGCTCGCGAGACGTGCGAGCGCAGGGTGAAGCCTCCACGCCCTGCCCTCACTTCGGTGAGCCAGGTGTCTCCCGCAGGGTGCGCGTGGCTTCGGCCACGACCTGAAGGAGGCACGGGTCGACGGTACCCAGTGGACGGGTATCGTCGCGGAGCTTCGGCTCCGGACCGCGTCCACTGCGGTCCGCACTCTGC
"""

seq2 = """
GCTCGAATTCTAATACGACTCACTATAGggGGAATTAGAGTGTGTTCCTGAACTGCTTCGGCGGTTCGCTACGTTCTTCGGAATGTATATAGTGTTCGCATTATACCGTAGTCCAA
"""


def clean_sequence(sequence):
    sequence = "".join(sequence.split()).upper().replace("T", "U")
    invalid = set(sequence) - set("ACGU")
    if invalid:
        raise ValueError(f"Invalid nucleotide characters: {invalid}")
    return sequence


seq1 = clean_sequence(seq1)
seq2 = clean_sequence(seq2)

print(f"3H-AE length: {len(seq1)} nt")
print(f"5H-TB length: {len(seq2)} nt")

# ============================================================
# EXPERIMENTAL CONDITIONS -- CHANGE AS NEEDED
# ============================================================

conc1 = 1e-6  # 3HAE concentration in M (1 µM)
conc2 = 1e-6  # 5HTB concentration in M (1 µM)

T_MIN = 20
T_MAX = 90
T_STEP = 2

# ============================================================
# DEFINE STRANDS
# ============================================================

A = Strand(seq1, name="3H-AE")
B = Strand(seq2, name="5H-TB")
complex_A = Complex([A], name="3H-AE")
complex_B = Complex([B], name="5H-TB")
complex_AA = Complex([A, A], name="3H-AE:3H-AE")
complex_BB = Complex([B, B], name="5H-TB:5H-TB")

temperatures = np.arange(T_MIN, T_MAX + T_STEP / 2, T_STEP)
fractions_paired_A = []
fractions_paired_B = []

# ============================================================
# TEMPERATURE SCAN
# ============================================================

for temp in temperatures:
    print(f"Calculating at {temp:.1f} °C ...")

    model = Model(
        material="rna",
        celsius=float(temp)
    )

    result = complex_analysis(
        complexes=[complex_A, complex_B],
        model=model,
        compute=["pairs"]
    )

    # The diagonal of each pair-probability matrix is the probability that
    # each nucleotide is unpaired. Report the fraction paired within that
    # strand as one minus the mean unpaired probability.
    pairs_A = result[complex_A].pairs
    pairs_B = result[complex_B].pairs
    fraction_paired_A = 1.0 - float(np.asarray(pairs_A.diagonal).mean())
    fraction_paired_B = 1.0 - float(np.asarray(pairs_B.diagonal).mean())
    fractions_paired_A.append(fraction_paired_A)
    fractions_paired_B.append(fraction_paired_B)

# ============================================================
# ESTIMATE 50% MELTING TEMPERATURE FOR EACH STRAND
# ============================================================

fractions_paired_A = np.asarray(fractions_paired_A)
fractions_paired_B = np.asarray(fractions_paired_B)


def estimate_tm(fractions, strand_name):
    crossings = np.where(
        (fractions[:-1] >= 0.5) &
        (fractions[1:] < 0.5)
    )[0]

    if len(crossings) > 0:
        i = crossings[0]

        # Linear interpolation between adjacent temperatures.
        t1, t2 = temperatures[i], temperatures[i + 1]
        f1, f2 = fractions[i], fractions[i + 1]
        tm = t1 + (0.5 - f1) * (t2 - t1) / (f2 - f1)
        print(f"Estimated {strand_name} 50% paired Tm: {tm:.2f} °C")
        return tm

    if fractions[0] < 0.5:
        print(f"{strand_name} is already below 50% paired at the minimum temperature.")
    elif fractions[-1] >= 0.5:
        print(f"{strand_name} remains at or above 50% paired at the maximum temperature.")
    else:
        print(f"No downward 50% paired crossing was detected for {strand_name}.")
    return None


tm_A = estimate_tm(fractions_paired_A, "3H-AE")
tm_B = estimate_tm(fractions_paired_B, "5H-TB")


# Homodimers and their Gibbs free energies of association
fractions_AA = []
fractions_BB = []
complex_free_energy_AA = []
complex_free_energy_BB = []
delta_g_assoc_AA = []
delta_g_assoc_BB = []

GAS_CONSTANT_KCAL = 0.00198717  # kcal / (mol K), matching NUPACK
STANDARD_CONCENTRATION_M = 1.0

for temp in temperatures:
    print(f"Calculating at {temp:.1f} °C ...")

    model = Model(material="rna", celsius=float(temp))

    # Tube A contains only strand A; no A:B complexes are possible.
    tube_A = Tube(
        strands={A: conc1},
        complexes=SetSpec(max_size=2),
        name="3H-AE only"
    )

    # Tube B contains only strand B; no A:B complexes are possible.
    tube_B = Tube(
        strands={B: conc2},
        complexes=SetSpec(max_size=2),
        name="5H-TB only"
    )

    result = tube_analysis(
        tubes=[tube_A, tube_B],
        model=model
    )

    conc_AA = float(
        result.tubes[tube_A].complex_concentrations.get(complex_AA, 0.0)
    )
    conc_BB = float(
        result.tubes[tube_B].complex_concentrations.get(complex_BB, 0.0)
    )

    # Each homodimer consumes two molecules of its strand.
    fraction_A_in_AA = min(1.0, 2 * conc_AA / conc1)
    fraction_B_in_BB = min(1.0, 2 * conc_BB / conc2)

    fractions_AA.append(fraction_A_in_AA)
    fractions_BB.append(fraction_B_in_BB)

    # NUPACK's complex free energies are ensemble Gibbs free energies in
    # kcal/mol. Convert the dimerization free energy to the 1 M standard state:
    # ΔG°assoc = G(AA) - 2G(A) + RT ln(ρ_water / 1 M).
    temperature_kelvin = float(model.temperature)
    water_concentration = water_molarity(temperature_kelvin)
    g_A = float(result[complex_A].free_energy)
    g_B = float(result[complex_B].free_energy)
    g_AA = float(result[complex_AA].free_energy)
    g_BB = float(result[complex_BB].free_energy)

    complex_free_energy_AA.append(g_AA)
    complex_free_energy_BB.append(g_BB)
    delta_g_assoc_AA.append(
        g_AA - 2 * g_A
        + GAS_CONSTANT_KCAL * temperature_kelvin
        * np.log(water_concentration / STANDARD_CONCENTRATION_M)
    )
    delta_g_assoc_BB.append(
        g_BB - 2 * g_B
        + GAS_CONSTANT_KCAL * temperature_kelvin
        * np.log(water_concentration / STANDARD_CONCENTRATION_M)
    )

    print(f"  3H-AE in A:A homodimer: {100*fraction_A_in_AA:.2f}%")
    print(f"  5H-TB in B:B homodimer: {100*fraction_B_in_BB:.2f}%")

fractions_AA = np.asarray(fractions_AA)
fractions_BB = np.asarray(fractions_BB)
complex_free_energy_AA = np.asarray(complex_free_energy_AA)
complex_free_energy_BB = np.asarray(complex_free_energy_BB)
delta_g_assoc_AA = np.asarray(delta_g_assoc_AA)
delta_g_assoc_BB = np.asarray(delta_g_assoc_BB)


def estimate_homodimer_simulation_tm(fractions, complex_name):
    crossings = np.where((fractions[:-1] >= 0.5) & (fractions[1:] < 0.5))[0]
    if len(crossings) == 0:
        print(f"No 50% simulation Tm crossing detected for {complex_name}.")
        return None

    i = crossings[0]
    t1, t2 = temperatures[i], temperatures[i + 1]
    f1, f2 = fractions[i], fractions[i + 1]
    tm = t1 + (0.5 - f1) * (t2 - t1) / (f2 - f1)
    print(f"{complex_name} simulation Tm (50% in homodimer): {tm:.2f} °C")
    return tm


def estimate_gibbs_tm(delta_g_standard, total_strand_concentration, complex_name):
    # At 50% of strand molecules in dimers, [AA]/[A]^2 = 1/C_total.
    # Thus ΔG°assoc = RT ln(C_total / 1 M); find that crossing in temperature.
    temperature_kelvin = temperatures + 273.15
    concentration_threshold = (
        GAS_CONSTANT_KCAL * temperature_kelvin
        * np.log(total_strand_concentration / STANDARD_CONCENTRATION_M)
    )
    residual = delta_g_standard - concentration_threshold
    crossings = np.where((residual[:-1] <= 0) & (residual[1:] >= 0))[0]
    if len(crossings) == 0:
        print(f"No Gibbs-derived theoretical Tm crossing detected for {complex_name}.")
        return None, residual

    i = crossings[0]
    t1, t2 = temperatures[i], temperatures[i + 1]
    r1, r2 = residual[i], residual[i + 1]
    tm = t1 - r1 * (t2 - t1) / (r2 - r1)
    print(f"{complex_name} Gibbs-derived theoretical Tm: {tm:.2f} °C")
    return tm, residual


tm_AA_sim = estimate_homodimer_simulation_tm(fractions_AA, "3H-AE:3H-AE")
tm_BB_sim = estimate_homodimer_simulation_tm(fractions_BB, "5H-TB:5H-TB")
tm_AA_gibbs, gibbs_residual_AA = estimate_gibbs_tm(
    delta_g_assoc_AA, conc1, "3H-AE:3H-AE"
)
tm_BB_gibbs, gibbs_residual_BB = estimate_gibbs_tm(
    delta_g_assoc_BB, conc2, "5H-TB:5H-TB"
)


# ============================================================
# SAVE RESULTS
# ============================================================

with open("rna_melting_results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Temperature_C",
        "3H-AE_fraction_paired",
        "5H-TB_fraction_paired",
        "3H-AE_fraction_in_homodimer",
        "5H-TB_fraction_in_homodimer",
        "3H-AE_AA_complex_free_energy_kcal_per_mol",
        "5H-TB_BB_complex_free_energy_kcal_per_mol",
        "3H-AE_AA_association_delta_G_standard_kcal_per_mol",
        "5H-TB_BB_association_delta_G_standard_kcal_per_mol",
        "3H-AE_Gibbs_Tm_criterion_residual_kcal_per_mol",
        "5H-TB_Gibbs_Tm_criterion_residual_kcal_per_mol"
    ])

    for values in zip(
        temperatures,
        fractions_paired_A,
        fractions_paired_B,
        fractions_AA,
        fractions_BB,
        complex_free_energy_AA,
        complex_free_energy_BB,
        delta_g_assoc_AA,
        delta_g_assoc_BB,
        gibbs_residual_AA,
        gibbs_residual_BB
    ):
        writer.writerow(values)

plt.figure(figsize=(8, 5))
plt.plot(temperatures, fractions_paired_A, "o-", label="3H-AE (single strand)")
plt.plot(temperatures, fractions_paired_B, "s-", label="5H-TB (single strand)")
plt.axhline(0.5, color="gray", linestyle="--", label="50% paired")

if tm_A is not None:
    plt.axvline(tm_A, linestyle=":", label=f"3H-AE Tm = {tm_A:.2f} °C")
if tm_B is not None:
    plt.axvline(tm_B, linestyle="-.", label=f"5H-TB Tm = {tm_B:.2f} °C")

plt.xlabel("Temperature (°C)")
plt.ylabel("Fraction of nucleotides paired within each strand")
plt.title("NUPACK Single-Strand RNA Melting Curves")
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("rna_melting_curve.png", dpi=300)

delta_g_AA_at_37 = float(np.interp(37.0, temperatures, delta_g_assoc_AA))
delta_g_BB_at_37 = float(np.interp(37.0, temperatures, delta_g_assoc_BB))

plt.figure(figsize=(9, 6))
plt.plot(
    temperatures, fractions_AA * 100, "o-", color="tab:blue",
    label=("3H-AE:3H-AE, "
           f"ΔG°assoc at 37 °C = {delta_g_AA_at_37:.2f} kcal/mol")
)
plt.plot(
    temperatures, fractions_BB * 100, "s-", color="tab:orange",
    label=("5H-TB:5H-TB, "
           f"ΔG°assoc at 37 °C = {delta_g_BB_at_37:.2f} kcal/mol")
)
plt.axhline(50, color="gray", linestyle="-", label="50% of strand molecules in homodimer")

if tm_AA_sim is not None:
    plt.axvline(
        tm_AA_sim, color="tab:blue", linestyle="--",
        label=f"3H-AE:3H-AE simulation Tm = {tm_AA_sim:.2f} °C"
    )
if tm_AA_gibbs is not None:
    plt.axvline(
        tm_AA_gibbs, color="tab:blue", linestyle=":",
        label=f"3H-AE:3H-AE Gibbs theoretical Tm = {tm_AA_gibbs:.2f} °C"
    )
if tm_BB_sim is not None:
    plt.axvline(
        tm_BB_sim, color="tab:orange", linestyle="--",
        label=f"5H-TB:5H-TB simulation Tm = {tm_BB_sim:.2f} °C"
    )
if tm_BB_gibbs is not None:
    plt.axvline(
        tm_BB_gibbs, color="tab:orange", linestyle=":",
        label=f"5H-TB:5H-TB Gibbs theoretical Tm = {tm_BB_gibbs:.2f} °C"
    )

plt.xlabel("Temperature (°C)")
plt.ylabel("Fraction of strand molecules in homodimers (%)")
plt.title("RNA Homodimer Melting Curves")
plt.ylim(-5, 105)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("rna_homodimer_melting_curve.png", dpi=300)

plt.figure(figsize=(8, 5))
plt.plot(
    temperatures, gibbs_residual_AA, "o-", color="tab:blue",
    label="3H-AE:3H-AE Gibbs criterion"
)
plt.plot(
    temperatures, gibbs_residual_BB, "s-", color="tab:orange",
    label="5H-TB:5H-TB Gibbs criterion"
)
plt.axhline(0, color="gray", linestyle="--", label="Theoretical 50% condition")
if tm_AA_gibbs is not None:
    plt.axvline(
        tm_AA_gibbs, color="tab:blue", linestyle=":",
        label=f"3H-AE:3H-AE Gibbs theoretical Tm = {tm_AA_gibbs:.2f} °C"
    )
if tm_BB_gibbs is not None:
    plt.axvline(
        tm_BB_gibbs, color="tab:orange", linestyle=":",
        label=f"5H-TB:5H-TB Gibbs theoretical Tm = {tm_BB_gibbs:.2f} °C"
    )
if tm_AA_sim is not None:
    plt.axvline(
        tm_AA_sim, color="tab:blue", linestyle="--",
        label=f"3H-AE:3H-AE simulation Tm = {tm_AA_sim:.2f} °C"
    )
if tm_BB_sim is not None:
    plt.axvline(
        tm_BB_sim, color="tab:orange", linestyle="--",
        label=f"5H-TB:5H-TB simulation Tm = {tm_BB_sim:.2f} °C"
    )
plt.xlabel("Temperature (°C)")
plt.ylabel("ΔG°assoc − RT ln(Ctotal / 1 M) (kcal/mol)")
plt.title("Gibbs Free-Energy Criterion for Homodimer Melting")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("rna_homodimer_gibbs_energy.png", dpi=300)

plt.show()

print("\nSaved rna_melting_results.csv")
print("Saved rna_melting_curve.png")
print("Saved rna_homodimer_melting_curve.png")
print("Saved rna_homodimer_gibbs_energy.png")
