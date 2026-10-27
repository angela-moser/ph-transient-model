#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, math, shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit

# Inputs ======================================================================
def inputs(filename):
    # User Inputs
    ms = {}

    # Column physical / numerical properties
    ms['N'] = 100  # Number of axial discretizations

    # Ligand densities (M packed column)
    ms['sCEX_ligand_density'] = 0.0
    ms['wCEX_ligand_density'] = 0.0082769
    ms['sAEX_ligand_density'] = 0.0
    ms['wAEX_ligand_density'] = 0.0070152

    # pKas for weak acid and weak base ligands
    ms['pKa_acid'] = 6.60925
    ms['pKb_base'] = 11.4326

    # Column information
    ms['L'] = 15.8 # Column length (cm)
    ms['D'] = 1.0 # Column diameter (cm)
    ms['Ee'] = 0.34 # Extraparticle porosity
    ms['Ep'] = 0.94 # Intraparticle porosity
    ms['F'] = 3.371 # Flow rate (mL/min)

    # Preliminary calculations
    ms['Ac'] = 0.25 * np.pi * ms['D'] ** 2  # Cross sectional area (cm^2)
    ms['col_vol'] = ms['Ac'] * ms['L'] # Column volume (mL)
    ms['Et'] = ms['Ee'] + (1 - ms['Ee']) * ms['Ep']  # Total porosity
    ms['inv_dx'] = ms['N'] / ms['L']
    ms['solid_phase_ratio'] = 1 / ms['Et']
    ms['sup_vel'] = ms['F'] / ms['Ac'] # Superficial velocity (cm/min)
    ms['int_vel'] = ms['sup_vel'] / ms['Et'] # Interstitial velocity (cm/min)
    ms['K_acid'] = 10 ** -ms['pKa_acid']
    ms['K_base'] = 10 ** -ms['pKb_base']

    # Locate input files relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, 'input_profiles')
    ms['inlet_data_path'] = os.path.join(input_dir, filename)

    return ms

# Differential equation =======================================================
@njit
def dsys(
        t, y, time, cNa0, cCl0, cT0, cG0, N, inv_dx, int_vel,
        solid_phase, sCEX_ligand_density, wCEX_ligand_density,
        sAEX_ligand_density, wAEX_ligand_density, K_acid, K_base
    ):

    # Unpack y vector
    cT_tot = y[0*N:1*N+1]
    cG_tot = y[1*N+1:2*N+2]
    cCl = y[2*N+2:3*N+3]
    cNa = y[3*N+3:4*N+4]
    qNa = y[4*N+4:5*N+5]
    qTH = y[5*N+5:6*N+6]
    qCl = y[6*N+6:7*N+7]
    qGH = y[7*N+7:8*N+8]
    qG1 = y[8*N+8:9*N+9]

    # Zeros dot vectors
    dydt = np.zeros(9*N+9)
    dcTdt = np.zeros(N+1)
    dcGdt = np.zeros(N+1)
    dcCldt = np.zeros(N+1)
    dcNadt = np.zeros(N+1)
    dqNadt = np.zeros(N+1)
    dqTHdt = np.zeros(N+1)
    dqCldt = np.zeros(N+1)
    dqGHdt = np.zeros(N+1)
    dqG1dt = np.zeros(N+1)

    # Find the index i where the given time point falls
    j = np.searchsorted(time, t) - 1

    # Get the multiplier for interpolation
    mult = (t - time[j]) / (time[j+1] - time[j])

    # Create IS vector
    IS = np.zeros(N+1)

    # Interpolate the concentrations at the inlet for the given time point
    cNa_in = cNa0[j] + (cNa0[j+1] - cNa0[j]) * mult
    cCl_in = cCl0[j] + (cCl0[j+1] - cCl0[j]) * mult
    cG_tot_in = cG0[j] + (cG0[j+1] - cG0[j]) * mult
    cT_tot_in = cT0[j] + (cT0[j+1] - cT0[j]) * mult

    # Precompute constants
    k_kin = 1e2
    conv = int_vel * inv_dx

    # Calculations for dqdt[0] ----------------------------------------------
    # Get equilibrated IS and cH
    IS[0], Kw_in, Ka_G1_in, Ka_G2_in, Ka_T_in, cH_in = equilibrate(cNa_in, cCl_in, cG_tot_in, cT_tot_in)

    # Calculate ion concentrations
    cTH_in, cGH_in, cG1_in = ion_concs(cH_in, cT_tot_in, cG_tot_in, Ka_G1_in, Ka_G2_in, Ka_T_in)

    # Calculate bound concentrations
    qNa_eq_s, qTH_eq_s, qGH_eq_s = sCEX_isotherm(cH_in, cNa_in, cTH_in, cGH_in, sCEX_ligand_density)
    qNa_eq_w, qTH_eq_w, qGH_eq_w = wCEX_isotherm(cH_in, cNa_in, cTH_in, cGH_in, wCEX_ligand_density, K_acid)
    qCl_eq_s, qG1_eq_s = sAEX_isotherm(cH_in, cCl_in, cG1_in, sAEX_ligand_density, Kw_in)
    qCl_eq_w, qG1_eq_w = wAEX_isotherm(cH_in, cCl_in, cG1_in, wAEX_ligand_density, Kw_in, K_base)

    qNa_eq = qNa_eq_s + qNa_eq_w
    qTH_eq = qTH_eq_s + qTH_eq_w
    qCl_eq = qCl_eq_s + qCl_eq_w
    qGH_eq = qGH_eq_s + qGH_eq_w
    qG1_eq = qG1_eq_s + qG1_eq_w

    # Linear driving force for interacting ions
    dqNadt[0] = k_kin * (qNa_eq - qNa[0])
    dqTHdt[0] = k_kin * (qTH_eq - qTH[0])
    dqCldt[0] = k_kin * (qCl_eq - qCl[0])
    dqGHdt[0] = k_kin * (qGH_eq - qGH[0])
    dqG1dt[0] = k_kin * (qG1_eq - qG1[0])

    #--------------------------------------------------------------------------
    # Inlet boundary condition using inlet concentrations
    dcNadt[0] = (cNa_in - cNa[0]) * conv - solid_phase * dqNadt[0]
    dcTdt[0]  = (cT_tot_in - cT_tot[0]) * conv - solid_phase * dqTHdt[0]
    dcCldt[0] = (cCl_in - cCl[0]) * conv - solid_phase * dqCldt[0]
    dcGdt[0]  = (cG_tot_in - cG_tot[0]) * conv - solid_phase * (dqGHdt[0] + dqG1dt[0])

    # -------------------------------------------------------------------------
    # Set up differential balances for all discretization cells
    for i in range(1, N+1):
        # Pull into local variables for speed
        cNa_i = cNa[i]
        cCl_i = cCl[i]
        cG_tot_i = cG_tot[i]
        cT_tot_i = cT_tot[i]
        qNa_i = qNa[i]
        qTH_i = qTH[i]
        qCl_i = qCl[i]
        qGH_i = qGH[i]
        qG1_i = qG1[i]

        # Get equilibrated IS and cH
        IS[i], Kw_i, Ka_G1_i, Ka_G2_i, Ka_T_i, cH_i = equilibrate(cNa_i, cCl_i, cG_tot_i, cT_tot_i)

        # Calculate ion concentrations
        cTH_i, cGH_i, cG1_i = ion_concs(cH_i, cT_tot_i, cG_tot_i, Ka_G1_i, Ka_G2_i, Ka_T_i)

        # Calculate bound concentrations
        qNa_eq_s, qTH_eq_s, qGH_eq_s = sCEX_isotherm(cH_i, cNa_i, cTH_i, cGH_i, sCEX_ligand_density)
        qNa_eq_w, qTH_eq_w, qGH_eq_w = wCEX_isotherm(cH_i, cNa_i, cTH_i, cGH_i, wCEX_ligand_density, K_acid)
        qCl_eq_s, qG1_eq_s = sAEX_isotherm(cH_i, cCl_i, cG1_i, sAEX_ligand_density, Kw_i)
        qCl_eq_w, qG1_eq_w = wAEX_isotherm(cH_i, cCl_i, cG1_i, wAEX_ligand_density, Kw_i, K_base)

        qNa_eq = qNa_eq_s + qNa_eq_w
        qTH_eq = qTH_eq_s + qTH_eq_w
        qCl_eq = qCl_eq_s + qCl_eq_w
        qGH_eq = qGH_eq_s + qGH_eq_w
        qG1_eq = qG1_eq_s + qG1_eq_w

        # Linear driving force for interacting ions
        dqNadt[i] = k_kin * (qNa_eq - qNa_i)
        dqTHdt[i] = k_kin * (qTH_eq - qTH_i)
        dqCldt[i] = k_kin * (qCl_eq - qCl_i)
        dqGHdt[i] = k_kin * (qGH_eq - qGH_i)
        dqG1dt[i] = k_kin * (qG1_eq - qG1_i)

        # Transport of ions
        dcNadt[i] = -solid_phase * dqNadt[i] + (cNa[i-1] - cNa_i) * conv
        dcTdt[i] = -solid_phase * dqTHdt[i] + (cT_tot[i-1] - cT_tot_i) * conv
        dcCldt[i] = -solid_phase * dqCldt[i] + (cCl[i-1] - cCl_i) * conv
        dcGdt[i] = -solid_phase * (dqGHdt[i] + dqG1dt[i]) + (cG_tot[i-1] - cG_tot_i) * conv

    # -------------------------------------------------------------------------
    # Transfer back to ydot vector for solving
    dydt[0*N:1*N+1] = dcTdt
    dydt[1*N+1:2*N+2] = dcGdt
    dydt[2*N+2:3*N+3] = dcCldt
    dydt[3*N+3:4*N+4] = dcNadt
    dydt[4*N+4:5*N+5] = dqNadt
    dydt[5*N+5:6*N+6] = dqTHdt
    dydt[6*N+6:7*N+7] = dqCldt
    dydt[7*N+7:8*N+8] = dqGHdt
    dydt[8*N+8:9*N+9] = dqG1dt

    return dydt


def dsys_wrapper(t, y, ms):
    return dsys(t, y,
                    ms['inlet_time'],
                    ms['cNa'],
                    ms['cCl'],
                    ms['cT_total'],
                    ms['cG_total'],
                    ms['N'],
                    ms['inv_dx'],
                    ms['int_vel'],
                    ms['solid_phase_ratio'],
                    ms['sCEX_ligand_density'],
                    ms['wCEX_ligand_density'],
                    ms['sAEX_ligand_density'],
                    ms['wAEX_ligand_density'],
                    ms['K_acid'],
                    ms['K_base'],)


# Electroneutrality for differential equation =================================
@njit
def residual_and_deriv_logH(logH, cNa, cCl, Kw, Ka_G1, Ka_G2, Ka_T, cG, cT):
    """
    Electroneutrality residual and derivative of electroneutrality residual
    """
    ln10 = 2.302585092994046
    cH = 10.0 ** logH

    # Calculate ion concentrations
    cOH = Kw / cH
    cTH, cGH, cG1 = ion_concs(cH, cT, cG, Ka_G1, Ka_G2, Ka_T)

    # Residual
    f = cCl + cOH + cG1 - cNa - cTH - cGH - cH

    # Derivatives wrt cH
    dOH_dH = -Kw / (cH**2)
    dTH_dH = cT * Ka_T / (cH + Ka_T)**2

    DG = cH**2 + Ka_G1 * cH + Ka_G1 * Ka_G2
    dDG_dH = 2*cH + Ka_G1

    dGH_dH = cG * (2*cH*DG - cH**2 * dDG_dH) / (DG**2)
    dG1_dH = -Ka_G1 * Ka_G2 * cG * dDG_dH / (DG**2)

    df_dH = dOH_dH + dG1_dH - dTH_dH - dGH_dH - 1.0

    # Chain rule: dH/dlogH = ln(10) * H
    df_dlogH = df_dH * (ln10 * cH)

    return f, df_dlogH


@njit
def residual_logH(logH, cNa, cCl, Kw, Ka_G1, Ka_G2, Ka_T, cG, cT):
    cH = 10.0 ** logH

    cOH = Kw / cH
    cTH, cGH, cG1 = ion_concs(cH, cT, cG, Ka_G1, Ka_G2, Ka_T)

    f = cCl + cOH + cG1 - cNa - cTH - cGH - cH
    return f


@njit
def solve_cH_newton(cNa, cCl, Kw, Ka_G1, Ka_G2, Ka_T, cG, cT,
                      logH0=-8.0,
                      tol_f=1e-12,
                      tol_step=1e-10,
                      maxiter=20,
                      clamp_lo=-9.0,
                      clamp_hi=-2.0):
    """
    Damped Newton in log10(cH) with simple safeguards.
    Returns cH (may be NaN if it can't converge robustly).
    """
    logH = logH0

    # Clamp initial guess
    if logH < clamp_lo:
        logH = clamp_lo
    elif logH > clamp_hi:
        logH = clamp_hi

    f, df = residual_and_deriv_logH(logH, cNa, cCl, Kw, Ka_G1, Ka_G2, Ka_T, cG, cT)

    for _ in range(maxiter):
        # Converged by residual
        if abs(f) < tol_f:
            return 10.0 ** logH

        # Bad derivative -> break to fallback
        if df == 0.0 or not np.isfinite(df) or not np.isfinite(f):
            break

        step = -f / df

        # Converged by step size
        if abs(step) < tol_step:
            return 10.0 ** logH

        # Damping / line search: ensure we improve |f|
        alpha = 1.0
        improved = False
        f_abs = abs(f)

        for _ls in range(8):  # Up to 8 halvings
            cand = logH + alpha * step

            # Clamp to keep cH sane
            if cand < clamp_lo:
                cand = clamp_lo
            elif cand > clamp_hi:
                cand = clamp_hi

            f_cand, df_cand = residual_and_deriv_logH(cand, cNa, cCl, Kw, Ka_G1, Ka_G2, Ka_T, cG, cT)

            if np.isfinite(f_cand) and abs(f_cand) < f_abs:
                logH = cand
                f = f_cand
                df = df_cand
                improved = True
                break

            alpha *= 0.5

        if not improved:
            break  # Go to fallback

    # Fallback: a few bisection iterations around current logH
    # Try to bracket within +/- 2 decades
    lo = logH - 2.0
    hi = logH + 2.0
    if lo < clamp_lo: lo = clamp_lo
    if hi > clamp_hi: hi = clamp_hi

    f_lo = residual_logH(lo, cNa, cCl, Kw, Ka_G1, Ka_G2, Ka_T, cG, cT)
    f_hi = residual_logH(hi, cNa, cCl, Kw, Ka_G1, Ka_G2, Ka_T, cG, cT)

    # If not bracketed, expand once
    if f_lo * f_hi > 0.0:
        lo = clamp_lo
        hi = clamp_hi
        f_lo = residual_logH(lo, cNa, cCl, Kw, Ka_G1, Ka_G2, Ka_T, cG, cT)
        f_hi = residual_logH(hi, cNa, cCl, Kw, Ka_G1, Ka_G2, Ka_T, cG, cT)
        if f_lo * f_hi > 0.0:
            return np.nan

    for _ in range(30):
        mid = 0.5 * (lo + hi)
        f_mid = residual_logH(mid, cNa, cCl, Kw, Ka_G1, Ka_G2, Ka_T, cG, cT)
        if abs(f_mid) < tol_f:
            return 10.0 ** mid
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

    return 10.0**(0.5 * (lo + hi))


# Basic Equations =============================================================
@njit
def get_Ka(IS, A=0.5114):
    # Ensure IS positive and reasonably large
    if IS < 1e-6:
        IS = 1e-6

    sqrtIS = math.sqrt(IS)

    # Glycine
    Ka_G1 = 10.0 ** (-2.34)
    Ka_G2 = 10.0 ** -(9.60 + 2 * (0 - 1) * (A * sqrtIS / (1 + sqrtIS) - 0.1 * IS))

    # Tris (not dependent on IS)
    Ka_T = 10.0 ** (-8.07)

    return Ka_G1, Ka_G2, Ka_T


@njit
def get_IS(cH, cNa, cCl, Kw, Ka_G1, Ka_G2, Ka_T, cG, cT):
    # Water
    cOH = Kw / cH

    # Calculate ion concentrations
    cTH, cGH, cG1 = ion_concs(cH, cT, cG, Ka_G1, Ka_G2, Ka_T)

    # Ionic strength (0.5 * sum c_i z_i^2)
    IS = 0.5 * (cCl + cG1 + cOH + cGH + cTH + cNa + cH)

    return IS


@njit
def corrected_Kw(IS):
    A = 0.5114  # at ~25 C
    b = 0.1

    sqrtIS = math.sqrt(IS)

    # Davies equation for monovalent ions
    log10_gamma = -A * sqrtIS / (1 + sqrtIS) - b * IS

    pKw_cor = 14.0 + 2 * log10_gamma
    Kw_cor = 10 ** (-pKw_cor)

    return Kw_cor


@njit
def ion_concs(cH, cT_tot, cG_tot, Ka_G1, Ka_G2, Ka_T):
    # Calculate TrisH+ ion concentration
    cTH = cT_tot * cH / (cH + Ka_T)

    # Calculate glycine ion concentrations
    D = cH**2 + Ka_G1 * cH + Ka_G1 * Ka_G2
    cGH = cG_tot * cH**2 / D
    cG1 = Ka_G1 * Ka_G2 * cG_tot / D

    return cTH, cGH, cG1

# Isotherms ===================================================================

@njit
def sCEX_isotherm(cH, cNa, cTH, cGH, sCEX_ligand_density):
    if sCEX_ligand_density == 0:
        return 0, 0, 0

    # Ensure all concentrations positive
    cH = max(cH, 1e-14)
    cNa = max(cNa, 1e-14)
    cTH = max(cTH, 1e-14)
    cGH = max(cGH, 1e-14)

    denom = cH + cNa + cTH + cGH

    # Donnan potential relationship
    qNa = sCEX_ligand_density * cNa / denom
    qTH = sCEX_ligand_density * cTH / denom
    qGH = sCEX_ligand_density * cGH / denom

    return qNa, qTH, qGH


@njit
def wCEX_isotherm(cH, cNa, cTH, cGH, wCEX_ligand_density, K_acid,):
    if wCEX_ligand_density == 0:
        return 0, 0, 0

    # Ensure all concentrations positive
    cH = max(cH, 1e-14)
    cNa = max(cNa, 1e-14)
    cTH = max(cTH, 1e-14)
    cGH = max(cGH, 1e-14)

    cCat = cH + cNa + cTH + cGH

    ratio = K_acid * cCat / cH
    qCat = 0.5 * (-ratio + math.sqrt(ratio**2 + 4 * wCEX_ligand_density * ratio))

    # Donnan potential relationship
    qNa = cNa * qCat / cCat
    qTH = cTH * qCat / cCat
    qGH = cGH * qCat / cCat

    return qNa, qTH, qGH


@njit
def sAEX_isotherm(cH, cCl, cG1, sAEX_ligand_density, Kw):
    if sAEX_ligand_density == 0:
        return 0, 0

    # Ensure all concentrations positive
    cOH = max(Kw / cH, 1e-14) # cOH = Kw / cH
    cCl = max(cCl, 1e-14)
    cG1 = max(cG1, 1e-14)

    denom = cOH + cCl + cG1

    # Donnan potential relationship
    qCl = sAEX_ligand_density * cCl / denom
    qG1 = sAEX_ligand_density * cG1 / denom

    return qCl, qG1


@njit
def wAEX_isotherm(cH, cCl, cG1, wAEX_ligand_density, Kw, K_base):
    if wAEX_ligand_density == 0:
        return 0, 0

    # Ensure all concentrations positive
    cOH = max(Kw / cH, 1e-14) # cOH = Kw / cH
    cCl = max(cCl, 1e-14)
    cG1 = max(cG1, 1e-14)

    cAnion = cOH + cCl + cG1

    ratio = K_base * cAnion / cOH
    qAnion = 0.5 * (-ratio + math.sqrt(ratio**2 + 4 * wAEX_ligand_density * ratio))

    # Donnan potential relationship
    qCl = cCl * qAnion / cAnion
    qG1 = cG1 * qAnion / cAnion

    return qCl, qG1


# =============================================================================

@njit
def equilibrate(cNa, cCl, cG_tot, cT_tot):
    IS = max(cNa, cCl)
    DeltaIS = 1e3
    tol = 1e-4

    # Iterate over IS and cH to get final IS and cH values
    while DeltaIS > tol:
        # Get cH from electroneutrality
        Kw = corrected_Kw(IS)
        Ka_G1, Ka_G2, Ka_T = get_Ka(IS)
        cH = solve_cH_newton(cNa, cCl, Kw, Ka_G1, Ka_G2, Ka_T, cG_tot, cT_tot)
        IS_new = get_IS(cH, cNa, cCl,  Kw, Ka_G1, Ka_G2, Ka_T, cG_tot, cT_tot)

        # Convergence based change in IS
        DeltaIS = abs(IS_new - IS)

        # Update ionic strength
        IS = IS_new

    # Get final cH from electroneutrality
    Kw = corrected_Kw(IS)
    Ka_G1, Ka_G2, Ka_T = get_Ka(IS)
    cH = solve_cH_newton(cNa, cCl, Kw, Ka_G1, Ka_G2, Ka_T, cG_tot, cT_tot)

    return IS, Kw, Ka_G1, Ka_G2, Ka_T, cH


def activity_coeff(IS, A=0.5114, b=0.1, z=1):
    g = 10 ** (-((z ** 2) * (A * np.sqrt(IS) / (1 + np.sqrt(IS)) - b * IS)))
    return g

# Main model runner function ==================================================
def resin_titration(ms):
    N = ms['N']

    # Inlet concentration profiles generated by the upstream mixing model
    # Concentrations are converted from mM to mol/L
    data = np.loadtxt(ms['inlet_data_path'], delimiter=",", skiprows=1)
    time = data[:, 0] / 60 # Time is converted from seconds to minutes
    cH = data[:, 3] * 1e-3
    cNa = data[:, 5] * 1e-3
    cCl = data[:, 6] * 1e-3
    cTrisH = data[:, 7] * 1e-3
    cTris0 = data[:, 8] * 1e-3
    cGH = data[:, 9] * 1e-3
    cG0 = data[:, 10] * 1e-3
    cG1 = data[:, 11] * 1e-3
    cT_tot = cTris0 + cTrisH
    cG_tot = cGH + cG0 + cG1

    ms['inlet_time'] = time
    ms['cH'] = cH
    ms['cNa'] = cNa
    ms['cCl'] = cCl
    ms['cT_total'] = cT_tot
    ms['cG_total'] = cG_tot

    # Get equilibrated starting values
    IS0, Kw0, Ka_G10, Ka_G20, Ka_T0, cH0, = equilibrate(cNa[0], cCl[0], cG_tot[0], cT_tot[0])

    # Calculate ion concentrations
    cTH0, cGH0, cG10 = ion_concs(cH0, cT_tot[0], cG_tot[0], Ka_G10, Ka_G20, Ka_T0)

    # Solid phase (Adsorbed) concentrations at inlet
    qNa0_s, qTH0_s, qGH0_s = sCEX_isotherm(cH0, cNa[0], cTH0, cGH0, ms['sCEX_ligand_density'])
    qNa0_w, qTH0_w, qGH0_w = wCEX_isotherm(cH0, cNa[0], cTH0, cGH0, ms['wCEX_ligand_density'], ms['K_acid'])
    qCl0_s, qG10_s = sAEX_isotherm(cH0, cCl[0], cG10, ms['sAEX_ligand_density'], Kw0)
    qCl0_w, qG10_w = wAEX_isotherm(cH0, cCl[0], cG10, ms['wAEX_ligand_density'], Kw0, ms['K_base'])

    qNa0 = qNa0_s + qNa0_w
    qTH0 = qTH0_s + qTH0_w
    qCl0 = qCl0_s + qCl0_w
    qGH0 = qGH0_s + qGH0_w
    qG10 = qG10_s + qG10_w

    # Apply initial conditions to the axial cells; concentrations at t=0.
    cT0 = np.ones(N + 1) * cT_tot[0]
    cG0 = np.ones(N + 1) * cG_tot[0]
    cCl0 = np.ones(N + 1) * cCl[0]
    cNa0 = np.ones(N + 1) * cNa[0]
    qNa0 = np.ones(N + 1) * qNa0
    qTH0 = np.ones(N + 1) * qTH0
    qCl0 = np.ones(N + 1) * qCl0
    qGH0 = np.ones(N + 1) * qGH0
    qG10 = np.ones(N + 1) * qG10

    y0 = np.concatenate([cT0, cG0, cCl0, cNa0, qNa0, qTH0, qCl0, qGH0, qG10])

    # -------------------------------------------------------------------------
    # Solving the model
    sol = solve_ivp(dsys_wrapper, t_span=(0, time[-2]), t_eval=time[:-1], y0=y0,
                    args=(ms,), rtol=1e-6, atol=1e-9, method='BDF')

    print(f'\nSolver success={sol.success}, Status={sol.status}, Message={sol.message}')

    # -------------------------------------------------------------------------
    # Get time and solution vectors from solver
    t = sol.t
    y = sol.y.T

    # Parse column profiles
    cT_tot = y[:, 0*N:1*N+1]  # Liquid phase Tris
    cG_tot = y[:, 1*N+1:2*N+2]  # Liquid phase Glycine
    cCl = y[:, 2*N+2:3*N+3]  # Liquid phase Cl
    cNa = y[:, 3*N+3:4*N+4]  # Liquid phase Na
    qNa = y[:, 4*N+4:5*N+5]  # Solid phase Na
    qTH = y[:, 5*N+5:6*N+6]  # Solid phase TH
    qCl = y[:, 6*N+6:7*N+7]  # Solid phase Cl
    qGH = y[:, 7*N+7:8*N+8]  # Solid phase GH
    qG1 = y[:, 8*N+8:9*N+9]  # Solid phase Gion

    # Get values at column outlet
    cell = -1
    cNa_out = cNa[:, cell]
    cCl_out = cCl[:, cell]
    cG_tot_out = cG_tot[:, cell]
    cT_tot_out = cT_tot[:, cell]
    qNa_out = qNa[:, cell]
    qTH_out = qTH[:, cell]
    qCl_out = qCl[:, cell]
    qGH_out = qGH[:, cell]
    qG1_out = qG1[:, cell]

    # Get final equilibrated outlet profiles
    IS_out = np.zeros_like(cNa_out)
    Kw_out = np.zeros_like(cNa_out)
    Ka_G1_out = np.zeros_like(cNa_out)
    Ka_G2_out = np.zeros_like(cNa_out)
    Ka_T_out = np.zeros_like(cNa_out)
    cH_out = np.zeros_like(cNa_out)

    for i in range(len(cNa_out)):
        IS_out[i], Kw_out[i], Ka_G1_out[i], Ka_G2_out[i], Ka_T_out[i], cH_out[i] = equilibrate(cNa_out[i], cCl_out[i], cG_tot_out[i], cT_tot_out[i])

    # Calculate ion concentrations
    cOH_out = Kw_out / cH_out
    cTH_out, cGH_out, cG1_out = ion_concs(cH_out, cT_tot_out, cG_tot_out, Ka_G1_out, Ka_G2_out, Ka_T_out)

    g = activity_coeff(IS_out)
    pH_out = np.real(-np.log10(g*cH_out))

    # Assign outlet profiles
    ms['cCl_out'] = cCl_out * 1e3
    ms['cNa_out'] = cNa_out * 1e3
    ms['qNa_out'] = qNa_out * 1e3
    ms['qTH_out'] = qTH_out * 1e3
    ms['qCl_out'] = qCl_out * 1e3
    ms['qGH_out'] = qGH_out * 1e3
    ms['qG1_out'] = qG1_out * 1e3
    ms['cTH_out'] = cTH_out * 1e3
    ms['cT0_out'] = (cT_tot_out - cTH_out) * 1e3
    ms['cH_out'] = cH_out * 1e3
    ms['cOH_out'] = cOH_out * 1e3
    ms['cGH_out'] = cGH_out * 1e3
    ms['cG1_out'] = cG1_out * 1e3
    ms['cG0_out'] = (cG_tot_out - cGH_out - cG1_out) * 1e3

    ms['pH_out'] = pH_out
    ms['CV_out'] = t * ms['F'] / ms['col_vol']
    ms['IS_out'] = IS_out * 1e3

    ms['column_outlet_df'] = pd.DataFrame({
        'CV': ms['CV_out'],
        'IS': ms['IS_out'],
        'pH': ms['pH_out'],
        'H+': ms['cH_out'],
        'OH-': ms['cOH_out'],
        'Na+': ms['cNa_out'],
        'Cl-': ms['cCl_out'],
        'TrisH+': ms['cTH_out'],
        'Tris0': ms['cT0_out'],
        'GlycineH+': ms['cGH_out'],
        'Glycine0': ms['cG0_out'],
        'Glycine-': ms['cG1_out'],
        'qNa': ms['qNa_out'],
        'qTrisH': ms['qTH_out'],
        'qCl': ms['qCl_out'],
        'qGlycineH+': ms['qGH_out'],
        'qGlycine-': ms['qG1_out'],
    })

    return ms


# Handle outputs ==============================================================
def make_results_folder():
    folder = 'results_' + time.strftime("%Y%m%d%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder_path = os.path.join(script_dir, "output")
    new_folder_path = os.path.join(output_folder_path, folder)
    # Check if the "output" folder exists; if not, create it
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)
    os.mkdir(new_folder_path)
    return new_folder_path


def make_csv_column(ms):
    # Define the file path with .csv extension
    path = os.path.join(ms['folder'], "column_out.csv")

    # Write the DataFrame to a CSV file
    df = ms['column_outlet_df']
    df.to_csv(path, index=False)

    print(f"\n CSV saved to {path}")


def generate_plot(ms):
    font_size = 14
    end_CV = 10

    RPI_cherry = (0.816, 0, 0.086)
    Na = (0/255, 150/255, 255/255) # Bright Blue
    Cl = (255/255, 117/255, 24/255) # Pumpkin Orange
    TH = (0/255, 128/255, 128/255) # Teal
    T0 = (150/255, 222/255, 209/255) # Robin Egg Blue
    GH = (128/255, 0/255, 128/255) # Purple
    G0 = (191/255, 64/255, 191/255) # Bright Purple
    G1 = (207/255, 159/255, 255/255) # Light Violet

    CV = ms['CV_out'] - 2.178 # Subtract 2.178 CV to have step start at 0
    pH = ms['pH_out']
    ionic_strength = ms['IS_out']

    # Make plots in 1x4 layout
    fig_combined, (axA, axB, axC, axD) = plt.subplots(1, 4, figsize=(14, 4), dpi=300)

    axA.plot(CV, pH, color=RPI_cherry, linewidth=2)
    axA.set_title("pH", fontsize=font_size)
    axA.set_xlabel("Column Volumes", fontsize=font_size)
    axA.set_xlim([0, end_CV])
    axA.set_ylim([3, 8.2])

    axB.plot(CV, ionic_strength, color=RPI_cherry, linewidth=2)
    axB.set_title("Ionic Strength (mM)", fontsize=font_size)
    axB.set_xlabel("Column Volumes", fontsize=font_size)
    axB.set_xlim([0, end_CV])
    axB.set_ylim([0, ionic_strength.max() * 1.1])

    axC.plot(CV, ms['cNa_out'], '-', linewidth=2, label='Na+', color=Na)
    axC.plot(CV, ms['cCl_out'], '-', linewidth=2, label='Cl-', color=Cl)
    axC.plot(CV, ms['cTH_out'], '-', linewidth=2, label='TH+', color=TH)
    axC.plot(CV, ms['cT0_out'], '-', linewidth=2, label='T0', color=T0)
    axC.plot(CV, ms['cGH_out'], '-', linewidth=2, label='GH+', color=GH)
    axC.plot(CV, ms['cG0_out'], '-', linewidth=2, label='G0', color=G0)
    axC.plot(CV, ms['cG1_out'], '-', linewidth=2, label='G1-', color=G1)
    axC.set_title("Concentrations (mM)", fontsize=font_size)
    axC.set_xlabel("Column Volumes", fontsize=font_size)
    axC.set_xlim([0, end_CV])
    axC.set_ylim([0, None])

    axD.plot(CV, ms['qNa_out'], '-', linewidth=2, label='Na+', color=Na)
    axD.plot(CV, ms['qCl_out'], '-', linewidth=2, label='Cl-', color=Cl)
    axD.plot(CV, ms['qTH_out'], '-', linewidth=2, label='TH+', color=TH)
    axD.plot(CV, ms['qGH_out'], '-', linewidth=2, label='GH+', color=GH)
    axD.plot(CV, ms['qG1_out'], '-', linewidth=2, label='G1-', color=G1)
    axD.set_title("Q (mM packed bed)", fontsize=font_size)
    axD.set_xlabel("Column Volumes", fontsize=font_size)
    axD.set_xlim([0, end_CV])
    axD.set_ylim([0, None])

    sCEX = ms['sCEX_ligand_density']
    wCEX = ms['wCEX_ligand_density']
    sAEX = ms['sAEX_ligand_density']
    wAEX = ms['wAEX_ligand_density']
    fig_combined.suptitle(f'sCEX={sCEX}, wCEX={wCEX}, sAEX={sAEX}, wAEX={wAEX}', fontsize=font_size)

    plt.tight_layout()
    path = os.path.join(ms['folder'], 'column_outlet_plot.png')
    fig_combined.savefig(path, bbox_inches="tight")
    save_legend(axC, ms['folder'], "c_legend.png")
    save_legend(axD, ms['folder'], "q_legend.png")
    plt.close('all')

    print(f"\n Plot saved to {path}")


def save_legend(ax, path, filename):

    save_path = os.path.join(path, filename)

    fig_legend = plt.figure(figsize=(1, 1))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis("off")

    ax_legend.legend(
        *ax.get_legend_handles_labels(),
        loc="center",
        frameon=False
    )

    fig_legend.savefig(
        save_path,
        bbox_inches="tight",
        transparent=True,
        dpi=300,
    )

    plt.close(fig_legend)

# =============================================================================
if __name__ == '__main__':
    filenames = [
        '41_50t_50g.csv',
        '42_50t_150g.csv',
        ]

    for file in filenames:
        ms = inputs(file)

        # Create the output folder
        ms['folder'] = make_results_folder()

        # Save a copy of this script into the results folder
        current_script = os.path.abspath(__file__)
        shutil.copy2(current_script, ms['folder'])

        # Run resin titration simulation
        tic = time.time()
        ms = resin_titration(ms)
        toc = time.time()

        # Make plots
        generate_plot(ms)

        # Make csv output file
        make_csv_column(ms)

        elapsed_time = toc - tic
        print(f'\n Simulation run time: {elapsed_time:.2f} seconds')
