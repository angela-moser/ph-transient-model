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
        t, y, time, cNa0, cCl0, cP0, cA0, N, inv_dx, int_vel,
        solid_phase, sCEX_ligand_density, wCEX_ligand_density,
        sAEX_ligand_density, wAEX_ligand_density, K_acid, K_base
    ):

    # Unpack y vector
    cP_tot = y[0*N:1*N+1]
    cA_tot = y[1*N+1:2*N+2]
    cCl = y[2*N+2:3*N+3]
    cNa = y[3*N+3:4*N+4]
    qNa = y[4*N+4:5*N+5]
    qCl = y[5*N+5:6*N+6]
    qP1 = y[6*N+6:7*N+7]
    qP2 = y[7*N+7:8*N+8]
    qP3 = y[8*N+8:9*N+9]
    qA1 = y[9*N+9:10*N+10]

    # Zeros dot vectors
    dydt = np.zeros(10*N+10)
    dcPdt = np.zeros(N+1)
    dcAdt = np.zeros(N+1)
    dcCldt = np.zeros(N+1)
    dcNadt = np.zeros(N+1)
    dqNadt = np.zeros(N+1)
    dqCldt = np.zeros(N+1)
    dqP1dt = np.zeros(N+1)
    dqP2dt = np.zeros(N+1)
    dqP3dt = np.zeros(N+1)
    dqA1dt = np.zeros(N+1)

    # Find the index i where the given time point falls
    j = np.searchsorted(time, t) - 1

    # Get the multiplier for interpolation
    mult = (t - time[j]) / (time[j+1] - time[j])

    # Create IS vector
    IS = np.zeros(N+1)

    # Interpolate the concentrations at the inlet for the given time point
    cNa_in = cNa0[j] + (cNa0[j+1] - cNa0[j]) * mult
    cCl_in = cCl0[j] + (cCl0[j+1] - cCl0[j]) * mult
    cA_tot_in = cA0[j] + (cA0[j+1] - cA0[j]) * mult
    cP_tot_in = cP0[j] + (cP0[j+1] - cP0[j]) * mult

    # Precompute constants
    k_kin = 1e2
    conv = int_vel * inv_dx

    # Calculations for dqdt[0] ----------------------------------------------
    # Get equilibrated IS and cH
    IS[0], Kw_in, Ka_P1_in, Ka_P2_in, Ka_P3_in, Ka_A1_in, cH_in = equilibrate(cNa_in, cCl_in, cP_tot_in, cA_tot_in)

    # Calculate ion concentration
    cP1_in, cP2_in, cP3_in, cA1_in = ion_concs(cH_in, cP_tot_in, cA_tot_in, Ka_P1_in, Ka_P2_in, Ka_P3_in, Ka_A1_in)

    # Calculate bound concentrations
    qNa_eq_s = sCEX_isotherm(cH_in, cNa_in, sCEX_ligand_density)
    qNa_eq_w = wCEX_isotherm(cH_in, cNa_in, wCEX_ligand_density, K_acid)
    qCl_eq_s, qP1_eq_s, qP2_eq_s, qP3_eq_s, qA1_eq_s = sAEX_isotherm(cH_in, cCl_in, cP1_in, cP2_in, cP3_in, cA1_in, sAEX_ligand_density, Kw_in)
    qCl_eq_w, qP1_eq_w, qP2_eq_w, qP3_eq_w, qA1_eq_w = wAEX_isotherm(cH_in, cCl_in, cP1_in, cP2_in, cP3_in, cA1_in, wAEX_ligand_density, Kw_in, K_base)

    qNa_eq = qNa_eq_s + qNa_eq_w
    qCl_eq = qCl_eq_s + qCl_eq_w
    qP1_eq = qP1_eq_s + qP1_eq_w
    qP2_eq = qP2_eq_s + qP2_eq_w
    qP3_eq = qP3_eq_s + qP3_eq_w
    qA1_eq = qA1_eq_s + qA1_eq_w

    # Linear driving force for interacting ions
    dqNadt[0] = k_kin * (qNa_eq - qNa[0])
    dqCldt[0] = k_kin * (qCl_eq - qCl[0])
    dqP1dt[0] = k_kin * (qP1_eq - qP1[0])
    dqP2dt[0] = k_kin * (qP2_eq - qP2[0])
    dqP3dt[0] = k_kin * (qP3_eq - qP3[0])
    dqA1dt[0] = k_kin * (qA1_eq - qA1[0])

    #--------------------------------------------------------------------------
    # Inlet boundary condition using inlet concentrations
    dcNadt[0] = (cNa_in - cNa[0]) * conv - solid_phase * dqNadt[0]
    dcCldt[0] = (cCl_in - cCl[0]) * conv - solid_phase * dqCldt[0]
    dcPdt[0]  = (cP_tot_in - cP_tot[0]) * conv - solid_phase * (dqP1dt[0] + dqP2dt[0] + dqP3dt[0])
    dcAdt[0]  = (cA_tot_in - cA_tot[0]) * conv - solid_phase * dqA1dt[0]

    # -------------------------------------------------------------------------
    # Set up differential balances for all discretization cells
    for i in range(1, N+1):
        # Pull into local variables for speed
        cNa_i = cNa[i]
        cCl_i = cCl[i]
        cA_tot_i = cA_tot[i]
        cP_tot_i = cP_tot[i]
        qNa_i = qNa[i]
        qCl_i = qCl[i]
        qP1_i = qP1[i]
        qP2_i = qP2[i]
        qP3_i = qP3[i]
        qA1_i = qA1[i]

        # Get equilibrated IS and cH
        IS[i], Kw_i, Ka_P1_i, Ka_P2_i, Ka_P3_i, Ka_A1_i, cH_i = equilibrate(cNa_i, cCl_i, cP_tot_i, cA_tot_i)

        # Calculate ion concentrations
        cP1_i, cP2_i, cP3_i, cA1_i = ion_concs(cH_i, cP_tot_i, cA_tot_i, Ka_P1_i, Ka_P2_i, Ka_P3_i, Ka_A1_i)

        # Calculate bound concentrations
        qNa_eq_s = sCEX_isotherm(cH_i, cNa_i, sCEX_ligand_density)
        qNa_eq_w = wCEX_isotherm(cH_i, cNa_i, wCEX_ligand_density, K_acid)
        qCl_eq_s, qP1_eq_s, qP2_eq_s, qP3_eq_s, qA1_eq_s = sAEX_isotherm(cH_i, cCl_i, cP1_i, cP2_i, cP3_i, cA1_i, sAEX_ligand_density, Kw_i)
        qCl_eq_w, qP1_eq_w, qP2_eq_w, qP3_eq_w, qA1_eq_w = wAEX_isotherm(cH_i, cCl_i, cP1_i, cP2_i, cP3_i, cA1_i, wAEX_ligand_density, Kw_i, K_base)

        qNa_eq = qNa_eq_s + qNa_eq_w
        qCl_eq = qCl_eq_s + qCl_eq_w
        qP1_eq = qP1_eq_s + qP1_eq_w
        qP2_eq = qP2_eq_s + qP2_eq_w
        qP3_eq = qP3_eq_s + qP3_eq_w
        qA1_eq = qA1_eq_s + qA1_eq_w

        # Linear driving force for interacting ions
        dqNadt[i] = k_kin * (qNa_eq - qNa_i)
        dqCldt[i] = k_kin * (qCl_eq - qCl_i)
        dqP1dt[i] = k_kin * (qP1_eq - qP1_i)
        dqP2dt[i] = k_kin * (qP2_eq - qP2_i)
        dqP3dt[i] = k_kin * (qP3_eq - qP3_i)
        dqA1dt[i] = k_kin * (qA1_eq - qA1_i)

        # Transport of ions
        dcNadt[i] = -solid_phase * dqNadt[i] + (cNa[i-1] - cNa_i) * conv
        dcCldt[i] = -solid_phase * dqCldt[i] + (cCl[i-1] - cCl_i) * conv
        dcPdt[i] = -solid_phase * (dqP1dt[i] + dqP2dt[i] + dqP3dt[i]) + (cP_tot[i-1] - cP_tot_i) * conv
        dcAdt[i] = -solid_phase * dqA1dt[i] + (cA_tot[i-1] - cA_tot_i) * conv

    # -------------------------------------------------------------------------
    # Transfer back to ydot vector for solving
    dydt[0*N:1*N+1] = dcPdt
    dydt[1*N+1:2*N+2] = dcAdt
    dydt[2*N+2:3*N+3] = dcCldt
    dydt[3*N+3:4*N+4] = dcNadt
    dydt[4*N+4:5*N+5] = dqNadt
    dydt[5*N+5:6*N+6] = dqCldt
    dydt[6*N+6:7*N+7] = dqP1dt
    dydt[7*N+7:8*N+8] = dqP2dt
    dydt[8*N+8:9*N+9] = dqP3dt
    dydt[9*N+9:10*N+10] = dqA1dt

    return dydt


def dsys_wrapper(t, y, ms):
    return dsys(t, y,
                    ms['inlet_time'],
                    ms['cNa'],
                    ms['cCl'],
                    ms['cP_total'],
                    ms['cA_total'],
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
def residual_and_deriv_logH(logH, cNa, cCl, cP, cA, Kw, Ka_P1, Ka_P2, Ka_P3, Ka_A1):
    """
    Electroneutrality residual and derivative of electroneutrality residual
    """
    ln10 = 2.302585092994046
    cH = 10.0 ** logH

    # Calculate ion concentrations
    cOH = Kw / cH
    cP1, cP2, cP3, cA1 = ion_concs(cH, cP, cA, Ka_P1, Ka_P2, Ka_P3, Ka_A1)

    f = cCl + cP1 + 2*cP2 + 3*cP3 + cA1 + cOH - cNa - cH

    # Derivatives wrt cH
    dOH_dH = -Kw / (cH**2)
    dA1_dH  = -Ka_A1 * cA / (cH + Ka_A1)**2

    DP = cH**3 + Ka_P1 * cH**2 + Ka_P1 * Ka_P2 * cH + Ka_P1 * Ka_P2 * Ka_P3
    dDP_dH = cH**2 + Ka_P1 * cH + Ka_P1 * Ka_P2

    dP1_dH = (
        (2*cH * Ka_P1 * cP) * DP
        - (cH**2 * Ka_P1 * cP) * dDP_dH
    ) / (DP**2)

    dP2_dH = (
        (Ka_P1 * Ka_P2 * cP) * DP
        - (cH * Ka_P1 * Ka_P2 * cP) * dDP_dH
    ) / (DP**2)

    dP3_dH = (
        - (Ka_P1 * Ka_P2 * Ka_P3 * cP) * dDP_dH
    ) / (DP**2)

    df_dH = dOH_dH + dP1_dH + 2*dP2_dH + 3*dP3_dH + dA1_dH - 1.0

    # Chain rule: dH/dlogH = ln(10) * H
    df_dlogH = df_dH * (ln10 * cH)

    return f, df_dlogH


@njit
def residual_logH(logH, cNa, cCl, cP, cA, Kw, Ka_P1, Ka_P2, Ka_P3, Ka_A1):
    cH = 10.0 ** logH

    cOH = Kw / cH
    cP1, cP2, cP3, cA1 = ion_concs(cH, cP, cA, Ka_P1, Ka_P2, Ka_P3, Ka_A1)

    f = cCl + cP1 + 2*cP2 + 3*cP3 + cA1 + cOH - cNa - cH
    return f


@njit
def solve_cH_newton(cNa, cCl, cP, cA, Kw, Ka_P1, Ka_P2, Ka_P3, Ka_A1,
                      logH0=-8.0,
                      tol_f=1e-12,
                      tol_step=1e-10,
                      maxiter=20,
                      clamp_lo=-8.0,
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

    f, df = residual_and_deriv_logH(logH, cNa, cCl, cP, cA, Kw, Ka_P1, Ka_P2, Ka_P3, Ka_A1)

    for _ in range(maxiter):
        # Converged by residual
        if abs(f) < tol_f:
            return 10 ** logH

        # Bad derivative -> break to fallback
        if df == 0.0 or not np.isfinite(df) or not np.isfinite(f):
            break

        step = -f / df

        # Converged by step size
        if abs(step) < tol_step:
            return 10 ** logH

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

            f_cand, df_cand = residual_and_deriv_logH(cand, cNa, cCl, cP, cA, Kw, Ka_P1, Ka_P2, Ka_P3, Ka_A1)

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

    f_lo = residual_logH(lo, cNa, cCl, cP, cA, Kw, Ka_P1, Ka_P2, Ka_P3, Ka_A1)
    f_hi = residual_logH(hi, cNa, cCl, cP, cA, Kw, Ka_P1, Ka_P2, Ka_P3, Ka_A1)

    # If not bracketed, expand once
    if f_lo * f_hi > 0.0:
        lo = clamp_lo
        hi = clamp_hi
        f_lo = residual_logH(lo, cNa, cCl, cP, cA, Kw, Ka_P1, Ka_P2, Ka_P3, Ka_A1)
        f_hi = residual_logH(hi, cNa, cCl, cP, cA, Kw, Ka_P1, Ka_P2, Ka_P3, Ka_A1)
        if f_lo * f_hi > 0.0:
            return np.nan

    for _ in range(30):
        mid = 0.5 * (lo + hi)
        f_mid = residual_logH(mid, cNa, cCl, cP, cA, Kw, Ka_P1, Ka_P2, Ka_P3, Ka_A1)
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

    # Phosphate
    Ka_P1 = 10.0 ** -(2.15 + 2 * (0 - 1) * (A * sqrtIS / (1 + sqrtIS) - 0.1 * IS))
    Ka_P2 = 10.0 ** -(7.21 + 2 * (-1 - 1) * (A * sqrtIS / (1 + sqrtIS) - 0.07 * IS))
    Ka_P3 = 10.0 ** -(12.33 + 2 * (-2 - 1) * (A * sqrtIS / (1 + sqrtIS) - 0.1 * IS))

    # Acetate
    Ka_A1 = 10.0 ** -(4.76 + 2 * (0 - 1) * (A * sqrtIS / (1 + sqrtIS) - 0.16 * IS))

    return Ka_P1, Ka_P2, Ka_P3, Ka_A1


@njit
def get_IS(cH, cNa, cCl, cP, cA, Kw, Ka_P1, Ka_P2, Ka_P3, Ka_A1):
    # Water
    cOH = Kw / cH

    # Calculate ion concentrations
    cP1, cP2, cP3, cA1 = ion_concs(cH, cP, cA, Ka_P1, Ka_P2, Ka_P3, Ka_A1)

    # Ionic strength (0.5 * sum c_i z_i^2)
    IS = 0.5 * (cCl + cA1 + cOH + cP1 + 4*cP2 + 9*cP3 + cNa + cH)

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
def ion_concs(cH, cP_tot, cA_tot, Ka_P1, Ka_P2, Ka_P3, Ka_A1):
    # Calculate phosphate ion concentrations
    DP = cH**3 + Ka_P1 * cH**2 + Ka_P1 * Ka_P2 * cH + Ka_P1 * Ka_P2 * Ka_P3
    cP1 = cH**2 * Ka_P1 * cP_tot / DP
    cP2 = cH * Ka_P1 * Ka_P2 * cP_tot / DP
    cP3 = Ka_P1 * Ka_P2 * Ka_P3 * cP_tot / DP

    # Calculate acetate ion concentrations
    DA = cH + Ka_A1
    cA1 = Ka_A1 * cA_tot / DA

    return cP1, cP2, cP3, cA1

# Isotherms ===================================================================

@njit
def sCEX_isotherm(cH, cNa, sCEX_ligand_density):
    # Explicit form, all cations are monovalent
    if sCEX_ligand_density == 0:
        return 0

    # Ensure all concentrations positive
    cH = max(cH, 1e-14)
    cNa = max(cNa, 1e-14)

    denom = cH + cNa

    qNa = sCEX_ligand_density * cNa / denom

    return qNa


@njit
def wCEX_isotherm(cH, cNa, wCEX_ligand_density, K_acid,):
    # Explicit form, all cations are monovalent
    if wCEX_ligand_density == 0:
        return 0

    # Ensure all concentrations positive
    cH = max(cH, 1e-14)
    cNa = max(cNa, 1e-14)

    cCat = cH + cNa

    ratio = K_acid * cCat / cH
    qCat = 0.5 * (-ratio + math.sqrt(ratio**2 + 4 * wCEX_ligand_density * ratio))

    qNa = cNa * qCat / cCat

    return qNa


@njit
def sAEX_isotherm(cH, cCl, cP1, cP2, cP3, cA, sAEX_ligand_density, Kw):
    # Implicit form, not all anions are monovalent
    if sAEX_ligand_density == 0:
        return 0, 0, 0, 0, 0

    # Ensure all concentrations positive
    cOH = max(Kw / cH, 1e-14)
    cCl = max(cCl, 1e-14)
    cP1 = max(cP1, 1e-14)
    cP2 = max(cP2, 1e-14)
    cP3 = max(cP3, 1e-14)
    cA = max(cA, 1e-14)

    L = sAEX_ligand_density

    a1 = cOH + cCl + cP1 + cA
    a2 = 2.0 * cP2
    a3 = 3.0 * cP3

    # Initial guess
    S = 0.5 * L

    for _ in range(20):
        # theta
        theta = (L - S) / cOH

        # Function f(S)
        f = a1 * theta + a2 * theta**2 + a3 * theta**3 - S

        # dtheta/dS
        dtheta = - 1 / cOH

        # Derivative df/dS (chain rule)
        df = dtheta * (a1 + 2.0 * a2 * theta + 3.0 * a3 * theta**2) - 1.0

        S_new = S - f / df

        if S_new < 0.0:
            S_new = 0.0
        elif S_new > L:
            S_new = L

        if abs(S_new - S) < 1e-12:
            S = S_new
            break

        S = S_new

    theta = (L - S) / cOH

    qCl = cCl * theta
    qP1 = cP1 * theta
    qP2 = cP2 * theta**2
    qP3 = cP3 * theta**3
    qA = cA * theta

    return qCl, qP1, qP2, qP3, qA


@njit
def wAEX_isotherm(cH, cCl, cP1, cP2, cP3, cA1, wAEX_ligand_density, Kw, K_base):
    # Implicit form, not all anions are monovalent
    if wAEX_ligand_density == 0:
        return 0, 0, 0, 0, 0

    # Ensure all concentrations positive
    cOH = max(Kw / cH, 1e-14)
    cCl = max(cCl, 1e-14)
    cP1 = max(cP1, 1e-14)
    cP2 = max(cP2, 1e-14)
    cP3 = max(cP3, 1e-14)
    cA1 = max(cA1, 1e-14)

    L = wAEX_ligand_density

    a1 = cOH + cCl + cP1 + cA1
    a2 = 2.0 * cP2
    a3 = 3.0 * cP3

    # Initial guess
    S = 0.5 * L

    for _ in range(20):
        # theta
        theta = (K_base * (L - S)) / (cOH * S)

        # Function f(S)
        f = a1 * theta + a2 * theta**2 + a3 * theta**3 - S

        # dtheta/dS
        dtheta = - K_base * L / (cOH * S**2)

        # Derivative df/dS (chain rule)
        df = dtheta * (a1 + 2.0 * a2 * theta + 3.0 * a3 * theta**2) - 1.0

        S_new = S - f / df

        if S_new < 0.0:
            S_new = 0.0
        elif S_new > L:
            S_new = L

        if abs(S_new - S) < 1e-12:
            S = S_new
            break

        S = S_new

    theta = (K_base * (L - S)) / (cOH * S)

    qCl = cCl * theta
    qP1 = cP1 * theta
    qP2 = cP2 * theta**2
    qP3 = cP3 * theta**3
    qA1 = cA1 * theta

    return qCl, qP1, qP2, qP3, qA1


# =============================================================================

@njit
def equilibrate(cNa, cCl, cP_tot, cA_tot):
    IS = max(cNa, cCl)
    DeltaIS = 1e3
    tol = 1e-4

    # Iterate over IS and cH to get final IS and cH values
    while DeltaIS > tol:
        # Get cH from electroneutrality
        Kw = corrected_Kw(IS)
        Ka_P1, Ka_P2, Ka_P3, Ka_A1 = get_Ka(IS)
        cH = solve_cH_newton(cNa, cCl, cP_tot, cA_tot, Kw, Ka_P1, Ka_P2, Ka_P3, Ka_A1)
        IS_new = get_IS(cH, cNa, cCl, cP_tot, cA_tot, Kw, Ka_P1, Ka_P2, Ka_P3, Ka_A1)

        # Convergence based change in IS
        DeltaIS = abs(IS_new - IS)

        # Update ionic strength
        IS = IS_new

    # Get final cH from electroneutrality
    Kw = corrected_Kw(IS)
    Ka_P1, Ka_P2, Ka_P3, Ka_A1 = get_Ka(IS)
    cH = solve_cH_newton(cNa, cCl, cP_tot, cA_tot, Kw, Ka_P1, Ka_P2, Ka_P3, Ka_A1)

    return IS, Kw, Ka_P1, Ka_P2, Ka_P3, Ka_A1, cH


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
    cP0 = data[:, 7] * 1e-3
    cP1 = data[:, 8] * 1e-3
    cP2 = data[:, 9] * 1e-3
    cP3 = data[:, 10] * 1e-3
    cA0 = data[:, 11] * 1e-3
    cA1 = data[:, 12] * 1e-3
    cP_tot = cP0 + cP1 + cP2 + cP3
    cA_tot = cA0 + cA1

    ms['inlet_time'] = time
    ms['cH'] = cH
    ms['cNa'] = cNa
    ms['cCl'] = cCl
    ms['cP_total'] = cP_tot
    ms['cA_total'] = cA_tot

    # Get equilibrated starting values
    IS0, Kw0, Ka_P10, Ka_P20, Ka_P30, Ka_A10, cH0 = equilibrate(cNa[0], cCl[0], cP_tot[0], cA_tot[0])

    # Calculate ion concentrations
    cP10, cP20, cP30, cA10 = ion_concs(cH0, cP_tot[0], cA_tot[0], Ka_P10, Ka_P20, Ka_P30, Ka_A10)

    # Solid phase (Adsorbed) conccentrations at inlet
    qNa0_s = sCEX_isotherm(cH0, cNa[0], ms['sCEX_ligand_density'])
    qNa0_w = wCEX_isotherm(cH0, cNa[0], ms['wCEX_ligand_density'], ms['K_acid'])
    qCl0_s, qP10_s, qP20_s, qP30_s, qA10_s = sAEX_isotherm(cH0, cCl[0], cP10, cP20, cP30, cA10, ms['sAEX_ligand_density'], Kw0)
    qCl0_w, qP10_w, qP20_w, qP30_w, qA10_w = wAEX_isotherm(cH0, cCl[0], cP10, cP20, cP30, cA10, ms['wAEX_ligand_density'], Kw0, ms['K_base'])

    qNa0 = qNa0_s + qNa0_w
    qCl0 = qCl0_s + qCl0_w
    qP10 = qP10_s + qP10_w
    qP20 = qP20_s + qP20_w
    qP30 = qP30_s + qP30_w
    qA10 = qA10_s + qA10_w

    # Apply initial conditions to the axial cells; concentrations at t=0.
    cP_tot_0 = np.ones(N + 1) * cP_tot[0]
    cA_tot_0 = np.ones(N + 1) * cA_tot[0]
    cCl0 = np.ones(N + 1) * cCl[0]
    cNa0 = np.ones(N + 1) * cNa[0]
    qNa0 = np.ones(N + 1) * qNa0
    qCl0 = np.ones(N + 1) * qCl0
    qP10 = np.ones(N + 1) * qP10
    qP20 = np.ones(N + 1) * qP20
    qP30 = np.ones(N + 1) * qP30
    qA10 = np.ones(N + 1) * qA10

    y0 = np.concatenate([cP_tot_0, cA_tot_0, cCl0, cNa0, qNa0, qCl0, qP10, qP20, qP30, qA10])

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
    cP_tot = y[:, 0*N:1*N+1]  # Liquid phase Phosphate
    cA_tot = y[:, 1*N+1:2*N+2]  # Liquid phase Acetate
    cCl = y[:, 2*N+2:3*N+3]  # Liquid phase Cl
    cNa = y[:, 3*N+3:4*N+4]  # Liquid phase Na
    qNa = y[:, 4*N+4:5*N+5]  # Solid phase Na
    qCl = y[:, 5*N+5:6*N+6]  # Solid phase Cl
    qP1 = y[:, 6*N+6:7*N+7]  # Solid phase P1-
    qP2 = y[:, 7*N+7:8*N+8]  # Solid phase P2-
    qP3 = y[:, 8*N+8:9*N+9]  # Solid phase P3-
    qA1 = y[:, 9*N+9:10*N+10]  # Solid phase A-

    # Get values at column outlet
    cell = -1
    cNa_out = cNa[:, cell]
    cCl_out = cCl[:, cell]
    cA_tot_out = cA_tot[:, cell]
    cP_tot_out = cP_tot[:, cell]
    qNa_out = qNa[:, cell]
    qCl_out = qCl[:, cell]
    qP1_out = qP1[:, cell]
    qP2_out = qP2[:, cell]
    qP3_out = qP3[:, cell]
    qA1_out = qA1[:, cell]

    # Get final equilibrated outlet profiles
    IS_out = np.zeros_like(cNa_out)
    Kw_out = np.zeros_like(cNa_out)
    Ka_P1_out = np.zeros_like(cNa_out)
    Ka_P2_out = np.zeros_like(cNa_out)
    Ka_P3_out = np.zeros_like(cNa_out)
    Ka_A1_out = np.zeros_like(cNa_out)
    cH_out = np.zeros_like(cNa_out)

    for i in range(len(cNa_out)):
        IS_out[i], Kw_out[i], Ka_P1_out[i], Ka_P2_out[i], Ka_P3_out[i], Ka_A1_out[i], cH_out[i], = equilibrate(cNa_out[i], cCl_out[i], cP_tot_out[i], cA_tot_out[i])

    # Calculate ion concentrations
    cOH_out = Kw_out / cH_out
    cP1_out, cP2_out, cP3_out, cA1_out = ion_concs(cH_out, cP_tot_out, cA_tot_out, Ka_P1_out, Ka_P2_out, Ka_P3_out, Ka_A1_out)

    g = activity_coeff(IS_out)
    pH_out = np.real(-np.log10(g*cH_out))

    # Assign outlet profiles
    ms['cCl_out'] = cCl_out * 1e3
    ms['cNa_out'] = cNa_out * 1e3
    ms['qNa_out'] = qNa_out * 1e3
    ms['qCl_out'] = qCl_out * 1e3
    ms['qP1_out'] = qP1_out * 1e3
    ms['qP2_out'] = qP2_out * 1e3
    ms['qP3_out'] = qP3_out * 1e3
    ms['qA1_out'] = qA1_out * 1e3
    ms['cP1_out'] = cP1_out * 1e3
    ms['cP2_out'] = cP2_out * 1e3
    ms['cP3_out'] = cP3_out * 1e3
    ms['cP0_out'] = (cP_tot_out - cP1_out - cP2_out - cP3_out) * 1e3
    ms['cA1_out'] = cA1_out * 1e3
    ms['cA0_out'] = (cA_tot_out - cA1_out) * 1e3
    ms['cH_out'] = cH_out * 1e3
    ms['cOH_out'] = cOH_out * 1e3

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
        'Acetate0': ms['cA0_out'],
        'Acetate-': ms['cA1_out'],
        'Phosphate0': ms['cP0_out'],
        'Phosphate-': ms['cP1_out'],
        'Phosphate2-': ms['cP2_out'],
        'Phosphate3-': ms['cP3_out'],
        'qNa': ms['qNa_out'],
        'qCl': ms['qCl_out'],
        'qPhosphate-': ms['qP1_out'],
        'qPhosphate2-': ms['qP2_out'],
        'qPhosphate3-': ms['qP3_out'],
        'qAcetate-': ms['qA1_out'],
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
    P0 = (53/255, 94/255, 59/255) # Hunter Green
    P1 = (79/255, 121/255, 66/255) # Fern Green
    P2 = (138/255, 154/255, 91/255) # Moss Green
    P3 = (147/255, 197/255, 114/255) # Pistachio
    A0 = (227/255, 11/255, 92/255) # Raspberry
    A1 = (255/255, 192/255, 203/255) # Pink

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
    axC.plot(CV, ms['cP0_out'], '-', linewidth=2, label='P0', color=P0)
    axC.plot(CV, ms['cP1_out'], '-', linewidth=2, label='P1-', color=P1)
    axC.plot(CV, ms['cP2_out'], '-', linewidth=2, label='P2-', color=P2)
    axC.plot(CV, ms['cP3_out'], '-', linewidth=2, label='P3-', color=P3)
    axC.plot(CV, ms['cA0_out'], '-', linewidth=2, label='A0', color=A0)
    axC.plot(CV, ms['cA1_out'], '-', linewidth=2, label='A1-', color=A1)
    axC.set_title("Concentrations (mM)", fontsize=font_size)
    axC.set_xlabel("Column Volumes", fontsize=font_size)
    axC.set_xlim([0, end_CV])
    axC.set_ylim([0, None])

    axD.plot(CV, ms['qNa_out'], '-', linewidth=2, label='Na+', color=Na)
    axD.plot(CV, ms['qCl_out'], '-', linewidth=2, label='Cl-', color=Cl)
    axD.plot(CV, ms['qP1_out'], '-', linewidth=2, label='P1-', color=P1)
    axD.plot(CV, ms['qP2_out'], '-', linewidth=2, label='P2-', color=P2)
    axD.plot(CV, ms['qP3_out'], '-', linewidth=2, label='P3-', color=P3)
    axD.plot(CV, ms['qA1_out'], '-', linewidth=2, label='A1-', color=A1)
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
        '13_50p_50a.csv',
        '22_50p_50a50s.csv',
        '23_150p_50a.csv',
        '53_50p150s_50a.csv',
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
