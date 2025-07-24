#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   Aquisition_function.py
# Time    :   2025/05/22 19:33:28
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

# Noisy Expected Improvement (NEI) calculation
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from icecream.icecream import ic

class YuKernel:
    def __init__(self, v0, w, a0, a1, sigma2):
        self.v0, self.w, self.a0, self.a1, self.sigma2 = v0, np.atleast_1d(w), a0, a1, sigma2

    def __call__(self, X, Y=None):
        X, Y = np.atleast_2d(X), np.atleast_2d(Y) if Y is not None else np.atleast_2d(X)
        diff = (X[:, None, :] - Y[None, :, :]) ** 2
        exp_term = self.v0 * np.exp(-np.sum(diff * self.w, axis=2))
        linear_term = self.a0 + self.a1 * np.dot(X, Y.T)
        noise = self.sigma2 * np.eye(X.shape[0]) if X.shape == Y.shape and np.allclose(X, Y) else 0.0
        return exp_term + linear_term + noise

def corrected_expected_improvement(X_cand, mu_cand, var_cand, x_plus, mu_plus, var_plus, kernel, jitter=1e-9):
    cov_vec = kernel(X_cand, np.atleast_2d(x_plus))[:, 0]
    sigma_tilde_sq = var_cand + var_plus - 2 * cov_vec
    sigma_tilde = np.sqrt(np.maximum(sigma_tilde_sq, 0.0)) + jitter
    u = mu_cand - mu_plus
    z = u / sigma_tilde
    return sigma_tilde * norm.pdf(z) + u * norm.cdf(z)

def parse_x(xstr):
    return np.fromstring(xstr.replace('[', '').replace(']', ''), sep=' ')

def plot_ei(ei_vals):
    plt.plot(range(1, len(ei_vals) + 1), ei_vals, marker='o', label='Noisy EI')
    plt.xlabel('Pareto Points')
    plt.ylabel('Noisy EI Value')
    plt.legend()
    plt.tight_layout()
    plt.xticks(range(1, len(ei_vals) + 1))
    # plt.savefig('Multi-Objective Optimisation\\Pareto_animation\\fulldata-s-epoch3\\Noisy_EI_Pareto_Points-epoch3.png', dpi=300, bbox_inches='tight') # epoch 3
    # plt.savefig('Multi-Objective Optimisation\\Pareto_animation\\fulldata-s-epoch2\\Noisy_EI_Pareto_Points-epoch2.png', dpi=300, bbox_inches='tight')  # epoch 2
    plt.savefig('Multi-Objective Optimisation\\Pareto_animation\\fulldata-s\\Noisy_EI_Pareto_Points.png', dpi=300, bbox_inches='tight')  # epoch 1
    plt.show()

def hv_2d(points, ref):
    sorted_pts = sorted(points, key=lambda pt: pt[0], reverse=True)
    hv, prev_x = 0.0, ref[0]
    for x, y in sorted_pts:
        hv += (prev_x - x) * (ref[1] - y)
        prev_x = x
    return hv

def plot_hv_contrib(x_vals, y_vals):
    x_vals = range(1, len(x_vals) + 1)

    if any(y < 0 for y in y_vals):
        y_vals = [0] * len(y_vals)

    plt.plot(x_vals, y_vals, marker='o', label='HV Contribution')
    plt.xlabel('Pareto Point Order (sorted by HV contribution)')
    plt.ylabel('Hypervolume Contribution')
    plt.legend()
    plt.tight_layout()
    plt.xticks(x_vals)
    # plt.savefig('Multi-Objective Optimisation\\Pareto_animation\\fulldata-s-epoch3\\HV_contribution-epoch3.png', dpi=300) # epoch 3
    # plt.savefig('Multi-Objective Optimisation\\Pareto_animation\\fulldata-s-epoch2\\HV_contribution-epoch2.png', dpi=300)  # epoch 2
    plt.savefig('Multi-Objective Optimisation\\Pareto_animation\\fulldata-s\\HV_contribution.png', dpi=300)  # epoch 1
    plt.show()

def plot_ei_and_hv(ei_vals, hv_vals):
    fig, ax1 = plt.subplots()
    color1 = 'tab:blue'
    color2 = 'tab:red'
    x = np.arange(1, len(ei_vals) + 1)

    ax1.set_xlabel('Pareto Points')
    ax1.set_ylabel('Noisy EI', color=color1)
    ax1.plot(x, ei_vals, marker='o', color=color1, label='Noisy EI')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel('HV Contribution', color=color2)
    ax2.plot(x, hv_vals, marker='s', color=color2, label='HV Contribution')
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.tight_layout()
    plt.title('Noisy EI and HV Contribution')
    plt.xticks(x)
    # plt.savefig('Multi-Objective Optimisation\\Pareto_animation\\fulldata-s-epoch3\\EI_HV_Combined-epoch3.png', dpi=300, bbox_inches='tight')  # epoch 3
    # plt.savefig('Multi-Objective Optimisation\\Pareto_animation\\fulldata-s-epoch2\\EI_HV_Combined-epoch2.png', dpi=300, bbox_inches='tight')  # epoch 2
    plt.savefig('Multi-Objective Optimisation\\Pareto_animation\\fulldata-s\\EI_HV_Combined.png', dpi=300, bbox_inches='tight')    # epoch 1
    plt.show()

def main():
    # Load Pareto front and hyperparameters
    # pareto_front = pd.read_excel('Multi-Objective Optimisation\\Dataset\\fulldata-s-epoch3\\pareto_front_fulldata.xlsx')   # Use this for epoch 3
    # pareto_front = pd.read_excel('Multi-Objective Optimisation\\Dataset\\fulldata-s-epoch2\\pareto_front_fulldata.xlsx')   # Use this for epoch 2
    pareto_front = pd.read_excel('Multi-Objective Optimisation\\Dataset\\fulldata-s\\pareto_front_fulldata.xlsx')     # Use this for epoch 1
    X_cand = np.vstack(pareto_front['X'].apply(parse_x).values)
    mu_cand, std_cand = pareto_front['Mean'].values, pareto_front['Std'].values
    var_cand = std_cand ** 2

    # hyper_df = pd.read_excel('Multi-Objective Optimisation\\Dataset\\fulldata-s-epoch3\\Bayesian_Optimisation_Results_fulldata_epoch3.xlsx')   # epoch 3
    # hyper_df = pd.read_excel('Multi-Objective Optimisation\\Dataset\\fulldata-s-epoch2\\Bayesian_Optimisation_Results_fulldata_epoch2.xlsx')   # epoch 2
    hyper_df = pd.read_excel('Multi-Objective Optimisation\\Dataset\\fulldata-s\\Bayesian_Optimisation_Results_fulldata.xlsx')   # epoch 1
    params = dict(
        v0=hyper_df.loc[0, 'v0'],
        a0=hyper_df.loc[0, 'a0'],
        a1=hyper_df.loc[0, 'a1'],
        sigma2=1e-6,
        w=[hyper_df.loc[0, 'wl1'], hyper_df.loc[0, 'wl2'], hyper_df.loc[0, 'wl3']]
    )

    # Epoch 1
    x_plus = np.array([22.7, 14.4, 16.0])
    mu_plus, std_plus = 2705.36, 157.69
    var_plus = std_plus ** 2

    # Epoch 2
    # x_plus = np.array([20.00, 11.50, 19.98])
    # mu_plus, std_plus = 3008.20, 157.69
    # var_plus = std_plus ** 2

    # Epoch 3
    # x_plus = np.array([20.00, 10.11, 19.97])
    # mu_plus, std_plus = 3305.14, 116.68
    # var_plus = std_plus ** 2

    kernel = YuKernel(**params)
    ei_vals = corrected_expected_improvement(X_cand, mu_cand, var_cand, x_plus, mu_plus, var_plus, kernel)
    sorted_indices = np.argsort(ei_vals)[::-1]

    print("Candidates sorted by EI (descending):")
    for i in sorted_indices:
        print(f"Candidate {i+1}: EI = {ei_vals[i]:.4f}")

    plot_ei(ei_vals)

    # Hypervolume Contribution
    excel_path = "Multi-Objective Optimisation\\Dataset\\Comparison of raw and optimisation in IVRT FS - Corrected.xlsx"

    df = pd.read_excel(excel_path, sheet_name="R-Epoch-1")      # Choose the correct epoch number: R-Epoch-1, R-Epoch-2-2, R-Epoch-3
    ref_point = (df["Mean"].min() - 1.0, df["Std"].min() - 1.0)  # TODO: Since the optimization maximizes these two objectives, the reference point should be set to the last best point.
    
    # ref_point = (mu_plus + 1e-6, std_plus + 1e-6)  # Use the last best point as reference point
    ic("Reference Point:", ref_point)

    points = df[["Mean", "Std"]].values
    total_hv = hv_2d(points, ref_point)
    contribs = [max(total_hv - hv_2d(points[np.arange(len(df)) != i], ref_point), 0) for i in range(len(df))]
    df["HV_Contribution"] = contribs
    df_sorted = df.sort_values("HV_Contribution", ascending=False).reset_index(drop=True)
    print(df_sorted[["Num", "Mean", "Std", "HV_Contribution"]].head(5).to_string(index=False))
    plot_hv_contrib(df['Num'].values, df["HV_Contribution"].values)
    plot_ei_and_hv(ei_vals, df["HV_Contribution"].values)

if __name__ == "__main__":

    main()