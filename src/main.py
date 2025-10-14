#!/usr/bin/env python
import physics_models as pm
import numpy as np
import ec
import matplotlib.pyplot as plt
import pickle
import pmm

if __name__=="__main__":
    # load gaussian model and EC
    N = 32
    gauss = pm.gaussian.Gaussian(N)
    gauss_ec = ec.EC(gauss)

    # import ground state energies and eigenvectors
    with open("../data/gauss_energies.pkl", "rb") as f:
        data = pickle.load(f)

    Ls_actual = data["Ls"]
    Es_actual = data["energies"][:,0]
    states = data["eigenstates"][:,0,:]

    pmm_gauss = pmm.PMM(gauss, 2)
    # print(pmm_gauss.temp())

    """
    # plot ground state eigenvector
    L_index = 10
    L = Ls_actual[L_index]
    a_phys = L / N
    psi = states[L_index]
    psi2 = np.abs(psi)**2 / a_phys**3
    
    ys2d, xs2d = np.indices((N, N))
    xs2d = (xs2d.flatten() - N / 2) * a_phys
    ys2d = (ys2d.flatten() - N / 2) * a_phys
    plt.scatter(xs2d, ys2d, c=psi2[:N**2]*a_phys, cmap='viridis')
    plt.colorbar()
    plt.show()

    # dilate ground state eigenvector and plot
    L_target = 10
    a_phys = L_target / N
    psi_dilated = ec.EC.dilate(L, L_target, psi)
    psi_dilated2 = np.abs(psi_dilated)**2 / a_phys**3

    ys2d, xs2d = np.indices((N, N))
    xs2d = (xs2d.flatten() - N / 2) * a_phys
    ys2d = (ys2d.flatten() - N / 2) * a_phys 
    plt.scatter(xs2d, ys2d, c=psi_dilated2[:N**2]*a_phys, cmap='viridis')
    plt.colorbar()
    plt.show()
    """
    
    """
    # use EC to find predictions for ground state energies
    Ls_train = np.arange(5, 13) 
    Es_train = gauss.get_eigenvalues(Ls_train)
    Es_predict, _ = gauss_ec.solve(Ls_train, Ls_actual, k_num_sample=6, k_num_predict=1, dilate=False)
    
    # find index
    L_star = 20
    temp = np.ones_like(Ls_actual) * L_star
    error = abs(Ls_actual - temp)
    i = np.argmin(error)
    print(Ls_actual[i])
    print(Es_predict[i])

    fig, ax = plt.subplots()
    ax.plot(Ls_train, Es_train, 'o', label='training values')
    ax.plot(Ls_actual, Es_actual, '-', label='test energies')
    ax.plot(Ls_actual, Es_predict, '--', label='prediction')
    ax.set_xlabel('System Size L')
    ax.set_ylabel('Ground State Energy E')
    ax.set_title('Ground State Energy vs System Size')
    ax.legend()
    plt.show() 
    """

