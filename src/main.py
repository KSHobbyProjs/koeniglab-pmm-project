#!/usr/bin/env python
import physics_models as pm
import numpy as np
import ec
import matplotlib.pyplot as plt
import pickle
import pmm

if __name__=="__main__":
    # load Gaussian model
    N = 32
    gauss = pm.gaussian.Gaussian(N)

    # import ground state energies and eigenvectors
    with open("../data/gauss_energies.pkl", "rb") as f:
        data = pickle.load(f)

    gs_actual = data["Ls"]
    Es_actual = data["energies"][:,0]

    gs_sample = 5 + np.linspace(0, 1, 20)**1.5 * (20 - 5)
    Es_sample = gauss.get_eigenvalues(gs_sample, k_num=1)
   # with open("../data/gauss_sample_energies.pkl", "wb") as f:
        #pickle.dump({"Ls" : gs_sample, "energies" : Es_sample}, f)
    
    epochs = 15000
    store_loss = 100
    dim = 5
    pmm_gauss = pmm.PMM(dim=dim)
    pmm_gauss.sample(file="../data/gauss_sample_energies.pkl")
    #pmm_gauss.load("../data/gauss_state.pkl")
    _, loss = pmm_gauss.train(epochs=epochs, store_loss=store_loss) 
    pmm_gauss.store(path="../data/gauss_state.pkl")
    Es_predict = pmm_gauss.predict(gs_actual)

    fig, ax = plt.subplots()
    ax.plot(store_loss * np.arange(len(loss)), np.log10(loss), '-')
    print(loss[-1])
    plt.show()
    
    fig, ax = plt.subplots()
    ax.plot(gs_actual, Es_actual, '-')
    ax.plot(gs_sample, Es_sample, 'o')
    ax.plot(gs_actual, Es_predict, '--')
    plt.show()

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

