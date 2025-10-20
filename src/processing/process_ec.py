
def process_ec_predicted_eigenpairs(
        model_name,
        sample_Ls,
        target_Ls,
        k_num_sample=1,
        k_num_predict=1,
        dilate=False,
        plot_kwargs=None,
        **model_kwargs
        ):

    # try loading existing exact data
    model_string =  utils.make_model_string(model_name, **model_kwargs)
    try:
        file_path = os.path.join(DATA_DIR, "exact_eigenpairs__" + model_string + ".pkl")
        exact_Ls, exact_energies, _ = load_eigenpairs(file_path)
    except FileNotFoundError:
        # if missing, compute
        print(f"[INFO] Exact eigenpairs not found for {model_name}, computing now... Run `get_exact_data.py` to preload data")
        exact_Ls = target_Ls
        exact_energies, _ = compute_exact_eigenpairs(model_name, exact_Ls, k_num_predict, **model_kwargs)

    # grab model
    submodule_name, class_name = model_name.split(".", 1)
    submodule = getattr(pm, submodule_name)
    ModelClass = getattr(submodule, class_name)
    model_instance = ModelClass(**model_kwargs)
    
    # define ec instance and grab ec predictions
    ec_instance = ec.EC(model_instance)
    predicted_energies, predicted_states = ec_instance.run_ec(sample_Ls, target_Ls, k_num_sample=k_num_sample, k_num_predict=k_num_predict, dilate=dilate)

    # save predicted energies and states
    ec_dir = os.path.join(RESULTS_DIR, "ec", model_string)
    os.makedirs(ec_dir, exist_ok=True)
    file_path = os.path.join(ec_dir, "ec_predicted_eigenpairs.pkl")
    utils.save_eigenpairs(file_path, sample_Ls, predicted_energies, predicted_states)

    # save metadata
    metadata_path = os.path.join(ec_dir, "ec_metadata.json")
    metadata = {
            "algorithm" : "ec",
            "model_name" : model_string,
            "sample_Ls" : f"np.linspace({sample_Ls[0]}, {sample_Ls[-1]}, {len(sample_Ls)}",
            "k_num_sample" : k_num_sample,
            "k_num_predict" : k_num_predict,
            "dilate" : dilate
            }
    utils.save_metadata(metadata_path, metadata)

    # wrap exact energies, sample energies, and predicted energies into a dictionary
    energies = {"exact" : exact_energies[:, k_num_predict],
                "predicted" : predicted_energies}
    Ls = {"exact" : exact_Ls,
          "predicted" : target_Ls}
    
    # create plot directory
    results_plot_dir = os.path.join(RESULTS_DIR, 'plots', model_string, "ec")
    os.makedirs(results_plot_dir, exist_ok=True)
    utils.plot_eigenvalues_separately(results_plot_dir, Ls, energies, k_indices=list(range(k_num_predict)), show=False, **(plot_kwargs or {}))
