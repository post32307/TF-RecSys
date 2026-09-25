import sys

with open('main.py', 'r') as f:
    lines = f.read().splitlines()

new_content = """    # ---------------------------------------------------------
    # 2. Independent validation search for IGF_EASE
    # ---------------------------------------------------------
    search_dataset = load_dataset_split(
        dataloader.Loader, dataset_path, "val.txt"
    ) if val_true == 1 else None
    if val_true == 0:
        search_dataset = dataset
    search_model = model.TFmodel(world.config, search_dataset)
    cache = {
        "eigen": {}, "max_eigen_entries": 1,
        "normalized": {}, "max_normalized_entries": 1,
        "convolution": {}, "max_convolution_entries": 1,
        "dan": {}, "max_dan_entries": 1,
    }
    saved_parameters = load_saved_parameters()
    saved_igf = saved_parameters.get("IGF_EASE")
    
    def metric(scores, evaluator, target):
        return evaluator.Test_RatingMatrix(scores, target)

    def score_model(recmodel, model_name, params, model_cache):
        if model_name == "IGF_EASE":
            return recmodel.S_IGF_EASE(cache=model_cache, **params)
        raise ValueError(f"Unknown model: {model_name}")

    def default_model_parameters(model_name):
        if model_name == "IGF_EASE":
            return {"beta_1": 1.0, "beta_2": 1.1, "alpha": 1.0,
                    "eps": 0.5, "lambda_": 50.0, "gamma": 1.0, "k": int(search_k_values[0])}
        raise ValueError(f"Unknown model: {model_name}")

    def model_parameter_space(model_name):
        if model_name == "IGF_EASE":
            return [dict(beta_1=b1, beta_2=b2, alpha=alpha, eps=eps,
                         lambda_=lambda_, gamma=gamma, k=k)
                    for alpha in (0.0, 1.0, 20.0)
                    for eps in (0.1, 0.3, 0.5)
                    for k in search_k_values
                    for lambda_ in (50, 500, 1000)
                    for b1, b2, gamma in product((0.8, 1.0), (1.1, 1.3),
                                                   (0.1, 0.5, 1.0))]
        raise ValueError(f"Unknown model: {model_name}")

    best_score = -np.inf
    best_params = default_model_parameters("IGF_EASE")
    best_result = None

    if val_true:
        candidates = model_parameter_space("IGF_EASE")
        for index, params in enumerate(candidates, 1):
            scores = score_model(search_model, "IGF_EASE", params, cache)
            result = metric(scores, search_model, search_dataset)
            score = float(result["recall"][0])
            if score > best_score:
                best_score = score
                best_params = params
                best_result = result
            del scores
            if index % 10 == 0 or index == len(candidates):
                world.cprint(f"IGF_EASE search {index}/{len(candidates)}: best Recall@20={best_score:.6f}")
        save_best_parameters("IGF_EASE", best_params)
    elif saved_igf:
        best_params = saved_igf[0] if isinstance(saved_igf, list) else saved_igf

    # ---------------------------------------------------------
    # 3. Final test of the model
    # ---------------------------------------------------------
    test_scores = score_model(Recmodel, "IGF_EASE", best_params, cache)
    test_result = Recmodel.Test_RatingMatrix(test_scores, dataset)
    if val_true or saved_igf is not None:
        save_best_parameters("IGF_EASE", best_params, test_results=test_result)
        
    world.cprint("=== Validation / search results ===")
    world.cprint(f"IGF_EASE validation: {best_result}")
    world.cprint("=== Final test results ===")
    world.cprint(f"IGF_EASE parameters: {best_params}")
    world.cprint(f"IGF_EASE test: {test_result}")
    
    del test_scores
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()"""

# Replace lines 240 through 673 (0-indexed) with the new content
final_lines = lines[:240] + new_content.splitlines() + lines[674:]

with open('main.py', 'w') as f:
    f.write('\n'.join(final_lines) + '\n')
