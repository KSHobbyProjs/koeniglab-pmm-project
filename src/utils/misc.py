def make_model_string(model_name, **kwargs):
    class_name = model_name.split(".", 1)[1]
    file_name = class_name + "__" + "__".join(f"{k}_{v}" for k, v in kwargs.items())
    return file_name
