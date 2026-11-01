class BasePipeline:
    def load_diffusion_model(self):
        raise NotImplementedError

    def prepare_inputs(self, inputs):
        raise NotImplementedError

    def to_layers(self):
        raise NotImplementedError

    def get_param_groups(self, parameters):
        return [{'params': parameters}]

    def get_loss_fn(self):
        raise NotImplementedError

    def save_model(self, save_dir, state_dict):
        raise NotImplementedError
