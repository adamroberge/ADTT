import torch.nn as nn
writer = None

class BaseLoss(nn.Module):


    def __init__(
            self,
            weight=1.0,
            input_dict={
                'input': 'input'},
            **kwargs):
        super().__init__()
        self.weight = weight
        self.input_dict = input_dict
        self.loss_func = lambda: 0
        self.writer = writer


    def forward(self, inputs):
        actual_inputs = {}
        for input_key, input_val in self.input_dict.items():
            if input_val not in inputs:
                raise KeyError(f"Input key '{input_val}' not found in model outputs. Available keys: {list(inputs.keys())}")
            actual_inputs.update({input_key: inputs[input_val]})
        return self.weight * self.loss_func(**actual_inputs)
