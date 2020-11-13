import torch

class Prox():
   #TODO Figure out exact syntax for running operations on GPU (cuda) or CPU with default being GPU

    def __init__(self, device = None):
        self._device = device
        #sigpy has size/shape input parameter, but I don't see why we would need it?: Maybe for now we do not need it
        pass

    
    def __call__(self, input): 
        #sigpy also has alpha value, maybe add that here after implementing basic functionality
        return self._apply(input)

class L1Regularizer(Prox):
    # Please add the latex expression of what is happening here:
    def __init__(self, Lambda, device):
        super().__init__(device)
        self.Lambda = float(Lambda)

    def _apply(self, input):
        # Closed form solution from
        # https://archive.siam.org/books/mo25/mo25_ch6.pdf

        with self._device: # This is a simple function, input and output are on the same device. Maybe we do not need this
            abs = torch.abs(input, device=self._device)
            sign = input.div(abs)
            sign[sign == float("Inf")] = 0
            # I suggest using torch.sign here. If Inf exist there should be something wrong. We do not expect operation to handle this

            abs = abs - self.Lambda
            abs[abs < 0] = 0 # this operation seems to be not differentiable to me. You can double check it.
            # Suggestion: check https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#Softshrink for how they are dealing with the gradient

            return torch.abs * sign



class L2Regularizer(Prox):
    def __init__(self, Lambda, device):


        super().__init__(device)
        self.
        Lambda = float(Lambda)

    def _apply(self, input):
        # Closed form solution from
        # https://archive.siam.org/books/mo25/mo25_ch6.pdf
        # Again, pls add the math expression
        with self._device:
            scale = 1 - self.lambda / max(lambda, torch.linalg.norm(input))
            return scale * input

