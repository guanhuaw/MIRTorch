import torch

class Prox():
   #TODO Figure out exact syntax for running operations on GPU (cuda) or CPU with default being GPU

    def __init__(self, device = None):
        self._device = device
        #sigpy has size/shape input parameter, but I don't see why we would need it?
        pass

    
    def __call__(self, input): 
        #sigpy also has alpha value, maybe add that here after implementing basic functionality
        return self._apply(input)

class L1Regularizer(Prox):
    def __init__(self, lambda, device):
        self.lambda = float(lambda)

        super().__init__(device)


    def _apply(self, input):
        # Closed form solution from
        # https://archive.siam.org/books/mo25/mo25_ch6.pdf

        with self._device:
            abs = torch.abs(input, device=self._device)
            sign = input.div(abs)
            sign[sign == float("Inf")] = 0

            abs = abs - self.lambda
            abs[abs < 0] = 0

            return torch.abs * sign



class L2Regularizer(Prox):
    def __init__(self, lambda, device):
        self.lambda = float(lambda)

        super().__init__(device)

    def _apply(self, input):
        # Closed form solution from
        # https://archive.siam.org/books/mo25/mo25_ch6.pdf
        with self._device:
            scale = 1 - self.lambda / max(lambda, torch.linalg.norm(input))
            return scale * input

