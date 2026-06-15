import torch


class Adam:
    def __init__(
        self,
        learning_rate=1.0,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        """
        Initialize the Adam optimizer.

        Parameters:
        - learning_rate: The learning rate (float).
        - beta1: The exponential decay rate for the first moment estimates (float).
        - beta2: The exponential decay rate for the second moment estimates (float).
        - epsilon: A small constant for numerical stability (float).
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment vector
        self.v = {}  # Second moment vector
        self.t = {}

    def step(self, params, grads):
        """
        Update parameters using Adam optimization.

        Parameters:
        - params: A dictionary of parameters to be updated.
        - grads: A dictionary of gradients corresponding to the parameters.

        Returns:
        - updated_params: The updated parameters.
        """

        for key in grads:
            if key not in self.m:
                self.m[key] = torch.zeros_like(grads[key])
                self.v[key] = torch.zeros_like(grads[key])
                self.t[key] = torch.ones(
                    (len(grads[key]),) + (1,) * (len(grads[key].shape) - 1),
                    device=grads[key].device,
                )

        # Increment time step
        updated_params = {}

        for key in params:
            dims = tuple(range(1, len(grads[key].shape)))
            skip_update = torch.all(
                torch.abs(grads[key]) < 1e-8, dim=dims, keepdim=True
            )
            skip_update = skip_update.float()

            # Update biased first and second moment estimates
            m_update = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            v_update = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key] ** 2

            self.m[key] = (1 - skip_update) * m_update + skip_update * self.m[key]
            self.v[key] = (1 - skip_update) * v_update + skip_update * self.v[key]

            # Correct bias in the first and second moment
            m_hat = self.m[key] / (1 - self.beta1 ** self.t[key])
            v_hat = self.v[key] / (1 - self.beta2 ** self.t[key])

            # Update parameters
            ug = m_hat / (v_hat**0.5 + self.epsilon)
            # updated_params[key] = params[key] - self.learning_rate * ug * (1 - skip_update)
            # see Eq.7 of Energy Guided Diffusion for Generating Neurally Exciting Images
            updated_params[key] = params[key] + self.learning_rate * ug * (
                1 - skip_update
            )
            self.t[key] += 1 - skip_update

        return updated_params
