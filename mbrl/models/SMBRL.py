from .model import Model
import torch
from torch.nn import functional as F
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import feyn
from sklearn.model_selection import train_test_split


class SMBRLModel(Model):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        ensemble_size: int,
        device: Union[str, torch.device],
    ):
        super().__init__(device)
        self.in_size = in_size
        self.out_size = out_size
        self.ql = feyn.QLattice()
        self.models_matrix = None
        self.ensemble_size = 1

    def step(self, stateaction):
        stateaction = stateaction.cpu()
        stateaction_pd = self.numpy_check_to_DataFrame(stateaction.numpy())
        # print("Nan count: ", stateaction_pd.isna().sum().sum())
        # stateaction_pd.fillna(0.0)
        #stateaction_pd.replace(np.nan, 0.0)
        # stateaction_pd.interpolate()
        if self.models_matrix is None:
            # +1 for reward
            return torch.rand(stateaction.shape[0], self.out_size)
        else:
            deltas = self.models_matrix[0][0].predict(stateaction_pd).reshape(-1, 1)
            for i in range(1, self.out_size):
                delta_i = self.models_matrix[i][0].predict(stateaction_pd).reshape(-1, 1)
                deltas = np.append(deltas, delta_i, axis=1)
        return torch.from_numpy(deltas).to(self.device)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        """Computes the output of the dynamics model.

        Args:
            x (tensor): the input to the model.

        Returns:
            (tuple of tensors): all tensors predicted by the model (e.g., .mean and logvar).
        """
        return self.step(x), None

    def loss(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes a loss that can be used to update the model using backpropagation.

        Args:
            model_in (tensor or batch of transitions): the inputs to the model.
            target (tensor, optional): the expected output for the given inputs, if it
                cannot be computed from ``model_in``.

        Returns:
            (tuple of tensor and optional dict): the loss tensor and, optionally,
                any additional metadata computed by the model,
                 as a dictionary from strings to objects with metadata computed by
                 the model (e.g., reconstruction, entropy) that will be used for logging.
        """
        return 0

    def eval_score(
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes an evaluation score for the model over the given input/target.

        This method should compute a non-reduced score for the model, intended mostly for
        logging/debugging purposes (so, it should not keep gradient information).
        For example, the following could be a valid
        implementation of ``eval_score``:

        .. code-block:: python

           with torch.no_grad():
               return torch.functional.mse_loss(model(model_in), target, reduction="none")


        Args:
            model_in (tensor or batch of transitions): the inputs to the model.
            target (tensor or sequence of tensors): the expected output for the given inputs, if it
                cannot be computed from ``model_in``.

        Returns:
            (tuple of tensor and optional dict): a non-reduced tensor score, and a dictionary
                from strings to objects with metadata computed by the model
                (e.g., reconstructions, entropy, etc.) that will be used for logging.
        """
        assert model_in.ndim == 2 and target.ndim == 2
        with torch.no_grad():
            pred_mean, _ = self.forward(model_in, use_propagation=False)
            #target = target.repeat((self.num_members, 1, 1))
            return F.mse_loss(pred_mean.to(self.device), target.to(self.device), reduction="none"), {}

    def reset(
        self, obs: torch.Tensor, rng: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """Prepares the model for simulating using :class:`mbrl.models.ModelEnv`."""
        raise NotImplementedError(
            "ModelEnv requires 1-D models must be wrapped into a OneDTransitionRewardModel."
        )

    def reset_1d(
        self, obs: torch.Tensor, rng: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """Initializes the model to start a new simulated trajectory.

        Returns a dictionary with one keys: "propagation_indices". If
        `self.propagation_method == "fixed_model"`, its value will be the
        computed propagation indices. Otherwise, its value is set to ``None``.

        Args:
            obs (tensor): the observation from which the trajectory will be
                started. The actual value is ignore, only the shape is used.
            rng (`torch.Generator`, optional): an optional random number generator
                to use.

        Returns:
            (dict(str, tensor)): the model state necessary to continue the simulation.
        """
        return {"obs": obs}

    def sample(
        self,
        act: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Dict[str, torch.Tensor]],
    ]:
        raise NotImplementedError(
            "ModelEnv requires 1-D models must be wrapped into a OneDTransitionRewardModel."
        )

    def sample_1d(
        self,
        model_input: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Samples an output from the model using .

        This method will be used by :class:`ModelEnv` to simulate a transition of the form.
            outputs_t+1, s_t+1 = sample(model_input_t, s_t), where

            - model_input_t: observation and action at time t, concatenated across axis=1.
            - s_t: model state at time t (as returned by :meth:`reset()` or :meth:`sample()`.
            - outputs_t+1: observation and reward at time t+1, concatenated across axis=1.

        The default implementation returns `s_t+1=s_t`.

        Args:
            model_input (tensor): the observation and action at.
            model_state (tensor): the model state st. Must contain a key
                "propagation_indices" to use for uncertainty propagation.
            deterministic (bool): if ``True``, the model returns a deterministic
                "sample" (e.g., the mean prediction). Defaults to ``False``.
            rng (`torch.Generator`, optional): an optional random number generator
                to use.

        Returns:
            (tuple): predicted observation, rewards, terminal indicator and model
                state dictionary. Everything but the observation is optional, and can
                be returned with value ``None``.
        """
        return (self.forward(model_input)[0], model_state)

    def numpy_check_to_DataFrame(self, X, y=None):
        if isinstance(X, np.ndarray):
            columns = [f'x{i}' for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=columns)

        if y is None:
            return X
        elif isinstance(y, np.ndarray):
            if y.ndim == 1:
                y = pd.Series(y, name='target')
            elif y.ndim == 2:
                columns = [f'y{i}' for i in range(y.shape[1])]
                y = pd.DataFrame(y, columns=columns)
            return pd.concat([X, y], axis=1)

    def update(self, batch):
        print("Number of samples: ", batch.__len__())
        s, a, ns, r, _ = self._process_batch(batch)
        x = torch.cat((s, a), dim=1).cpu().numpy()
        delta = (ns - s).cpu().numpy()
        y = np.append(delta, r.cpu().numpy().reshape(-1, 1), axis=1)

        data = self.numpy_check_to_DataFrame(x, y)
        #train, test = train_test_split(data, test_size=0.1)

        # Create best models array for every output dimension
        models_matrix = []
        for i in range(y.shape[1]):
            new_data = data
            for j in range(y.shape[1]):
                if i != j:
                    new_data = new_data.drop('y' + str(j), axis=1)
            train, test = train_test_split(new_data, test_size=0.1)
            starting_models = self.models_matrix[i] if self.models_matrix is not None else None
            models_i = self.ql.auto_run(train, output_name='y' + str(i),
                                        starting_models=starting_models, n_epochs=10)
            models_matrix.append(models_i)
            # models_i[0].plot(data=new_data)
            print("Dynamics function for dimension ", i, ": ", models_i[0].sympify(signif=3))
            print("R2 Score: ", models_i[0].r2_score(test))
        self.models_matrix = models_matrix
        for i in range(y.shape[1]):
            print(self.models_matrix[i][0].sympify(signif=3))
