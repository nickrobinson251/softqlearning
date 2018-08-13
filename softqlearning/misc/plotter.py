import numpy as np
import matplotlib.pyplot as plt


class QFPolicyPlotter:
    def __init__(
            self,
            q_function,
            policy,
            observations,
            default_action,
            n_samples):
        self._q_function = q_function
        self._policy = policy
        self._observations = observations
        self._default_action = default_action
        self._n_samples = n_samples

        self._var_inds = np.where(np.isnan(default_action))[0]
        # assert len(self._var_inds) == 2

        n_plots = len(observations)
        y_size = 5
        x_size = y_size * n_plots

        # fig = plt.figure(figsize=(x_size, y_size))
        # self._axes = []
        # for i in range(1, n_plots+1):
        #     ax = fig.add_subplot(1, n_plots, i)
        #     ax.set_xlim((-1, 1))
        #     ax.set_ylim((-1, 1))
        #     ax.grid(True)
        #     self._axes.append(ax)

        fig, self._axes = plt.subplots(
            nrows=1, ncols=n_plots, figsize=(x_size, y_size))
        for ax in self._axes:
            ax.set_xlim((-1, 1))
            ax.set_ylim((-1, 1))
            ax.grid(True)

        self._line_objects = list()

    def draw(self):
        # noinspection PyArgumentList
        [h.remove() for h in self._line_objects]
        self._line_objects = list()
        self._plot_level_curves()
        self._plot_action_samples()
        plt.draw()
        plt.pause(0.001)

    def _plot_level_curves(self):
        # Create mesh grid
        xs = np.linspace(-1, 1, 50)
        ys = np.linspace(-1, 1, 50)
        xgrid, ygrid = np.meshgrid(xs, ys)
        N = len(xs)*len(ys)

        # Copy default values along the first axis and replace nans with
        # the mesh grid points.
        actions = np.tile(self._default_action, (N, 1))
        actions[:, self._var_inds[0]] = xgrid.ravel()
        actions[:, self._var_inds[1]] = ygrid.ravel()

        for ax, obs in zip(self._axes, self._observations):
            qs = self._q_function.eval(obs[None], actions)
            qs = qs.reshape(xgrid.shape)

            n_intervals = 20
            contour_set = ax.contour(xgrid, ygrid, qs, n_intervals)
            self._line_objects += contour_set.collections
            self._line_objects += ax.clabel(
                contour_set, inline=True, fontsize=10, fmt='%.2f')

    def _plot_action_samples(self):
        for ax, obs in zip(self._axes, self._observations):
            actions = self._policy.get_actions(
                np.ones((self._n_samples, 1)) * obs[None, :])
            x, y = actions[:, 0], actions[:, 1]
            self._line_objects += ax.plot(x, y, 'b*')
