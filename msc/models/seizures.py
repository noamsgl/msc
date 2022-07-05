import gpytorch
import torch
import pytorch_lightning as pl
import pyro

class SeizuresModule(pl.LightningModule):
    """
    PyTorch Lightning Model
    On training, optimizer is called and the marginal log likelihood of the model is maximized.
    The parameters will be estimated and analyzed for the statistical validity of the model.
    """
    def __init__(self, hparams, train_x, train_y) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        assert train_y().dim() == 1, f"error: {train_y().dim()=} should be 1"
        self.train_x = train_x
        self.train_y = train_y
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gpmodel = SeizuresGPModel(train_x, train_y, self.likelihood)


    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        loss = self.svi.evaluate_loss(batch)
        loss = torch.tensor(loss).requires_grad_(True)
        tensorboard_logs = {'running/loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def test_step(batch, batch_idx):
        raise NotImplementedError("no need for test")

    def configure_optimizers(self):
        self.elbo = pyro.infer.Trace_ELBO(num_particals=self.hparams.num_particles, vectorize_particles=True, retain_graph=True)
        self.svi = pyro.infer.SVI(self.gpmodel.model, self.gpmodel.guide, self.configure_optimizers() ,self.elbo)
        return [self.svi]

    @pl.data_loader
    def train_dataloader(self):
        x = None
        y = None
        ds = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(ds, batch_size=self.hparams.batch_size)
        return dataloader
    

class SeizuresGPModel(gpytorch.models.ApproximateGP):
    """"""
    def __init__(self, num_arrivals, max_time, num_inducing, name_prefix="seizure_gp_model"):
        self.name_prefix = name_prefix
        self.max_time = max_time
        self.mean_intensity = (num_arrivals / max_time)

        # Define the variational distribution and strategy of the GP
        # We will initiailize the inducing points with to lie on a grid from 0 to T
        inducing_points = torch.linspace(0, max_time, num_inducing).unsqueeze(-1)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=num_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution)
        
        # initialize gpytorch.models.ApproximateGP
        super().__init__(variational_strategy=variational_strategy)

        # Define mean and kernel
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, times):
        """
        Forward pass of the model.
        Computes the prior GP mean and covariance at the supplied times.
        """
        mean_output = self.mean_module(times)
        covar_output = self.covar_module(times)
        return gpytorch.distributions.MultivariateNormal(mean_output, covar_output)
    

    def guide(self, arrival_times, quadrature_times):
        """
        Guide pass of the model.
        define the approximate GP posterior at both arrival times and quadrature times.
        """
        function_distribution = self.pyro_guide(torch.cat([arrival_times, quadrature_times], -1))

        # Draw samples from q(f) at arrival_times
        # Also draw samples from q(f) at evenly-spaced points (quadrature_times)
        with pyro.plate(self.name_prefix + ".times_plate", dim=-1):
            pyro.sample(
                self.name_prefix + ".function_samples",
                function_distribution
            )

    def model(self, arrival_times, quadrature_times):
        """
        Model pass of the model.
        1. Computes the GP prior at arrival_times and quadrature_times.
        2. Converts GP function samples into intensity function samples, using the exponential link function.
        3. Computes the likelihood of the arrivals. We will use ```pyro.factor()``` to define the likelihood."""
        pyro.module(self.name_prefix + ".gp", self)

        # get p(f) - prior distribution of latent function
        function_distribution = self.pyro_model(torch.cat([arrival_times, quadrature_times], -1))
        
        # Draw samples from p(f) at arrival times
        # Also draw samples from p(f) at evenly-spaced points (quadrature_times)
        # Use a plate here to indicate conditional independencies
        with pyro.plate(self.name_prefix + ".times_plate", dim=-1):
            # sample from latent distribution function
            function_samples = pyro.sample(
                self.name_prefix + ".function_samples",
                function_distribution
            )
        
        ####
        # Convert function samples into intensity samples, using the link function
        ####
        intensity_samples = function_samples.exp() * self.mean_intensity
        
        # Divide the intensity samples into arrival_intensity_samples and quadrature_intensity_samples
        arrival_intensity_samples, quadrature_intensity_samples = intensity_samples.split([
            arrival_times.size(-1), quadrature_times.size(-1)
        ], dim=-1)

        ####
        # Compute the log_likelihood, using the likelihood of a Cox process.
        # This is instead of sampling from a known distribution with an obs argument.
        ####
        arrival_log_intensities = arrival_intensity_samples.log().sum(dim=-1)
        est_num_arrivals = quadrature_intensity_samples.mean(dim=-1).mul(self.max_time)
        log_likelihood = arrival_log_intensities - est_num_arrivals
        pyro.factor(self.name_prefix + ".log_likelihood", log_likelihood)