class DDPMModule(LightningModule):
    def __init__(
        self, 
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        beta_small: float,
        beta_large: float,
        in_size: int, # width*height
        t_range: int,
        img_depth: int, # channels 
    ) -> None:
        super().__init__()
        
        self.beta_small = beta_small # beta_1 
        self.beta_large = beta_large # beta_T 
        self.in_size = in_size
        self.t_range = t_range
        self.img_depth = img_depth
        
        self.save_hyperparameters(logger=False)
        self.net = net(img_depth=img_depth)
        
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        
    def forward(self, x, t):
        x = x.to(self.device)
        t = t.to(self.device)
        return self.net(x, t)
    
    def on_train_start(self) -> None:
        self.val_loss.reset()
        
####################################################################################################
##################################### Diffusion step ###############################################

    def beta(self, t):
        return self.beta_small + (t / self.t_range) * (
            self.beta_large - self.beta_small
        )
    def alpha(self, t):
        return 1 - self.beta(t)
    
    def alpha_bar(self, t):
        return math.prod([self.alpha(j) for j in range(t)])
    
    def model_step(self, batch):
        """
        Corresponds to Algorithm 1 from (Ho et al., 2020).
        """
        batch_size = batch.shape[0]
        ts = torch.randint(0, self.t_range, size=[batch_size], device=self.device)
        noise_imgs = []
        epsilon = torch.randn(size=batch.shape, device=self.device)
        for i in range(len(ts)):
            a_hat = self.alpha_bar(ts[i])
            x_0 = batch[i]
            x_t = math.sqrt(a_hat) * x_0 + math.sqrt(1 - a_hat) * epsilon[i]
            noise_imgs.append(x_t)
            
        noise_imgs = torch.stack(noise_imgs, dim=0)

        # reverse diffusion process using our model
        epsilon_hat = self.forward(noise_imgs, ts.unsqueeze(-1).type(torch.float))

        # calculate loss
        loss = F.mse_loss(input = epsilon_hat.reshape(-1, self.in_size), 
                          target = epsilon.reshape(-1, self.in_size))
        
        return loss
    
    @torch.no_grad()
    def denoise_sample(self, x_t, t):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        Denoising process from x_{t} to x_{t-1}.
        """
        x_t = x_t.to(self.device)
        if t > 1:
            z = torch.rand(size=x.shape, device=self.device)
        else:
            z = 0
            
        # calculate x_{t-1}
        epsilon_hat = self.forward(x_t, t.view(1, 1).repeat(x_t.shape[0], 1))
        pre_scale = 1 / math.sqrt(self.alpha_bar(t))
        epsilon_scale = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
        post_sigma = math.sqrt(self.beta(t)) * z
        x_t_minus_1 = pre_scale * (x_t - epsilon_scale * epsilon_hat) + post_sigma
        return x_t_minus_1
    
    def generate_sample(self):
        # config
        gif_shape = [3, 3]
        sample_batch_size = gif_shape[0] * gif_shape[1]
        n_hold_final = 10
        
        # generation process
        gen_samples = []
        x = torch.randn((sample_batch_size, self.img_depth, int(math.sqrt(self.in_size)), int(math.sqrt(self.in_size))), device=self.device)
        sample_steps = torch.arange(self.t_range - 1, 0, -1, device=self.device)
        for t in sample_steps:
            x = self.denoise_sample(x, t)
            if t % 50 == 0:
                gen_samples.append(x)
        for _ in range(n_hold_final):
            gen_samples.append(x)
        
        # post - process
        gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1) # (frame, 9, width, height)
        gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2 # (frame, 9, width, height)
        gen_samples = (gen_samples * 255).type(torch.uint8)
        gen_samples = gen_samples.reshape(-1, gif_shape[0], gif_shape[1], int(math.sqrt(self.in_size)), int(math.sqrt(self.in_size)), self.img_depth) # (frame, 3, 3, width, height, depth)
        
        return gen_samples
        
####################################################################################################
##################################### Diffusion step ###############################################
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.model_step(batch)
        self.train_loss(loss)
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss = self.model_step(batch)
        self.val_loss(loss)
        # log metrics
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
    def on_validation_epoch_end(self) -> None:
        gen_samples = self.generate_sample()
        gen_samples_np = gen_samples.to('cpu').numpy()
        
        frames = []
        for frame_idx in range(gen_samples_np.shape[0]):
            # Arrange sequences in a 3x3 grid for each frame
            # vstack and hstack like concatenate vertically and horizontally
            # :,:,: is all the data dimension
            grid = np.vstack([np.hstack(gen_samples_np[frame_idx, i, j, :, :, :] for j in range(3)) for i in range(3)]) # (30 x 30)
            frames.append(grid)

        # Save the GIF
        file_path = "data/pred.gif"
        imageio.mimsave(file_path, frames, fps=5)
        
        wandb.log({"samples": wandb.Image(file_path)})
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        pass
    
    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
        
        
if __name__ == "__main__":
    import pyrootutils
    from omegaconf import DictConfig
    import hydra
    
    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root"
    )
    
    config_path = str(path/"configs"/"model")
    
    output_path = str(path/"output")
    
    print("paths", path, config_path, output_path)

    def test_module(cfg):
        module = hydra.utils.instantiate(cfg)
        input = 2*torch.rand((32,1,32,32)) - 1
        t = torch.randint(0, 1000, size=(32, 1))
        # net = module.net
        output = module(input, t)
        print(output.shape)
        
    @hydra.main(version_base="1.3", config_path=config_path, config_name="ddpm.yaml")    
    def main(cfg: DictConfig):
        print(cfg)
        test_module(cfg)
            
    main()