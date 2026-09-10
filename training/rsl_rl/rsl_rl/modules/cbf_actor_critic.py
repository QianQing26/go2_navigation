import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

from .cbf_lse_layer import ExactLSECBFLayer

class DifferentiableSafeActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  
                    num_actions,
                    actor_hidden_dims=[256, 256, 256],
                    critic_hidden_dims=[256, 256, 256],
                    encoder_hidden_dims=[512, 256, 128],
                    activation='elu',
                    init_noise_std=1.5,
                    num_props=12,
                    num_rays=41,
                    his_len=10,
                 **kwargs):
        super().__init__()

        activation = get_activation(activation)
        
        self.his_len = his_len
        self.num_rays = num_rays
        self.num_props = num_props
        self.num_obs_one_step = num_rays + num_props + 2  # 2 for target x,y
        self.num_obs_hist = self.num_obs_one_step * his_len
        self.num_actions = num_actions
        self.num_latent = 16
        self.enable_shield = True

        mlp_input_dim_a = self.num_obs_one_step + self.num_latent
        mlp_input_dim_c = self.num_obs_one_step + self.num_latent
        mlp_input_dim_e = self.num_obs_hist

        # 1. Shared Backbone
        backbone_layers = []
        backbone_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        backbone_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                pass
            else:
                backbone_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                backbone_layers.append(activation)
        self.backbone = nn.Sequential(*backbone_layers)

        # 2. Navigation Head
        self.nav_head = nn.Sequential(
            nn.Linear(actor_hidden_dims[-1], 128),
            activation,
            nn.Linear(128, num_actions)
        )

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)


        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(mlp_input_dim_e, encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], self.num_latent))
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)


        # 3. Adaptive Safety Head (Alpha Head)
        self.alpha_head = nn.Sequential(
            nn.Linear(actor_hidden_dims[-1], 64),
            activation,
            nn.Linear(64, 1)
        )

        # 4. Closed-form CBF Layer
        self.cbf_layer = ExactLSECBFLayer(num_rays=num_rays)

        self.std = nn.Parameter(1.5 * torch.ones(num_actions))

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    

    def extract(self, observations):
        obs_hist = observations
        obs_buf = observations[:, -self.num_obs_one_step:]
        props = obs_buf[:, :self.num_props]
        rays = obs_buf[:, self.num_props : self.num_props + self.num_rays]
        goals = obs_buf[:, -2:]
        return obs_buf, obs_hist, props, rays, goals

    def forward(self, observations, safety_drift=None, shield_rays=None):
        """
        Directly return the safety mean used to build the distribution.
        Following the original paper concept, CBF is the final layer of the network.

        The actor observation remains the delayed observation history.  When
        supplied, ``shield_rays`` is a synchronized physical-distance tensor
        used only by the shield; it does not change the actor or critic input.
        """
        obs_buf, obs_hist, props, rays, goals = self.extract(observations)
        
        latent = self.encoder(obs_hist)
        obs_cat = torch.cat((obs_buf, latent.detach()), dim=-1)
        
        # 1. Extract shared deep features
        shared_features = self.backbone(obs_cat)
        
        # 2. Dual-head parallel inference
        u_bar = self.nav_head(shared_features)
        alpha_raw = self.alpha_head(shared_features)
        
        # 3. Calculate adaptive parameter \alpha, using softplus to ensure \alpha > 0 mathematically
        rays_real = torch.exp2(rays) if shield_rays is None else shield_rays
        if not isinstance(rays_real, torch.Tensor):
            rays_real = torch.as_tensor(
                rays_real, device=observations.device, dtype=observations.dtype
            )
        else:
            rays_real = rays_real.to(
                device=observations.device, dtype=observations.dtype
            )
        expected_shape = (observations.shape[0], self.num_rays)
        if tuple(rays_real.shape) != expected_shape:
            raise ValueError(
                'shield_rays must have shape {}, got {}'.format(
                    expected_shape, tuple(rays_real.shape)
                )
            )
        alpha = F.softplus(alpha_raw)
        self.alpha = alpha 
        
        # 4. Get u_s through differentiable safety layer
        u_s = self.cbf_layer(
            u_bar, rays_real, alpha, safety_drift=safety_drift
        )
        
        # Save u_bar and u_s for calculating Intervention Loss
        self.u_bar = u_bar
        self.u_s = u_s
        
        return u_s
    
    def update_distribution(self, observations, safety_drift=None, shield_rays=None, **kwargs):
        mean = self.forward(
            observations,
            safety_drift=safety_drift,
            shield_rays=shield_rays,
        )
        self.distribution = Normal(mean, mean*0. + self.std)
        self.mean = mean

    def act(self, observations, **kwargs):
        self.update_distribution(
            observations,
            safety_drift=kwargs.pop('safety_drift', None),
            shield_rays=kwargs.pop('shield_rays', None),
            **kwargs
        )
        actions = self.distribution.sample()
        return actions

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, observations, **kwargs):
        obs_buf, obs_hist, props, rays, goals = self.extract(observations)
        latent = self.encoder(obs_hist)
        observations = torch.cat(
                            (obs_buf, latent), dim=-1) 
        value = self.critic(observations)
        return value
    
    def act_inference(self, observations, **kwargs):
        """ Used effectively during evaluation or deployment, without exploration noise """
        u_s = self.forward(
            observations,
            safety_drift=kwargs.pop('safety_drift', None),
            shield_rays=kwargs.pop('shield_rays', None),
        )
        return u_s


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
