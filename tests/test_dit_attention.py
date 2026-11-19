"""Attention tracing preserves real DiT sampling and validates its probabilities."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from mujoco_tools.dit_attention import DiTAttentionTrace, RGBDSpatialAttentionTrace

torch = pytest.importorskip('torch')
pytest.importorskip('diffusers')

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy, SpatialSoftmax


def make_policy(**overrides):
    options = dict(device='cpu', denoiser_type='dit',
        input_features={'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(3,)),
                        'observation.environment_state': PolicyFeature(type=FeatureType.ENV, shape=(2,))},
        output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        n_obs_steps=1, horizon=7, n_action_steps=4, drop_n_last_frames=0,
        dit_hidden_dim=32, dit_num_layers=2, dit_num_heads=4, dit_mlp_ratio=2., dit_dropout=0.,
        diffusion_step_embed_dim=16, down_dims=(16,32), n_groups=4,
        num_train_timesteps=8, num_inference_steps=4, pretrained_backbone_weights=None, push_to_hub=False)
    options.update(overrides)
    policy = DiffusionPolicy(DiffusionConfig(**options)).eval()
    if options['denoiser_type'] == 'dit':
        # The untrained AdaLN-zero model would hide perturbations behind zero
        # output weights and closed gates. Open these before parity testing.
        with torch.no_grad():
            for block in policy.diffusion.denoiser.blocks:
                block.conditioning[-1].weight.normal_(std=.04)
                block.conditioning[-1].bias.normal_(std=.1)
            policy.diffusion.denoiser.output_projection.weight.normal_(std=.15)
    return policy


def no_hooks(model):
    for module in model.modules():
        assert not module._forward_pre_hooks
        assert not module._forward_hooks


@pytest.mark.parametrize('scheduler', ['DDPM','DDIM'])
def test_actual_sampling_actions_rng_weights_and_modes_unchanged(scheduler):
    torch.manual_seed(173)
    policy = make_policy(noise_scheduler_type=scheduler)
    model = policy.diffusion
    condition = torch.arange(model.denoiser.global_cond_dim).reshape(1,-1).float()/7
    parameters = {name:value.clone() for name,value in model.state_dict().items()}
    modes = {name:module.training for name,module in model.named_modules()}
    rng = torch.get_rng_state().clone()
    with torch.inference_mode():
        baseline = model.conditional_sample(1, global_cond=condition)
        final_rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        with DiTAttentionTrace(model, timesteps=(0,6,2)) as trace:
            actual = model.conditional_sample(1, global_cond=condition)
            assert torch.equal(actual, baseline)
            assert torch.equal(torch.get_rng_state(), final_rng)
            assert trace.capture_count == 1 and trace.denoiser_call_count == 4
            capture = trace.latest
            np.testing.assert_array_equal(capture.timesteps,[6,2,0])
            np.testing.assert_array_equal(capture.sampling_timesteps,[6,4,2,0])
            assert capture.weights.shape == (3,2,4,7,7)
            assert capture.mean_weights.shape == (3,2,7,7)
            assert capture.attention_gates.shape == (3,2,32)
            assert capture.num_heads == 4 and capture.action_horizon == 7
            np.testing.assert_array_equal(capture.layer_indices,[0,1])
            np.testing.assert_allclose(capture.weights.sum(-1),1,atol=2e-7)
            np.testing.assert_array_equal(capture.mean_weights,capture.weights.mean(2))
            assert capture.reference_weights_max_abs_error.max() < 2e-7
            assert capture.live_output_max_abs_error.max() < 2e-6
            assert np.any(capture.attention_gates != 0)
            for value in capture.__dict__.values():
                if isinstance(value,np.ndarray):
                    assert not value.flags.writeable and np.isfinite(value).all()
    no_hooks(model)
    for name,value in model.state_dict().items():
        assert torch.equal(value,parameters[name])
    assert modes == {name:module.training for name,module in model.named_modules()}


def test_action_queue_retains_capture_between_predictions_and_clear_reuse():
    torch.manual_seed(3)
    policy = make_policy()
    observation = {'observation.state':torch.tensor([[.1,.2,.3]]),
                   'observation.environment_state':torch.tensor([[.4,.5]])}
    trace = DiTAttentionTrace(policy.diffusion,timesteps=(6,0))
    with torch.inference_mode(),trace:
        policy.select_action(dict(observation))
        first=trace.latest
        for _ in range(3):
            policy.select_action({key:value+.1 for key,value in observation.items()})
            assert trace.latest is first and trace.capture_count == 1
        policy.select_action(dict(observation))
        assert trace.latest is not first and trace.capture_count == 2
        trace.clear()
        assert trace.latest is None and trace.capture_count == 2
    no_hooks(policy)
    policy.reset()
    with torch.inference_mode(),trace:
        policy.select_action(dict(observation))
        assert trace.latest is not None and trace.capture_count == 3
    no_hooks(policy)


def test_live_attention_calls_keep_need_weights_false_and_run_once():
    policy=make_policy()
    seen=[]
    attention=policy.diffusion.denoiser.blocks[0].attention
    handle=attention.register_forward_pre_hook(lambda module,args,kwargs:seen.append(dict(kwargs)),with_kwargs=True)
    try:
        with torch.inference_mode(),DiTAttentionTrace(policy.diffusion,timesteps=(6,0)):
            policy.diffusion.conditional_sample(1,global_cond=torch.zeros(1,5))
        assert seen == [{'need_weights':False}]*4
        assert len(attention._forward_pre_hooks) == 1
    finally:
        handle.remove()
    no_hooks(policy)


def test_failing_denoiser_keeps_previous_capture_and_removes_hooks():
    policy=make_policy()
    trace=DiTAttentionTrace(policy.diffusion,timesteps=(6,0))
    with torch.inference_mode(),trace:
        policy.diffusion.conditional_sample(1,global_cond=torch.zeros(1,5))
        first=trace.latest
        def fail(module,args):
            raise RuntimeError('injected block failure')
        handle=policy.diffusion.denoiser.blocks[1].register_forward_pre_hook(fail)
        try:
            with pytest.raises(RuntimeError,match='injected block failure'):
                policy.diffusion.conditional_sample(1,global_cond=torch.zeros(1,5))
            assert trace.latest is first and trace.capture_count == 1
            assert not trace._handles
        finally:
            handle.remove()
    no_hooks(policy)


def test_corrupted_actual_attention_output_is_detected():
    policy=make_policy()
    attention=policy.diffusion.denoiser.blocks[0].attention
    handle=attention.register_forward_hook(lambda module,args,output:(output[0]+.05,output[1]))
    try:
        with torch.inference_mode(),DiTAttentionTrace(policy.diffusion,timesteps=(6,0)) as trace:
            with pytest.raises(ValueError,match='actual live MHA output'):
                policy.diffusion.conditional_sample(1,global_cond=torch.zeros(1,5))
            assert trace.latest is None and trace.capture_count == 0 and not trace._handles
    finally:
        handle.remove()
    no_hooks(policy)


@pytest.mark.parametrize('timesteps',[(),(-1,),(1.5,),(True,),(6,6)])
def test_bad_timestep_selection_rejected_without_hooks(timesteps):
    policy=make_policy()
    with pytest.raises(ValueError,match='timesteps'):
        DiTAttentionTrace(policy.diffusion,timesteps=timesteps)
    no_hooks(policy)


def test_requested_missing_schedule_step_rejected():
    policy=make_policy()
    with DiTAttentionTrace(policy.diffusion,timesteps=(7,0)) as trace:
        with pytest.raises(ValueError,match='absent'):
            policy.diffusion.conditional_sample(1,global_cond=torch.zeros(1,5))
        assert trace.latest is None and not trace._handles
    no_hooks(policy)


@pytest.mark.parametrize('when',['before','after'])
def test_training_mode_rejected_without_changing_mode(when):
    policy=make_policy()
    if when=='before':
        policy.train()
        with pytest.raises(ValueError,match='model.eval'):
            with DiTAttentionTrace(policy.diffusion,timesteps=(6,0)):
                pass
    else:
        with DiTAttentionTrace(policy.diffusion,timesteps=(6,0)):
            policy.train()
            with pytest.raises(ValueError,match='model.eval'):
                policy.diffusion.conditional_sample(1,global_cond=torch.zeros(1,5))
    assert policy.training
    no_hooks(policy)


def test_batch_two_rejected():
    policy=make_policy()
    with DiTAttentionTrace(policy.diffusion,timesteps=(6,0)):
        with pytest.raises(ValueError,match='batch size one'):
            policy.diffusion.conditional_sample(2,global_cond=torch.zeros(2,5))
    no_hooks(policy)


def test_unet_not_implicitly_treated_as_dit():
    policy=make_policy(denoiser_type='unet',horizon=8)
    with pytest.raises(ValueError,match='denoiser_type'):
        DiTAttentionTrace(policy.diffusion)
    no_hooks(policy)


class TinyEncoder(torch.nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.resize,self.do_crop=None,False
        self.backbone=torch.nn.Conv2d(channels,4,1)
        self.pool=SpatialSoftmax((4,2,3),num_kp=2)

    def forward(self,image):
        return self.pool(self.backbone(image)).flatten(1)


class TinyRGBDModel(torch.nn.Module):
    def __init__(self,*,depth=True,channels=None):
        super().__init__()
        rgb=PolicyFeature(type=FeatureType.VISUAL,shape=(3,2,3))
        depth_feature=PolicyFeature(type=FeatureType.VISUAL,shape=(1,2,3))
        self.config=SimpleNamespace(type='diffusion',n_obs_steps=1,use_separate_rgb_encoder_per_camera=True,
            crop_shape=None,resize_shape=None,image_features={'observation.images.head':rgb},
            input_features={'observation.images.head':rgb,'observation.depth.head':depth_feature},
            depth_camera_map={'observation.images.head':'observation.depth.head'} if depth else {},depth_unit='m')
        self.rgb_encoder=torch.nn.ModuleList([TinyEncoder(channels or (4 if depth else 3))])
        self.eval()

    def forward(self,image):
        return self.rgb_encoder[0](image)


@pytest.mark.parametrize('depth',[False,True])
def test_spatial_pooling_supports_declared_rgbd_without_changing_inputs_or_config(depth):
    torch.manual_seed(4)
    model=TinyRGBDModel(depth=depth)
    image=torch.randn(1,4 if depth else 3,2,3)
    saved=image.clone();config=deepcopy(model.config.__dict__)
    with torch.inference_mode():
        baseline=model(image)
        rng=torch.get_rng_state().clone()
        with RGBDSpatialAttentionTrace(model) as trace:
            actual=model(image)
            assert torch.equal(actual,baseline) and torch.equal(image,saved)
            assert torch.equal(torch.get_rng_state(),rng)
            assert trace.capture_count == 1
            key='observation.images.head'
            assert trace.latest.camera_weights[key].shape==(2,2,3)
            np.testing.assert_allclose(trace.latest.camera_weights[key].sum((-2,-1)),1,atol=1e-7)
            assert trace.depth_camera_map==model.config.depth_camera_map
    assert model.config.__dict__==config
    no_hooks(model)


@pytest.mark.parametrize('depth,channels',[(False,4),(True,3)])
def test_encoder_channels_must_match_declared_depth_map(depth,channels):
    model=TinyRGBDModel(depth=depth,channels=channels)
    with pytest.raises(ValueError,match='channels disagree'):
        RGBDSpatialAttentionTrace(model)
    no_hooks(model)


@pytest.mark.parametrize('setting,value,match',[
    ('depth_unit','mm','metric depth'),
    ('depth_camera_map',{'observation.images.head':'missing'},'Missing depth'),
])
def test_invalid_rgbd_declaration_rejected(setting,value,match):
    model=TinyRGBDModel()
    setattr(model.config,setting,value)
    with pytest.raises(ValueError,match=match):
        RGBDSpatialAttentionTrace(model)
    no_hooks(model)
