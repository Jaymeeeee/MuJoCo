"""Rendering safeguards for prediction-image and fixed-scale interpretation."""
import numpy as np
import pytest

from examples.insert_plug.policy.dit_attention_render import FRAME_SIZE, render_frame, render_sensitivity_frame


def inputs():
    keys = ('observation.images.head', 'observation.images.wrist')
    return dict(live_images={key:np.full((256,256,3),30,dtype=np.uint8) for key in keys},
        source_images={key:np.full((256,256,3),80,dtype=np.uint8) for key in keys},
        spatial_maps={key:np.full((8,8),1/64,dtype=np.float32) for key in keys},
        keypoints={key:np.zeros((32,2),dtype=np.float32) for key in keys},
        grid_shapes={key:(8,8) for key in keys},
        action_attention=np.full((64,64),1/64,dtype=np.float32),
        action_offset=4,time_s=3.6,source_time_s=3.2,chunk_index=1)


def test_source_overlay_does_not_follow_changed_live_camera_during_cached_action():
    data=inputs()
    first=render_frame(**data)
    data['live_images']={key:np.full((256,256,3),230,dtype=np.uint8) for key in data['live_images']}
    second=render_frame(**data)
    assert first.shape==(FRAME_SIZE[1],FRAME_SIZE[0],3)
    assert first.dtype==np.uint8
    assert not np.array_equal(first[120:440,20:340],second[120:440,20:340])
    assert np.array_equal(first[485:805,20:676],second[485:805,20:676])


def test_other_row_peak_does_not_rescale_unchanged_matrix_cells():
    data=inputs()
    first=render_frame(**data)
    data['action_attention'][20]=0
    data['action_attention'][20,40]=1
    second=render_frame(**data)
    # q=8, k=8 is unchanged and outside the query highlight and axis boundary.
    assert np.array_equal(first[171+8*8+4,774+8*8+4],second[171+8*8+4,774+8*8+4])
    assert not np.array_equal(first[171+20*8+4,774+40*8+4],second[171+20*8+4,774+40*8+4])


def test_rejects_nonprobability_attention_and_unexecuted_query():
    data=inputs()
    data['action_attention'][0]*=2
    with pytest.raises(ValueError,match='sum to one'):
        render_frame(**data)
    data=inputs();data['action_offset']=32
    with pytest.raises(ValueError,match='0..31'):
        render_frame(**data)


def sensitivity_inputs():
    data=inputs()
    return dict(live_images=data['live_images'],source_images=data['source_images'],
        arm_maps={key:np.full((4,4),.5,dtype=np.float32) for key in data['source_images']},
        gripper_maps={key:np.full((4,4),2.,dtype=np.float32) for key in data['source_images']},
        arm_vmax_deg=1.,gripper_vmax_mm=4.,time_s=3.6,source_time_s=3.2,chunk_index=1)


def test_sensitivity_keeps_source_image_and_physical_metric_scales_separate():
    data=sensitivity_inputs()
    first=render_sensitivity_frame(**data)
    data['live_images']={key:np.full((256,256,3),240,dtype=np.uint8) for key in data['live_images']}
    data['gripper_vmax_mm']=8.
    second=render_sensitivity_frame(**data)
    assert first.shape==(FRAME_SIZE[1],FRAME_SIZE[0],3)
    assert not np.array_equal(first[118:374,20:276],second[118:374,20:276])
    # Changing a millimetre scale must not recolor degree-valued source maps.
    assert np.array_equal(first[477:797,20:676],second[477:797,20:676])
    assert not np.array_equal(first[477:797,692:1348],second[477:797,692:1348])


def test_sensitivity_accepts_unnormalized_magnitudes_but_rejects_negative_effects():
    data=sensitivity_inputs()
    assert render_sensitivity_frame(**data).dtype==np.uint8
    data['arm_maps']['observation.images.head'][0,0]=-.01
    with pytest.raises(ValueError,match='nonnegative'):
        render_sensitivity_frame(**data)
