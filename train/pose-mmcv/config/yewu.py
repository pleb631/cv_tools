dataset_info = dict(
    dataset_name='yewu',
    paper_info=dict(
        author='Wu, Wayne and Qian, Chen and Yang, Shuo and Wang, '
        'Quan and Cai, Yici and Zhou, Qiang',
        title='Look at boundary: A boundary-aware face alignment algorithm',
        container='Proceedings of the IEEE conference on computer '
        'vision and pattern recognition',
        year='2018',
        homepage='https://wywu.github.io/projects/LAB/WFLW.html',
    ),
    keypoint_info={
        0:
        dict(
            name='kpt-0', id=0, color=[255, 255, 255], type='', swap='kpt-1'),
        1:
        dict(
            name='kpt-1', id=1, color=[255, 255, 255], type='', swap='kpt-0'),
        2:
        dict(
            name='kpt-2', id=2, color=[255, 255, 255], type='', swap=''),
        3:
        dict(
            name='kpt-3', id=3, color=[255, 255, 255], type='', swap='kpt-4'),
        4:
        dict(
            name='kpt-4', id=4, color=[255, 255, 255], type='', swap='kpt-3'),

    },
    skeleton_info={},
    joint_weights=[1.] * 5,
    sigmas=[])
