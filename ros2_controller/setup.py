from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'ogre_policy_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/models', glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Brad',
    maintainer_email='your@email.com',
    description='ROS2 node for running trained RL policy as Nav2 controller',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'policy_controller = ogre_policy_controller.policy_controller_node:main',
        ],
    },
)
