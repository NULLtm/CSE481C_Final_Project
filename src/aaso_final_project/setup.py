import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'aaso_final_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        # Include the config folder
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')), 
    ],
    install_requires=['setuptools', 'interfaces'],
    zip_safe=True,
    maintainer='Owen Boseley',
    maintainer_email='nulltm01@gmail.com',
    description='A ROS2 Package to Play Chess!',
    license='TODO: License declaration',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'chess_driver = aaso_final_project.chess_driver:main',
            'aruco_test = aaso_final_project.aruco_driver:main'
        ],
    },
)
