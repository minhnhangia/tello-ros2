from setuptools import find_packages, setup

package_name = 'tello'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'resource/ost.txt', 'resource/ost.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tentone',
    maintainer_email='tentone@outlook.com',
    description='DJI Tello control package for ROS 2',
    license='MIT',
    tests_require=[],
    entry_points={
        'console_scripts': [
            'tello = tello.node:main'
        ],
    },
)
