from setuptools import setup, find_packages

exec(open('optimal_step_nicp/version.py').read())

setup(
    name='optimal-step-nicp',
    packages=find_packages(),
    version=__version__,
    include_package_data=True,
    package_data={
        'optimal_step_nicp': ['data/*', 'models/*', 'config/*'],
    },
    entry_points={
        'console_scripts': [
            'optimal-step-nicp=optimal_step_nicp.demo:main',
        ],
    },
    license='MIT',
    description='Optimal Step Non-Rigid ICP for mesh registration',
    author='Kirill Goltsman',
    author_email='goltsmank@gmail.com',
    url='h',
    long_description_content_type='text/markdown',
    keywords=['3d mesh registration', 'machine learning', 'non-rigid icp'],
    install_requires=[
        'torch>=2.0',
        'open3d==0.18.0',
        'matplotlib',
        'trimesh',
        'numpy',
        'pillow',
        'scipy',
        'scikit-learn ',
        'tqdm',
        'mediapipe',
        'opencv-python',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :  MIT License',
        'Programming Language :: Python :: 3.11.9',
    ],
)
