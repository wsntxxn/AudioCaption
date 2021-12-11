import setuptools

setuptools.setup(
    name="audiocaptioning",
    version="0.0.1",
    author="Xuenan Xu",
    author_email="wsntxxn@gmail.com",
    install_requires=[
        "efficient_latent@git+https://github.com/richermans/HEAR2021_EfficientLatent.git"
    ],
    packages=setuptools.find_packages(),
)
