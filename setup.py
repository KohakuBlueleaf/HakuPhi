from setuptools import setup, find_packages


setup(
    name="hakuphi",
    packages=find_packages(exclude=["data*"]),
    version="0.0.1.dev3",
    url="https://github.com/KohakuBlueleaf/HakuPhi",
    description="A playground to place lot of things about phi.",
    author="KohakuBlueLeaf",
    author_email="apolloyeh0123@gmail.com",
    zip_safe=False,
    install_requires=["torch", "transformers", "pytorch-lightning", "einops"],
)
