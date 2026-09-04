from setuptools import find_packages, setup

package_name = "grippers_vla"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ubuntu",
    maintainer_email="11306260+liangfuyuan@user.noreply.gitee.com",
    description="VLA 정책 추론 노드 — 정책이 낸 관절 청크를 arm_driver 로 재생시킨다",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vla_inference_node = grippers_vla.vla_inference_node:main",
        ],
    },
)
