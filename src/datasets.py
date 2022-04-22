from collections import namedtuple
from pathlib import Path

# Dataset root directory
_DATASET_ROOT = Path('../data')

Dataset = namedtuple('Dataset', ['name', 'src', 'bug_repo', 'repo_url', 'features'])

# Source codes and bug repositories

aspectj = Dataset(
    'aspectj',
    _DATASET_ROOT / 'org.aspectj-bug433351/',
    _DATASET_ROOT / 'AspectJ.txt',
    "https://github.com/eclipse/org.aspectj/tree/bug433351.git",
    _DATASET_ROOT / 'features_aspectj/'
)

eclipse = Dataset(
    'eclipse',
    _DATASET_ROOT / 'eclipse.platform.ui-johna-402445/',
    _DATASET_ROOT / 'Eclipse_Platform_UI.txt',
    "https://github.com/eclipse/eclipse.platform.ui.git",
    _DATASET_ROOT / 'features_eclipse/'
)

swt = Dataset(
    'swt',
    _DATASET_ROOT / 'eclipse.platform.swt-xulrunner-31/',
    _DATASET_ROOT / 'SWT.txt',
    "https://github.com/eclipse/eclipse.platform.swt.git",
    _DATASET_ROOT / 'features_swt/'
)

tomcat = Dataset(
    'tomcat',
    _DATASET_ROOT / 'tomcat-7.0.51/',
    _DATASET_ROOT / 'Tomcat.txt',
    "https://github.com/apache/tomcat.git",
    _DATASET_ROOT / 'features_tomcat/'
)

jdt = Dataset(
    'jdt',
    _DATASET_ROOT / 'eclipse.jdt.ui-mmathew-BETA_JAVA8/',
    _DATASET_ROOT / 'JDT.txt',
    "https://github.com/eclipse/eclipse.jdt.ui/tree/mmathew/BETA_JAVA8.git",
    _DATASET_ROOT / 'features_jdt/'
)

birt = Dataset(
    'birt',
    _DATASET_ROOT / 'birt-20140211-1400/',
    _DATASET_ROOT / 'Birt.txt',
    "https://github.com/eclipse/birt.git",
    _DATASET_ROOT / 'features_birt/'
)

### Current dataset in use. (change this name to change the dataset)
DATASET = tomcat

if __name__ == '__main__':
    print(DATASET.name, DATASET.src, DATASET.bug_repo)
