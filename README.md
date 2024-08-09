# Moire Lattice & Nanotube Generator
Python scripts to create [LAMMPS](https://docs.lammps.org/2001/data_format.html) input data files for various 2D materials, including Graphene, TMDC, and hBN. 
A Python script is also available for generating TMDC nanotubes.

![](./image.png)

#### Requirements
- python ( vesrion >= 3.8)
- [numpy](https://pypi.org/project/numpy/)
- [scipy](https://pypi.org/project/scipy/)
- [matplotlib](https://pypi.org/project/matplotlib/) 
- [termcolor](https://pypi.org/project/termcolor/)

**Example :**
```sh
python moire_TMDC_rect.py -m=3 -r=1
```
Here, **m** and **r** correspond to the commensurate angle of the moire lattice.

```sh
python TMDC_nanotube.py -n=3 -m=4
```
Here, **n** and **m** correspond to the chirality of the nanotube.

#### Command line arguments
Several command line arguments are available in addition to **m** and **r**. To see more details, run the corresponding Python script with the ```--help``` flag.

#### License

[GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html)
